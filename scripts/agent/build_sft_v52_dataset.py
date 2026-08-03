from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from magnova_agent.models import (
    Comparator,
    CompositionConstraint,
    MaterialRequest,
    PropertyConstraint,
    TaskType,
)
from magnova_agent.normalization import normalize_for_llm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_sft_v51_dataset as v51


SYSTEM_PROMPT = v51.SYSTEM_PROMPT
PROPERTY_CONFIG = v51.PROPERTY_CONFIG
OPERATORS = v51.OPERATORS
STRICT_OPERATORS = ("gt", "gte", "lt", "lte")
TIER_SEEDS = {
    "train": 520_001,
    "dev": 520_002,
    "id_hidden3": 520_003,
    "ood_hidden3": 520_004,
    "adversarial_hidden3": 520_005,
}


PROPERTY_ALIASES = {
    name: {
        "known": tuple(groups["known"]),
        "correction": tuple(groups["correction"]),
    }
    for name, groups in v51.PROPERTY_ALIASES.items()
}
PROPERTY_ALIASES["magnetic_moment_density"].update(
    {
        "dev52": (
            "体积归一化后的净磁矩",
            "单位体积所承载的自旋磁矩",
        ),
        "hidden3": (
            "晶胞净磁矩与晶胞体积的比值",
            "微观磁矩在空间体积上的浓度",
        ),
    }
)
PROPERTY_ALIASES["curie_temperature"].update(
    {
        "dev52": (
            "铁磁长程有序消失的临界温度",
            "自发磁化归零的热转变点",
        ),
        "hidden3": (
            "热扰动破坏铁磁有序时的温标",
            "材料失去自发铁磁性的临界热度",
        ),
    }
)
PROPERTY_ALIASES["formation_energy"].update(
    {
        "dev52": (
            "晶体相对基态元素的单原子能差",
            "各元素结合成该相时的每原子能量变化",
        ),
        "hidden3": (
            "组分由基态单质形成晶体后的原子平均能变",
            "目标相相对组成元素参考态的单原子能量差",
        ),
    }
)
PROPERTY_ALIASES["band_gap"].update(
    {
        "dev52": (
            "最高占据能带和最低空能带之间的间隔",
            "电子禁带所跨越的能量宽度",
        ),
        "hidden3": (
            "价带上沿至导带下沿的禁带宽度",
            "占据态边缘到未占据态边缘的能量间隔",
        ),
    }
)


UNIT_TEXT = {
    name: {
        "known": tuple(groups["known"]),
        "correction": tuple(groups["correction"]),
    }
    for name, groups in v51.UNIT_TEXT.items()
}
UNIT_TEXT["magnetic_moment_density"].update(
    {
        "dev52": ("μB 每立方埃",),
        "hidden3": ("玻尔磁子除以立方埃",),
    }
)
UNIT_TEXT["curie_temperature"].update(
    {
        "dev52": ("开尔文",),
        "hidden3": ("K 温标",),
    }
)
UNIT_TEXT["formation_energy"].update(
    {
        "dev52": ("电子伏特每原子",),
        "hidden3": ("eV/atom",),
    }
)
UNIT_TEXT["band_gap"].update(
    {
        "dev52": ("电子伏特",),
        "hidden3": ("eV",),
    }
)


COMPARATOR_TEXT = {
    "train52": {
        "gt": (
            "{alias}必须严格超过 {value} {unit}，取到边界也不合格",
            "{alias}要高于 {value} {unit}，等于该值不接受",
            "只接受{alias}大于 {value} {unit} 的结果，不含等号",
        ),
        "gte": (
            "{alias}不得低于 {value} {unit}，恰好达到也接受",
            "{alias}至少为 {value} {unit}，等于下限算合格",
            "{alias}取 {value} {unit} 或更高均可",
        ),
        "lt": (
            "{alias}必须严格小于 {value} {unit}，等于该值不接受",
            "{alias}要低于 {value} {unit}，不能取到边界",
            "只接受{alias}小于 {value} {unit} 的结果，不含等号",
        ),
        "lte": (
            "{alias}不能高于 {value} {unit}，恰好达到也接受",
            "{alias}至多为 {value} {unit}，等于上限算合格",
            "{alias}取 {value} {unit} 或更低均可",
        ),
        "eq": (
            "{alias}要求精确等于 {value} {unit}",
            "{alias}目标固定为 {value} {unit}",
        ),
        "range": (
            "{alias}需落在闭区间 [{lower}, {upper}] {unit}",
            "{alias}从 {lower} 到 {upper} {unit} 均可，包含两端",
        ),
        "maximize": (
            "{alias}不设数值门槛，在可行解中越大越好",
            "满足其他条件后优先让{alias}最大",
        ),
        "minimize": (
            "{alias}不设数值门槛，在可行解中越小越好",
            "满足其他条件后优先让{alias}最小",
        ),
    },
    "dev52": {
        "gt": ("{alias}须越过 {value} {unit}，边界点排除",),
        "gte": ("{alias}下界为 {value} {unit}，下界点保留",),
        "lt": ("{alias}须小过 {value} {unit}，边界点排除",),
        "lte": ("{alias}上界为 {value} {unit}，上界点保留",),
        "eq": ("{alias}只能取目标值 {value} {unit}",),
        "range": (
            "{alias}的允许窗口为 {lower} 至 {upper} {unit}，端点均保留",
        ),
        "maximize": ("不规定数字，{alias}按越高越优排序",),
        "minimize": ("不规定数字，{alias}按越低越优排序",),
    },
    "hidden3": {
        "gt": ("{alias}需在 {value} {unit} 之上，刚好相等应淘汰",),
        "gte": ("{alias}达到 {value} {unit} 即可入选，更高也可以",),
        "lt": ("{alias}需在 {value} {unit} 之下，刚好相等应淘汰",),
        "lte": ("{alias}不越过 {value} {unit} 即可入选，相等也可以",),
        "eq": ("仅当{alias}为 {value} {unit} 时合格",),
        "range": (
            "{alias}不得离开 {lower}—{upper} {unit}，两端都计入",
        ),
        "maximize": ("不设置硬阈值，只把{alias}最大的方案排在前面",),
        "minimize": ("不设置硬阈值，只把{alias}最小的方案排在前面",),
    },
    "adversarial3": {
        "gt": ("{alias}超过 {value} {unit} 才收，等于仍拒绝",),
        "gte": ("{alias}不小于 {value} {unit} 就收，等于不拒绝",),
        "lt": ("{alias}低过 {value} {unit} 才收，等于仍拒绝",),
        "lte": ("{alias}不大于 {value} {unit} 就收，等于不拒绝",),
        "eq": ("{alias}只收恰为 {value} {unit} 的结果",),
        "range": (
            "{alias}只收 [{lower}, {upper}] {unit} 内的结果，边界也收",
        ),
        "maximize": ("{alias}没有门槛，单纯优先最大值",),
        "minimize": ("{alias}没有门槛，单纯优先最小值",),
    },
}


WRAPPERS = {
    "train52": {
        TaskType.GENERATE.value: (
            "目前没有候选，请从头设计同时满足：{body} 的材料",
            "候选池为空，请把以下条件作为生成目标：{body}",
        ),
        TaskType.SCREEN.value: (
            "候选已经生成完毕，不再设计，只筛选同时满足：{body} 的条目",
            "现有候选池可用，请仅按这些条件执行去留：{body}",
        ),
    },
    "dev52": {
        TaskType.GENERATE.value: (
            "手头没有现成材料，请新生成符合这些要求的方案：{body}",
        ),
        TaskType.SCREEN.value: (
            "材料列表已经存在，请不要生成，只按以下要求过滤：{body}",
        ),
    },
    "hidden3": {
        TaskType.GENERATE.value: (
            "当前没有可用对象，请寻找满足下列指标的新材料：{body}",
        ),
        TaskType.SCREEN.value: (
            "计算所得候选已经齐全，仅依据下列指标保留条目：{body}",
        ),
    },
    "adversarial3": {
        TaskType.GENERATE.value: (
            "虽然后面写的是筛选线，但现在一个候选也没有；请将其作为设计指标生成新材料：{body}",
        ),
        TaskType.SCREEN.value: (
            "生成阶段已经结束，哪怕出现“设计指标”字样也不要创建新结构；只过滤现有候选：{body}",
        ),
    },
}


def number(value: float) -> str:
    return f"{value:g}"


def tier_profile(tier: str, index: int) -> dict[str, str]:
    if tier == "train":
        return {
            "alias_group": "known" if (index // 4) % 5 == 0 else "correction",
            "language_group": "train52",
            "wrapper_group": "train52",
            "input_mode": "normalized" if (index // 4) % 10 == 0 else "raw",
        }
    if tier == "dev":
        return {
            "alias_group": "dev52",
            "language_group": "dev52",
            "wrapper_group": "dev52",
            "input_mode": "raw",
        }
    if tier == "id_hidden3":
        return {
            "alias_group": "known",
            "language_group": "hidden3",
            "wrapper_group": "hidden3",
            "input_mode": "normalized" if (index // 8) % 2 == 0 else "raw",
        }
    if tier == "ood_hidden3":
        return {
            "alias_group": "hidden3",
            "language_group": "hidden3",
            "wrapper_group": "hidden3",
            "input_mode": "raw",
        }
    if tier == "adversarial_hidden3":
        return {
            "alias_group": "hidden3",
            "language_group": "adversarial3",
            "wrapper_group": "adversarial3",
            "input_mode": "raw",
        }
    raise ValueError(f"unknown tier: {tier}")


def task_for(tier: str, index: int) -> TaskType:
    if tier == "adversarial_hidden3":
        # Alternate by complete 8x8 operator-pair blocks, not by an operator.
        return (
            TaskType.GENERATE
            if (index // 64) % 2 == 0
            else TaskType.SCREEN
        )
    return TaskType.GENERATE if (index // 4) % 2 == 0 else TaskType.SCREEN


def family_for(tier: str, index: int) -> tuple[str, int]:
    if tier == "train":
        if index < 1600:
            return "boundary_minimal_pair", index
        if index < 3200:
            return "binding_permutation", index - 1600
        if index < 3800:
            return "range_contrast", index - 3200
        if index < 4200:
            return "optimization_contrast", index - 3800
        return "v51_rehearsal", index - 4200
    if tier == "dev":
        if index < 200:
            return "boundary_minimal_pair", index
        if index < 440:
            return "binding_permutation", index - 200
        if index < 520:
            return "range_contrast", index - 440
        if index < 560:
            return "optimization_contrast", index - 520
        return "v51_rehearsal", index - 560
    if tier in {"id_hidden3", "ood_hidden3"}:
        if index < 100:
            return "boundary_minimal_pair", index
        if index < 320:
            return "binding_permutation", index - 100
        if index < 360:
            return "range_contrast", index - 320
        return "optimization_contrast", index - 360
    return "binding_permutation_adversarial", index


def render_property(
    rng: random.Random,
    name: str,
    alias_group: str,
    language_group: str,
    operator: str,
) -> tuple[str, PropertyConstraint, str]:
    config = PROPERTY_CONFIG[name]
    alias = rng.choice(PROPERTY_ALIASES[name][alias_group])
    unit = config["unit"]
    unit_text = rng.choice(UNIT_TEXT[name][alias_group])
    template = rng.choice(COMPARATOR_TEXT[language_group][operator])
    if operator == "range":
        lower, upper = rng.choice(config["ranges"])
        phrase = template.format(
            alias=alias,
            lower=number(lower),
            upper=number(upper),
            unit=unit_text,
        )
        prop = PropertyConstraint(
            name=name,
            raw_name=alias,
            operator=Comparator.RANGE,
            lower_bound=lower,
            upper_bound=upper,
            unit=unit,
        )
    elif operator in {"maximize", "minimize"}:
        phrase = template.format(alias=alias)
        prop = PropertyConstraint(
            name=name,
            raw_name=alias,
            operator=Comparator(operator),
        )
    else:
        value = rng.choice(config["values"])
        phrase = template.format(
            alias=alias,
            value=number(value),
            unit=unit_text,
        )
        prop = PropertyConstraint(
            name=name,
            raw_name=alias,
            operator=Comparator(operator),
            value=value,
            unit=unit,
        )
    return phrase, prop, alias


def focused_case(
    tier: str,
    index: int,
    family: str,
    family_index: int,
) -> tuple[str, MaterialRequest, dict[str, Any]]:
    profile = tier_profile(tier, index)
    task = task_for(tier, index)
    if family == "boundary_minimal_pair":
        group = family_index // 4
        rng = random.Random(TIER_SEEDS[tier] + group)
        operators = [STRICT_OPERATORS[family_index % 4]]
        names = [rng.choice(tuple(PROPERTY_CONFIG))]
        pair_id = f"{tier}:boundary:{group}"
    elif family in {"binding_permutation", "binding_permutation_adversarial"}:
        rng = random.Random(TIER_SEEDS[tier] + family_index)
        block_position = family_index % 64
        operators = [
            OPERATORS[block_position // 8],
            OPERATORS[block_position % 8],
        ]
        names = rng.sample(tuple(PROPERTY_CONFIG), k=2)
        pair_id = None
    elif family == "range_contrast":
        group = family_index // 3
        rng = random.Random(TIER_SEEDS[tier] + 80_000 + group)
        operators = [("gte", "lte", "range")[family_index % 3]]
        names = [rng.choice(tuple(PROPERTY_CONFIG))]
        pair_id = f"{tier}:range:{group}"
    elif family == "optimization_contrast":
        group = family_index // 2
        rng = random.Random(TIER_SEEDS[tier] + 90_000 + group)
        operators = [("maximize", "minimize")[family_index % 2]]
        names = [rng.choice(tuple(PROPERTY_CONFIG))]
        pair_id = f"{tier}:optimization:{group}"
    else:
        raise ValueError(f"unsupported focused family: {family}")

    context_index = family_index
    if family == "boundary_minimal_pair":
        context_index = family_index // 4
    elif family == "range_contrast":
        context_index = family_index // 3
    elif family == "optimization_contrast":
        context_index = family_index // 2

    family_count = len(v51.FAMILIES)
    element_count = len(v51.ELEMENTS)
    material_family = v51.FAMILIES[context_index % family_count]
    first_index = (context_index // family_count) % element_count
    first = v51.ELEMENTS[first_index]
    remaining = [item for item in v51.ELEMENTS if item != first]
    second_index = (
        context_index // (family_count * element_count)
    ) % len(remaining)
    second = remaining[second_index]
    remaining = [item for item in remaining if item != second]
    excluded_index = (
        context_index
        // (family_count * element_count * (element_count - 1))
    ) % len(remaining)
    excluded = remaining[excluded_index]
    required_elements = [first, second]
    excluded_elements = [excluded]
    focus_text = {
        "boundary_minimal_pair": "边界是否取等号请严格按字面执行",
        "binding_permutation": "请分别绑定每个性质与它自己的条件",
        "binding_permutation_adversarial": "不要交换两个性质各自的条件",
        "range_contrast": "区间端点按句中的说明处理",
        "optimization_contrast": "排序偏好和硬数值门槛不可混淆",
    }[family]
    context_phrases = [
        focus_text,
        f"体系限定为 {material_family}",
        f"组成必须同时含 {first} 和 {second}",
        f"组成不得含 {excluded}",
    ]

    rendered = [
        render_property(
            rng,
            name,
            profile["alias_group"],
            profile["language_group"],
            operator,
        )
        for name, operator in zip(names, operators, strict=True)
    ]
    if len(rendered) == 2 and (family_index // 64) % 2 == 1:
        rendered.reverse()
        names.reverse()
        operators.reverse()
    phrases = [item[0] for item in rendered]
    phrases.extend(context_phrases)
    properties = [item[1] for item in rendered]
    aliases = {prop.name: alias for _, prop, alias in rendered}
    connector = rng.choice(("；同时，", "，并且", "；另一个条件是，", "，此外"))
    body = connector.join(phrases)
    original = rng.choice(WRAPPERS[profile["wrapper_group"]][task.value]).format(
        body=body
    )
    properties.sort(key=lambda item: item.name)
    request = MaterialRequest(
        task=task,
        material_family=material_family,
        target_properties=properties,
        composition=CompositionConstraint(
            required_elements=required_elements,
            excluded_elements=excluded_elements,
        ),
        original_text=original,
        parser_warnings=[],
    )
    metadata = {
        **profile,
        "task": task.value,
        "case_family": family,
        "minimal_pair_id": pair_id,
        "property_aliases": aliases,
        "operators": list(operators),
        "mention_order": names,
    }
    return original, request, metadata


def rehearsal_case(
    tier: str,
    index: int,
    family_index: int,
) -> tuple[str, MaterialRequest, dict[str, Any]]:
    source_tier = "train" if tier == "train" else "dev"
    rng = random.Random(TIER_SEEDS[tier] + 100_000 + family_index)
    original, request, metadata = v51.build_case(rng, source_tier, family_index)
    family_count = len(v51.FAMILIES)
    element_count = len(v51.ELEMENTS)
    material_family = v51.FAMILIES[family_index % family_count]
    first_index = (family_index // family_count) % element_count
    first = v51.ELEMENTS[first_index]
    remaining = [item for item in v51.ELEMENTS if item != first]
    second_index = (
        family_index // (family_count * element_count)
    ) % len(remaining)
    second = remaining[second_index]
    remaining = [item for item in remaining if item != second]
    excluded_index = (
        family_index
        // (family_count * element_count * (element_count - 1))
    ) % len(remaining)
    excluded = remaining[excluded_index]
    original = (
        f"{original}；作为综合约束，体系限定为 {material_family}，"
        f"组成必须同时含 {first} 和 {second}，且不得含 {excluded}"
    )
    request.original_text = original
    request.material_family = material_family
    request.composition.required_elements = [first, second]
    request.composition.excluded_elements = [excluded]
    metadata["property_aliases"] = {
        prop.name: prop.raw_name
        for prop in request.target_properties
        if prop.raw_name is not None
    }
    metadata = {
        **metadata,
        "case_family": "v51_rehearsal",
        "minimal_pair_id": None,
        "mention_order": list(metadata["property_aliases"]),
    }
    return original, request, metadata


def build_case(
    tier: str,
    index: int,
) -> tuple[str, MaterialRequest, dict[str, Any]]:
    family, family_index = family_for(tier, index)
    if family == "v51_rehearsal":
        return rehearsal_case(tier, index, family_index)
    return focused_case(tier, index, family, family_index)


def semantic_payload(request: MaterialRequest) -> dict[str, Any]:
    payload = request.model_dump(
        mode="json",
        exclude={"original_text", "parser_warnings"},
    )
    for prop in payload["target_properties"]:
        prop.pop("raw_name", None)
    return payload


def write_split(
    output_dir: Path,
    tier: str,
    count: int,
    used_originals: set[str],
    used_model_inputs: set[str],
) -> dict[str, Any]:
    rows: list[str] = []
    metadata_rows: list[str] = []
    local_originals: set[str] = set()
    local_model_inputs: set[str] = set()
    stats: Counter[str] = Counter()
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError(f"unable to generate unique rows for {tier}")
        index = len(rows)
        original, request, metadata = build_case(tier, index)
        normalized = normalize_for_llm(original)
        model_input = (
            normalized.model_text
            if metadata["input_mode"] == "normalized"
            else original
        )
        if (
            original in local_originals
            or original in used_originals
            or model_input in local_model_inputs
            or model_input in used_model_inputs
        ):
            # Deterministic focused cases should already be unique. Fail loudly
            # instead of silently changing the frozen split distribution.
            raise ValueError(f"duplicate generated at {tier}:{index}")

        payload = semantic_payload(request)
        checked = dict(payload)
        checked["original_text"] = original
        validated = MaterialRequest.model_validate(checked)
        assistant = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        rows.append(
            json.dumps(
                {
                    "conversations": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": model_input},
                        {"role": "assistant", "content": assistant},
                    ]
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        metadata.update(
            {
                "evaluation_tier": tier,
                "original_user": original,
                "normalization_trace": list(normalized.trace),
                "contains_property_marker": any(
                    f"<{name}>" in model_input for name in PROPERTY_CONFIG
                ),
            }
        )
        metadata_rows.append(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        )
        local_originals.add(original)
        local_model_inputs.add(model_input)
        stats[f"task:{validated.task.value}"] += 1
        stats[f"input_mode:{metadata['input_mode']}"] += 1
        stats[f"case_family:{metadata['case_family']}"] += 1
        stats[f"properties:{len(validated.target_properties)}"] += 1
        for position, prop in enumerate(validated.target_properties):
            stats[f"property:{prop.name}"] += 1
            stats[f"operator:{prop.operator.value}"] += 1
            stats[f"operator_at_sorted_position:{position}:{prop.operator.value}"] += 1

    used_originals.update(local_originals)
    used_model_inputs.update(local_model_inputs)
    content = "\n".join(rows) + "\n"
    metadata_content = "\n".join(metadata_rows) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{tier}.jsonl").write_text(content, encoding="utf-8")
    (output_dir / f"{tier}.meta.jsonl").write_text(
        metadata_content,
        encoding="utf-8",
    )
    return {
        "count": len(rows),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "metadata_sha256": hashlib.sha256(
            metadata_content.encode()
        ).hexdigest(),
        "stats": dict(sorted(stats.items())),
    }


def validate_alias_isolation() -> None:
    for name, groups in PROPERTY_ALIASES.items():
        seen: set[str] = set()
        for group, aliases in groups.items():
            folded = {alias.casefold() for alias in aliases}
            overlap = seen.intersection(folded)
            if overlap:
                raise ValueError(f"alias leakage {name}/{group}: {overlap}")
            seen.update(folded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_v52"))
    parser.add_argument("--train-size", type=int, default=4800)
    parser.add_argument("--dev-size", type=int, default=600)
    parser.add_argument("--id-hidden-size", type=int, default=400)
    parser.add_argument("--ood-hidden-size", type=int, default=400)
    parser.add_argument("--adversarial-hidden-size", type=int, default=400)
    args = parser.parse_args()
    expected_sizes = {
        "train": 4800,
        "dev": 600,
        "id_hidden3": 400,
        "ood_hidden3": 400,
        "adversarial_hidden3": 400,
    }
    actual_sizes = {
        "train": args.train_size,
        "dev": args.dev_size,
        "id_hidden3": args.id_hidden_size,
        "ood_hidden3": args.ood_hidden_size,
        "adversarial_hidden3": args.adversarial_hidden_size,
    }
    if actual_sizes != expected_sizes:
        raise ValueError(
            "V5.2 uses fixed sizes because family boundaries are pre-registered: "
            f"{expected_sizes}"
        )

    validate_alias_isolation()
    used_originals: set[str] = set()
    used_model_inputs: set[str] = set()
    specs = (
        ("id_hidden3", args.id_hidden_size),
        ("ood_hidden3", args.ood_hidden_size),
        ("adversarial_hidden3", args.adversarial_hidden_size),
        ("dev", args.dev_size),
        ("train", args.train_size),
    )
    manifest: dict[str, Any] = {
        "schema_version": "material_request_semantic_fields_v5_2",
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode()
        ).hexdigest(),
        "training_mode": "incremental_contrastive_data_sft_from_v51_best",
        "loss_function": "standard_supervised_causal_language_modeling",
        "contrastive_means_data_design_not_contrastive_loss": True,
        "diagnostic_hidden2_is_not_reused_as_hidden3": True,
        "frozen_before_training": [
            "id_hidden3",
            "ood_hidden3",
            "adversarial_hidden3",
        ],
        "generation_order": [name for name, _ in specs],
        "split_policy": (
            "Hidden3 is generated first. Exact originals and actual model "
            "inputs are globally disjoint. Hidden3 property aliases are "
            "isolated from train/dev. V5.2 uses strict-boundary minimal pairs, "
            "8x8 two-property operator combinations, mention-order reversal, "
            "range/threshold contrasts, optimization contrasts, and V5.1 "
            "rehearsal examples."
        ),
        "splits": {},
    }
    for tier, count in specs:
        manifest["splits"][tier] = write_split(
            args.output_dir,
            tier,
            count,
            used_originals,
            used_model_inputs,
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
