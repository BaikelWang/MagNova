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
import build_sft_v5_dataset as v5


SYSTEM_PROMPT = v5.SYSTEM_PROMPT
PROPERTY_CONFIG = v5.PROPERTY_CONFIG
FAMILIES = v5.FAMILIES
ELEMENTS = v5.ELEMENTS


PROPERTY_ALIASES = {
    name: {
        "known": tuple(groups["known"]),
        "correction": tuple(
            dict.fromkeys(groups["train_ood"] + groups["hidden_ood"])
        ),
    }
    for name, groups in v5.PROPERTY_ALIASES.items()
}
PROPERTY_ALIASES["magnetic_moment_density"].update(
    {
        "dev51": ("每一体积单元对应的磁矩", "磁矩按体积归一后的数值"),
        "hidden2": ("磁偶极矩的体积浓度", "按晶胞体积折算的总磁矩"),
    }
)
PROPERTY_ALIASES["curie_temperature"].update(
    {
        "dev51": ("铁磁有序终止的临界温度", "自发磁化热消失点"),
        "hidden2": ("铁磁序发生热退化的温度点", "自发磁性终止温度"),
    }
)
PROPERTY_ALIASES["formation_energy"].update(
    {
        "dev51": ("由单质生成晶体的单位原子能差", "相对元素参考态的稳定能"),
        "hidden2": (
            "由基态单质构成晶体时的单位原子能变",
            "相对组成元素参考态的能量收益",
        ),
    }
)
PROPERTY_ALIASES["band_gap"].update(
    {
        "dev51": ("价带顶和导带底之间的能量窗口", "占据态到空态的最低能量差"),
        "hidden2": (
            "价带最高点与导带最低点之间的能量窗",
            "电子占据带和空带之间的能量间隔",
        ),
    }
)


UNIT_TEXT = {
    name: {
        "known": tuple(groups["known"]),
        "correction": tuple(
            dict.fromkeys(groups["train_ood"] + groups["hidden_ood"])
        ),
    }
    for name, groups in v5.UNIT_TEXT.items()
}
UNIT_TEXT["magnetic_moment_density"].update(
    {
        "dev51": ("μB 按每 Å³ 计",),
        "hidden2": ("μB 除以 Å 的三次方", "玻尔磁子·立方埃的倒数"),
    }
)
UNIT_TEXT["curie_temperature"].update(
    {
        "dev51": ("热力学绝对温标",),
        "hidden2": ("绝对温标单位",),
    }
)
UNIT_TEXT["formation_energy"].update(
    {
        "dev51": ("每原子所对应的电子伏特",),
        "hidden2": ("每个原子对应的 eV",),
    }
)
UNIT_TEXT["band_gap"].update(
    {
        "dev51": ("电子伏特制",),
        "hidden2": ("eV 能量制",),
    }
)


COMPARATOR_TEXT = {
    "correction": {
        "gt": (
            "必须超过但不能等于",
            "只接受严格超过",
            "必须落在阈值之上，等于不合格",
            "严格大于",
        ),
        "gte": (
            "允许等于下限且不能更低",
            "等于门槛也算合格，不能更低",
            "达到该数或更高均可",
            "不得小于",
            "起码达到",
        ),
        "lt": (
            "必须低于且不取等号",
            "只接受严格低于",
            "只能处于该值之下，不能相等",
            "严格小于",
        ),
        "lte": (
            "可等于上限但不能更高",
            "等于上限也算合格，不能更高",
            "不高过该上限，等于可以",
            "不得超过",
        ),
        "eq": (
            "精确目标设为",
            "数值需要正好是",
            "要求与给定数值一致",
        ),
        "maximize": (
            "在可行范围内数值越大越优",
            "其他条件满足时优先最大",
            "以取得更大数值为优化方向",
        ),
        "minimize": (
            "在可行范围内数值越小越优",
            "其他条件满足时优先最小",
            "以取得更小数值为优化方向",
        ),
    },
    "dev51": {
        "gt": ("需要高过该阈值且不包括相等",),
        "gte": ("下限取该值并允许相等",),
        "lt": ("需要小过该阈值且不包括相等",),
        "lte": ("上限取该值并允许相等",),
        "eq": ("应当恰好达到",),
        "maximize": ("可行解中优先数值最大的",),
        "minimize": ("可行解中优先数值最小的",),
    },
    "id2": {
        "gt": ("高于给定数值且相等不算",),
        "gte": ("至少达到给定数值，相等可接受",),
        "lt": ("低于给定数值且相等不算",),
        "lte": ("最多达到给定数值，相等可接受",),
        "eq": ("目标与给定数字相同",),
        "maximize": ("在约束内追求尽量大的结果",),
        "minimize": ("在约束内追求尽量小的结果",),
    },
    "hidden2": {
        "gt": ("必须落在该阈值之上，等于也不行",),
        "gte": ("达到该数或更高均可",),
        "lt": ("只能处于该值之下，不能相等",),
        "lte": ("不高过该上限，等于可以",),
        "eq": ("要求与给定数值一致",),
        "maximize": ("满足其余条件后取尽可能大的值",),
        "minimize": ("满足其余条件后取尽可能小的值",),
    },
    "adversarial2": {
        "gt": ("超过门槛才合格，刚好等于不合格",),
        "gte": ("不低于门槛即可，刚好等于合格",),
        "lt": ("低过门槛才合格，刚好等于不合格",),
        "lte": ("不高于门槛即可，刚好等于合格",),
        "eq": ("仅接受与目标数字相同的值",),
        "maximize": ("不设具体数字，以最大者优先",),
        "minimize": ("不设具体数字，以最小者优先",),
    },
}


RANGE_TEXT = {
    "correction": (
        "{alias}闭区间限定为 [{lower}, {upper}]",
        "{alias}需要处于包含端点的 {lower} 至 {upper} 区间",
        "{alias}上下界分别为 {lower} 和 {upper}，两个端点都允许",
    ),
    "dev51": (
        "{alias}可接受窗口从 {lower} 到 {upper}，包含上下界",
    ),
    "id2": (
        "{alias}限制在含端点的 {lower}—{upper} 范围",
    ),
    "hidden2": (
        "{alias}上下界都包含在 {lower} 至 {upper} 的可接受窗口内",
    ),
    "adversarial2": (
        "{alias}不得越出 [{lower}, {upper}]，上下端点均合格",
    ),
}


WRAPPERS = {
    "correction": {
        "generate_material": (
            "候选池目前为空，无法筛选，请生成满足{body}的新材料",
            "现在没有现成结果，筛选阈值就是设计目标，请研发满足{body}的方案",
            "当前没有任何可用候选，请从零提出符合{body}的材料",
        ),
        "screen_candidates": (
            "生成步骤已经结束，只在现成清单中按{body}淘汰",
            "已有候选可用，不要重新生成，只保留满足{body}的项",
            "设计阶段已经完成，请对当前结果按{body}做准入判断",
        ),
    },
    "dev51": {
        "generate_material": (
            "暂时无材料可筛，请把{body}作为目标提出全新组成",
        ),
        "screen_candidates": (
            "候选计算已结束，不创建新项，只按{body}过滤现有结果",
        ),
    },
    "id2": {
        "generate_material": (
            "手头没有可用方案，请从头设计符合{body}的材料",
        ),
        "screen_candidates": (
            "材料池已经备好，请依据{body}留下合格对象",
        ),
    },
    "hidden2": {
        "generate_material": (
            "数据库中还没有可筛选条目，请寻找满足{body}的新组成",
        ),
        "screen_candidates": (
            "现有条目已经齐备，仅根据{body}执行去留判断",
        ),
    },
    "adversarial2": {
        "generate_material": (
            "现在没有候选，不能做过滤；请把这些筛选阈值转成设计目标并生成满足{body}的材料",
        ),
        "screen_candidates": (
            "新结构生成工作已经完成，不再设计；仅从已有集合中筛出满足{body}的对象",
        ),
    },
}


RARE_EARTH = {
    "known": v5.RARE_EARTH["known"],
    "correction": tuple(
        dict.fromkeys(v5.RARE_EARTH["train_ood"] + v5.RARE_EARTH["hidden_ood"])
    ),
    "dev51": ("配方对所有稀土元素执行零容忍",),
    "hidden2": ("任何属于稀土族的元素都不能进入组成",),
}


FREE_TEXT = {
    "known": v5.FREE_TEXT["known"],
    "correction": {
        canonical: tuple(
            dict.fromkeys(
                v5.FREE_TEXT["train_ood"][canonical]
                + v5.FREE_TEXT["hidden_ood"][canonical]
            )
        )
        for canonical in v5.FREE_TEXT["known"]
    },
    "dev51": {
        "室温稳定": ("普通室内温度下相态保持不变",),
        "低成本": ("原料和加工的总费用要低",),
        "无毒": ("使用与制备过程不能产生毒害",),
        "可合成": ("必须存在现实可执行的合成路径",),
    },
    "hidden2": {
        "室温稳定": ("在约 298 K 环境下保持相稳定",),
        "低成本": ("采购和制造两部分开销都要小",),
        "无毒": ("不得对生物体产生有害毒性",),
        "可合成": ("实验室工艺应当能够真正制得",),
    },
}


OPERATORS = (
    "gt",
    "gte",
    "lt",
    "lte",
    "eq",
    "range",
    "maximize",
    "minimize",
)


def number(value: float) -> str:
    return f"{value:g}"


def profile_for(tier: str, index: int) -> dict[str, str]:
    if tier == "train":
        focuses = (
            "operator",
            "operator",
            "binding",
            "binding",
            "task_contrast",
            "task_contrast",
            "constraints",
            "constraints",
        )
        position = index % 16
        return {
            "input_mode": "normalized" if position >= 14 else "raw",
            "alias_group": "known" if position >= 14 else "correction",
            "language_group": "correction",
            "wrapper_group": "correction",
            "case_family": (
                "marked_reinforcement"
                if position >= 14
                else focuses[(position // 2) % len(focuses)]
            ),
            "task": (
                TaskType.GENERATE.value
                if position % 2 == 0
                else TaskType.SCREEN.value
            ),
        }
    if tier == "dev":
        position = index % 4
        return {
            "input_mode": "normalized" if position == 0 else "raw",
            "alias_group": "known" if position == 0 else "dev51",
            "language_group": "dev51",
            "wrapper_group": "dev51",
            "case_family": ("marked_reinforcement" if position == 0 else "focused_dev"),
            "task": (
                TaskType.GENERATE.value
                if position in (0, 1)
                else TaskType.SCREEN.value
            ),
        }
    if tier == "id_hidden2":
        position = index % 4
        return {
            "input_mode": "normalized" if position in (0, 2) else "raw",
            "alias_group": "known",
            "language_group": "id2",
            "wrapper_group": "id2",
            "case_family": (
                "marked_known" if position in (0, 2) else "raw_known"
            ),
            "task": (
                TaskType.GENERATE.value
                if position in (0, 1)
                else TaskType.SCREEN.value
            ),
        }
    if tier == "ood_hidden2":
        return {
            "input_mode": "raw",
            "alias_group": "hidden2",
            "language_group": "hidden2",
            "wrapper_group": "hidden2",
            "case_family": "raw_hidden2",
            "task": (
                TaskType.GENERATE.value
                if index % 2 == 0
                else TaskType.SCREEN.value
            ),
        }
    if tier == "adversarial_hidden2":
        return {
            "input_mode": "raw",
            "alias_group": "hidden2",
            "language_group": "adversarial2",
            "wrapper_group": "adversarial2",
            "case_family": "task_contrast_hidden2",
            "task": (
                TaskType.GENERATE.value
                if index % 2 == 0
                else TaskType.SCREEN.value
            ),
        }
    raise ValueError(f"unknown tier: {tier}")


def property_phrase(
    rng: random.Random,
    name: str,
    alias_group: str,
    language_group: str,
    operator: str,
    omit_unit: bool,
) -> tuple[str, PropertyConstraint, str]:
    config = PROPERTY_CONFIG[name]
    alias = rng.choice(PROPERTY_ALIASES[name][alias_group])
    unit_text = None if omit_unit else rng.choice(UNIT_TEXT[name][alias_group])
    unit = None if omit_unit else config["unit"]
    if operator == "range":
        lower, upper = rng.choice(config["ranges"])
        phrase = rng.choice(RANGE_TEXT[language_group]).format(
            alias=alias,
            lower=number(lower),
            upper=number(upper),
        )
        if unit_text:
            phrase += f" {unit_text}"
        prop = PropertyConstraint(
            name=name,
            raw_name=alias,
            operator=Comparator.RANGE,
            lower_bound=lower,
            upper_bound=upper,
            unit=unit,
        )
    elif operator in ("maximize", "minimize"):
        phrase = f"{alias}{rng.choice(COMPARATOR_TEXT[language_group][operator])}"
        prop = PropertyConstraint(
            name=name,
            raw_name=alias,
            operator=Comparator(operator),
        )
    else:
        value = rng.choice(config["values"])
        phrase = (
            f"{alias}{rng.choice(COMPARATOR_TEXT[language_group][operator])} "
            f"{number(value)}"
        )
        if unit_text:
            phrase += f" {unit_text}"
        prop = PropertyConstraint(
            name=name,
            raw_name=alias,
            operator=Comparator(operator),
            value=value,
            unit=unit,
        )
    return phrase, prop, alias


def build_case(
    rng: random.Random,
    tier: str,
    index: int,
) -> tuple[str, MaterialRequest, dict[str, Any]]:
    profile = profile_for(tier, index)
    task = TaskType(profile["task"])
    missing_property = tier in {"train", "dev", "id_hidden2"} and index % 47 == 0
    missing_unit = not missing_property and index % 13 == 0
    if missing_property:
        property_count = 0
    elif profile["case_family"] in {
        "binding",
        "task_contrast",
        "task_contrast_hidden2",
    }:
        property_count = 2
    else:
        property_count = 2 if rng.random() < 0.65 else 1

    phrases: list[str] = []
    properties: list[PropertyConstraint] = []
    aliases: dict[str, str] = {}
    operators: list[str] = []
    if property_count:
        names = rng.sample(tuple(PROPERTY_CONFIG), k=property_count)
        for property_index, name in enumerate(names):
            operator = OPERATORS[(index + property_index * 3) % len(OPERATORS)]
            phrase, prop, alias = property_phrase(
                rng,
                name,
                profile["alias_group"],
                profile["language_group"],
                operator,
                omit_unit=missing_unit and property_index == 0,
            )
            phrases.append(phrase)
            properties.append(prop)
            aliases[name] = alias
            operators.append(operator)
    properties.sort(key=lambda item: item.name)

    family = rng.choice(FAMILIES) if rng.random() < 0.22 else None
    required: list[str] = []
    excluded: list[str] = []
    force_constraints = profile["case_family"] == "constraints"
    exclude_rare_earth = force_constraints or rng.random() < 0.42
    free_constraints: list[str] = []
    if family:
        phrases.append(f"材料体系限定为 {family}")
    if rng.random() < 0.22:
        required = rng.sample(ELEMENTS[:-1], k=2)
        phrases.append(f"组成必须同时含有 {' 和 '.join(required)}")
    if rng.random() < 0.22:
        available = [item for item in ELEMENTS if item not in required]
        excluded = [rng.choice(available)]
        phrases.append(f"配方中排除元素 {excluded[0]}")
    if exclude_rare_earth:
        phrases.append(rng.choice(RARE_EARTH[profile["alias_group"]]))
    if force_constraints or rng.random() < 0.42:
        canonical = rng.choice(tuple(FREE_TEXT[profile["alias_group"]]))
        free_constraints.append(canonical)
        phrases.append(rng.choice(FREE_TEXT[profile["alias_group"]][canonical]))
    if not phrases:
        phrases.append("适用于后续材料研究")

    rng.shuffle(phrases)
    body = rng.choice(("，并且", "；同时", "、", "，还要")).join(phrases)
    original = rng.choice(WRAPPERS[profile["wrapper_group"]][task.value]).format(
        body=body
    )
    request = MaterialRequest(
        task=task,
        material_family=family,
        target_properties=properties,
        composition=CompositionConstraint(
            required_elements=required,
            excluded_elements=excluded,
            exclude_rare_earth=exclude_rare_earth,
        ),
        free_text_constraints=free_constraints,
        original_text=original,
        parser_warnings=[],
    )
    metadata = {
        **profile,
        "property_aliases": aliases,
        "operators": operators,
    }
    return original, request, metadata


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
    seed: int,
    used_originals: set[str],
    used_model_inputs: set[str],
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows: list[str] = []
    metadata_rows: list[str] = []
    local_originals: set[str] = set()
    local_model_inputs: set[str] = set()
    stats: Counter[str] = Counter()
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 600:
            raise RuntimeError(f"unable to generate unique rows for {tier}")
        index = len(rows)
        original, request, metadata = build_case(rng, tier, index)
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
            continue
        payload = semantic_payload(request)
        check = dict(payload)
        check["original_text"] = original
        validated = MaterialRequest.model_validate(check)
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
        stats[f"alias_group:{metadata['alias_group']}"] += 1
        stats[f"properties:{len(validated.target_properties)}"] += 1
        for prop in validated.target_properties:
            stats[f"property:{prop.name}"] += 1
            stats[f"operator:{prop.operator.value}"] += 1

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
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_v51"))
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--dev-size", type=int, default=400)
    parser.add_argument("--id-hidden-size", type=int, default=400)
    parser.add_argument("--ood-hidden-size", type=int, default=400)
    parser.add_argument("--adversarial-hidden-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    validate_alias_isolation()
    used_originals: set[str] = set()
    used_model_inputs: set[str] = set()
    specs = (
        ("id_hidden2", args.id_hidden_size),
        ("ood_hidden2", args.ood_hidden_size),
        ("adversarial_hidden2", args.adversarial_hidden_size),
        ("dev", args.dev_size),
        ("train", args.train_size),
    )
    manifest: dict[str, Any] = {
        "schema_version": "material_request_semantic_fields_v5_1",
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode()
        ).hexdigest(),
        "training_mode": "incremental_correction_sft_from_v5_best",
        "diagnostic_v5_hidden_is_not_reused_as_hidden2": True,
        "frozen_before_training": [
            "id_hidden2",
            "ood_hidden2",
            "adversarial_hidden2",
        ],
        "generation_order": [name for name, _ in specs],
        "task_input_factorization": (
            "Task and marked/raw input mode are crossed independently in ID hidden2."
        ),
        "split_policy": (
            "Hidden2 tiers are generated first. Exact originals and actual model "
            "inputs are globally disjoint. Hidden2 aliases are isolated from "
            "correction train/dev. Correction data targets operator semantics, "
            "multi-property binding, task contrast, and constraint paraphrases."
        ),
        "splits": {},
    }
    for offset, (tier, count) in enumerate(specs):
        manifest["splits"][tier] = write_split(
            args.output_dir,
            tier,
            count,
            args.seed + offset,
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
