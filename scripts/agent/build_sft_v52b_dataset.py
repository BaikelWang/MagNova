from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from magnova_agent.models import (
    Comparator,
    CompositionConstraint,
    MaterialRequest,
    PropertyConstraint,
    TaskType,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_sft_v52_dataset as v52


SYSTEM_PROMPT = """你是 MagNova 材料需求解析器。用户输入既可能是原始自然语言，也可能含有 <canonical_name> 规范标记；没有标记时也必须依据完整语义识别需求。把需求转换成一个合法 JSON 对象，不解释、不使用 Markdown。
只输出 task、material_family、target_properties、composition、free_text_constraints。
task 只能是 generate_material、screen_candidates 或 predict_properties。当前没有候选、需要提出新组成或从零设计时用 generate_material；已经有候选且只要求保留、淘汰、过滤或核验时用 screen_candidates。
target_properties 每项只能包含 name、operator、value、lower_bound、upper_bound、unit，并按 name 字母顺序排列；不要输出 raw_name。operator 语义必须严格遵守：gt 表示严格大于 >，gte 表示大于等于 >=，lt 表示严格小于 <，lte 表示小于等于 <=，eq 表示精确等于，range 表示包含上下端点的闭区间，maximize 表示无硬阈值并尽量增大，minimize 表示无硬阈值并尽量减小。gt、gte、lt、lte、eq 使用 value 且上下界为 null；range 的 value 为 null 并填写 lower_bound、upper_bound；maximize、minimize 的 value 和上下界都为 null。
规范性质名只有 magnetic_moment_density、curie_temperature、formation_energy、band_gap。不能猜测用户未给出的数值或单位；缺失值用 null，缺失列表用 []。
composition 只能包含 required_elements、excluded_elements、exclude_rare_earth。free_text_constraints 只保留规范短语：室温稳定、低成本、无毒、可合成。
不要输出 original_text、parser_warnings、raw_name 或任何其他字段。"""

PROPERTY_CONFIG = v52.PROPERTY_CONFIG
OPERATORS = v52.OPERATORS
STRICT_OPERATORS = v52.STRICT_OPERATORS
SEEDS = {
    "train": 521_001,
    "dev": 521_002,
    "id_hidden4": 521_003,
    "ood_hidden4": 521_004,
    "adversarial_hidden4": 521_005,
}


PROPERTY_ALIASES = {
    name: {
        "known": tuple(groups["known"]),
        "correction": tuple(groups["correction"]),
    }
    for name, groups in v52.PROPERTY_ALIASES.items()
}
PROPERTY_ALIASES["magnetic_moment_density"].update(
    {
        "dev52b": ("净磁矩除以材料体积", "每单位空间体积的磁偶极矩"),
        "hidden4": ("晶体磁矩的体积归一值", "体积尺度上的净自旋磁矩强度"),
    }
)
PROPERTY_ALIASES["curie_temperature"].update(
    {
        "dev52b": ("自发铁磁序的热学终点", "铁磁相转为顺磁相的临界温标"),
        "hidden4": ("长程铁磁排列被热涨落破坏的温度", "铁磁性随升温消退的转变点"),
    }
)
PROPERTY_ALIASES["formation_energy"].update(
    {
        "dev52b": ("相对元素基态的平均原子生成能", "晶相生成时每个原子的能量差"),
        "hidden4": ("元素参考态结合为目标相的原子平均能变", "目标晶相相对单质基准的每原子能差"),
    }
)
PROPERTY_ALIASES["band_gap"].update(
    {
        "dev52b": ("导带底与价带顶的能量距离", "电子态谱中的禁带宽度"),
        "hidden4": ("最低未占据带到最高占据带的能量间距", "价带边缘和导带边缘之间的空能区宽度"),
    }
)


UNIT_TEXT = {
    name: {
        "known": tuple(groups["known"]),
        "correction": tuple(groups["correction"]),
    }
    for name, groups in v52.UNIT_TEXT.items()
}
UNIT_TEXT["magnetic_moment_density"].update(
    {"dev52b": ("μB/Å³",), "hidden4": ("玻尔磁子每立方埃",)}
)
UNIT_TEXT["curie_temperature"].update(
    {"dev52b": ("K",), "hidden4": ("绝对温标 K",)}
)
UNIT_TEXT["formation_energy"].update(
    {"dev52b": ("eV/atom",), "hidden4": ("每原子电子伏特",)}
)
UNIT_TEXT["band_gap"].update(
    {"dev52b": ("eV",), "hidden4": ("电子伏特",)}
)


COMPARATOR_TEXT = {
    "train52b": {
        "gt": (
            "{alias}严格大于 {value} {unit}",
            "{alias}必须超过 {value} {unit}，相等不算",
            "{alias}需高于 {value} {unit} 且不含边界",
            "只收{alias}在 {value} {unit} 之上的结果",
        ),
        "gte": (
            "{alias}大于等于 {value} {unit}",
            "{alias}不得低于 {value} {unit}，相等可算",
            "{alias}至少达到 {value} {unit}",
            "{alias}以 {value} {unit} 为含端点下界",
        ),
        "lt": (
            "{alias}严格小于 {value} {unit}",
            "{alias}必须低过 {value} {unit}，相等不算",
            "{alias}需低于 {value} {unit} 且不含边界",
            "只收{alias}在 {value} {unit} 之下的结果",
        ),
        "lte": (
            "{alias}小于等于 {value} {unit}",
            "{alias}不得高于 {value} {unit}，相等可算",
            "{alias}至多达到 {value} {unit}",
            "{alias}以 {value} {unit} 为含端点上界",
        ),
        "eq": ("{alias}精确等于 {value} {unit}", "{alias}固定为 {value} {unit}"),
        "range": (
            "{alias}位于闭区间 [{lower}, {upper}] {unit}",
            "{alias}介于 {lower} 和 {upper} {unit}，两端均包含",
        ),
        "maximize": ("不设阈值，{alias}越大越好", "优先最大化{alias}"),
        "minimize": ("不设阈值，{alias}越小越好", "优先最小化{alias}"),
    },
    "dev52b": {
        "gt": ("{alias}要越过 {value} {unit}，等号不取",),
        "gte": ("{alias}从 {value} {unit} 起算，边界保留",),
        "lt": ("{alias}要小过 {value} {unit}，等号不取",),
        "lte": ("{alias}到 {value} {unit} 为止，边界保留",),
        "eq": ("{alias}只接受目标点 {value} {unit}",),
        "range": ("{alias}限制在含端点的 {lower}—{upper} {unit}",),
        "maximize": ("{alias}不设门槛并按高值优先",),
        "minimize": ("{alias}不设门槛并按低值优先",),
    },
    "hidden4": {
        "gt": ("{alias}需超出 {value} {unit}，卡在该值不合格",),
        "gte": ("{alias}达到 {value} {unit} 便合格，以上也行",),
        "lt": ("{alias}需低过 {value} {unit}，卡在该值不合格",),
        "lte": ("{alias}不超过 {value} {unit} 便合格，等于也行",),
        "eq": ("{alias}仅取 {value} {unit} 这个点",),
        "range": ("{alias}处于 {lower} 到 {upper} {unit} 的闭合窗口",),
        "maximize": ("不给定数字，只让{alias}尽可能高",),
        "minimize": ("不给定数字，只让{alias}尽可能低",),
    },
    "adversarial4": {
        "gt": ("{alias}越过 {value} {unit} 才保留，恰等时删除",),
        "gte": ("{alias}不低过 {value} {unit} 就保留，恰等时也保留",),
        "lt": ("{alias}小过 {value} {unit} 才保留，恰等时删除",),
        "lte": ("{alias}不高过 {value} {unit} 就保留，恰等时也保留",),
        "eq": ("{alias}与 {value} {unit} 完全一致才保留",),
        "range": ("{alias}落在 [{lower}, {upper}] {unit} 就保留，端点有效",),
        "maximize": ("{alias}没有硬边界，只做降序优选",),
        "minimize": ("{alias}没有硬边界，只做升序优选",),
    },
}


WRAPPERS = {
    "train52b": {
        TaskType.GENERATE.value: ("目前无候选，请设计满足{body}的新材料",),
        TaskType.SCREEN.value: ("已有候选，不再生成，只保留满足{body}的条目",),
    },
    "dev52b": {
        TaskType.GENERATE.value: ("现无材料可选，请从零寻找满足{body}的组成",),
        TaskType.SCREEN.value: ("候选已备好，请仅筛出满足{body}的材料",),
    },
    "hidden4": {
        TaskType.GENERATE.value: ("手头没有方案，请提出符合{body}的新材料",),
        TaskType.SCREEN.value: ("现有清单已经生成，请按{body}执行去留",),
    },
    "adversarial4": {
        TaskType.GENERATE.value: ("虽然给了筛选条件，但当前池为空，请转为设计目标生成满足{body}的材料",),
        TaskType.SCREEN.value: ("虽然条件可用于设计，但当前候选已生成，请不要新建，只筛选满足{body}的项",),
    },
}


def number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def deterministic_value(name: str, code: int) -> float:
    if name == "band_gap":
        return round(0.4 + (code % 3200) * 0.001, 3)
    if name == "curie_temperature":
        return float(220 + code % 1100)
    if name == "formation_energy":
        return round(-2.2 + (code % 2000) * 0.001, 3)
    if name == "magnetic_moment_density":
        return round(0.03 + (code % 1200) * 0.001, 3)
    raise ValueError(name)


def profile(tier: str, index: int) -> dict[str, str]:
    if tier == "train":
        return {
            "alias_group": "known" if (index // 4) % 5 == 0 else "correction",
            "language_group": "train52b",
            "wrapper_group": "train52b",
            "input_mode": "normalized" if (index // 4) % 10 == 0 else "raw",
        }
    if tier == "dev":
        return {"alias_group": "dev52b", "language_group": "dev52b", "wrapper_group": "dev52b", "input_mode": "raw"}
    if tier == "id_hidden4":
        return {"alias_group": "known", "language_group": "hidden4", "wrapper_group": "hidden4", "input_mode": "normalized" if (index // 8) % 2 == 0 else "raw"}
    if tier == "ood_hidden4":
        return {"alias_group": "hidden4", "language_group": "hidden4", "wrapper_group": "hidden4", "input_mode": "raw"}
    if tier == "adversarial_hidden4":
        return {"alias_group": "hidden4", "language_group": "adversarial4", "wrapper_group": "adversarial4", "input_mode": "raw"}
    raise ValueError(tier)


def task_for(tier: str, index: int) -> TaskType:
    if tier == "adversarial_hidden4":
        return TaskType.GENERATE if (index // 64) % 2 == 0 else TaskType.SCREEN
    return TaskType.GENERATE if (index // 4) % 2 == 0 else TaskType.SCREEN


def family_for(tier: str, index: int) -> tuple[str, int]:
    if tier == "train":
        bounds = ((1600, "boundary_minimal_pair"), (3200, "binding_permutation"), (3800, "range_contrast"), (4200, "optimization_contrast"), (4800, "v51_rehearsal"))
    elif tier == "dev":
        bounds = ((200, "boundary_minimal_pair"), (440, "binding_permutation"), (520, "range_contrast"), (560, "optimization_contrast"), (600, "v51_rehearsal"))
    elif tier in {"id_hidden4", "ood_hidden4"}:
        bounds = ((100, "boundary_minimal_pair"), (320, "binding_permutation"), (360, "range_contrast"), (400, "optimization_contrast"))
    else:
        return "binding_permutation_adversarial", index
    previous = 0
    for bound, family in bounds:
        if index < bound:
            return family, index - previous
        previous = bound
    raise ValueError((tier, index))


def render_property(rng: random.Random, name: str, alias_group: str, language_group: str, operator: str, code: int) -> tuple[str, PropertyConstraint, str]:
    alias = rng.choice(PROPERTY_ALIASES[name][alias_group])
    unit = PROPERTY_CONFIG[name]["unit"]
    unit_text = rng.choice(UNIT_TEXT[name][alias_group])
    template = rng.choice(COMPARATOR_TEXT[language_group][operator])
    if operator == "range":
        lower = deterministic_value(name, code)
        width = {"band_gap": 0.4, "curie_temperature": 120.0, "formation_energy": 0.35, "magnetic_moment_density": 0.18}[name]
        upper = round(lower + width, 3)
        phrase = template.format(alias=alias, lower=number(lower), upper=number(upper), unit=unit_text)
        prop = PropertyConstraint(name=name, raw_name=alias, operator=Comparator.RANGE, lower_bound=lower, upper_bound=upper, unit=unit)
    elif operator in {"maximize", "minimize"}:
        phrase = template.format(alias=alias)
        prop = PropertyConstraint(name=name, raw_name=alias, operator=Comparator(operator))
    else:
        value = deterministic_value(name, code)
        phrase = template.format(alias=alias, value=number(value), unit=unit_text)
        prop = PropertyConstraint(name=name, raw_name=alias, operator=Comparator(operator), value=value, unit=unit)
    return phrase, prop, alias


def focused_case(tier: str, index: int, family: str, family_index: int) -> tuple[str, MaterialRequest, dict[str, Any]]:
    cfg = profile(tier, index)
    task = task_for(tier, index)
    if family == "boundary_minimal_pair":
        group = family_index // 4
        rng = random.Random(SEEDS[tier] + group)
        operators = [STRICT_OPERATORS[family_index % 4]]
        names = [tuple(PROPERTY_CONFIG)[group % len(PROPERTY_CONFIG)]]
        codes = [group * 7 + 3]
        pair_id = f"{tier}:boundary:{group}"
    elif family in {"binding_permutation", "binding_permutation_adversarial"}:
        rng = random.Random(SEEDS[tier] + family_index)
        pair_position = family_index % 64
        operators = [OPERATORS[pair_position // 8], OPERATORS[pair_position % 8]]
        names = rng.sample(tuple(PROPERTY_CONFIG), k=2)
        codes = [family_index * 2 + 11, family_index * 2 + 12]
        pair_id = None
    elif family == "range_contrast":
        group = family_index // 3
        rng = random.Random(SEEDS[tier] + 80_000 + group)
        operators = [("gte", "lte", "range")[family_index % 3]]
        names = [tuple(PROPERTY_CONFIG)[group % len(PROPERTY_CONFIG)]]
        codes = [group * 9 + 5]
        pair_id = f"{tier}:range:{group}"
    elif family == "optimization_contrast":
        group = family_index // 2
        rng = random.Random(SEEDS[tier] + 90_000 + group)
        operators = [("maximize", "minimize")[family_index % 2]]
        names = [tuple(PROPERTY_CONFIG)[group % len(PROPERTY_CONFIG)]]
        codes = [group]
        pair_id = f"{tier}:optimization:{group}"
    else:
        raise ValueError(family)

    rendered = [render_property(rng, name, cfg["alias_group"], cfg["language_group"], operator, code) for name, operator, code in zip(names, operators, codes, strict=True)]
    if len(rendered) == 2 and (family_index // 64) % 2 == 1:
        rendered.reverse()
        names.reverse()
        operators.reverse()
    family_prefix = {
        "boundary_minimal_pair": "请严格判断是否包含边界",
        "binding_permutation": "各性质分别绑定各自条件",
        "binding_permutation_adversarial": "两个性质的条件不可互换",
        "range_contrast": "请区分单边阈值与闭区间",
        "optimization_contrast": "请区分无阈值排序偏好",
    }[family]
    phrases = [family_prefix, *(item[0] for item in rendered)]
    properties = [item[1] for item in rendered]
    aliases = {prop.name: alias for _, prop, alias in rendered}
    material_family = None
    composition = CompositionConstraint()
    if family == "optimization_contrast":
        group = family_index // 2
        family_count = len(v52.v51.FAMILIES)
        element_count = len(v52.v51.ELEMENTS)
        material_family = v52.v51.FAMILIES[group % family_count]
        required_index = (group // family_count) % element_count
        required = v52.v51.ELEMENTS[required_index]
        remaining = [item for item in v52.v51.ELEMENTS if item != required]
        excluded_index = (
            group // (family_count * element_count)
        ) % len(remaining)
        excluded = remaining[excluded_index]
        phrases.extend((f"体系为 {material_family}", f"必须含 {required}", f"排除 {excluded}"))
        composition = CompositionConstraint(required_elements=[required], excluded_elements=[excluded])
    body = rng.choice(("，并且", "；同时", "、", "，另要求")).join(phrases)
    original = rng.choice(WRAPPERS[cfg["wrapper_group"]][task.value]).format(body=body)
    properties.sort(key=lambda item: item.name)
    request = MaterialRequest(task=task, material_family=material_family, target_properties=properties, composition=composition, original_text=original, parser_warnings=[])
    metadata = {**cfg, "task": task.value, "case_family": family, "minimal_pair_id": pair_id, "property_aliases": aliases, "operators": list(operators), "mention_order": names}
    return original, request, metadata


def rehearsal_case(tier: str, family_index: int) -> tuple[str, MaterialRequest, dict[str, Any]]:
    source_tier = "train" if tier == "train" else "dev"
    rng = random.Random(SEEDS[tier] + 100_000 + family_index)
    original, request, metadata = v52.v51.build_case(rng, source_tier, family_index + 10_000)
    family_count = len(v52.v51.FAMILIES)
    element_count = len(v52.v51.ELEMENTS)
    material_family = v52.v51.FAMILIES[family_index % family_count]
    first_index = (family_index // family_count) % element_count
    first = v52.v51.ELEMENTS[first_index]
    remaining = [item for item in v52.v51.ELEMENTS if item != first]
    second_index = (
        family_index // (family_count * element_count)
    ) % len(remaining)
    second = remaining[second_index]
    remaining = [item for item in remaining if item != second]
    excluded_index = (
        family_index // (family_count * element_count * (element_count - 1))
    ) % len(remaining)
    excluded = remaining[excluded_index]
    original = (
        f"{original}；综合约束：体系为 {material_family}，"
        f"必须含 {first} 和 {second}，排除 {excluded}"
    )
    request.original_text = original
    request.material_family = material_family
    request.composition.required_elements = [first, second]
    request.composition.excluded_elements = [excluded]
    metadata = {**metadata, "case_family": "v51_rehearsal", "minimal_pair_id": None, "mention_order": list(metadata["property_aliases"])}
    return original, request, metadata


def build_case(tier: str, index: int) -> tuple[str, MaterialRequest, dict[str, Any]]:
    family, family_index = family_for(tier, index)
    if family == "v51_rehearsal":
        return rehearsal_case(tier, family_index)
    return focused_case(tier, index, family, family_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_v52b"))
    args = parser.parse_args()
    v52.SYSTEM_PROMPT = SYSTEM_PROMPT
    v52.PROPERTY_ALIASES = PROPERTY_ALIASES
    v52.build_case = build_case
    v52.validate_alias_isolation()
    specs = (("id_hidden4", 400), ("ood_hidden4", 400), ("adversarial_hidden4", 400), ("dev", 600), ("train", 4800))
    used_originals: set[str] = set()
    used_model_inputs: set[str] = set()
    manifest: dict[str, Any] = {
        "schema_version": "material_request_semantic_fields_v5_2b",
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "training_mode": "incremental_focused_sft_from_v51_best",
        "parent_weight": "magnova_sft_v51_768.pth",
        "v52_degraded_branch_is_not_parent": True,
        "frozen_before_training": ["id_hidden4", "ood_hidden4", "adversarial_hidden4"],
        "generation_order": [name for name, _ in specs],
        "split_policy": "Hidden4 is frozen first and aliases are isolated. Focused samples remove unrelated composition load, use deterministic continuous values for uniqueness, strict-boundary minimal pairs, 8x8 operator pairs, and 12.5% V5.1 rehearsal.",
        "splits": {},
    }
    for tier, count in specs:
        manifest["splits"][tier] = v52.write_split(args.output_dir, tier, count, used_originals, used_model_inputs)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
