from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from magnova_agent.models import CompositionConstraint, MaterialRequest, TaskType

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_sft_v52_dataset as writer
import build_sft_v52b_dataset as v52b


SYSTEM_PROMPT = v52b.SYSTEM_PROMPT
PROPERTY_CONFIG = v52b.PROPERTY_CONFIG
OPERATORS = v52b.OPERATORS
SEEDS = {"train": 530_001, "dev": 530_002, "id_hidden5": 530_003, "ood_hidden5": 530_004, "adversarial_hidden5": 530_005}


PROPERTY_ALIASES = {
    name: {
        "known": tuple(groups["known"]),
        "correction": tuple(groups["correction"]),
        "train53": tuple(groups["dev52b"]),
    }
    for name, groups in v52b.PROPERTY_ALIASES.items()
}
PROPERTY_ALIASES["magnetic_moment_density"].update({"dev53": ("磁矩与体积相除后的强度",), "hidden5": ("单位晶体体积对应的总自旋磁矩", "材料净磁矩的空间密度")})
PROPERTY_ALIASES["curie_temperature"].update({"dev53": ("铁磁有序的最高存续温度",), "hidden5": ("自发磁化能够维持到的热学上限", "铁磁长程关联消失时的温度点")})
PROPERTY_ALIASES["formation_energy"].update({"dev53": ("晶体生成相对元素基准的每原子能差",), "hidden5": ("从基态元素形成目标晶相的原子平均能量变化", "该相相对各单质参考能的单原子差值")})
PROPERTY_ALIASES["band_gap"].update({"dev53": ("价带顶到导带底的能量空窗",), "hidden5": ("最高已占据带与最低空带之间的禁带距离", "电子能带中无允许态区域的宽度")})


UNIT_TEXT = {
    name: {
        "known": tuple(groups["known"]),
        "correction": tuple(groups["correction"]),
        "train53": tuple(groups["dev52b"]),
    }
    for name, groups in v52b.UNIT_TEXT.items()
}
UNIT_TEXT["magnetic_moment_density"].update({"dev53": ("μB/Å³",), "hidden5": ("玻尔磁子·立方埃的倒数",)})
UNIT_TEXT["curie_temperature"].update({"dev53": ("K",), "hidden5": ("开尔文",)})
UNIT_TEXT["formation_energy"].update({"dev53": ("eV/atom",), "hidden5": ("电子伏特每原子",)})
UNIT_TEXT["band_gap"].update({"dev53": ("eV",), "hidden5": ("电子伏特",)})


LANGUAGE = {
    "train53": v52b.COMPARATOR_TEXT["train52b"],
    "dev53": v52b.COMPARATOR_TEXT["dev52b"],
    "hidden5": {
        "gt": ("{alias}必须跨过 {value} {unit}，停在边界不收",),
        "gte": ("{alias}从 {value} {unit} 开始均合格，边界也收",),
        "lt": ("{alias}必须落到 {value} {unit} 以下，停在边界不收",),
        "lte": ("{alias}截至 {value} {unit} 均合格，边界也收",),
        "eq": ("{alias}必须正好落在 {value} {unit}",),
        "range": ("{alias}应在含两端的 {lower} 至 {upper} {unit} 窗口内",),
        "maximize": ("{alias}不设门槛，只按数值从高到低优选",),
        "minimize": ("{alias}不设门槛，只按数值从低到高优选",),
    },
    "adversarial5": {
        "gt": ("{alias}高出 {value} {unit} 才合格，等于不合格",),
        "gte": ("{alias}不少于 {value} {unit} 就合格，等于合格",),
        "lt": ("{alias}低出 {value} {unit} 才合格，等于不合格",),
        "lte": ("{alias}不多于 {value} {unit} 就合格，等于合格",),
        "eq": ("{alias}只接受与 {value} {unit} 相同的点",),
        "range": ("{alias}只接受闭区间 [{lower}, {upper}] {unit}",),
        "maximize": ("{alias}没有合格线，直接选最高",),
        "minimize": ("{alias}没有合格线，直接选最低",),
    },
}


TASK_WRAPPERS = {
    "train53": {
        TaskType.GENERATE.value: ("尽管下面写的是筛选标准，但候选池为空，无法筛选；请把{body}作为目标生成新材料", "没有任何现成条目，即使给出过滤线也要从零设计满足{body}的材料"),
        TaskType.SCREEN.value: ("尽管这些条件也可叫设计指标，但候选已经生成；不要创造新结构，只筛选满足{body}的条目", "现有材料池已经完备，即使出现研发字样也仅按{body}执行去留"),
    },
    "dev53": {
        TaskType.GENERATE.value: ("当前列表是空的，筛选无对象，请转而生成符合{body}的新组成",),
        TaskType.SCREEN.value: ("列表已有结果，设计阶段结束，请只过滤符合{body}的对象",),
    },
    "hidden5": {
        TaskType.GENERATE.value: ("手头零候选，所谓筛选线只能当生成目标；请提出满足{body}的材料",),
        TaskType.SCREEN.value: ("候选计算已收尾，所谓设计线只用于淘汰；请留下满足{body}的材料",),
    },
    "adversarial5": {
        TaskType.GENERATE.value: ("先别被‘筛选’二字误导：池中没有条目，请新建满足{body}的材料",),
        TaskType.SCREEN.value: ("先别被‘生成指标’误导：池中已有条目，请仅保留满足{body}的材料",),
    },
}


def profile(tier: str) -> dict[str, str]:
    if tier == "train": return {"alias_group": "correction", "language_group": "train53", "wrapper_group": "train53", "input_mode": "raw"}
    if tier == "dev": return {"alias_group": "dev53", "language_group": "dev53", "wrapper_group": "dev53", "input_mode": "raw"}
    if tier == "id_hidden5": return {"alias_group": "known", "language_group": "hidden5", "wrapper_group": "hidden5", "input_mode": "raw"}
    if tier == "ood_hidden5": return {"alias_group": "hidden5", "language_group": "hidden5", "wrapper_group": "hidden5", "input_mode": "raw"}
    if tier == "adversarial_hidden5": return {"alias_group": "hidden5", "language_group": "adversarial5", "wrapper_group": "adversarial5", "input_mode": "raw"}
    raise ValueError(tier)


def family_for(tier: str, index: int) -> tuple[str, int]:
    if tier == "train": bounds = ((800, "task_contrast"), (1400, "double_range"), (2000, "operator_pair"), (2400, "eq_contrast"), (2800, "rare_earth"), (3200, "rehearsal"))
    elif tier == "dev": bounds = ((100, "task_contrast"), (180, "double_range"), (260, "operator_pair"), (310, "eq_contrast"), (360, "rare_earth"), (400, "rehearsal"))
    else: bounds = ((80, "task_contrast"), (140, "double_range"), (220, "operator_pair"), (260, "eq_contrast"), (300, "rare_earth"))
    previous = 0
    for bound, family in bounds:
        if index < bound: return family, index - previous
        previous = bound
    raise ValueError((tier, index))


def task_for(tier: str, index: int) -> TaskType:
    return TaskType.GENERATE if index % 2 == 0 else TaskType.SCREEN


def render(rng: random.Random, tier: str, name: str, operator: str, code: int):
    cfg = profile(tier)
    alias = rng.choice(PROPERTY_ALIASES[name][cfg["alias_group"]])
    unit_text = rng.choice(UNIT_TEXT[name][cfg["alias_group"]])
    templates = LANGUAGE[cfg["language_group"]]
    old_aliases = v52b.PROPERTY_ALIASES
    old_units = v52b.UNIT_TEXT
    old_text = v52b.COMPARATOR_TEXT
    try:
        v52b.PROPERTY_ALIASES = PROPERTY_ALIASES
        v52b.UNIT_TEXT = UNIT_TEXT
        v52b.COMPARATOR_TEXT = LANGUAGE
        local_rng = random.Random(code * 17 + SEEDS[tier])
        # Preserve the selected alias/unit by rendering directly with the same semantics.
        template = rng.choice(templates[operator])
        value = v52b.deterministic_value(name, code)
        config = PROPERTY_CONFIG[name]
        if operator == "range":
            width = {"band_gap": 0.35, "curie_temperature": 100.0, "formation_energy": 0.3, "magnetic_moment_density": 0.15}[name]
            lower, upper = value, round(value + width, 3)
            phrase = template.format(alias=alias, lower=v52b.number(lower), upper=v52b.number(upper), unit=unit_text)
            from magnova_agent.models import Comparator, PropertyConstraint
            prop = PropertyConstraint(name=name, raw_name=alias, operator=Comparator.RANGE, lower_bound=lower, upper_bound=upper, unit=config["unit"])
        elif operator in {"maximize", "minimize"}:
            phrase = template.format(alias=alias)
            from magnova_agent.models import Comparator, PropertyConstraint
            prop = PropertyConstraint(name=name, raw_name=alias, operator=Comparator(operator))
        else:
            phrase = template.format(alias=alias, value=v52b.number(value), unit=unit_text)
            from magnova_agent.models import Comparator, PropertyConstraint
            prop = PropertyConstraint(name=name, raw_name=alias, operator=Comparator(operator), value=value, unit=config["unit"])
        return phrase, prop, alias
    finally:
        v52b.PROPERTY_ALIASES, v52b.UNIT_TEXT, v52b.COMPARATOR_TEXT = old_aliases, old_units, old_text


def focused_case(tier: str, index: int, family: str, local_index: int) -> tuple[str, MaterialRequest, dict[str, Any]]:
    cfg = profile(tier)
    task = task_for(tier, index)
    rng = random.Random(SEEDS[tier] + local_index + index * 13)
    names: list[str]
    operators: list[str]
    pair_id = None
    if family == "task_contrast":
        names = [tuple(PROPERTY_CONFIG)[local_index % 4]]
        operators = [OPERATORS[(local_index // 4) % 8]]
    elif family == "double_range":
        names = rng.sample(tuple(PROPERTY_CONFIG), 2)
        operators = ["range", "range"]
    elif family == "operator_pair":
        names = rng.sample(tuple(PROPERTY_CONFIG), 2)
        position = local_index % 64
        operators = [OPERATORS[position // 8], OPERATORS[position % 8]]
    elif family == "eq_contrast":
        group = local_index // 5
        names = [tuple(PROPERTY_CONFIG)[group % 4]]
        operators = [("eq", "gt", "gte", "lt", "lte")[local_index % 5]]
        pair_id = f"{tier}:eq:{group}"
    elif family == "rare_earth":
        names = [tuple(PROPERTY_CONFIG)[local_index % 4]]
        operators = [OPERATORS[(local_index // 4) % 8]]
    else:
        raise ValueError(family)

    codes = [local_index * 3 + position + 101 for position in range(len(names))]
    rendered = [render(rng, tier, name, operator, code) for name, operator, code in zip(names, operators, codes, strict=True)]
    if len(rendered) == 2 and local_index % 2:
        rendered.reverse(); names.reverse(); operators.reverse()
    family_prefix = {
        "task_contrast": "请依据候选是否存在判断任务",
        "double_range": "两个性质都使用各自的闭区间",
        "operator_pair": "两个性质分别绑定自己的比较条件",
        "eq_contrast": "请严格区分等于与单边比较",
        "rare_earth": "请单独判断稀土是否需要排除",
    }[family]
    phrases = [family_prefix, *(item[0] for item in rendered)]
    properties = [item[1] for item in rendered]
    aliases = {prop.name: alias for _, prop, alias in rendered}
    composition = CompositionConstraint()
    material_family = None
    if any(operator in {"maximize", "minimize"} for operator in operators):
        context_index = index
        family_count = len(writer.v51.FAMILIES)
        element_count = len(writer.v51.ELEMENTS)
        material_family = writer.v51.FAMILIES[context_index % family_count]
        required_index = (context_index // family_count) % element_count
        required = writer.v51.ELEMENTS[required_index]
        remaining = [item for item in writer.v51.ELEMENTS if item != required]
        excluded_index = (
            context_index // (family_count * element_count)
        ) % len(remaining)
        excluded = remaining[excluded_index]
        phrases.extend((f"体系为 {material_family}", f"必须含 {required}", f"排除 {excluded}"))
        composition.required_elements = [required]
        composition.excluded_elements = [excluded]
    if family == "rare_earth":
        exclude = local_index % 2 == 0
        if exclude:
            phrases.append(rng.choice(("任何稀土元素都不得进入组成", "配方必须完全排除稀土族", "不允许含有任一稀土元素")))
        else:
            phrases.append(rng.choice(("稀土元素不作额外限制", "可以使用稀土元素", "无需排除稀土族")))
        composition.exclude_rare_earth = exclude
    body = rng.choice(("，并且", "；同时", "、", "，另有")).join(phrases)
    original = rng.choice(TASK_WRAPPERS[cfg["wrapper_group"]][task.value]).format(body=body)
    properties.sort(key=lambda item: item.name)
    request = MaterialRequest(task=task, material_family=material_family, target_properties=properties, composition=composition, original_text=original, parser_warnings=[])
    meta = {**cfg, "task": task.value, "case_family": family, "minimal_pair_id": pair_id, "property_aliases": aliases, "operators": list(operators), "mention_order": names}
    return original, request, meta


def build_case(tier: str, index: int):
    family, local_index = family_for(tier, index)
    if family == "rehearsal":
        original, request, meta = v52b.build_case("train" if tier == "train" else "dev", local_index)
        meta = {**meta, "case_family": "v52b_rehearsal"}
        return original, request, meta
    return focused_case(tier, index, family, local_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_v53"))
    args = parser.parse_args()
    writer.SYSTEM_PROMPT = SYSTEM_PROMPT
    writer.PROPERTY_ALIASES = PROPERTY_ALIASES
    writer.build_case = build_case
    writer.validate_alias_isolation()
    specs = (("id_hidden5", 300), ("ood_hidden5", 300), ("adversarial_hidden5", 300), ("dev", 400), ("train", 3200))
    used_originals: set[str] = set(); used_inputs: set[str] = set()
    manifest: dict[str, Any] = {"schema_version": "material_request_semantic_fields_v5_3", "system_prompt": SYSTEM_PROMPT, "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), "training_mode": "targeted_incremental_sft_from_v52b_best", "frozen_before_training": ["id_hidden5", "ood_hidden5", "adversarial_hidden5"], "generation_order": [name for name, _ in specs], "focus": ["task_contrast", "double_range", "operator_pair", "eq_contrast", "rare_earth", "v52b_rehearsal"], "splits": {}}
    for tier, count in specs:
        manifest["splits"][tier] = writer.write_split(args.output_dir, tier, count, used_originals, used_inputs)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
