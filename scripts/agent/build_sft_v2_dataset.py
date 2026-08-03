from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from magnova_agent.models import (
    Comparator,
    CompositionConstraint,
    MaterialRequest,
    PropertyConstraint,
    TaskType,
)


V2_SYSTEM_PROMPT = """你是 MagNova 材料需求解析器。把用户文本转换成一个合法 JSON 对象，不解释、不使用 Markdown。
只输出以下业务字段：task、material_family、target_properties、composition、free_text_constraints。
task 只能是 generate_material、screen_candidates 或 predict_properties。用户要求设计、生成或寻找新材料时用 generate_material；用户要求在已有候选、列表或材料池中筛选、保留、剔除、核验时用 screen_candidates。
target_properties 每项只能包含 name、operator、value、lower_bound、upper_bound、unit、raw_name。operator 只能是 gt、gte、lt、lte、eq、range、maximize、minimize。
支持的规范性质名：magnetic_moment_density、curie_temperature、formation_energy、band_gap。不能猜测用户没给出的数值或单位，缺失值用 null，缺失列表用 []。
composition 只能包含 required_elements、excluded_elements、exclude_rare_earth。
free_text_constraints 只保留规范短语：室温稳定、低成本、无毒、可合成。
不要输出 original_text、parser_warnings 或任何其他字段。"""


@dataclass(frozen=True)
class PropertyDomain:
    name: str
    aliases: tuple[str, ...]
    canonical_unit: str
    unit_variants: dict[str, tuple[str, ...]]
    values: tuple[float, ...]
    ranges: tuple[tuple[float, float], ...]
    operators: tuple[Comparator, ...]


PROPERTIES = (
    PropertyDomain(
        "magnetic_moment_density",
        ("磁矩密度", "磁化密度", "单位体积磁矩"),
        "μB/Å³",
        {
            "train": ("μB/Å³", "μB/Å3"),
            "dev": ("uB/A^3",),
            "test": ("μB / Å³",),
        },
        (0.12, 0.18, 0.2, 0.25, 0.3),
        ((0.15, 0.25), (0.2, 0.35)),
        (Comparator.GT, Comparator.GTE, Comparator.EQ, Comparator.RANGE, Comparator.MAXIMIZE),
    ),
    PropertyDomain(
        "curie_temperature",
        ("居里温度", "磁性相变温度", "Tc"),
        "K",
        {"train": ("K",), "dev": ("开尔文",), "test": ("K",)},
        (300.0, 450.0, 500.0, 600.0, 750.0),
        ((300.0, 500.0), (450.0, 700.0)),
        (Comparator.GT, Comparator.GTE, Comparator.EQ, Comparator.RANGE, Comparator.MAXIMIZE),
    ),
    PropertyDomain(
        "formation_energy",
        ("形成能", "生成能", "formation energy"),
        "eV/atom",
        {
            "train": ("eV/atom", "eV 每原子"),
            "dev": ("eV/atom",),
            "test": ("eV per atom",),
        },
        (-1.2, -0.8, -0.5, -0.3),
        ((-1.2, -0.6), (-0.8, -0.3)),
        (Comparator.LT, Comparator.LTE, Comparator.EQ, Comparator.RANGE, Comparator.MINIMIZE),
    ),
    PropertyDomain(
        "band_gap",
        ("带隙", "禁带宽度", "band gap"),
        "eV",
        {"train": ("eV",), "dev": ("电子伏特",), "test": ("eV",)},
        (0.8, 1.0, 1.5, 2.0, 2.5),
        ((0.8, 1.5), (1.5, 2.5)),
        (Comparator.GT, Comparator.GTE, Comparator.LTE, Comparator.EQ, Comparator.RANGE),
    ),
)

FAMILIES = ("Fe-Co", "Fe-Ni", "Mn-Al", "Co-B", "Fe-P")
ELEMENTS = ("Fe", "Co", "Ni", "Mn", "Al", "B", "C", "P")

WRAPPERS = {
    "train": {
        TaskType.GENERATE: (
            "请设计一种{body}的新材料",
            "帮我生成满足{body}的材料",
            "我想寻找{body}的材料方案",
            "以{body}为目标开发新材料",
            "能不能构造一类{body}的候选",
            "需要创建满足这些要求的材料：{body}",
            "围绕{body}做一次材料设计",
            "给出符合{body}的新材料",
        ),
        TaskType.SCREEN: (
            "从现有候选材料中选出{body}的",
            "对候选池进行过滤，只保留{body}的材料",
            "检查候选列表，筛掉不满足条件的项；条件是{body}",
            "在已有材料清单里挑选{body}的候选",
            "请过滤当前候选，留下满足{body}的结果",
            "针对候选集合执行筛选，要求{body}",
            "帮我从给定材料中选出符合{body}的",
            "候选已经生成好了，现在按{body}进行过滤",
        ),
    },
    "dev": {
        TaskType.GENERATE: (
            "希望获得一种全新的材料，它需要{body}",
            "请围绕这些性能要求提出材料：{body}",
            "我的设计目标包括{body}",
        ),
        TaskType.SCREEN: (
            "在已有候选范围内找出达到{body}的材料",
            "对材料列表做一次条件过滤：{body}",
            "请从现有结果中挑出满足{body}的部分",
        ),
    },
    "test": {
        TaskType.GENERATE: (
            "面向{body}这一目标提出新材料",
            "从零设计一种能够{body}的材料",
            "我需要尚未生成的材料方案，约束为{body}",
            "请探索满足下列指标的新材料：{body}",
        ),
        TaskType.SCREEN: (
            "候选池里哪些材料满足{body}",
            "请淘汰不符合条件的项，条件为{body}",
            "在给定材料列表内检索{body}的对象",
            "按照{body}给候选材料做初筛",
        ),
    },
}

COMPARATOR_TEXT = {
    Comparator.GT: ("大于", "高于", "超过"),
    Comparator.GTE: ("至少达到", "不低于", "大于等于"),
    Comparator.LT: ("小于", "低于"),
    Comparator.LTE: ("不超过", "至多为", "小于等于"),
    Comparator.EQ: ("约为", "等于", "接近"),
    Comparator.MAXIMIZE: ("尽可能高", "越高越好", "需要最大化"),
    Comparator.MINIMIZE: ("尽可能低", "越低越好", "需要最小化"),
}

FREE_VARIANTS = {
    "train": {
        "室温稳定": ("室温稳定", "在室温下保持稳定"),
        "低成本": ("低成本", "成本尽量低"),
        "无毒": ("无毒", "避免有毒成分"),
        "可合成": ("可合成", "具备可合成性"),
    },
    "dev": {
        "室温稳定": ("常温条件下稳定",),
        "低成本": ("制造成本不能太高",),
        "无毒": ("不使用有毒成分",),
        "可合成": ("实验上能够合成",),
    },
    "test": {
        "室温稳定": ("室温环境中不能失稳",),
        "低成本": ("预算友好",),
        "无毒": ("不能有毒",),
        "可合成": ("应当具有合成可行性",),
    },
}

RARE_EARTH_TEXT = {
    "train": ("不含稀土元素", "排除稀土", "避免使用稀土"),
    "dev": ("不采用任何稀土成分", "把稀土元素排除在外"),
    "test": ("不依赖稀土成分", "组成里不能出现稀土"),
}


def number_text(value: float) -> str:
    return f"{value:g}"


def property_phrase(
    rng: random.Random,
    domain: PropertyDomain,
    split: str,
    omit_unit: bool,
) -> tuple[str, PropertyConstraint]:
    alias = rng.choice(domain.aliases)
    operator = rng.choice(domain.operators)
    raw_unit = None if omit_unit else rng.choice(domain.unit_variants[split])
    canonical_unit = None if omit_unit else domain.canonical_unit

    if operator == Comparator.RANGE:
        lower, upper = rng.choice(domain.ranges)
        phrase = f"{alias}在 {number_text(lower)} 到 {number_text(upper)}"
        if raw_unit:
            phrase += f" {raw_unit}"
        return (
            phrase,
            PropertyConstraint(
                name=domain.name,
                raw_name=alias,
                operator=operator,
                lower_bound=lower,
                upper_bound=upper,
                unit=canonical_unit,
            ),
        )

    if operator in {Comparator.MAXIMIZE, Comparator.MINIMIZE}:
        return (
            f"{alias}{rng.choice(COMPARATOR_TEXT[operator])}",
            PropertyConstraint(
                name=domain.name,
                raw_name=alias,
                operator=operator,
            ),
        )

    value = rng.choice(domain.values)
    phrase = f"{alias}{rng.choice(COMPARATOR_TEXT[operator])} {number_text(value)}"
    if raw_unit:
        phrase += f" {raw_unit}"
    return (
        phrase,
        PropertyConstraint(
            name=domain.name,
            raw_name=alias,
            operator=operator,
            value=value,
            unit=canonical_unit,
        ),
    )


def build_case(rng: random.Random, split: str) -> tuple[str, MaterialRequest]:
    task = TaskType.SCREEN if rng.random() < 0.45 else TaskType.GENERATE
    missing_property = rng.random() < 0.08
    missing_unit = not missing_property and rng.random() < 0.12

    phrases: list[str] = []
    properties: list[PropertyConstraint] = []
    if not missing_property:
        property_count = 2 if rng.random() < 0.32 else 1
        for index, domain in enumerate(rng.sample(PROPERTIES, k=property_count)):
            phrase, prop = property_phrase(
                rng,
                domain,
                split,
                omit_unit=missing_unit and index == 0,
            )
            phrases.append(phrase)
            properties.append(prop)

    family = rng.choice(FAMILIES) if rng.random() < 0.22 else None
    required: list[str] = []
    excluded: list[str] = []
    exclude_rare_earth = rng.random() < 0.28
    free_constraints: list[str] = []

    if family:
        phrases.append(f"限定在 {family} 体系")
    if rng.random() < 0.18:
        required = rng.sample(ELEMENTS, k=2)
        phrases.append(f"成分必须有 {' 和 '.join(required)}")
    if rng.random() < 0.16:
        available = [element for element in ELEMENTS if element not in required]
        excluded = [rng.choice(available)]
        phrases.append(f"成分中不要 {excluded[0]}")
    if exclude_rare_earth:
        phrases.append(rng.choice(RARE_EARTH_TEXT[split]))

    for canonical in rng.sample(
        tuple(FREE_VARIANTS[split]),
        k=1 if rng.random() < 0.32 else 0,
    ):
        free_constraints.append(canonical)
        phrases.append(rng.choice(FREE_VARIANTS[split][canonical]))

    if not phrases:
        phrases.append("适合后续材料研究")
    rng.shuffle(phrases)
    separator = rng.choice(("、", "，并且", "；同时"))
    body = separator.join(phrases)
    user_text = rng.choice(WRAPPERS[split][task]).format(body=body)

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
        original_text=user_text,
        parser_warnings=[],
    )
    return user_text, request


def assistant_payload(request: MaterialRequest) -> str:
    payload = request.model_dump(
        mode="json",
        exclude={"original_text", "parser_warnings"},
    )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_split(
    path: Path,
    split: str,
    count: int,
    seed: int,
    all_inputs: set[str],
) -> dict[str, Any]:
    rng = random.Random(seed)
    lines: list[str] = []
    local_inputs: set[str] = set()
    stats: Counter[str] = Counter()
    attempts = 0

    while len(lines) < count:
        attempts += 1
        if attempts > count * 150:
            raise RuntimeError(f"unable to create {count} unique {split} rows")
        user_text, request = build_case(rng, split)
        if user_text in local_inputs or user_text in all_inputs:
            continue
        output = assistant_payload(request)
        decoded = json.loads(output)
        decoded["original_text"] = user_text
        MaterialRequest.model_validate(decoded)
        row = {
            "conversations": [
                {"role": "system", "content": V2_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": output},
            ]
        }
        lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        local_inputs.add(user_text)
        stats[f"task:{request.task.value}"] += 1
        stats[f"status:{'ready' if request.ready_for_submission else 'clarify'}"] += 1
        stats[f"properties:{len(request.target_properties)}"] += 1
        for prop in request.target_properties:
            stats[f"property:{prop.name}"] += 1
            stats[f"operator:{prop.operator.value}"] += 1

    all_inputs.update(local_inputs)
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "count": len(lines),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "stats": dict(sorted(stats.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_v2"))
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--dev-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    used_inputs: set[str] = set()
    manifest: dict[str, Any] = {
        "schema_version": "material_request_business_fields_v2",
        "seed": args.seed,
        "system_prompt": V2_SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(
            V2_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "audit_fields_generated_by_model": False,
        "old_v1_test_reused": False,
        "split_policy": (
            "Task wrappers, free-constraint paraphrases and selected unit "
            "variants are disjoint across train/dev/test. Exact user text is "
            "globally deduplicated."
        ),
        "splits": {},
    }
    sizes = {
        "train": args.train_size,
        "dev": args.dev_size,
        "test": args.test_size,
    }
    for offset, (split, count) in enumerate(sizes.items()):
        manifest["splits"][split] = write_split(
            args.output_dir / f"{split}.jsonl",
            split,
            count,
            args.seed + offset,
            used_inputs,
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
