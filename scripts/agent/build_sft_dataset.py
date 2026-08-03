from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from magnova_agent.interpreter import SYSTEM_PROMPT
from magnova_agent.models import (
    Comparator,
    CompositionConstraint,
    MaterialRequest,
    PropertyConstraint,
    TaskType,
)


@dataclass(frozen=True)
class PropertyDomain:
    name: str
    aliases: tuple[str, ...]
    unit: str
    values: tuple[float, ...]
    ranges: tuple[tuple[float, float], ...]
    operators: tuple[Comparator, ...]


PROPERTIES = (
    PropertyDomain(
        "magnetic_moment_density",
        ("磁矩密度", "磁化密度"),
        "μB/Å³",
        (0.12, 0.18, 0.2, 0.25, 0.3),
        ((0.15, 0.25), (0.2, 0.35)),
        (Comparator.GT, Comparator.GTE, Comparator.EQ, Comparator.RANGE, Comparator.MAXIMIZE),
    ),
    PropertyDomain(
        "curie_temperature",
        ("居里温度", "磁性相变温度"),
        "K",
        (300.0, 450.0, 500.0, 600.0, 750.0),
        ((300.0, 500.0), (450.0, 700.0)),
        (Comparator.GT, Comparator.GTE, Comparator.EQ, Comparator.RANGE, Comparator.MAXIMIZE),
    ),
    PropertyDomain(
        "formation_energy",
        ("形成能", "生成能"),
        "eV/atom",
        (-1.2, -0.8, -0.5, -0.3),
        ((-1.2, -0.6), (-0.8, -0.3)),
        (Comparator.LT, Comparator.LTE, Comparator.EQ, Comparator.RANGE, Comparator.MINIMIZE),
    ),
    PropertyDomain(
        "band_gap",
        ("带隙", "禁带宽度"),
        "eV",
        (0.8, 1.0, 1.5, 2.0, 2.5),
        ((0.8, 1.5), (1.5, 2.5)),
        (Comparator.GT, Comparator.GTE, Comparator.LTE, Comparator.EQ, Comparator.RANGE),
    ),
)

FAMILIES = ("Fe-Co", "Fe-Ni", "Mn-Al", "Co-B", "Fe-P")
ELEMENTS = ("Fe", "Co", "Ni", "Mn", "Al", "B", "C", "P")
FREE_CONSTRAINTS = ("室温稳定", "低成本", "无毒", "可合成")

WRAPPERS = {
    "train": {
        TaskType.GENERATE: (
            "请生成{body}的材料",
            "帮我设计满足{body}的候选材料",
            "我需要{body}的新材料",
            "寻找{body}的材料",
            "请给出一种{body}的材料",
        ),
        TaskType.SCREEN: (
            "请从候选中筛选{body}的材料",
            "过滤出满足{body}的候选材料",
            "帮我筛选出{body}的材料",
        ),
    },
    "dev": {
        TaskType.GENERATE: (
            "能否给出符合{body}要求的材料方案",
            "目标是{body}，请进行材料生成",
        ),
        TaskType.SCREEN: (
            "从已有结果里挑出{body}的材料",
            "请在候选集合中查找满足{body}的条目",
        ),
    },
    "test": {
        TaskType.GENERATE: (
            "麻烦找一找{body}这样的材料",
            "候选材料需要做到{body}",
            "以{body}为约束开展材料设计",
        ),
        TaskType.SCREEN: (
            "哪些候选能够同时达到{body}",
            "请保留那些符合{body}要求的候选",
            "依据{body}对材料列表进行筛查",
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

RARE_EARTH_TEXT = {
    "train": ("不含稀土元素", "排除稀土", "避免使用稀土"),
    "dev": ("不采用任何稀土成分", "把稀土元素排除在外"),
    "test": ("成分中不要出现稀土", "无需依赖稀土元素"),
}


def number_text(value: float) -> str:
    return f"{value:g}"


def property_phrase(
    rng: random.Random,
    domain: PropertyDomain,
    *,
    omit_unit: bool = False,
) -> tuple[str, PropertyConstraint]:
    alias = rng.choice(domain.aliases)
    operator = rng.choice(domain.operators)
    unit = None if omit_unit else domain.unit

    if operator == Comparator.RANGE:
        lower, upper = rng.choice(domain.ranges)
        phrase = f"{alias}在 {number_text(lower)} 到 {number_text(upper)}"
        if unit:
            phrase += f" {unit}"
        prop = PropertyConstraint(
            name=domain.name,
            raw_name=alias,
            operator=operator,
            lower_bound=lower,
            upper_bound=upper,
            unit=unit,
        )
        return phrase, prop

    if operator in {Comparator.MAXIMIZE, Comparator.MINIMIZE}:
        phrase = f"{alias}{rng.choice(COMPARATOR_TEXT[operator])}"
        return (
            phrase,
            PropertyConstraint(
                name=domain.name,
                raw_name=alias,
                operator=operator,
            ),
        )

    value = rng.choice(domain.values)
    phrase = f"{alias}{rng.choice(COMPARATOR_TEXT[operator])} {number_text(value)}"
    if unit:
        phrase += f" {unit}"
    return (
        phrase,
        PropertyConstraint(
            name=domain.name,
            raw_name=alias,
            operator=operator,
            value=value,
            unit=unit,
        ),
    )


def build_case(rng: random.Random, split: str, index: int) -> tuple[str, MaterialRequest]:
    task = TaskType.SCREEN if rng.random() < 0.2 else TaskType.GENERATE
    missing_property = rng.random() < 0.08
    missing_unit = not missing_property and rng.random() < 0.12

    property_phrases: list[str] = []
    properties: list[PropertyConstraint] = []
    if not missing_property:
        count = 2 if rng.random() < 0.28 else 1
        domains = rng.sample(PROPERTIES, k=count)
        for prop_index, domain in enumerate(domains):
            phrase, prop = property_phrase(
                rng,
                domain,
                omit_unit=missing_unit and prop_index == 0,
            )
            property_phrases.append(phrase)
            properties.append(prop)

    material_family = rng.choice(FAMILIES) if rng.random() < 0.2 else None
    required_elements: list[str] = []
    excluded_elements: list[str] = []
    exclude_rare_earth = rng.random() < 0.25
    free_constraints = (
        rng.sample(FREE_CONSTRAINTS, k=1 if rng.random() < 0.25 else 0)
    )

    body_parts = list(property_phrases)
    if material_family:
        body_parts.append(f"属于 {material_family} 体系")
    if rng.random() < 0.16:
        required_elements = rng.sample(ELEMENTS, k=2)
        body_parts.append(f"必须包含 {' 和 '.join(required_elements)}")
    if rng.random() < 0.14:
        available = [element for element in ELEMENTS if element not in required_elements]
        excluded_elements = [rng.choice(available)]
        body_parts.append(f"不含 {excluded_elements[0]}")
    if exclude_rare_earth:
        body_parts.append(rng.choice(RARE_EARTH_TEXT[split]))
    body_parts.extend(free_constraints)
    if not body_parts:
        body_parts.append("适合进一步研究")

    body = "、".join(body_parts)
    wrapper = rng.choice(WRAPPERS[split][task])
    user_text = wrapper.format(body=body)

    request = MaterialRequest(
        task=task,
        material_family=material_family,
        target_properties=properties,
        composition=CompositionConstraint(
            required_elements=required_elements,
            excluded_elements=excluded_elements,
            exclude_rare_earth=exclude_rare_earth,
        ),
        free_text_constraints=free_constraints,
        original_text=user_text,
        parser_warnings=[],
    )
    return user_text, request


def write_split(
    output_path: Path,
    *,
    split: str,
    count: int,
    seed: int,
    forbidden_inputs: set[str],
) -> dict[str, Any]:
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[str] = []
    stats: Counter[str] = Counter()
    attempts = 0

    while len(rows) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError(f"could not create {count} unique {split} samples")
        user_text, request = build_case(rng, split, len(rows))
        if user_text in seen or user_text in forbidden_inputs:
            continue
        request_json = request.model_dump_json()
        MaterialRequest.model_validate_json(request_json)
        row = {
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": request_json},
            ]
        }
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        seen.add(user_text)
        stats[f"task:{request.task.value}"] += 1
        stats[f"status:{'ready' if request.ready_for_submission else 'clarify'}"] += 1
        stats[f"properties:{len(request.target_properties)}"] += 1
        for prop in request.target_properties:
            stats[f"property:{prop.name}"] += 1
            stats[f"operator:{prop.operator.value}"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(rows) + "\n"
    output_path.write_text(content, encoding="utf-8")
    forbidden_inputs.update(seen)
    return {
        "count": len(rows),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "stats": dict(sorted(stats.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_v1"))
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--dev-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    used_inputs: set[str] = set()
    split_sizes = {
        "train": args.train_size,
        "dev": args.dev_size,
        "test": args.test_size,
    }
    manifest: dict[str, Any] = {
        "schema_version": "material_request_v1",
        "seed": args.seed,
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "split_policy": (
            "User wrapper templates and rare-earth paraphrases are disjoint "
            "across train/dev/test; exact user text is globally deduplicated."
        ),
        "splits": {},
    }
    for offset, (split, count) in enumerate(split_sizes.items()):
        path = args.output_dir / f"{split}.jsonl"
        manifest["splits"][split] = write_split(
            path,
            split=split,
            count=count,
            seed=args.seed + offset,
            forbidden_inputs=used_inputs,
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
