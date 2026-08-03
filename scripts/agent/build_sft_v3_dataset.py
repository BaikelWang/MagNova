from __future__ import annotations

import argparse
import hashlib
import json
import random
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


SYSTEM_PROMPT = """你是 MagNova 材料需求解析器。把用户文本转换成一个合法 JSON 对象，不解释、不使用 Markdown。
只输出以下业务字段：task、material_family、target_properties、composition、free_text_constraints。
task 只能是 generate_material、screen_candidates 或 predict_properties。设计、生成、开发或寻找新材料用 generate_material；在已有候选、列表、结果或材料池中筛选、初筛、保留、淘汰、排除或核验用 screen_candidates。
target_properties 每项只能包含 name、operator、value、lower_bound、upper_bound、unit、raw_name，并按 name 字母顺序排列。operator 只能是 gt、gte、lt、lte、eq、range、maximize、minimize。
支持的规范性质名：magnetic_moment_density、curie_temperature、formation_energy、band_gap。不能猜测用户没给出的数值或单位，缺失值用 null，缺失列表用 []。
composition 只能包含 required_elements、excluded_elements、exclude_rare_earth。
free_text_constraints 只保留规范短语：室温稳定、低成本、无毒、可合成。
不要输出 original_text、parser_warnings 或任何其他字段。"""


PROPERTY_CONFIG = {
    "magnetic_moment_density": {
        "aliases": {
            "train": ("磁矩密度", "磁化密度", "单位体积磁矩", "体积磁矩"),
            "dev": ("单位体积的磁矩", "magnetic moment density"),
            "test": ("体积磁化强度指标", "每单位体积磁矩"),
        },
        "unit": "μB/Å³",
        "units": {
            "train": ("μB/Å³", "μB / Å³", "μB/Å3", "uB/A^3"),
            "dev": ("μB·Å^-3",),
            "test": ("μB·Å⁻³", "玻尔磁子/立方埃"),
        },
        "values": (0.12, 0.18, 0.2, 0.25, 0.3),
        "ranges": ((0.15, 0.25), (0.2, 0.35)),
        "operators": ("gt", "gte", "eq", "range", "maximize"),
    },
    "curie_temperature": {
        "aliases": {
            "train": ("居里温度", "磁性相变温度", "Tc"),
            "dev": ("铁磁转变温度", "Curie temperature"),
            "test": ("居里点", "铁磁有序温度"),
        },
        "unit": "K",
        "units": {
            "train": ("K", "开尔文"),
            "dev": ("kelvin",),
            "test": ("Kelvin",),
        },
        "values": (300.0, 450.0, 500.0, 600.0, 750.0),
        "ranges": ((300.0, 500.0), (450.0, 700.0)),
        "operators": ("gt", "gte", "eq", "range", "maximize"),
    },
    "formation_energy": {
        "aliases": {
            "train": ("形成能", "生成能", "formation energy"),
            "dev": ("每原子形成能", "formation-energy"),
            "test": ("热力学生成能", "单原子形成能"),
        },
        "unit": "eV/atom",
        "units": {
            "train": ("eV/atom", "eV per atom", "eV 每原子", "eV atom^-1"),
            "dev": ("eV·atom⁻¹",),
            "test": ("每原子 eV", "eV/原子"),
        },
        "values": (-1.2, -0.8, -0.5, -0.3),
        "ranges": ((-1.2, -0.6), (-0.8, -0.3)),
        "operators": ("lt", "lte", "eq", "range", "minimize"),
    },
    "band_gap": {
        "aliases": {
            "train": ("带隙", "禁带宽度", "band gap"),
            "dev": ("能带间隙", "band-gap"),
            "test": ("电子能隙", "禁带间隔"),
        },
        "unit": "eV",
        "units": {
            "train": ("eV", "电子伏特"),
            "dev": ("electron volt",),
            "test": ("electron-volt",),
        },
        "values": (0.8, 1.0, 1.5, 2.0, 2.5),
        "ranges": ((0.8, 1.5), (1.5, 2.5)),
        "operators": ("gt", "gte", "lte", "eq", "range"),
    },
}

COMPARATOR_TEXT = {
    "train": {
        "gt": ("大于", "高于", "超过", "需超出"),
        "gte": ("至少", "不低于", "大于等于", "下限是"),
        "lt": ("小于", "低于", "少于"),
        "lte": ("不超过", "至多", "小于等于", "上限是"),
        "eq": ("约为", "等于", "接近", "目标值为"),
        "maximize": ("尽可能高", "越高越好", "需要最大化"),
        "minimize": ("尽可能低", "越低越好", "需要最小化"),
    },
    "dev": {
        "gt": ("必须高过",),
        "gte": ("最低达到",),
        "lt": ("必须低过",),
        "lte": ("最高只能到",),
        "eq": ("希望达到",),
        "maximize": ("优先取高值",),
        "minimize": ("优先取低值",),
    },
    "test": {
        "gt": ("严格高于",),
        "gte": ("不得小于", "起码达到"),
        "lt": ("严格少于",),
        "lte": ("不得超过",),
        "eq": ("以此数值为目标",),
        "maximize": ("数值越大越理想",),
        "minimize": ("数值越小越理想",),
    },
}

WRAPPERS = {
    "train": {
        "generate_material": (
            "请设计满足{body}的新材料",
            "帮我生成一种{body}的材料",
            "从零开发符合{body}的材料",
            "寻找尚不存在的材料方案，要求{body}",
            "以{body}为目标提出新材料",
            "围绕{body}进行材料设计",
        ),
        "screen_candidates": (
            "按照{body}给候选材料做初筛",
            "请淘汰不符合条件的项，条件为{body}",
            "从候选池中剔除不满足{body}的材料",
            "已有一批候选，只留下符合{body}的",
            "对当前材料结果执行过滤，要求{body}",
            "核验候选清单并保留达到{body}的项",
            "在现有材料列表中筛查{body}的对象",
            "候选已经生成，按{body}二次筛选",
        ),
    },
    "dev": {
        "generate_material": (
            "需要构思一种全新材料，使其{body}",
            "请提出能够{body}的新配方",
            "尚无候选，请依据{body}开展设计",
        ),
        "screen_candidates": (
            "针对已有结果做条件审查：{body}",
            "在备选材料中排除不符合{body}的项",
            "给现有候选做复筛，判据是{body}",
        ),
    },
    "test": {
        "generate_material": (
            "请创造一种材料，预期它能{body}",
            "当前没有候选，请按这些指标研发：{body}",
            "为满足{body}探索新的材料组成",
            "希望得到一个全新方案，其约束是{body}",
        ),
        "screen_candidates": (
            "材料候选已备好，请依据{body}做准入判断",
            "从备选集合中排除未达到{body}的条目",
            "对已有计算结果按{body}进行合格性核查",
            "不再生成新材料，只在当前清单中找出{body}的项",
        ),
    },
}

RARE_EARTH = {
    "train": (
        "不依赖稀土成分",
        "组成里不能出现稀土",
        "不含稀土元素",
        "排除所有稀土",
        "避免使用任何稀土",
    ),
    "dev": ("稀土元素不得参与组成", "要求完全无稀土"),
    "test": ("稀土含量必须为零", "不允许含任何稀土元素", "采用零稀土设计"),
}

FREE_TEXT = {
    "train": {
        "室温稳定": ("室温稳定", "常温下不能失稳"),
        "低成本": ("低成本", "预算友好"),
        "无毒": ("无毒", "不能有毒"),
        "可合成": ("可合成", "应当具有合成可行性"),
    },
    "dev": {
        "室温稳定": ("在日常温度下保持稳定",),
        "低成本": ("制造费用要低",),
        "无毒": ("避免毒性风险",),
        "可合成": ("实验制备可实现",),
    },
    "test": {
        "室温稳定": ("室温条件下结构不发生破坏",),
        "低成本": ("原料与制备成本受控",),
        "无毒": ("不得具有毒性",),
        "可合成": ("现有实验条件能够制备",),
    },
}

FAMILIES = ("Fe-Co", "Fe-Ni", "Mn-Al", "Co-B", "Fe-P")
ELEMENTS = ("Fe", "Co", "Ni", "Mn", "Al", "B", "C", "P", "Pb")


def n(value: float) -> str:
    return f"{value:g}"


def property_phrase(
    rng: random.Random,
    split: str,
    name: str,
    omit_unit: bool,
) -> tuple[str, PropertyConstraint]:
    config = PROPERTY_CONFIG[name]
    alias = rng.choice(config["aliases"][split])
    operator = rng.choice(config["operators"])
    unit_text = None if omit_unit else rng.choice(config["units"][split])
    unit = None if omit_unit else config["unit"]

    if operator == "range":
        lower, upper = rng.choice(config["ranges"])
        phrase = f"{alias}介于 {n(lower)} 和 {n(upper)}"
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
        phrase = f"{alias}{rng.choice(COMPARATOR_TEXT[split][operator])}"
        prop = PropertyConstraint(
            name=name,
            raw_name=alias,
            operator=Comparator(operator),
        )
    else:
        value = rng.choice(config["values"])
        phrase = f"{alias}{rng.choice(COMPARATOR_TEXT[split][operator])} {n(value)}"
        if unit_text:
            phrase += f" {unit_text}"
        prop = PropertyConstraint(
            name=name,
            raw_name=alias,
            operator=Comparator(operator),
            value=value,
            unit=unit,
        )
    return phrase, prop


def build_case(rng: random.Random, split: str) -> tuple[str, MaterialRequest]:
    task = (
        TaskType.SCREEN if rng.random() < 0.5 else TaskType.GENERATE
    )
    missing_property = rng.random() < 0.04
    missing_unit = not missing_property and rng.random() < 0.1
    phrases: list[str] = []
    properties: list[PropertyConstraint] = []

    if not missing_property:
        property_count = 2 if rng.random() < 0.58 else 1
        names = rng.sample(tuple(PROPERTY_CONFIG), k=property_count)
        for index, name in enumerate(names):
            phrase, prop = property_phrase(
                rng, split, name, omit_unit=missing_unit and index == 0
            )
            phrases.append(phrase)
            properties.append(prop)
    properties.sort(key=lambda item: item.name)

    family = rng.choice(FAMILIES) if rng.random() < 0.2 else None
    required: list[str] = []
    excluded: list[str] = []
    exclude_rare_earth = rng.random() < 0.42
    free_constraints: list[str] = []

    if family:
        phrases.append(f"限定在 {family} 材料体系")
    if rng.random() < 0.2:
        required = rng.sample(ELEMENTS[:-1], k=2)
        phrases.append(f"组成必须包含 {' 和 '.join(required)}")
    if rng.random() < 0.2:
        available = [item for item in ELEMENTS if item not in required]
        excluded = [rng.choice(available)]
        phrases.append(f"成分中不得含 {excluded[0]}")
    if exclude_rare_earth:
        phrases.append(rng.choice(RARE_EARTH[split]))

    if rng.random() < 0.38:
        canonical = rng.choice(tuple(FREE_TEXT[split]))
        free_constraints.append(canonical)
        phrases.append(rng.choice(FREE_TEXT[split][canonical]))

    if not phrases:
        phrases.append("适用于进一步材料研究")
    rng.shuffle(phrases)
    body = rng.choice(("，并且", "；同时", "、")).join(phrases)
    user = rng.choice(WRAPPERS[split][task.value]).format(body=body)
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
        original_text=user,
        parser_warnings=[],
    )
    return user, request


def output_json(request: MaterialRequest) -> str:
    payload = request.model_dump(
        mode="json", exclude={"original_text", "parser_warnings"}
    )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_split(
    path: Path,
    split: str,
    count: int,
    seed: int,
    global_inputs: set[str],
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows: list[str] = []
    local: set[str] = set()
    stats: Counter[str] = Counter()
    while len(rows) < count:
        if len(local) > count * 200:
            raise RuntimeError(f"failed to generate unique {split} rows")
        user, request = build_case(rng, split)
        if user in local or user in global_inputs:
            continue
        assistant = output_json(request)
        check = json.loads(assistant)
        check["original_text"] = user
        MaterialRequest.model_validate(check)
        row = {
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        local.add(user)
        stats[f"task:{request.task.value}"] += 1
        stats[f"properties:{len(request.target_properties)}"] += 1
        stats[f"rare_earth:{request.composition.exclude_rare_earth}"] += 1
        stats[f"status:{'ready' if request.ready_for_submission else 'clarify'}"] += 1
        for prop in request.target_properties:
            stats[f"property:{prop.name}"] += 1
            stats[f"operator:{prop.operator.value}"] += 1
    global_inputs.update(local)
    content = "\n".join(rows) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "count": len(rows),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "stats": dict(sorted(stats.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_v3"))
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--dev-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    used: set[str] = set()
    manifest: dict[str, Any] = {
        "schema_version": "material_request_business_fields_v3",
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "test_frozen_before_training": True,
        "old_test_rows_reused": False,
        "target_property_order": "canonical_name_ascending",
        "split_policy": (
            "Task wrappers, property aliases, unit spellings, rare-earth phrases, "
            "free-text paraphrases and exact user text are isolated by split."
        ),
        "splits": {},
    }
    for offset, (split, count) in enumerate(
        (("test", args.test_size), ("dev", args.dev_size), ("train", args.train_size))
    ):
        manifest["splits"][split] = write_split(
            args.output_dir / f"{split}.jsonl",
            split,
            count,
            args.seed + offset,
            used,
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
