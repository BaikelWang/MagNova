from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from magnova_agent.models import MaterialRequest
from magnova_agent.normalization import normalize_for_llm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_sft_v3_dataset as v3


SYSTEM_PROMPT = """你是 MagNova 材料需求解析器。用户输入中的领域术语可能带有 <canonical_name> 标记。把用户需求转换成一个合法 JSON 对象，不解释、不使用 Markdown。
只输出 task、material_family、target_properties、composition、free_text_constraints。
task 只能是 generate_material、screen_candidates 或 predict_properties。设计、生成、开发或寻找新材料用 generate_material；对已有候选、列表、结果或材料池做筛选、初筛、保留、淘汰、排除或核验用 screen_candidates。
target_properties 每项只能包含 name、operator、value、lower_bound、upper_bound、unit，并按 name 字母顺序排列；不要输出 raw_name。operator 只能是 gt、gte、lt、lte、eq、range、maximize、minimize。
规范性质名只有 magnetic_moment_density、curie_temperature、formation_energy、band_gap。不能猜测用户未给出的数值或单位；缺失值用 null，缺失列表用 []。
composition 只能包含 required_elements、excluded_elements、exclude_rare_earth。
free_text_constraints 只保留规范短语：室温稳定、低成本、无毒、可合成。
不要输出 original_text、parser_warnings、raw_name 或任何其他字段。"""


OOD_REPLACEMENTS = {
    "体积磁化强度指标": "磁性体积强度",
    "每单位体积磁矩": "单位体积磁响应",
    "居里点": "磁有序临界温度",
    "铁磁有序温度": "铁磁序消失温度",
    "热力学生成能": "化合稳定能",
    "单原子形成能": "组分稳定能",
    "电子能隙": "价导带分离能",
    "禁带间隔": "导带价带间隔量",
    "μB·Å⁻³": "Bohr-magneton per cubic angstrom",
    "玻尔磁子/立方埃": "Bohr magneton Å^-3",
    "每原子 eV": "electronvolt per constituent atom",
    "eV/原子": "electronvolt atom inverse",
    "Kelvin": "absolute-degree",
    "electron-volt": "electronvolt",
    "室温条件下结构不发生破坏": "环境温度下相结构保持",
    "原料与制备成本受控": "经济性要好",
    "不得具有毒性": "生物安全风险低",
    "现有实验条件能够制备": "实验路线现实可行",
    "稀土含量必须为零": "镧系及相关稀土成分为零",
    "不允许含任何稀土元素": "不得采用镧系元素",
    "采用零稀土设计": "材料不借助镧系组分",
}


def without_raw_name(request: MaterialRequest) -> dict[str, Any]:
    payload = request.model_dump(
        mode="json", exclude={"original_text", "parser_warnings"}
    )
    for prop in payload["target_properties"]:
        prop.pop("raw_name", None)
    return payload


def standard_case(
    rng: random.Random, split: str
) -> tuple[str, MaterialRequest]:
    return v3.build_case(rng, split)


def ood_case(rng: random.Random) -> tuple[str, MaterialRequest]:
    raw, request = v3.build_case(rng, "test")
    for known, unknown in OOD_REPLACEMENTS.items():
        raw = raw.replace(known, unknown)
    request.original_text = raw
    for prop in request.target_properties:
        prop.raw_name = None
    return raw, request


def write_split(
    path: Path,
    tier: str,
    count: int,
    seed: int,
    builder: Callable[[random.Random], tuple[str, MaterialRequest]],
    used_original: set[str],
    used_model: set[str],
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows: list[str] = []
    metadata_rows: list[str] = []
    originals: set[str] = set()
    model_inputs: set[str] = set()
    stats: Counter[str] = Counter()
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 300:
            raise RuntimeError(f"unable to generate {tier}")
        original, request = builder(rng)
        normalized = normalize_for_llm(original)
        if (
            original in originals
            or original in used_original
            or normalized.model_text in model_inputs
            or normalized.model_text in used_model
        ):
            continue
        payload = without_raw_name(request)
        check = dict(payload)
        check["original_text"] = original
        validated = MaterialRequest.model_validate(check)
        assistant = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        )
        row = {
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": normalized.model_text},
                {"role": "assistant", "content": assistant},
            ],
        }
        metadata = {
            "evaluation_tier": tier,
            "original_user": original,
            "normalization_trace": list(normalized.trace),
        }
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        metadata_rows.append(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        )
        originals.add(original)
        model_inputs.add(normalized.model_text)
        stats[f"task:{validated.task.value}"] += 1
        stats[f"properties:{len(validated.target_properties)}"] += 1
        stats[f"rare_earth:{validated.composition.exclude_rare_earth}"] += 1
        stats[f"status:{'ready' if validated.ready_for_submission else 'clarify'}"] += 1
        stats["normalization_changes"] += len(normalized.trace)
        stats["property_markers"] += normalized.model_text.count("<")
    used_original.update(originals)
    used_model.update(model_inputs)
    content = "\n".join(rows) + "\n"
    metadata_content = "\n".join(metadata_rows) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    metadata_path = path.with_suffix(".meta.jsonl")
    metadata_path.write_text(metadata_content, encoding="utf-8")
    return {
        "count": len(rows),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_content.encode()).hexdigest(),
        "stats": dict(sorted(stats.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_v4"))
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--dev-size", type=int, default=400)
    parser.add_argument("--id-test-size", type=int, default=400)
    parser.add_argument("--ood-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    used_original: set[str] = set()
    used_model: set[str] = set()

    specs = (
        ("id_test", args.id_test_size, lambda rng: standard_case(rng, "test")),
        ("ood_challenge", args.ood_size, ood_case),
        ("dev", args.dev_size, lambda rng: standard_case(rng, "dev")),
        ("train", args.train_size, lambda rng: standard_case(rng, "train")),
    )
    manifest: dict[str, Any] = {
        "schema_version": "material_request_semantic_fields_v4",
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "model_generates_raw_name": False,
        "program_restores_raw_name": True,
        "frozen_before_training": ["id_test", "ood_challenge"],
        "split_policy": (
            "Exact original and normalized model inputs are globally disjoint. "
            "ID test uses ontology-known terms; OOD challenge replaces them "
            "with terms outside the normalizer lexicon."
        ),
        "splits": {},
    }
    for offset, (name, count, builder) in enumerate(specs):
        manifest["splits"][name] = write_split(
            args.output_dir / f"{name}.jsonl",
            name,
            count,
            args.seed + offset,
            builder,
            used_original,
            used_model,
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
