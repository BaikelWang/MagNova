from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from magnova_agent.interpreter import extract_json_object
from magnova_agent.models import MaterialRequest


SEMANTIC_FIELDS = (
    "task",
    "material_family",
    "target_properties",
    "composition",
    "free_text_constraints",
)

OOD_MARKERS = {
    "property:magnetic_moment_density": (
        "磁性体积强度",
        "单位体积磁响应",
    ),
    "property:curie_temperature": (
        "磁有序临界温度",
        "铁磁序消失温度",
    ),
    "property:formation_energy": (
        "化合稳定能",
        "组分稳定能",
    ),
    "property:band_gap": (
        "价导带分离能",
        "导带价带间隔量",
    ),
    "unit:magnetic_moment_density": (
        "Bohr-magneton per cubic angstrom",
        "Bohr magneton Å^-3",
    ),
    "unit:formation_energy": (
        "electronvolt per constituent atom",
        "electronvolt atom inverse",
    ),
    "unit:curie_temperature": ("absolute-degree",),
    "unit:band_gap": ("electronvolt",),
    "constraint:室温稳定": ("环境温度下相结构保持",),
    "constraint:低成本": ("经济性要好",),
    "constraint:无毒": ("生物安全风险低",),
    "constraint:可合成": ("实验路线现实可行",),
    "composition:exclude_rare_earth": (
        "镧系及相关稀土成分为零",
        "不得采用镧系元素",
        "材料不借助镧系组分",
    ),
}


def canonical_property(prop: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": prop.get("name"),
        "operator": prop.get("operator"),
        "value": prop.get("value"),
        "lower_bound": prop.get("lower_bound"),
        "upper_bound": prop.get("upper_bound"),
        "unit": prop.get("unit"),
    }


def canonical_properties(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    properties = payload.get("target_properties")
    if not isinstance(properties, list):
        return []
    return sorted(
        (canonical_property(prop) for prop in properties if isinstance(prop, dict)),
        key=lambda prop: str(prop["name"]),
    )


def canonical_field(payload: dict[str, Any] | None, field: str) -> Any:
    if payload is None:
        return None
    if field == "target_properties":
        return canonical_properties(payload)
    return payload.get(field)


def readiness_status(payload: dict[str, Any], original_text: str) -> str | None:
    try:
        check = dict(payload)
        check["original_text"] = original_text
        request = MaterialRequest.model_validate(check)
    except Exception:
        return None
    return "submitted" if request.ready_for_submission else "needs_clarification"


def property_mismatch_causes(
    expected: dict[str, Any],
    actual: dict[str, Any] | None,
) -> list[str]:
    expected_props = canonical_properties(expected)
    actual_props = canonical_properties(actual)
    expected_names = {item["name"] for item in expected_props}
    actual_names = {item["name"] for item in actual_props}
    if expected_names != actual_names:
        return ["property_name_set"]

    causes: set[str] = set()
    expected_by_name = {item["name"]: item for item in expected_props}
    actual_by_name = {item["name"]: item for item in actual_props}
    for name in expected_names:
        left = expected_by_name[name]
        right = actual_by_name[name]
        if left["operator"] != right["operator"]:
            causes.add("operator")
        if (
            left["value"] != right["value"]
            or left["lower_bound"] != right["lower_bound"]
            or left["upper_bound"] != right["upper_bound"]
        ):
            causes.add("value")
        if left["unit"] != right["unit"]:
            causes.add("unit")
    return sorted(causes)


def task_cue(text: str) -> str:
    generate_cues = (
        "当前没有候选",
        "探索新的材料组成",
        "全新方案",
        "创造一种材料",
        "研发",
    )
    screen_cues = (
        "不再生成新材料",
        "当前清单",
        "已有计算结果",
        "候选已备好",
        "备选集合",
        "准入判断",
    )
    generate = any(cue in text for cue in generate_cues)
    screen = any(cue in text for cue in screen_cues)
    if generate and screen:
        return "mixed"
    if generate:
        return "generate_cue"
    if screen:
        return "screen_cue"
    return "other"


def analyze(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report["results"]
    raw_valid = 0
    raw_schema_valid = 0
    raw_field_correct: Counter[str] = Counter()
    final_field_correct: Counter[str] = Counter()
    raw_mismatch_causes: Counter[str] = Counter()
    final_mismatch_causes: Counter[str] = Counter()
    task_confusion: Counter[str] = Counter()
    task_cue_stats: dict[str, Counter[str]] = defaultdict(Counter)
    gate_drops = 0
    gate_drop_examples: list[dict[str, Any]] = []
    guardrail_task_recoveries = 0
    guardrail_composition_recoveries = 0
    invalid_examples: list[dict[str, Any]] = []
    mismatch_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    marker_stats: dict[str, Counter[str]] = defaultdict(Counter)

    for item in results:
        text = item["input"]
        expected = item["expected"]
        final = item.get("actual")
        raw: dict[str, Any] | None = None
        raw_status: str | None = None
        try:
            raw = extract_json_object(item.get("raw_output") or "")
            raw_valid += 1
            check = dict(raw)
            check["original_text"] = text
            MaterialRequest.model_validate(check)
            raw_schema_valid += 1
            raw_status = readiness_status(raw, text)
        except Exception as exc:
            if len(invalid_examples) < 12:
                invalid_examples.append(
                    {
                        "input": text,
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_output": item.get("raw_output"),
                    }
                )

        for field in SEMANTIC_FIELDS:
            if canonical_field(raw, field) == canonical_field(expected, field):
                raw_field_correct[field] += 1
            if canonical_field(final, field) == canonical_field(expected, field):
                final_field_correct[field] += 1

        expected_status = expected.get("status")
        if raw_status == expected_status:
            raw_field_correct["status"] += 1
        if final is not None and final.get("status") == expected_status:
            final_field_correct["status"] += 1

        raw_task = None if raw is None else raw.get("task")
        final_task = None if final is None else final.get("task")
        expected_task = expected.get("task")
        if raw_task != expected_task:
            task_confusion[f"{expected_task}->{raw_task}"] += 1
        cue = task_cue(text)
        task_cue_stats[cue]["samples"] += 1
        if raw_task == expected_task:
            task_cue_stats[cue]["raw_correct"] += 1
        if final_task == expected_task:
            task_cue_stats[cue]["final_correct"] += 1

        raw_causes = property_mismatch_causes(expected, raw)
        final_causes = property_mismatch_causes(expected, final)
        raw_mismatch_causes.update(raw_causes)
        final_mismatch_causes.update(final_causes)
        for cause in raw_causes:
            if len(mismatch_examples[cause]) < 8:
                mismatch_examples[cause].append(
                    {
                        "input": text,
                        "expected": canonical_properties(expected),
                        "raw": canonical_properties(raw),
                        "final": canonical_properties(final),
                    }
                )

        raw_props_correct = (
            canonical_properties(raw) == canonical_properties(expected)
        )
        final_props_correct = (
            canonical_properties(final) == canonical_properties(expected)
        )
        if raw_props_correct and not final_props_correct:
            gate_drops += 1
            if len(gate_drop_examples) < 12:
                gate_drop_examples.append(
                    {
                        "input": text,
                        "expected": canonical_properties(expected),
                        "raw": canonical_properties(raw),
                        "final": canonical_properties(final),
                    }
                )

        if raw_task != expected_task and final_task == expected_task:
            guardrail_task_recoveries += 1
        if (
            canonical_field(raw, "composition")
            != canonical_field(expected, "composition")
            and canonical_field(final, "composition")
            == canonical_field(expected, "composition")
        ):
            guardrail_composition_recoveries += 1

        for marker, phrases in OOD_MARKERS.items():
            if not any(phrase in text for phrase in phrases):
                continue
            marker_stats[marker]["samples"] += 1
            if raw_props_correct:
                marker_stats[marker]["raw_all_properties_correct"] += 1
            if final_props_correct:
                marker_stats[marker]["final_all_properties_correct"] += 1
            if raw_task == expected_task:
                marker_stats[marker]["raw_task_correct"] += 1

    sample_count = len(results)

    def rates(counter: Counter[str]) -> dict[str, float]:
        return {
            field: round(counter[field] / sample_count, 4)
            for field in (*SEMANTIC_FIELDS, "status")
        }

    marker_summary = {}
    for marker, counts in sorted(marker_stats.items()):
        samples = counts["samples"]
        marker_summary[marker] = {
            "samples": samples,
            "raw_all_properties_correct_rate": round(
                counts["raw_all_properties_correct"] / samples,
                4,
            ),
            "final_all_properties_correct_rate": round(
                counts["final_all_properties_correct"] / samples,
                4,
            ),
            "raw_task_correct_rate": round(
                counts["raw_task_correct"] / samples,
                4,
            ),
        }

    cue_summary = {}
    for cue, counts in sorted(task_cue_stats.items()):
        samples = counts["samples"]
        cue_summary[cue] = {
            "samples": samples,
            "raw_task_accuracy": round(counts["raw_correct"] / samples, 4),
            "final_task_accuracy": round(counts["final_correct"] / samples, 4),
        }

    return {
        "source_report": str(report_path),
        "samples": sample_count,
        "reported_summary": report.get("summary"),
        "raw_model": {
            "json_valid_rate": round(raw_valid / sample_count, 4),
            "schema_valid_rate": round(raw_schema_valid / sample_count, 4),
            "field_accuracy": rates(raw_field_correct),
            "task_confusion": dict(task_confusion.most_common()),
            "property_mismatch_causes": dict(
                raw_mismatch_causes.most_common()
            ),
        },
        "final_after_guardrails": {
            "field_accuracy": rates(final_field_correct),
            "property_mismatch_causes": dict(
                final_mismatch_causes.most_common()
            ),
        },
        "boundary_effects": {
            "entity_gate_drops_after_raw_property_match": gate_drops,
            "task_guardrail_recoveries": guardrail_task_recoveries,
            "composition_guardrail_recoveries": (
                guardrail_composition_recoveries
            ),
        },
        "task_accuracy_by_cue": cue_summary,
        "ood_marker_breakdown": marker_summary,
        "examples": {
            "invalid_json_or_schema": invalid_examples,
            "entity_gate_drops": gate_drop_examples,
            "property_mismatch": dict(mismatch_examples),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = {
        report.stem: analyze(report)
        for report in args.reports
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
