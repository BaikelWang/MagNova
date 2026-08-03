from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROPERTIES = (
    "band_gap",
    "curie_temperature",
    "formation_energy",
    "magnetic_moment_density",
)
PROPERTY_FIELDS = (
    "operator",
    "value",
    "lower_bound",
    "upper_bound",
    "unit",
)


def property_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["name"]: item
        for item in payload["target_properties"]
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def analyze_split(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    counts: Counter[str] = Counter()
    task_confusion: Counter[str] = Counter()
    mismatch_samples: Counter[str] = Counter()
    mismatch_properties: Counter[str] = Counter()
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    by_input_mode: dict[str, Counter[str]] = defaultdict(Counter)
    free_expected: Counter[str] = Counter()
    free_recalled: Counter[str] = Counter()

    for row in results:
        expected = row["expected"]
        actual = row["raw_actual"]
        case_family = row.get("case_family") or "unknown"
        input_mode = row.get("input_mode") or "unknown"
        counts["samples"] += 1
        by_case[case_family]["samples"] += 1
        by_input_mode[input_mode]["samples"] += 1

        expected_names = set(property_map(expected))
        for name in expected_names:
            counts[f"property_expected:{name}"] += 1
        for name in expected["free_text_constraints"]:
            free_expected[name] += 1

        if actual is None:
            counts["invalid"] += 1
            task_confusion[f"{expected['task']} -> INVALID"] += 1
            continue

        counts["schema_valid"] += 1
        by_case[case_family]["schema_valid"] += 1
        by_input_mode[input_mode]["schema_valid"] += 1
        if actual == expected:
            counts["contract_exact"] += 1
        if actual["task"] == expected["task"]:
            counts["task_correct"] += 1
            by_case[case_family]["task_correct"] += 1
            by_input_mode[input_mode]["task_correct"] += 1
        task_confusion[f"{expected['task']} -> {actual['task']}"] += 1

        for field in (
            "material_family",
            "target_properties",
            "composition",
            "free_text_constraints",
            "status",
        ):
            if actual[field] == expected[field]:
                counts[f"field_correct:{field}"] += 1

        expected_props = property_map(expected)
        actual_props = property_map(actual)
        actual_names = set(actual_props)
        if actual_names == expected_names:
            counts["property_name_set_exact"] += 1
        else:
            mismatch_samples["property_name_set"] += 1
        for name in expected_names:
            if name in actual_names:
                counts[f"property_recalled:{name}"] += 1
            else:
                mismatch_properties[f"{name}:missing"] += 1
        for name in actual_names - expected_names:
            mismatch_properties[f"{name}:extra"] += 1

        sample_property_causes: set[str] = set()
        for name in expected_names.intersection(actual_names):
            expected_prop = expected_props[name]
            actual_prop = actual_props[name]
            for field in PROPERTY_FIELDS:
                if actual_prop[field] != expected_prop[field]:
                    sample_property_causes.add(field)
                    mismatch_properties[f"{name}:{field}"] += 1
                    mismatch_properties[f"all:{field}"] += 1
        for cause in sample_property_causes:
            mismatch_samples[cause] += 1

        expected_composition = expected["composition"]
        actual_composition = actual["composition"]
        for field in (
            "required_elements",
            "excluded_elements",
            "exclude_rare_earth",
        ):
            if actual_composition[field] == expected_composition[field]:
                counts[f"composition_correct:{field}"] += 1

        actual_free = set(actual["free_text_constraints"])
        for name in expected["free_text_constraints"]:
            if name in actual_free:
                free_recalled[name] += 1

    def grouped_summary(grouped: dict[str, Counter[str]]) -> dict[str, Any]:
        return {
            name: {
                "samples": values["samples"],
                "schema_valid_rate": ratio(
                    values["schema_valid"], values["samples"]
                ),
                "task_accuracy": ratio(
                    values["task_correct"], values["samples"]
                ),
            }
            for name, values in sorted(grouped.items())
        }

    return {
        "samples": total,
        "schema_valid_rate": ratio(counts["schema_valid"], total),
        "contract_exact_rate": ratio(counts["contract_exact"], total),
        "task_accuracy": ratio(counts["task_correct"], total),
        "field_accuracy": {
            field: ratio(counts[f"field_correct:{field}"], total)
            for field in (
                "material_family",
                "target_properties",
                "composition",
                "free_text_constraints",
                "status",
            )
        },
        "property_name_set_exact_rate": ratio(
            counts["property_name_set_exact"], total
        ),
        "property_recall_all_samples": {
            name: {
                "expected": counts[f"property_expected:{name}"],
                "recalled": counts[f"property_recalled:{name}"],
                "recall": ratio(
                    counts[f"property_recalled:{name}"],
                    counts[f"property_expected:{name}"],
                ),
            }
            for name in PROPERTIES
        },
        "property_mismatch_sample_causes": dict(
            mismatch_samples.most_common()
        ),
        "property_mismatch_detail": dict(
            mismatch_properties.most_common()
        ),
        "composition_subfield_accuracy": {
            field: ratio(counts[f"composition_correct:{field}"], total)
            for field in (
                "required_elements",
                "excluded_elements",
                "exclude_rare_earth",
            )
        },
        "free_text_recall": {
            name: {
                "expected": free_expected[name],
                "recalled": free_recalled[name],
                "recall": ratio(free_recalled[name], free_expected[name]),
            }
            for name in sorted(free_expected)
        },
        "task_confusion": dict(task_confusion.most_common()),
        "by_case_family": grouped_summary(by_case),
        "by_input_mode": grouped_summary(by_input_mode),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-report", type=Path, required=True)
    parser.add_argument("--v5-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {
        "v4": json.loads(args.v4_report.read_text(encoding="utf-8")),
        "v5": json.loads(args.v5_report.read_text(encoding="utf-8")),
    }
    analysis: dict[str, Any] = {}
    for model_name, report in reports.items():
        analysis[model_name] = {
            split: analyze_split(payload["results"])
            for split, payload in report["splits"].items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
