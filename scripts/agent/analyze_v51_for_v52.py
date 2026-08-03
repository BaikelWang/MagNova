from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROPERTY_FIELDS = (
    "operator",
    "value",
    "lower_bound",
    "upper_bound",
    "unit",
)
STRICT_OPERATORS = {"gt", "gte", "lt", "lte"}


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def property_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in payload["target_properties"]}


def compact_property(item: dict[str, Any]) -> dict[str, Any]:
    return {field: item.get(field) for field in ("name", *PROPERTY_FIELDS)}


def analyze_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    operator_confusion: Counter[str] = Counter()
    operator_expected: Counter[str] = Counter()
    operator_correct: Counter[str] = Counter()
    operator_by_property: dict[str, Counter[str]] = defaultdict(Counter)
    field_mismatch: Counter[str] = Counter()
    cross_property_transfer: Counter[str] = Counter()
    by_property_count: dict[int, Counter[str]] = defaultdict(Counter)
    operator_pair_exact: Counter[str] = Counter()
    operator_pair_total: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        counts["samples"] += 1
        expected = row["expected"]
        actual = row.get("raw_actual")
        expected_properties = property_map(expected)
        property_count = len(expected_properties)
        by_property_count[property_count]["samples"] += 1

        if actual is None:
            counts["invalid"] += 1
            by_property_count[property_count]["invalid"] += 1
            continue

        actual_properties = property_map(actual)
        if actual["target_properties"] == expected["target_properties"]:
            counts["target_properties_exact"] += 1
            by_property_count[property_count]["target_properties_exact"] += 1

        expected_pair = tuple(
            item["operator"] for item in expected["target_properties"]
        )
        pair_key = "+".join(expected_pair) if expected_pair else "none"
        operator_pair_total[pair_key] += 1
        pair_is_exact = True

        for name, expected_property in expected_properties.items():
            expected_operator = expected_property["operator"]
            operator_expected[expected_operator] += 1
            if name not in actual_properties:
                counts["property_missing"] += 1
                operator_confusion[f"{expected_operator}->MISSING"] += 1
                operator_by_property[name][f"{expected_operator}->MISSING"] += 1
                pair_is_exact = False
                continue

            actual_property = actual_properties[name]
            actual_operator = actual_property["operator"]
            confusion = f"{expected_operator}->{actual_operator}"
            operator_confusion[confusion] += 1
            operator_by_property[name][confusion] += 1
            if expected_operator == actual_operator:
                operator_correct[expected_operator] += 1
            else:
                pair_is_exact = False
                if len(examples[confusion]) < 3:
                    examples[confusion].append(
                        {
                            "index": row["index"],
                            "input": row["original_input"],
                            "property": name,
                            "expected": compact_property(expected_property),
                            "actual": compact_property(actual_property),
                        }
                    )

            for field in PROPERTY_FIELDS:
                expected_value = expected_property.get(field)
                actual_value = actual_property.get(field)
                if actual_value == expected_value:
                    continue
                field_mismatch[field] += 1
                for other_name, other_property in expected_properties.items():
                    if other_name == name:
                        continue
                    if actual_value == other_property.get(field):
                        cross_property_transfer[
                            f"{field}:{other_name}->{name}"
                        ] += 1
                        break

        if pair_is_exact:
            operator_pair_exact[pair_key] += 1

    operator_metrics = {}
    for operator in sorted(operator_expected):
        expected_count = operator_expected[operator]
        operator_metrics[operator] = {
            "expected": expected_count,
            "correct": operator_correct[operator],
            "accuracy": ratio(operator_correct[operator], expected_count),
            "is_strict_boundary_operator": operator in STRICT_OPERATORS,
        }

    property_count_metrics = {
        str(count): {
            "samples": values["samples"],
            "target_properties_exact": values["target_properties_exact"],
            "target_properties_exact_rate": ratio(
                values["target_properties_exact"],
                values["samples"],
            ),
            "invalid": values["invalid"],
        }
        for count, values in sorted(by_property_count.items())
    }

    operator_pair_metrics = {
        pair: {
            "samples": total,
            "operator_pair_exact": operator_pair_exact[pair],
            "operator_pair_exact_rate": ratio(operator_pair_exact[pair], total),
        }
        for pair, total in operator_pair_total.most_common()
    }

    return {
        "samples": counts["samples"],
        "target_properties_exact_rate": ratio(
            counts["target_properties_exact"],
            counts["samples"],
        ),
        "operator_metrics": operator_metrics,
        "operator_confusion": dict(operator_confusion.most_common()),
        "operator_confusion_by_property": {
            name: dict(confusions.most_common())
            for name, confusions in sorted(operator_by_property.items())
        },
        "field_mismatch_count": dict(field_mismatch.most_common()),
        "cross_property_transfer": dict(
            cross_property_transfer.most_common()
        ),
        "by_property_count": property_count_metrics,
        "operator_pair_metrics": operator_pair_metrics,
        "representative_operator_errors": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    analysis = {
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
