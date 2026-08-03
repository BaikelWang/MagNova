from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from magnova_agent.interpreter import extract_json_object


CANONICAL_PROPERTIES = (
    "band_gap",
    "curie_temperature",
    "formation_energy",
    "magnetic_moment_density",
)
CANONICAL_CONSTRAINTS = ("室温稳定", "低成本", "无毒", "可合成")


def properties(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    values = payload.get("target_properties")
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, dict)]


def names(payload: dict[str, Any] | None) -> set[str]:
    return {
        str(item.get("name"))
        for item in properties(payload)
        if item.get("name") is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    totals: Counter[str] = Counter()
    property_stats: dict[str, Counter[str]] = defaultdict(Counter)
    constraint_stats: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_names: Counter[str] = Counter()
    raw_name_set_sizes: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in report["results"]:
        expected = item["expected"]
        final = item.get("actual")
        try:
            raw = extract_json_object(item.get("raw_output") or "")
            totals["raw_json_valid"] += 1
        except Exception:
            raw = None
            totals["raw_invalid"] += 1

        expected_names = names(expected)
        raw_names = names(raw)
        final_names = names(final)
        raw_name_set_sizes[str(len(raw_names))] += 1
        if not raw_names:
            totals["raw_empty_property_set"] += 1
        if raw_names.intersection(expected_names):
            totals["raw_recognized_at_least_one_expected_property"] += 1
        if raw_names == expected_names:
            totals["raw_property_name_set_exact"] += 1
        if final_names == expected_names:
            totals["final_property_name_set_exact"] += 1

        for name in raw_names:
            if name not in CANONICAL_PROPERTIES:
                unknown_names[name] += 1

        for name in expected_names:
            stats = property_stats[name]
            stats["expected_mentions"] += 1
            if name in raw_names:
                stats["raw_name_recognized"] += 1
            if name in final_names:
                stats["final_name_recognized"] += 1
            if name not in raw_names and len(examples[name]) < 8:
                examples[name].append(
                    {
                        "input": item["input"],
                        "raw_names": sorted(raw_names),
                        "raw_output": item.get("raw_output"),
                    }
                )

        expected_constraints = set(expected.get("free_text_constraints") or [])
        raw_constraints = (
            set(raw.get("free_text_constraints") or [])
            if raw is not None
            else set()
        )
        final_constraints = (
            set(final.get("free_text_constraints") or [])
            if final is not None
            else set()
        )
        for constraint in CANONICAL_CONSTRAINTS:
            if constraint not in expected_constraints:
                continue
            stats = constraint_stats[constraint]
            stats["expected_mentions"] += 1
            if constraint in raw_constraints:
                stats["raw_recognized"] += 1
            if constraint in final_constraints:
                stats["final_recognized"] += 1

        expected_rare = expected["composition"]["exclude_rare_earth"]
        raw_rare = (
            raw.get("composition", {}).get("exclude_rare_earth")
            if raw is not None
            and isinstance(raw.get("composition"), dict)
            else None
        )
        final_rare = (
            final.get("composition", {}).get("exclude_rare_earth")
            if final is not None
            else None
        )
        if expected_rare:
            totals["rare_earth_positive"] += 1
            if raw_rare is True:
                totals["raw_rare_earth_true_positive"] += 1
            if final_rare is True:
                totals["final_rare_earth_true_positive"] += 1
        else:
            totals["rare_earth_negative"] += 1
            if raw_rare is False:
                totals["raw_rare_earth_true_negative"] += 1
            if final_rare is False:
                totals["final_rare_earth_true_negative"] += 1

    def with_rates(stats_by_key: dict[str, Counter[str]]) -> dict[str, Any]:
        output = {}
        for key, counts in sorted(stats_by_key.items()):
            total = counts["expected_mentions"]
            output[key] = {
                **dict(counts),
                "raw_recall": round(
                    counts.get("raw_name_recognized", counts.get("raw_recognized", 0))
                    / total,
                    4,
                ),
                "final_recall": round(
                    counts.get("final_name_recognized", counts.get("final_recognized", 0))
                    / total,
                    4,
                ),
            }
        return output

    result = {
        "source": str(args.report),
        "samples": len(report["results"]),
        "totals": dict(totals),
        "raw_property_name_set_size": dict(
            sorted(raw_name_set_sizes.items(), key=lambda item: int(item[0]))
        ),
        "property_recognition": with_rates(property_stats),
        "constraint_recognition": with_rates(constraint_stats),
        "unknown_raw_property_names": dict(unknown_names.most_common()),
        "examples": dict(examples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in result.items() if k != "examples"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
