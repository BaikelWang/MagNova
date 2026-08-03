from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from magnova_agent.interpreter import apply_critical_guardrails
from magnova_agent.models import MaterialRequest
from magnova_agent.normalization import normalize_for_llm


FIELDS = (
    "task",
    "material_family",
    "target_properties",
    "composition",
    "free_text_constraints",
    "status",
)


def request_from_comparable(
    payload: dict[str, Any],
    original_text: str,
) -> MaterialRequest:
    source = {key: value for key, value in payload.items() if key != "status"}
    source["original_text"] = original_text
    source["parser_warnings"] = []
    return MaterialRequest.model_validate(source)


def comparable(request: MaterialRequest) -> dict[str, Any]:
    payload = request.model_dump(mode="json")
    properties = []
    for prop in payload["target_properties"]:
        prop = dict(prop)
        prop.pop("raw_name", None)
        properties.append(prop)
    properties.sort(
        key=lambda item: json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    composition = dict(payload["composition"])
    composition["required_elements"] = sorted(composition["required_elements"])
    composition["excluded_elements"] = sorted(composition["excluded_elements"])
    return {
        "task": payload["task"],
        "material_family": payload["material_family"],
        "target_properties": properties,
        "composition": composition,
        "free_text_constraints": sorted(payload["free_text_constraints"]),
        "status": (
            "submitted"
            if request.ready_for_submission
            else "needs_clarification"
        ),
    }


def finalize(request: MaterialRequest, original_text: str) -> MaterialRequest:
    normalization = normalize_for_llm(original_text)
    recognized = {
        mention.canonical_name for mention in normalization.property_mentions
    }
    unlinked = [
        prop.name
        for prop in request.target_properties
        if prop.name not in recognized
    ]
    if unlinked:
        request.parser_warnings.append(
            "entity_gate_unverified_unlinked_properties:"
            + ",".join(sorted(set(unlinked)))
        )
    for prop in request.target_properties:
        if prop.raw_name is None:
            prop.raw_name = normalization.raw_name_for(prop.name)
    return apply_critical_guardrails(request, original_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    replay: dict[str, Any] = {"source_report": str(args.input), "splits": {}}

    for split, split_report in report["splits"].items():
        counts: Counter[str] = Counter()
        changes: Counter[str] = Counter()
        results = []
        source_results = split_report["results"]
        for row in source_results:
            expected = row["expected"]
            raw_actual = row["raw_actual"]
            if raw_actual is None:
                results.append(
                    {
                        "index": row["index"],
                        "final_actual": None,
                        "warnings": [],
                    }
                )
                continue
            request = request_from_comparable(
                raw_actual,
                row["original_input"],
            )
            finalized = finalize(request, row["original_input"])
            actual = comparable(finalized)
            for field in FIELDS:
                if actual[field] == expected[field]:
                    counts[f"field:{field}"] += 1
                if actual[field] != raw_actual[field]:
                    changes[f"changed:{field}"] += 1
            if actual == expected:
                counts["contract_exact"] += 1
            for warning in finalized.parser_warnings:
                changes[warning.split(":", 1)[0]] += 1
            results.append(
                {
                    "index": row["index"],
                    "final_actual": actual,
                    "warnings": finalized.parser_warnings,
                }
            )
        total = len(source_results)
        replay["splits"][split] = {
            "summary": {
                "samples": total,
                "contract_exact_rate": counts["contract_exact"] / total,
                "field_accuracy": {
                    field: counts[f"field:{field}"] / total
                    for field in FIELDS
                },
                "changes": dict(changes.most_common()),
            },
            "results": results,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(replay, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                split: payload["summary"]
                for split, payload in replay["splits"].items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
