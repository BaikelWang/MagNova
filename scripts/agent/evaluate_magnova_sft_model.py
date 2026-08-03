from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from magnova_agent.interpreter import (
    MiniMindInterpreter,
    SYSTEM_PROMPT,
    extract_json_object,
)
from magnova_agent.models import MaterialRequest


FIELDS = (
    "task",
    "material_family",
    "target_properties",
    "composition",
    "free_text_constraints",
    "status",
)


def _sort_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def comparable(request: MaterialRequest) -> dict[str, Any]:
    """Normalize fields whose order has no business meaning before comparison."""
    payload = request.model_dump(mode="json")
    result = {
        "task": payload["task"],
        "material_family": payload["material_family"],
        "target_properties": payload["target_properties"],
        "composition": payload["composition"],
        "free_text_constraints": payload["free_text_constraints"],
        "status": (
            "submitted"
            if request.ready_for_submission
            else "needs_clarification"
        ),
    }
    result["target_properties"] = sorted(
        result["target_properties"], key=_sort_key
    )
    result["composition"] = dict(result["composition"])
    result["composition"]["required_elements"] = sorted(
        result["composition"]["required_elements"]
    )
    result["composition"]["excluded_elements"] = sorted(
        result["composition"]["excluded_elements"]
    )
    result["free_text_constraints"] = sorted(result["free_text_constraints"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8503/v1/chat/completions")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
    ]
    metadata_path = args.data.with_suffix(".meta.jsonl")
    if metadata_path.exists():
        metadata_rows = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
        ]
        if len(metadata_rows) != len(rows):
            raise ValueError("metadata row count does not match dataset")
        rows = [
            dict(row, **metadata)
            for row, metadata in zip(rows, metadata_rows, strict=True)
        ]
    if args.limit:
        rows = rows[: args.limit]

    interpreter = MiniMindInterpreter(
        endpoint=args.endpoint,
        timeout_seconds=90,
        repair_attempts=0,
    )
    counts: Counter[str] = Counter()
    results: list[dict[str, Any]] = []
    latencies: list[float] = []

    for index, row in enumerate(rows):
        messages = row["conversations"]
        model_text = messages[1]["content"]
        user_text = row.get("original_user", model_text)
        expected_payload = json.loads(messages[2]["content"])
        expected_payload["original_text"] = user_text
        expected_request = MaterialRequest.model_validate(expected_payload)
        expected = comparable(expected_request)
        raw_output = ""
        actual = None
        error = None
        json_valid = False
        schema_valid = False
        started = time.perf_counter()
        try:
            raw_output = interpreter._complete(
                [
                    {"role": "system", "content": messages[0]["content"]},
                    {"role": "user", "content": model_text},
                ]
            )
            extract_json_object(raw_output)
            json_valid = True
            counts["json_valid"] += 1
            actual_request = interpreter._validate(raw_output, user_text)
            schema_valid = True
            counts["schema_valid"] += 1
            actual = comparable(actual_request)
            for field in FIELDS:
                if actual[field] == expected[field]:
                    counts[f"field:{field}"] += 1
            if actual == expected:
                counts["contract_exact"] += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started
        latencies.append(latency)
        results.append(
            {
                "index": index,
                "input": user_text,
                "expected": expected,
                "actual": actual,
                "json_valid": json_valid,
                "schema_valid": schema_valid,
                "error": error,
                "latency_seconds": round(latency, 3),
                "raw_output": raw_output,
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(
                f"evaluated={index + 1}/{len(rows)} "
                f"schema_valid={counts['schema_valid']} "
                f"exact={counts['contract_exact']}",
                flush=True,
            )

    total = len(rows)
    summary = {
        "samples": total,
        "json_valid_rate": counts["json_valid"] / total,
        "schema_valid_rate": counts["schema_valid"] / total,
        "contract_exact_rate": counts["contract_exact"] / total,
        "field_accuracy": {
            field: counts[f"field:{field}"] / total for field in FIELDS
        },
        "latency_mean_seconds": round(statistics.mean(latencies), 3),
        "latency_p95_seconds": round(
            sorted(latencies)[round((len(latencies) - 1) * 0.95)],
            3,
        ),
    }
    report = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
