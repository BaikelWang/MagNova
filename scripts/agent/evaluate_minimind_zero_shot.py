from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from magnova_agent.interpreter import (
    MiniMindInterpreter,
    SYSTEM_PROMPT,
    extract_json_object,
)


CASES = [
    {
        "id": "complete_magnetic",
        "input": "请生成磁矩密度大于 0.2 μB/Å³ 且不含稀土元素的材料",
        "expected_status": "submitted",
        "expected_task": "generate_material",
        "expected_properties": ["magnetic_moment_density"],
        "expected_exclude_rare_earth": True,
    },
    {
        "id": "band_gap",
        "input": "设计带隙至少 1.5 eV 的材料",
        "expected_status": "submitted",
        "expected_task": "generate_material",
        "expected_properties": ["band_gap"],
        "expected_exclude_rare_earth": False,
    },
    {
        "id": "maximize_curie",
        "input": "生成居里温度尽可能高的材料",
        "expected_status": "submitted",
        "expected_task": "generate_material",
        "expected_properties": ["curie_temperature"],
        "expected_exclude_rare_earth": False,
    },
    {
        "id": "missing_unit",
        "input": "设计磁矩密度大于 0.2 的材料",
        "expected_status": "needs_clarification",
        "expected_task": "generate_material",
        "expected_properties": ["magnetic_moment_density"],
        "expected_exclude_rare_earth": False,
    },
    {
        "id": "missing_property",
        "input": "请设计一种新材料",
        "expected_status": "needs_clarification",
        "expected_task": "generate_material",
        "expected_properties": [],
        "expected_exclude_rare_earth": False,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/minimind_zero_shot_v0.json"),
    )
    args = parser.parse_args()

    interpreter = MiniMindInterpreter(repair_attempts=0, timeout_seconds=90)
    results = []
    for case in CASES:
        started = time.perf_counter()
        raw_output = ""
        error = None
        json_valid = False
        schema_valid = False
        actual = None
        try:
            raw_output = interpreter._complete(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": case["input"]},
                ]
            )
            extract_json_object(raw_output)
            json_valid = True
            request = interpreter._validate(raw_output, case["input"])
            schema_valid = True
            actual = {
                "status": (
                    "submitted"
                    if request.ready_for_submission
                    else "needs_clarification"
                ),
                "task": request.task.value,
                "properties": sorted(
                    prop.name for prop in request.target_properties
                ),
                "exclude_rare_earth": request.composition.exclude_rare_earth,
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        expected = {
            "status": case["expected_status"],
            "task": case["expected_task"],
            "properties": sorted(case["expected_properties"]),
            "exclude_rare_earth": case["expected_exclude_rare_earth"],
        }
        results.append(
            {
                "id": case["id"],
                "input": case["input"],
                "latency_seconds": round(time.perf_counter() - started, 3),
                "json_valid": json_valid,
                "schema_valid": schema_valid,
                "contract_match": actual == expected,
                "actual": actual,
                "expected": expected,
                "error": error,
                "raw_output": raw_output,
            }
        )
        print(
            case["id"],
            f"json={json_valid}",
            f"schema={schema_valid}",
            f"contract={actual == expected}",
        )

    summary = {
        "cases": len(results),
        "json_valid_rate": sum(item["json_valid"] for item in results) / len(results),
        "schema_valid_rate": sum(item["schema_valid"] for item in results) / len(results),
        "contract_match_rate": sum(item["contract_match"] for item in results) / len(results),
        "average_latency_seconds": round(
            sum(item["latency_seconds"] for item in results) / len(results),
            3,
        ),
    }
    report = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
