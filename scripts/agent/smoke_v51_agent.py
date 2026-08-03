from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from magnova_agent.interpreter import MiniMindInterpreter


CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "generate_known_gte_and_rare_earth",
        "text": "当前没有候选，请设计带隙至少 1.5 eV 且不含稀土元素的新材料",
        "task": "generate_material",
        "properties": {"band_gap": ("gte", 1.5, "eV")},
        "exclude_rare_earth": True,
    },
    {
        "name": "screen_negative_formation_energy",
        "text": "已有候选材料，只保留形成能严格低于 -0.5 eV/atom 的条目",
        "task": "screen_candidates",
        "properties": {"formation_energy": ("lt", -0.5, "eV/atom")},
    },
    {
        "name": "generate_two_property_binding",
        "text": "目前没有现成方案，请研发磁矩密度严格大于 0.2 μB/Å³、居里温度不低于 500 K 的材料",
        "task": "generate_material",
        "properties": {
            "magnetic_moment_density": ("gt", 0.2, "μB/Å³"),
            "curie_temperature": ("gte", 500.0, "K"),
        },
    },
    {
        "name": "screen_two_ranges",
        "text": "计算结果已经生成，只筛选形成能位于 -1.2 到 -0.6 eV/atom、带隙位于 1.0 到 2.0 eV 的候选",
        "task": "screen_candidates",
        "ranges": {
            "formation_energy": (-1.2, -0.6, "eV/atom"),
            "band_gap": (1.0, 2.0, "eV"),
        },
    },
    {
        "name": "missing_unit_needs_clarification",
        "text": "没有候选，请设计居里温度至少 600 的新材料",
        "task": "generate_material",
        "properties": {"curie_temperature": ("gte", 600.0, None)},
        "ready": False,
    },
    {
        "name": "unseen_band_gap_paraphrase",
        "text": "请从零寻找电子占据态与未占据态之间能量差大于 1.2 eV 的材料",
        "task": "generate_material",
        "properties": {"band_gap": ("gt", 1.2, "eV")},
    },
)


def property_map(request) -> dict[str, Any]:
    return {prop.name: prop for prop in request.target_properties}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8504/v1/chat/completions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v51_agent_smoke.json"),
    )
    args = parser.parse_args()
    interpreter = MiniMindInterpreter(
        endpoint=args.endpoint,
        timeout_seconds=90,
        repair_attempts=0,
    )
    results = []
    passed = 0

    for case in CASES:
        errors: list[str] = []
        try:
            request = interpreter.interpret(case["text"])
            props = property_map(request)
            if request.task.value != case["task"]:
                errors.append(
                    f"task expected={case['task']} actual={request.task.value}"
                )
            for name, expected in case.get("properties", {}).items():
                if name not in props:
                    errors.append(f"missing_property:{name}")
                    continue
                prop = props[name]
                actual = (prop.operator.value, prop.value, prop.unit)
                if actual != expected:
                    errors.append(
                        f"property:{name} expected={expected} actual={actual}"
                    )
            for name, expected in case.get("ranges", {}).items():
                if name not in props:
                    errors.append(f"missing_property:{name}")
                    continue
                prop = props[name]
                actual = (
                    prop.lower_bound,
                    prop.upper_bound,
                    prop.unit,
                )
                if prop.operator.value != "range" or actual != expected:
                    errors.append(
                        f"range:{name} expected={expected} "
                        f"actual_operator={prop.operator.value} actual={actual}"
                    )
            if "exclude_rare_earth" in case and (
                request.composition.exclude_rare_earth
                != case["exclude_rare_earth"]
            ):
                errors.append("exclude_rare_earth_mismatch")
            if "ready" in case and request.ready_for_submission != case["ready"]:
                errors.append(
                    f"ready expected={case['ready']} "
                    f"actual={request.ready_for_submission}"
                )
            payload = request.model_dump(mode="json")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            payload = None
        if not errors:
            passed += 1
        results.append(
            {
                "name": case["name"],
                "input": case["text"],
                "passed": not errors,
                "errors": errors,
                "actual": payload,
            }
        )
        print(case["name"], "PASS" if not errors else f"FAIL {errors}")

    report = {
        "endpoint": args.endpoint,
        "passed": passed,
        "total": len(CASES),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"passed": passed, "total": len(CASES)}))
    raise SystemExit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
