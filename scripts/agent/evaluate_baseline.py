from __future__ import annotations

import argparse
import json
from pathlib import Path

from magnova_agent import MagNovaAgent


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "golden_cases.jsonl"


def load_cases(path: Path = GOLDEN_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate() -> dict:
    agent = MagNovaAgent()
    rows: list[dict] = []
    for case in load_cases():
        outcome = agent.run(case["input"])
        actual_properties = sorted(prop.name for prop in outcome.request.target_properties)
        checks = {
            "status": outcome.status.value == case["expected_status"],
            "task": outcome.request.task.value == case["expected_task"],
            "properties": actual_properties == sorted(case["expected_properties"]),
            "exclude_rare_earth": (
                outcome.request.composition.exclude_rare_earth
                == case["expected_exclude_rare_earth"]
            ),
        }
        rows.append(
            {
                "id": case["id"],
                "checks": checks,
                "all_correct": all(checks.values()),
                "actual_status": outcome.status.value,
                "actual_task": outcome.request.task.value,
                "actual_properties": actual_properties,
            }
        )

    total = len(rows)
    metrics = {
        key + "_accuracy": sum(row["checks"][key] for row in rows) / total
        for key in ("status", "task", "properties", "exclude_rare_earth")
    }
    metrics["all_fields_accuracy"] = sum(row["all_correct"] for row in rows) / total
    return {
        "baseline": "RuleBasedInterpreter",
        "evaluation_scope": "curated_contract_regression_not_generalization",
        "cases": total,
        "metrics": metrics,
        "failures": [row for row in rows if not row["all_correct"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
