from __future__ import annotations

import argparse
import json
from pathlib import Path

from magnova_agent.interpreter import extract_json_object
from magnova_agent.models import MaterialRequest


parser = argparse.ArgumentParser()
parser.add_argument("report", type=Path)
args = parser.parse_args()

report = json.loads(args.report.read_text(encoding="utf-8"))
json_valid = 0
schema_valid = 0
for item in report["results"]:
    try:
        payload = extract_json_object(item["raw_output"])
        json_valid += 1
        payload["original_text"] = item["input"]
        MaterialRequest.model_validate(payload)
        schema_valid += 1
    except Exception:
        pass

print(
    json.dumps(
        {
            "samples": len(report["results"]),
            "json_valid": json_valid,
            "schema_valid": schema_valid,
        }
    )
)
