from __future__ import annotations

import json
from pathlib import Path

from magnova_agent.models import MaterialRequest


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sft_v2"


def load_rows(split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA_DIR / f"{split}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def validated_request(row: dict) -> MaterialRequest:
    messages = row["conversations"]
    payload = json.loads(messages[2]["content"])
    assert "original_text" not in payload
    assert "parser_warnings" not in payload
    payload["original_text"] = messages[1]["content"]
    return MaterialRequest.model_validate(payload)


def test_v2_counts_schema_and_business_only_output() -> None:
    expected_counts = {"train": 2400, "dev": 300, "test": 300}
    for split, count in expected_counts.items():
        rows = load_rows(split)
        assert len(rows) == count
        for row in rows:
            assert set(row) == {"conversations"}
            assert [message["role"] for message in row["conversations"]] == [
                "system",
                "user",
                "assistant",
            ]
            validated_request(row)


def test_v2_has_no_exact_user_leakage() -> None:
    inputs = {
        split: {row["conversations"][1]["content"] for row in load_rows(split)}
        for split in ("train", "dev", "test")
    }
    assert inputs["train"].isdisjoint(inputs["dev"])
    assert inputs["train"].isdisjoint(inputs["test"])
    assert inputs["dev"].isdisjoint(inputs["test"])


def test_v2_task_balance_and_hard_cases() -> None:
    requests = [validated_request(row) for row in load_rows("train")]
    screen_ratio = sum(
        request.task.value == "screen_candidates" for request in requests
    ) / len(requests)
    assert 0.4 <= screen_ratio <= 0.5
    assert any(not request.ready_for_submission for request in requests)
    assert any(len(request.target_properties) == 2 for request in requests)
    assert any(request.free_text_constraints for request in requests)
    assert any(request.composition.exclude_rare_earth for request in requests)
