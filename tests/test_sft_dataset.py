from __future__ import annotations

import json
from pathlib import Path

from magnova_agent.interpreter import SYSTEM_PROMPT
from magnova_agent.models import MaterialRequest


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sft_v1"


def load_rows(split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA_DIR / f"{split}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_sft_splits_have_valid_schema_and_no_extra_columns() -> None:
    expected_counts = {"train": 1200, "dev": 150, "test": 150}
    for split, expected_count in expected_counts.items():
        rows = load_rows(split)
        assert len(rows) == expected_count
        for row in rows:
            assert set(row) == {"conversations"}
            messages = row["conversations"]
            assert [message["role"] for message in messages] == [
                "system",
                "user",
                "assistant",
            ]
            assert messages[0]["content"].strip()
            request = MaterialRequest.model_validate_json(messages[2]["content"])
            assert request.original_text == messages[1]["content"]


def test_sft_user_text_does_not_leak_across_splits() -> None:
    user_sets = {
        split: {row["conversations"][1]["content"] for row in load_rows(split)}
        for split in ("train", "dev", "test")
    }
    assert user_sets["train"].isdisjoint(user_sets["dev"])
    assert user_sets["train"].isdisjoint(user_sets["test"])
    assert user_sets["dev"].isdisjoint(user_sets["test"])


def test_sft_contains_ready_and_clarification_examples() -> None:
    requests = [
        MaterialRequest.model_validate_json(row["conversations"][2]["content"])
        for row in load_rows("train")
    ]
    assert any(request.ready_for_submission for request in requests)
    assert any(not request.ready_for_submission for request in requests)
    assert any(len(request.target_properties) == 2 for request in requests)
    assert any(request.composition.exclude_rare_earth for request in requests)
