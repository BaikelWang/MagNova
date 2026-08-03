from __future__ import annotations

import json
from pathlib import Path

from magnova_agent.models import MaterialRequest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sft_v3"


def load(split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_v3_sizes_and_global_exact_deduplication() -> None:
    expected = {"train": 4000, "dev": 400, "test": 400}
    seen: set[str] = set()
    for split, size in expected.items():
        rows = load(split)
        assert len(rows) == size
        inputs = {row["conversations"][1]["content"] for row in rows}
        assert len(inputs) == size
        assert not seen.intersection(inputs)
        seen.update(inputs)


def test_v3_all_targets_match_schema_and_are_sorted() -> None:
    for split in ("train", "dev", "test"):
        for row in load(split):
            messages = row["conversations"]
            payload = json.loads(messages[2]["content"])
            payload["original_text"] = messages[1]["content"]
            request = MaterialRequest.model_validate(payload)
            names = [item.name for item in request.target_properties]
            assert names == sorted(names)


def test_v3_split_specific_phrases_do_not_leak() -> None:
    train_text = "\n".join(
        row["conversations"][1]["content"] for row in load("train")
    )
    test_text = "\n".join(
        row["conversations"][1]["content"] for row in load("test")
    )
    assert "按照" in train_text
    assert "请淘汰不符合条件的项" in train_text
    assert "材料候选已备好" not in train_text
    assert "材料候选已备好" in test_text
    assert "稀土含量必须为零" not in train_text
    assert "稀土含量必须为零" in test_text


def test_v3_manifest_declares_frozen_hidden_test() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_frozen_before_training"] is True
    assert manifest["old_test_rows_reused"] is False
    assert manifest["target_property_order"] == "canonical_name_ascending"
