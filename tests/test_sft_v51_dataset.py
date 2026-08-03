from __future__ import annotations

import hashlib
import json
from pathlib import Path

from magnova_agent.models import MaterialRequest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sft_v51"
SIZES = {
    "train": 4000,
    "dev": 400,
    "id_hidden2": 400,
    "ood_hidden2": 400,
    "adversarial_hidden2": 400,
}
ORDER = ("id_hidden2", "ood_hidden2", "adversarial_hidden2", "dev", "train")


def load(split: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (DATA / f"{split}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    metadata = [
        json.loads(line)
        for line in (DATA / f"{split}.meta.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(rows) == len(metadata)
    return [
        dict(row, **meta)
        for row, meta in zip(rows, metadata, strict=True)
    ]


def test_sizes_schema_and_global_isolation() -> None:
    originals: set[str] = set()
    model_inputs: set[str] = set()
    for split in ORDER:
        rows = load(split)
        assert len(rows) == SIZES[split]
        split_originals = {row["original_user"] for row in rows}
        split_inputs = {row["conversations"][1]["content"] for row in rows}
        assert len(split_originals) == SIZES[split]
        assert len(split_inputs) == SIZES[split]
        assert not originals.intersection(split_originals)
        assert not model_inputs.intersection(split_inputs)
        originals.update(split_originals)
        model_inputs.update(split_inputs)
        for row in rows:
            payload = json.loads(row["conversations"][2]["content"])
            serialized = row["conversations"][2]["content"]
            assert "raw_name" not in serialized
            payload["original_text"] = row["original_user"]
            MaterialRequest.model_validate(payload)


def test_hidden2_aliases_are_isolated_from_train_and_dev() -> None:
    visible: set[str] = set()
    hidden: set[str] = set()
    for split in ("train", "dev"):
        for row in load(split):
            visible.update(
                alias.casefold() for alias in row["property_aliases"].values()
            )
    for split in ("ood_hidden2", "adversarial_hidden2"):
        for row in load(split):
            hidden.update(
                alias.casefold() for alias in row["property_aliases"].values()
            )
    assert hidden
    assert not visible.intersection(hidden)


def test_id_hidden2_crosses_task_and_input_mode_independently() -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in load("id_hidden2"):
        key = (row["task"], row["input_mode"])
        counts[key] = counts.get(key, 0) + 1
    assert counts == {
        ("generate_material", "normalized"): 100,
        ("generate_material", "raw"): 100,
        ("screen_candidates", "normalized"): 100,
        ("screen_candidates", "raw"): 100,
    }


def test_correction_train_balances_tasks_and_all_operators() -> None:
    task_counts: dict[str, int] = {}
    operator_counts: dict[str, int] = {}
    for row in load("train"):
        task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1
        for operator in row["operators"]:
            operator_counts[operator] = operator_counts.get(operator, 0) + 1
    assert task_counts == {
        "generate_material": 2000,
        "screen_candidates": 2000,
    }
    assert set(operator_counts) == {
        "gt",
        "gte",
        "lt",
        "lte",
        "eq",
        "range",
        "maximize",
        "minimize",
    }
    assert min(operator_counts.values()) >= 600


def test_manifest_hashes_and_freeze_order() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_order"] == list(ORDER)
    assert manifest["frozen_before_training"] == list(ORDER[:3])
    for split in SIZES:
        data = (DATA / f"{split}.jsonl").read_bytes()
        meta = (DATA / f"{split}.meta.jsonl").read_bytes()
        assert hashlib.sha256(data).hexdigest() == (
            manifest["splits"][split]["sha256"]
        )
        assert hashlib.sha256(meta).hexdigest() == (
            manifest["splits"][split]["metadata_sha256"]
        )
