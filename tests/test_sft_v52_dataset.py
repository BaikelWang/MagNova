from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from magnova_agent.models import MaterialRequest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sft_v52"
SIZES = {
    "train": 4800,
    "dev": 600,
    "id_hidden3": 400,
    "ood_hidden3": 400,
    "adversarial_hidden3": 400,
}
ORDER = ("id_hidden3", "ood_hidden3", "adversarial_hidden3", "dev", "train")
STRICT_OPERATORS = {"gt", "gte", "lt", "lte"}
ALL_OPERATORS = {
    "gt",
    "gte",
    "lt",
    "lte",
    "eq",
    "range",
    "maximize",
    "minimize",
}


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
            assert "raw_name" not in row["conversations"][2]["content"]
            payload["original_text"] = row["original_user"]
            MaterialRequest.model_validate(payload)


def test_hidden3_aliases_are_isolated_from_train_and_dev() -> None:
    visible: set[str] = set()
    hidden: set[str] = set()
    for split in ("train", "dev"):
        for row in load(split):
            visible.update(
                alias.casefold() for alias in row["property_aliases"].values()
            )
    for split in ("ood_hidden3", "adversarial_hidden3"):
        for row in load(split):
            hidden.update(
                alias.casefold() for alias in row["property_aliases"].values()
            )
    assert hidden
    assert not visible.intersection(hidden)


def test_train_boundary_minimal_pairs_are_complete() -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in load("train"):
        if row["case_family"] == "boundary_minimal_pair":
            groups[row["minimal_pair_id"]].append(row)
    assert len(groups) == 400
    for rows in groups.values():
        assert len(rows) == 4
        assert {row["operators"][0] for row in rows} == STRICT_OPERATORS
        assert len({tuple(row["property_aliases"]) for row in rows}) == 1
        assert len({tuple(row["mention_order"]) for row in rows}) == 1


def test_adversarial_hidden3_covers_all_operator_pairs() -> None:
    counts: Counter[tuple[str, str]] = Counter()
    rows = load("adversarial_hidden3")
    assert all(len(row["operators"]) == 2 for row in rows)
    for row in rows:
        counts[tuple(row["operators"])] += 1
    assert set(counts) == {
        (left, right) for left in ALL_OPERATORS for right in ALL_OPERATORS
    }
    assert min(counts.values()) >= 6


def test_train_contains_rehearsal_and_balanced_operator_coverage() -> None:
    families: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    for row in load("train"):
        families[row["case_family"]] += 1
        operators.update(row["operators"])
    assert families == {
        "boundary_minimal_pair": 1600,
        "binding_permutation": 1600,
        "range_contrast": 600,
        "optimization_contrast": 400,
        "v51_rehearsal": 600,
    }
    assert set(operators) == ALL_OPERATORS
    assert min(operators.values()) >= 500


def test_manifest_hashes_and_freeze_order() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_order"] == list(ORDER)
    assert manifest["frozen_before_training"] == list(ORDER[:3])
    assert manifest["contrastive_means_data_design_not_contrastive_loss"] is True
    for split in SIZES:
        data = (DATA / f"{split}.jsonl").read_bytes()
        meta = (DATA / f"{split}.meta.jsonl").read_bytes()
        assert hashlib.sha256(data).hexdigest() == (
            manifest["splits"][split]["sha256"]
        )
        assert hashlib.sha256(meta).hexdigest() == (
            manifest["splits"][split]["metadata_sha256"]
        )
