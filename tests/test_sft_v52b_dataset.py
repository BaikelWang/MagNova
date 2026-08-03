from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from magnova_agent.models import MaterialRequest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sft_v52b"
ORDER = ("id_hidden4", "ood_hidden4", "adversarial_hidden4", "dev", "train")
SIZES = {"id_hidden4": 400, "ood_hidden4": 400, "adversarial_hidden4": 400, "dev": 600, "train": 4800}
OPS = {"gt", "gte", "lt", "lte", "eq", "range", "maximize", "minimize"}


def load(split: str) -> list[dict]:
    rows = [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = [json.loads(line) for line in (DATA / f"{split}.meta.jsonl").read_text(encoding="utf-8").splitlines()]
    return [dict(row, **item) for row, item in zip(rows, meta, strict=True)]


def test_schema_sizes_and_global_isolation() -> None:
    originals: set[str] = set()
    inputs: set[str] = set()
    for split in ORDER:
        rows = load(split)
        assert len(rows) == SIZES[split]
        for row in rows:
            assert row["original_user"] not in originals
            assert row["conversations"][1]["content"] not in inputs
            originals.add(row["original_user"])
            inputs.add(row["conversations"][1]["content"])
            payload = json.loads(row["conversations"][2]["content"])
            payload["original_text"] = row["original_user"]
            MaterialRequest.model_validate(payload)


def test_hidden4_aliases_are_isolated() -> None:
    visible = {alias.casefold() for split in ("train", "dev") for row in load(split) for alias in row["property_aliases"].values()}
    hidden = {alias.casefold() for split in ("ood_hidden4", "adversarial_hidden4") for row in load(split) for alias in row["property_aliases"].values()}
    assert hidden and not visible.intersection(hidden)


def test_minimal_pairs_and_operator_pairs() -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in load("train"):
        if row["case_family"] == "boundary_minimal_pair":
            groups[row["minimal_pair_id"]].append(row)
    assert len(groups) == 400
    assert all({row["operators"][0] for row in rows} == {"gt", "gte", "lt", "lte"} for rows in groups.values())
    pairs = Counter(tuple(row["operators"]) for row in load("adversarial_hidden4"))
    assert set(pairs) == {(left, right) for left in OPS for right in OPS}
    assert min(pairs.values()) >= 6


def test_focused_samples_are_not_composition_heavy() -> None:
    focused = [row for row in load("train") if row["case_family"] != "v51_rehearsal"]
    with_composition = 0
    for row in focused:
        payload = json.loads(row["conversations"][2]["content"])
        comp = payload["composition"]
        if comp["required_elements"] or comp["excluded_elements"] or comp["exclude_rare_earth"]:
            with_composition += 1
    assert with_composition / len(focused) < 0.11


def test_manifest_hashes_and_freeze_order() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_order"] == list(ORDER)
    assert manifest["frozen_before_training"] == list(ORDER[:3])
    assert manifest["v52_degraded_branch_is_not_parent"] is True
    for split in SIZES:
        assert hashlib.sha256((DATA / f"{split}.jsonl").read_bytes()).hexdigest() == manifest["splits"][split]["sha256"]
        assert hashlib.sha256((DATA / f"{split}.meta.jsonl").read_bytes()).hexdigest() == manifest["splits"][split]["metadata_sha256"]
