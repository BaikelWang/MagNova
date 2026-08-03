from __future__ import annotations

import json
from pathlib import Path

from magnova_agent.models import MaterialRequest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sft_v4"
SIZES = {"train": 4000, "dev": 400, "id_test": 400, "ood_challenge": 200}


def load(split: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    metadata = [
        json.loads(line)
        for line in (DATA / f"{split}.meta.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(rows) == len(metadata)
    return [dict(row, **meta) for row, meta in zip(rows, metadata, strict=True)]


def test_v4_sizes_and_global_isolation() -> None:
    original_seen: set[str] = set()
    model_seen: set[str] = set()
    for split, size in SIZES.items():
        rows = load(split)
        assert len(rows) == size
        originals = {row["original_user"] for row in rows}
        model_inputs = {row["conversations"][1]["content"] for row in rows}
        assert len(originals) == size
        assert len(model_inputs) == size
        assert not original_seen.intersection(originals)
        assert not model_seen.intersection(model_inputs)
        original_seen.update(originals)
        model_seen.update(model_inputs)


def test_v4_model_never_generates_raw_name_or_audit_fields() -> None:
    for split in SIZES:
        for row in load(split):
            payload = json.loads(row["conversations"][2]["content"])
            serialized = row["conversations"][2]["content"]
            assert "raw_name" not in serialized
            assert "original_text" not in serialized
            assert "parser_warnings" not in serialized
            payload["original_text"] = row["original_user"]
            MaterialRequest.model_validate(payload)


def test_v4_id_has_markers_and_ood_contains_unknown_terminology() -> None:
    id_text = "\n".join(row["conversations"][1]["content"] for row in load("id_test"))
    ood_text = "\n".join(
        row["conversations"][1]["content"] for row in load("ood_challenge")
    )
    assert "<band_gap>" in id_text
    assert "<formation_energy>" in id_text
    assert "价导带分离能" in ood_text
    assert "化合稳定能" in ood_text
    assert "价导带分离能<band_gap>" not in ood_text


def test_v4_manifest_freezes_both_test_tiers() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_generates_raw_name"] is False
    assert manifest["frozen_before_training"] == ["id_test", "ood_challenge"]
