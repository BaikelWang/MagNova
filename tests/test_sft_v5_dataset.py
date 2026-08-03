from __future__ import annotations

import hashlib
import json
from pathlib import Path

from magnova_agent.models import MaterialRequest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sft_v5"
SIZES = {
    "train": 8000,
    "dev": 800,
    "id_hidden": 600,
    "ood_hidden": 400,
    "adversarial_hidden": 400,
}
HIDDEN = ("id_hidden", "ood_hidden", "adversarial_hidden")


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


def test_sizes_and_global_exact_isolation() -> None:
    original_seen: set[str] = set()
    model_seen: set[str] = set()
    for split in ("id_hidden", "ood_hidden", "adversarial_hidden", "dev", "train"):
        rows = load(split)
        assert len(rows) == SIZES[split]
        originals = {row["original_user"] for row in rows}
        model_inputs = {row["conversations"][1]["content"] for row in rows}
        assert len(originals) == SIZES[split]
        assert len(model_inputs) == SIZES[split]
        assert not original_seen.intersection(originals)
        assert not model_seen.intersection(model_inputs)
        original_seen.update(originals)
        model_seen.update(model_inputs)


def test_all_targets_match_schema_and_exclude_audit_fields() -> None:
    for split in SIZES:
        for row in load(split):
            serialized = row["conversations"][2]["content"]
            assert "raw_name" not in serialized
            assert "original_text" not in serialized
            assert "parser_warnings" not in serialized
            payload = json.loads(serialized)
            payload["original_text"] = row["original_user"]
            request = MaterialRequest.model_validate(payload)
            names = [prop.name for prop in request.target_properties]
            assert names == sorted(names)


def test_manifest_hashes_and_hidden_freeze_order() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frozen_before_training"] == list(HIDDEN)
    assert manifest["generation_order"][:3] == list(HIDDEN)
    assert manifest["previous_v4_tests_are_not_v5_hidden_tests"] is True
    for split in SIZES:
        data_bytes = (DATA / f"{split}.jsonl").read_bytes()
        metadata_bytes = (DATA / f"{split}.meta.jsonl").read_bytes()
        assert hashlib.sha256(data_bytes).hexdigest() == (
            manifest["splits"][split]["sha256"]
        )
        assert hashlib.sha256(metadata_bytes).hexdigest() == (
            manifest["splits"][split]["metadata_sha256"]
        )


def test_hidden_property_aliases_are_not_in_train_or_dev() -> None:
    visible_aliases: set[str] = set()
    hidden_aliases: set[str] = set()
    for split in ("train", "dev"):
        for row in load(split):
            visible_aliases.update(
                alias.casefold() for alias in row["property_aliases"].values()
            )
    for split in ("ood_hidden", "adversarial_hidden"):
        for row in load(split):
            hidden_aliases.update(
                alias.casefold() for alias in row["property_aliases"].values()
            )
    assert hidden_aliases
    assert not visible_aliases.intersection(hidden_aliases)


def test_training_mix_contains_marked_raw_paraphrase_and_contrast() -> None:
    rows = load("train")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["case_family"]] = counts.get(row["case_family"], 0) + 1
    assert counts == {
        "marked_known": 2800,
        "raw_known": 2000,
        "raw_paraphrase": 2000,
        "task_contrast": 1200,
    }
    marked = [row for row in rows if row["input_mode"] == "normalized"]
    raw = [row for row in rows if row["input_mode"] == "raw"]
    assert len(marked) == 2800
    assert len(raw) == 5200
    assert all(
        row["contains_property_marker"] or not row["property_aliases"]
        for row in marked
    )
    assert all(not row["contains_property_marker"] for row in raw)


def test_hidden_ood_and_adversarial_inputs_have_no_property_markers() -> None:
    for split in ("ood_hidden", "adversarial_hidden"):
        for row in load(split):
            model_input = row["conversations"][1]["content"]
            assert row["input_mode"] == "raw"
            assert row["contains_property_marker"] is False
            assert not any(
                marker in model_input
                for marker in (
                    "<band_gap>",
                    "<formation_energy>",
                    "<curie_temperature>",
                    "<magnetic_moment_density>",
                )
            )


def test_task_and_property_distribution_is_broad() -> None:
    for split in SIZES:
        rows = load(split)
        task_counts: dict[str, int] = {}
        property_counts: dict[str, int] = {}
        operator_counts: dict[str, int] = {}
        for row in rows:
            payload = json.loads(row["conversations"][2]["content"])
            task_counts[payload["task"]] = task_counts.get(payload["task"], 0) + 1
            for prop in payload["target_properties"]:
                property_counts[prop["name"]] = (
                    property_counts.get(prop["name"], 0) + 1
                )
                operator_counts[prop["operator"]] = (
                    operator_counts.get(prop["operator"], 0) + 1
                )
        assert abs(
            task_counts["generate_material"] - task_counts["screen_candidates"]
        ) <= 1
        assert set(property_counts) == {
            "band_gap",
            "curie_temperature",
            "formation_energy",
            "magnetic_moment_density",
        }
        assert min(property_counts.values()) >= max(20, SIZES[split] // 5)
        assert {"gt", "gte", "lt", "lte", "eq", "range", "maximize", "minimize"} <= (
            set(operator_counts)
        )


def test_adversarial_examples_contain_conflicting_task_words() -> None:
    rows = load("adversarial_hidden")
    for row in rows:
        text = row["original_user"]
        task = row["task"]
        if task == "generate_material":
            assert "筛选" in text
            assert "为空" in text or "没有现成结果" in text
        else:
            assert "生成" in text or "设计" in text
            assert "已有候选" in text or "现成清单" in text
