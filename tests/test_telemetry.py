from __future__ import annotations

import json

from magnova_agent.telemetry import InteractionRecorder, redact_text


def test_redaction_preserves_material_numbers_and_units() -> None:
    text = "联系 13812345678，带隙至少 1.5 eV，邮箱 a@example.com"
    redacted = redact_text(text)
    assert "13812345678" not in redacted
    assert "a@example.com" not in redacted
    assert "1.5 eV" in redacted


def test_recorder_appends_valid_jsonl(tmp_path) -> None:
    path = tmp_path / "interactions.jsonl"
    recorder = InteractionRecorder(path=path, model_version="test-v1")
    for value in (1.2, 1.5):
        recorder.record(
            prompt_hash="abc123",
            original_text=f"带隙至少 {value} eV",
            normalized_text=f"带隙至少 {value} eV",
            raw_output="{}",
            outcome="success",
            latency_ms=12.3456,
            validated_request={"task": "generate_material"},
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["schema_version"] == "magnova.interaction.v1"
    assert rows[0]["review"]["training_approved"] is False
    assert rows[1]["input"]["user_text"] == "带隙至少 1.5 eV"
