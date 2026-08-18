from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import requests

from magnova_agent import MagNovaAgent, MiniMindInterpreter, RuleBasedInterpreter
from magnova_agent.interpreter import InterpreterError, extract_json_object
from magnova_agent.agent import MockMagNovaTool
from magnova_agent.models import AgentStatus, MaterialRequest
from magnova_agent.telemetry import InteractionRecorder


ROOT = Path(__file__).resolve().parents[1]


def test_complete_request_calls_mock_tool() -> None:
    outcome = MagNovaAgent().run(
        "请生成磁矩密度大于 0.2 μB/Å³ 且不含稀土元素的材料"
    )
    assert outcome.status == AgentStatus.SUBMITTED
    assert outcome.tool_call is not None
    assert outcome.tool_call.name == "submit_magnova_task"
    assert len(outcome.candidates) == 3
    assert outcome.request.composition.exclude_rare_earth is True


def test_missing_unit_causes_clarification() -> None:
    outcome = MagNovaAgent().run("设计磁矩密度大于 0.2 的材料")
    assert outcome.status == AgentStatus.NEEDS_CLARIFICATION
    assert outcome.tool_call is None
    assert any("单位" in question for question in outcome.clarification_questions)


def test_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MaterialRequest.model_validate(
            {
                "original_text": "test",
                "unexpected": "must be rejected",
            }
        )


def test_mock_tool_is_deterministic() -> None:
    request = RuleBasedInterpreter().interpret("生成带隙为 1.5 eV 的材料")
    tool = MockMagNovaTool()
    first = tool.submit(request)
    second = tool.submit(request)
    assert first == second


def test_all_golden_cases_match_expected_contract() -> None:
    agent = MagNovaAgent()
    path = ROOT / "examples" / "golden_cases.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    failures: list[dict] = []
    for case in cases:
        outcome = agent.run(case["input"])
        actual = {
            "status": outcome.status.value,
            "task": outcome.request.task.value,
            "properties": sorted(prop.name for prop in outcome.request.target_properties),
            "exclude_rare_earth": outcome.request.composition.exclude_rare_earth,
        }
        expected = {
            "status": case["expected_status"],
            "task": case["expected_task"],
            "properties": sorted(case["expected_properties"]),
            "exclude_rare_earth": case["expected_exclude_rare_earth"],
        }
        if actual != expected:
            failures.append({"id": case["id"], "actual": actual, "expected": expected})
    assert not failures, failures


def test_extract_json_object_accepts_fenced_output() -> None:
    payload = extract_json_object(
        """模型输出如下：
```json
{"task": "generate_material", "target_properties": []}
```"""
    )
    assert payload["task"] == "generate_material"


def test_minimind_interpreter_validates_and_preserves_original_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter = MiniMindInterpreter(repair_attempts=0)
    monkeypatch.setattr(
        interpreter,
        "_complete",
        lambda messages: """{
          "task": "generate_material",
          "material_family": null,
          "target_properties": [{
            "name": "band_gap",
            "operator": "gte",
            "value": 1.5,
            "lower_bound": null,
            "upper_bound": null,
            "unit": "eV",
            "raw_name": "带隙"
          }],
          "composition": {
            "required_elements": [],
            "excluded_elements": [],
            "exclude_rare_earth": false
          },
          "free_text_constraints": [],
          "original_text": "模型不能覆盖这个字段",
          "parser_warnings": []
        }""",
    )

    request = interpreter.interpret("  生成带隙至少 1.5 eV 的材料  ")

    assert request.original_text == "生成带隙至少 1.5 eV 的材料"
    assert request.target_properties[0].name == "band_gap"


def test_minimind_interpreter_falls_back_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter = MiniMindInterpreter(
        repair_attempts=0,
        fallback=RuleBasedInterpreter(),
    )

    def fail(_messages: list[dict[str, str]]) -> str:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(interpreter, "_complete", fail)
    request = interpreter.interpret("生成带隙大于 1.5 eV 的材料")

    assert request.target_properties[0].name == "band_gap"
    assert request.parser_warnings == ["minimind_fallback:ConnectionError"]


def test_minimind_interpreter_records_successful_interaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "interactions.jsonl"
    interpreter = MiniMindInterpreter(
        repair_attempts=0,
        model_version="test-model",
        recorder=InteractionRecorder(trace_path, "test-model"),
    )
    monkeypatch.setattr(
        interpreter,
        "_complete",
        lambda messages: """{
          "task": "generate_material",
          "material_family": null,
          "target_properties": [{
            "name": "band_gap", "operator": "gte", "value": 1.5,
            "lower_bound": null, "upper_bound": null, "unit": "eV"
          }],
          "composition": {
            "required_elements": [], "excluded_elements": [],
            "exclude_rare_earth": false
          },
          "free_text_constraints": []
        }""",
    )

    interpreter.interpret("联系 13812345678，生成带隙至少 1.5 eV 的材料")

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["outcome"] == "success"
    assert trace["model_version"] == "test-model"
    assert trace["input"]["user_text"].startswith("联系 <PHONE>")
    assert trace["validation"]["request"]["target_properties"][0]["value"] == 1.5
    assert trace["review"]["training_approved"] is False


def test_extract_json_object_rejects_nested_fragment_of_malformed_outer() -> None:
    malformed = (
        '{"task":"generate_material","target_properties":'
        '[{"name":"band_gap","operator":"gte"}]'
    )
    with pytest.raises(InterpreterError):
        extract_json_object(malformed)


def test_minimind_guardrail_corrects_explicit_lead_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter = MiniMindInterpreter(repair_attempts=0)
    monkeypatch.setattr(
        interpreter,
        "_complete",
        lambda messages: """{
          "task": "generate_material",
          "material_family": null,
          "target_properties": [{
            "name": "band_gap", "operator": "gte", "value": 1.5,
            "lower_bound": null, "upper_bound": null, "unit": "eV",
            "raw_name": "带隙"
          }],
          "composition": {
            "required_elements": [], "excluded_elements": [],
            "exclude_rare_earth": true
          },
          "free_text_constraints": []
        }""",
    )

    request = interpreter.interpret("请设计带隙至少 1.5 eV 且不含铅的材料")

    assert request.composition.exclude_rare_earth is False
    assert request.composition.excluded_elements == ["Pb"]
    assert "guardrail_corrected:exclude_rare_earth" in request.parser_warnings
    assert "guardrail_added:excluded_element:Pb" in request.parser_warnings


def test_minimind_guardrail_recognizes_held_out_rare_earth_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter = MiniMindInterpreter(repair_attempts=0)
    monkeypatch.setattr(
        interpreter,
        "_complete",
        lambda messages: """{
          "task": "screen_candidates",
          "material_family": null,
          "target_properties": [{
            "name": "formation_energy", "operator": "lt", "value": -0.2,
            "lower_bound": null, "upper_bound": null, "unit": "eV/atom",
            "raw_name": "形成能"
          }],
          "composition": {
            "required_elements": [], "excluded_elements": [],
            "exclude_rare_earth": false
          },
          "free_text_constraints": []
        }""",
    )

    request = interpreter.interpret(
        "请淘汰不符合条件的项，条件为形成能低于 -0.2 eV/atom 且不依赖稀土成分"
    )

    assert request.composition.exclude_rare_earth is True
    assert "guardrail_corrected:exclude_rare_earth" in request.parser_warnings
