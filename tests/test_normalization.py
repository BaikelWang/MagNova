from __future__ import annotations

import pytest

from magnova_agent import MiniMindInterpreter, normalize_for_llm


def test_normalizer_maps_property_unit_and_constraint() -> None:
    result = normalize_for_llm(
        "电子能隙不得小于 1.5 electron-volt，原料与制备成本受控"
    )
    assert "带隙<band_gap>" in result.model_text
    assert "1.5 eV" in result.model_text
    assert "低成本" in result.model_text
    assert result.raw_name_for("band_gap") == "电子能隙"


def test_normalizer_maps_rare_earth_and_magnetic_unit() -> None:
    result = normalize_for_llm(
        "每单位体积磁矩高于 0.2 玻尔磁子/立方埃，稀土含量必须为零"
    )
    assert "磁矩密度<magnetic_moment_density>" in result.model_text
    assert "0.2 μB/Å³" in result.model_text
    assert "不含稀土元素" in result.model_text


def test_interpreter_can_hide_raw_name_from_model_and_restore_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter = MiniMindInterpreter(
        repair_attempts=0,
        normalize_input=True,
    )
    captured = {}

    def complete(messages):
        captured["user"] = messages[1]["content"]
        return """{
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
        }"""

    monkeypatch.setattr(interpreter, "_complete", complete)
    request = interpreter.interpret("电子能隙不得小于 1.5 electron-volt")

    assert captured["user"] == "带隙<band_gap>不得小于 1.5 eV"
    assert request.target_properties[0].raw_name == "电子能隙"
    assert request.original_text == "电子能隙不得小于 1.5 electron-volt"
    assert any(item.startswith("normalized:property:") for item in request.parser_warnings)


def test_explicit_task_and_rare_earth_helpers() -> None:
    from magnova_agent.normalization import (
        explicit_task_hint,
        has_rare_earth_exclusion,
    )

    assert explicit_task_hint("当前没有候选，请按指标研发") == "generate_material"
    assert explicit_task_hint("不再生成新材料，只在当前清单中找出合格项") == "screen_candidates"
    assert (
        explicit_task_hint("候选池目前为空，无法执行筛选；请从零生成新材料")
        == "generate_material"
    )
    assert (
        explicit_task_hint("生成步骤已经结束，只在现成清单中淘汰")
        == "screen_candidates"
    )
    assert (
        explicit_task_hint("现在没有候选，不能做过滤；请生成新材料")
        == "generate_material"
    )
    assert (
        explicit_task_hint("数据库中还没有可筛选条目，请寻找新的组成")
        == "generate_material"
    )
    assert (
        explicit_task_hint("新结构生成工作已经完成，仅从已有集合中筛选")
        == "screen_candidates"
    )
    assert explicit_task_hint("希望材料的带隙较高") is None
    assert has_rare_earth_exclusion("稀土含量必须为零")
    assert has_rare_earth_exclusion("不依赖稀土成分")
    assert has_rare_earth_exclusion("元素组成不得包含任何稀土族成员")
    assert has_rare_earth_exclusion("配方中不能引入任何稀土原子")
    assert not has_rare_earth_exclusion("不含铅")


def test_interpreter_guardrail_corrects_explicit_screen_task(
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
            "lower_bound": null, "upper_bound": null, "unit": "eV"
          }],
          "composition": {
            "required_elements": [], "excluded_elements": [],
            "exclude_rare_earth": false
          },
          "free_text_constraints": []
        }""",
    )
    request = interpreter.interpret(
        "不再生成新材料，只在当前清单中找出带隙至少 1.5 eV 的项"
    )
    assert request.task.value == "screen_candidates"
    assert "guardrail_corrected:task:screen_candidates" in request.parser_warnings


def test_entity_gate_keeps_canonical_property_for_unseen_paraphrase(
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
            "lower_bound": null, "upper_bound": null, "unit": "eV"
          }],
          "composition": {
            "required_elements": [], "excluded_elements": [],
            "exclude_rare_earth": false
          },
          "free_text_constraints": []
        }""",
    )
    request = interpreter.interpret("价导带分离能不得小于 1.5 electronvolt")
    assert [prop.name for prop in request.target_properties] == ["band_gap"]
    assert request.ready_for_submission is True
    assert any(
        item.startswith("entity_gate_unverified_unlinked_properties:")
        for item in request.parser_warnings
    )
