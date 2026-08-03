from __future__ import annotations

from magnova_agent import (
    MagNovaAgent,
    MultiTurnMagNovaAgent,
    RuleBasedInterpreter,
)
from magnova_agent.models import AgentStatus


def build_agent() -> MultiTurnMagNovaAgent:
    return MultiTurnMagNovaAgent(
        MagNovaAgent(interpreter=RuleBasedInterpreter())
    )


def test_missing_unit_is_filled_on_second_turn() -> None:
    agent = build_agent()
    first = agent.run("设计磁矩密度大于 0.2 的材料", session_id="a")
    assert first.status == AgentStatus.NEEDS_CLARIFICATION
    assert agent.state("a").pending_request is not None

    second = agent.run("单位是 μB/Å³", session_id="a")
    assert second.status == AgentStatus.SUBMITTED
    assert second.request.target_properties[0].unit == "μB/Å³"
    assert agent.state("a").pending_request is None


def test_named_units_fill_two_properties_over_multiple_turns() -> None:
    agent = build_agent()
    first = agent.run(
        "设计带隙大于 1.5 且居里温度高于 500 的材料",
        session_id="b",
    )
    assert first.status == AgentStatus.NEEDS_CLARIFICATION

    second = agent.run("带隙单位是 eV", session_id="b")
    assert second.status == AgentStatus.NEEDS_CLARIFICATION
    units = {prop.name: prop.unit for prop in second.request.target_properties}
    assert units["band_gap"] == "eV"
    assert units["curie_temperature"] is None

    third = agent.run("居里温度用 K", session_id="b")
    assert third.status == AgentStatus.SUBMITTED
    units = {prop.name: prop.unit for prop in third.request.target_properties}
    assert units == {"curie_temperature": "K", "band_gap": "eV"}


def test_ambiguous_unit_does_not_fill_multiple_slots() -> None:
    agent = build_agent()
    agent.run(
        "设计带隙大于 1.5 且居里温度高于 500 的材料",
        session_id="c",
    )
    second = agent.run("单位是 eV", session_id="c")
    assert second.status == AgentStatus.NEEDS_CLARIFICATION
    assert all(prop.unit is None for prop in second.request.target_properties)


def test_sessions_are_isolated_and_resettable() -> None:
    agent = build_agent()
    agent.run("设计带隙大于 1.5 的材料", session_id="one")
    assert agent.state("one").pending_request is not None
    assert agent.state("two").pending_request is None
    agent.reset("one")
    assert agent.state("one").pending_request is None
