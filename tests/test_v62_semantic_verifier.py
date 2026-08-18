from __future__ import annotations

from magnova_agent.models import MaterialRequest, TaskType
from magnova_agent.semantic_property_linker import SemanticPropertyMatch
from magnova_agent.v62_semantic_verifier import finalize_v62_semantic_slots


class FixedLinker:
    def __init__(self, label: str | None, *, accepted: bool = True) -> None:
        self.label = label
        self.accepted = accepted

    def link(self, text: str) -> SemanticPropertyMatch:
        return SemanticPropertyMatch(
            canonical_name=self.label if self.accepted else None,
            score=0.8,
            runner_up_score=0.4,
            margin=0.4,
            accepted=self.accepted,
            reason="accepted" if self.accepted else "ambiguous_margin",
            text=text,
        )


def apply(text: str) -> MaterialRequest:
    return finalize_v62_semantic_slots(
        MaterialRequest(original_text=text),
        text,
        property_linker=FixedLinker("band_gap"),  # type: ignore[arg-type]
        task_linker=FixedLinker("generate_material"),  # type: ignore[arg-type]
    )


def test_v62_repairs_explicit_symbol_without_semantic_guessing() -> None:
    request = apply("眼下没有候选，请设计价带到导带的距离 > 0.8 eV 的材料。")
    assert request.task == TaskType.GENERATE
    assert request.target_properties[0].name == "band_gap"
    assert request.target_properties[0].operator.value == "gt"
    assert request.target_properties[0].value == 0.8


def test_v62_semantically_links_unitless_optimization() -> None:
    request = apply("没有候选，请设计一种价带到导带的距离越大越好的材料。")
    assert request.target_properties[0].name == "band_gap"
    assert request.target_properties[0].operator.value == "maximize"


def test_v62_keeps_explicit_material_family() -> None:
    request = apply("请设计材料体系限定为 Mn-Al。")
    assert request.material_family == "Mn-Al"


def test_v62_recovers_explicit_rare_earth_exclusion() -> None:
    request = apply("请设计一种材料，并且所有稀土元素均须排除。")
    assert request.composition.exclude_rare_earth is True
    assert "v62_explicit_composition:exclude_rare_earth" in request.parser_warnings


def test_v62_recovers_explicit_required_aluminium() -> None:
    request = apply("请筛选带隙不低于1.4 eV且含Al的候选材料。")
    assert request.composition.required_elements == ["Al"]
    assert request.composition.excluded_elements == []


def test_v62_recovers_explicit_lead_exclusion_and_removes_conflict() -> None:
    seed = MaterialRequest(original_text="严禁含有Pb", composition={"required_elements": ["Pb"]})
    request = finalize_v62_semantic_slots(
        seed,
        "严禁含有Pb",
        property_linker=FixedLinker("band_gap"),  # type: ignore[arg-type]
        task_linker=FixedLinker("generate_material"),  # type: ignore[arg-type]
    )
    assert request.composition.required_elements == []
    assert request.composition.excluded_elements == ["Pb"]


def test_v62_does_not_treat_material_family_as_required_elements() -> None:
    request = apply("请设计Fe-Co体系的新材料。")
    assert request.composition.required_elements == []


def test_v62_repairs_explicit_task_and_low_cost_synonym() -> None:
    request = apply("从已计算完的候选材料中挑出带隙高于1.4 eV且便宜的材料。")
    assert request.task == TaskType.SCREEN
    assert request.free_text_constraints == ["低成本"]


def test_v62_keeps_low_cost_out_when_explicitly_unconstrained() -> None:
    request = apply("请从零设计带隙高于1.4 eV的材料，不考虑成本。")
    assert request.free_text_constraints == []


def test_v62_repairs_bound_text_operator() -> None:
    request = apply("请从零设计带隙不大于1.4 eV的新材料。")
    assert request.target_properties[0].operator.value == "lte"


def test_v62_repairs_formation_energy_unit_context() -> None:
    request = apply("请从零设计每原子形成能小于-1.25 eV且含Al的新材料。")
    assert request.target_properties[0].name == "formation_energy"
    assert request.target_properties[0].unit == "eV/atom"


def test_v62_repairs_fe_co_alias() -> None:
    request = apply("从计算完的候选中挑出铁钴材料。")
    assert request.material_family == "Fe-Co"


def test_v62_respects_explicit_rare_earth_permission() -> None:
    request = apply("请筛选材料，并且无需排除稀土族。")
    assert request.composition.exclude_rare_earth is False


def test_v62_keeps_strict_below_as_lt() -> None:
    request = apply("请筛选带隙必须落到0.6 eV以下，停在边界不收的材料。")
    assert request.target_properties[0].operator.value == "lt"
