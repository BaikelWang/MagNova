from magnova_agent.models import (
    Comparator,
    CompositionConstraint,
    MaterialRequest,
    PropertyConstraint,
    TaskType,
)
from magnova_agent.v6_slot_verifier import (
    SlotVerifiedInterpreter,
    extract_explicit_measurements,
    extract_explicit_optimizations,
    finalize_v61_slots,
    recover_v612_slots,
)


def request(*properties: PropertyConstraint) -> MaterialRequest:
    return MaterialRequest(
        task=TaskType.SCREEN,
        target_properties=list(properties),
        composition=CompositionConstraint(),
        original_text="placeholder",
    )


def test_extracts_negative_formation_energy_and_strict_operator() -> None:
    items = extract_explicit_measurements(
        "请设计形成能必须低于 -2.093 eV/atom 的材料"
    )
    assert len(items) == 1
    assert items[0].name == "formation_energy"
    assert items[0].operator == Comparator.LT
    assert items[0].value == -2.093
    assert items[0].unit == "eV/atom"


def test_extracts_closed_range() -> None:
    items = extract_explicit_measurements(
        "带隙应在含两端的 0.5 至 1.2 eV 窗口内"
    )
    assert len(items) == 1
    assert items[0].operator == Comparator.RANGE
    assert items[0].lower_bound == 0.5
    assert items[0].upper_bound == 1.2


def test_repairs_model_numeric_and_operator_errors() -> None:
    model_request = request(
        PropertyConstraint(
            name="curie_temperature",
            operator=Comparator.LT,
            value=59,
            unit="K",
        )
    )
    repaired = finalize_v61_slots(
        model_request,
        "列表已有结果，请只筛选居里温度不少于 324 K 的对象",
    )
    prop = repaired.target_properties[0]
    assert repaired.task == TaskType.SCREEN
    assert prop.operator == Comparator.GTE
    assert prop.value == 324


def test_adds_two_properties_from_unambiguous_units() -> None:
    repaired = finalize_v61_slots(
        request(),
        "候选池为空，请设计形成能低于 -1.2 eV/atom、带隙高于 0.8 eV 的材料",
    )
    assert repaired.task == TaskType.GENERATE
    assert [item.name for item in repaired.target_properties] == [
        "band_gap",
        "formation_energy",
    ]


def test_preserves_unmeasured_optimization_objective() -> None:
    model_request = request(
        PropertyConstraint(
            name="magnetic_moment_density",
            operator=Comparator.MINIMIZE,
        )
    )
    repaired = finalize_v61_slots(
        model_request,
        "没有任何候选，请生成磁矩密度不设门槛、越高越好的材料",
    )
    assert repaired.task == TaskType.GENERATE
    assert repaired.target_properties[0].operator == Comparator.MAXIMIZE


def test_repairs_wrong_optimization_property_name() -> None:
    model_request = request(
        PropertyConstraint(
            name="curie_temperature",
            operator=Comparator.MAXIMIZE,
        )
    )
    repaired = finalize_v61_slots(
        model_request,
        "没有候选，请设计单位体积磁矩不设门槛，只按数值从高到低优选的材料",
    )
    assert [(item.name, item.operator) for item in repaired.target_properties] == [
        ("magnetic_moment_density", Comparator.MAXIMIZE)
    ]


def test_binds_two_optimization_properties_independently() -> None:
    items = extract_explicit_optimizations(
        "请筛选形成能不设门槛，只按数值从低到高优选；"
        "带隙不设门槛，只按数值从高到低优选"
    )
    assert [(item.name, item.operator) for item in items] == [
        ("formation_energy", Comparator.MINIMIZE),
        ("band_gap", Comparator.MAXIMIZE),
    ]


def test_preserves_numeric_property_and_repairs_optimization_property() -> None:
    model_request = request(
        PropertyConstraint(
            name="magnetic_moment_density",
            operator=Comparator.MAXIMIZE,
        )
    )
    repaired = finalize_v61_slots(
        model_request,
        "带隙高于 0.8 eV；居里温度不设门槛，只按数值从低到高优选",
    )
    assert [(item.name, item.operator) for item in repaired.target_properties] == [
        ("band_gap", Comparator.GT),
        ("curie_temperature", Comparator.MINIMIZE),
    ]


def test_does_not_convert_plain_property_discussion_into_optimization() -> None:
    model_request = request(
        PropertyConstraint(
            name="band_gap",
            operator=Comparator.EQ,
            value=1.0,
            unit="eV",
        )
    )
    repaired = finalize_v61_slots(
        model_request,
        "我们先讨论带隙的物理意义，暂时不要设置设计目标",
    )
    assert repaired.target_properties[0].name == "band_gap"
    assert repaired.target_properties[0].operator == Comparator.EQ


def test_ood_scientific_alias_maps_to_curie_temperature() -> None:
    items = extract_explicit_optimizations(
        "自发磁化能够维持到的热学上限没有门槛，直接选最高"
    )
    assert len(items) == 1
    assert items[0].name == "curie_temperature"


def test_v613_scientific_alias_and_new_strict_comparator() -> None:
    repaired = finalize_v61_slots(
        request(),
        "候选集合仍是空的，请新建设计，并确保晶体相相对元素基态的每原子能量差"
        "务必严格高过 -1.8 eV/atom，达到等号仍不合格",
    )
    assert repaired.task == TaskType.GENERATE
    prop = repaired.target_properties[0]
    assert prop.name == "formation_energy"
    assert prop.operator == Comparator.GT
    assert prop.value == -1.8


def test_v613_existing_candidates_and_inclusive_upper_bound() -> None:
    repaired = finalize_v61_slots(
        request(),
        "已有一批计算完成的候选，请淘汰不符合价带顶与导带底之间的能量差"
        "最高允许到 1.4 eV，刚好等于仍可接收的材料",
    )
    assert repaired.task == TaskType.SCREEN
    prop = repaired.target_properties[0]
    assert prop.name == "band_gap"
    assert prop.operator == Comparator.LTE
    assert prop.value == 1.4


def test_v613_new_optimization_language() -> None:
    repaired = finalize_v61_slots(
        request(
            PropertyConstraint(
                name="band_gap",
                operator=Comparator.MINIMIZE,
            )
        ),
        "铁磁序消失的临界温度不设置硬门槛，排序时优先把这一指标推到更大",
    )
    assert [(item.name, item.operator) for item in repaired.target_properties] == [
        ("curie_temperature", Comparator.MAXIMIZE)
    ]


def test_slot_verified_interpreter_wraps_successful_base() -> None:
    class WrongBase:
        def interpret(self, text: str) -> MaterialRequest:
            return request(
                PropertyConstraint(
                    name="curie_temperature",
                    operator=Comparator.EQ,
                    value=1,
                    unit="K",
                )
            )

    wrapped = SlotVerifiedInterpreter(WrongBase()).interpret(
        "请设计带隙严格高过 0.9 eV 的材料"
    )
    assert wrapped.target_properties[0].name == "band_gap"
    assert wrapped.target_properties[0].operator == Comparator.GT
    assert wrapped.target_properties[0].value == 0.9


def test_recovers_explicit_slots_when_model_output_is_unusable() -> None:
    recovered = recover_v612_slots(
        "候选计算已收尾，请筛选铁磁长程关联消失时的温度点必须跨过 420 开尔文的材料"
    )
    assert recovered.task == TaskType.SCREEN
    assert len(recovered.target_properties) == 1
    prop = recovered.target_properties[0]
    assert prop.name == "curie_temperature"
    assert prop.operator == Comparator.GT
    assert prop.value == 420
    assert prop.unit == "K"
    assert recovered.ready_for_submission is True


def test_explicit_rare_earth_prohibition_overrides_model() -> None:
    model_request = request()
    model_request.composition.exclude_rare_earth = False
    repaired = finalize_v61_slots(
        model_request,
        "任何稀土元素都不得进入组成",
    )
    assert repaired.composition.exclude_rare_earth is True


def test_explicit_rare_earth_permission_overrides_keyword_match() -> None:
    model_request = request()
    model_request.composition.exclude_rare_earth = True
    repaired = finalize_v61_slots(
        model_request,
        "稀土元素不作额外限制，无需排除稀土族",
    )
    assert repaired.composition.exclude_rare_earth is False


def test_normalized_ev_per_atom_remains_formation_energy() -> None:
    items = extract_explicit_measurements(
        "形成能低于 -2.1 电子伏特每原子"
    )
    assert len(items) == 1
    assert items[0].name == "formation_energy"
    assert items[0].unit == "eV/atom"
    assert items[0].value == -2.1
