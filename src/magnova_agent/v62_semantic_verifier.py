from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Comparator, MaterialRequest, PropertyConstraint, TaskType
from .normalization import normalize_for_llm
from .semantic_property_linker import SemanticPropertyLinker
from .semantic_task_linker import SemanticTaskLinker
from .v6_slot_verifier import finalize_v61_slots, recover_v612_slots


def _explicit_symbol_operator(text: str) -> Comparator | None:
    patterns = (
        (r"(?:>=|≥)\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)", Comparator.GTE),
        (r"(?:<=|≤)\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)", Comparator.LTE),
        (r"(?<![>=])>\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)", Comparator.GT),
        (r"(?<![<=])<\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)", Comparator.LT),
    )
    matches = [operator for pattern, operator in patterns if re.search(pattern, text)]
    return matches[0] if len(matches) == 1 else None


def _explicit_optimization_operator(text: str) -> Comparator | None:
    high = ("越大越好", "越高越好", "最大化", "推到更大", "优先增大", "数值越大排序越靠前", "数值越高排序越靠前")
    low = ("越小越好", "越低越好", "最小化", "压到更小", "优先减小", "数值越小排序越靠前", "数值越低排序越靠前")
    high_requested = any(marker in text for marker in high)
    low_requested = any(marker in text for marker in low)
    if high_requested == low_requested:
        return None
    return Comparator.MAXIMIZE if high_requested else Comparator.MINIMIZE


def _explicit_family(text: str) -> str | None:
    match = re.search(
        r"(?:体系为|材料体系(?:限定)?为)\s*([A-Z][a-z]?(?:[-–][A-Z][a-z]?){1,4})",
        text,
    )
    return match.group(1).replace("–", "-") if match else None


def _has_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _apply_explicit_composition(request: MaterialRequest, original_text: str) -> None:
    """Repair composition only when the text contains unambiguous evidence."""
    compact = re.sub(r"\s+", "", original_text)
    composition = request.composition.model_copy(deep=True)
    rare_earth_allowed = any(
        marker in compact
        for marker in ("无需排除稀土", "可以使用稀土", "稀土元素不作额外限制", "无需限制稀土")
    )

    excludes_pb = _has_any_pattern(
        compact,
        (
            r"(?:不含|不包含|无|别含|不要含?|不得含|不得包含|不能含|不能有|严禁含|禁止使用|排除(?:任何)?含?|千万别含?)(?:有)?(?:铅元素|铅|Pb)",
            r"(?:铅元素|铅|Pb)(?:的)?(?:不要|不能有|不得有|必须排除|须排除|需排除)",
            r"除(?:铅|Pb)外",
        ),
    )
    excludes_rare_earth = _has_any_pattern(
        compact,
        (
            r"(?:不含|不包含|别用|不要|不得含|不能含|不能使用|严禁含|严禁包含|禁止使用|排除(?:所有)?含?|绝不能含?)(?:有)?(?:任何|所有)?稀土(?:元素)?",
            r"稀土(?:元素)?(?:均|都|一律)?(?:须|需|必须)?排除",
        ),
    )
    excludes_al = _has_any_pattern(
        compact,
        (r"(?:不含|不包含|无|别含|不要|不得含|不能含|严禁含|排除含?)(?:有)?(?:铝元素|铝|Al)",),
    )
    requires_al = not excludes_al and _has_any_pattern(
        compact,
        (
            r"(?:必须|务必|一定要|强制要求|还得|要)(?:有|包含|含有|含)(?:铝元素|铝|Al)",
            r"(?:包含|含有|含)(?:铝元素|铝|Al)",
        ),
    )

    if excludes_pb:
        composition.required_elements = [x for x in composition.required_elements if x != "Pb"]
        if "Pb" not in composition.excluded_elements:
            composition.excluded_elements.append("Pb")
        request.parser_warnings.append("v62_explicit_composition:exclude:Pb")
    if requires_al:
        composition.excluded_elements = [x for x in composition.excluded_elements if x != "Al"]
        if "Al" not in composition.required_elements:
            composition.required_elements.append("Al")
        request.parser_warnings.append("v62_explicit_composition:require:Al")
    if excludes_rare_earth and not rare_earth_allowed:
        composition.exclude_rare_earth = True
        request.parser_warnings.append("v62_explicit_composition:exclude_rare_earth")
    request.composition = composition


def _explicit_text_operator(text: str) -> Comparator | None:
    compact = re.sub(r"\s+", "", text)
    groups = (
        (Comparator.GTE, ("不低于", "不少于", "大于或等于", "大于等于", "及以上")),
        (Comparator.LTE, ("不高于", "不超过", "不大于", "不超", "小于或等于", "小于等于", "以下")),
        (Comparator.GT, ("严格大于", "高于", "超过", "大于")),
        (Comparator.LT, ("严格小于", "低于", "小于")),
    )
    matches = [operator for operator, markers in groups if any(marker in compact for marker in markers)]
    return matches[0] if len(matches) == 1 else None


def _explicit_task_v62(text: str) -> TaskType | None:
    compact = re.sub(r"\s+", "", text)
    generate_context = (
        "没有任何候选", "没有候选", "没候选", "候选池为空", "候选列表为空",
        "列表为空", "从零", "从头", "全新的材料生成",
    )
    screen_context = (
        "已计算", "计算完成", "计算完毕", "计算完", "已算完", "算好的",
        "候选已生成", "现有候选", "候选池中", "候选材料中", "候选集中",
        "候选集合中", "候选库里", "候选列表中", "计算任务均已结束",
        "已完成的那些", "现有计算结果中", "这些候选的计算结果",
    )
    screen_actions = ("筛选", "挑选", "挑出", "挑出来", "找出", "选出", "提取", "甄选")
    if any(context in compact for context in screen_context) and any(
        action in compact for action in screen_actions
    ):
        return TaskType.SCREEN
    if any(marker in compact for marker in generate_context):
        return TaskType.GENERATE
    generate_actions = ("请设计", "需要设计", "计划设计", "开发一种", "开发一款", "生成候选")
    if any(action in compact for action in generate_actions):
        return TaskType.GENERATE
    return None


def _apply_explicit_low_cost(request: MaterialRequest, original_text: str) -> None:
    compact = re.sub(r"\s+", "", original_text)
    negations = ("不考虑成本", "成本不作限制", "无需考虑成本", "不限制成本")
    markers = (
        "低成本", "成本低", "成本较低", "成本低廉", "控制成本", "降低成本",
        "便宜", "经济可行", "原料经济性", "预算有限", "成本也要低",
        "成本必须低", "成本要低", "成本应尽量低", "成本方面也要低",
    )
    if not any(marker in compact for marker in negations) and any(
        marker in compact for marker in markers
    ):
        if "低成本" not in request.free_text_constraints:
            request.free_text_constraints.append("低成本")
        request.parser_warnings.append("v62_explicit_constraint:low_cost")


def _apply_explicit_measurement_repairs(request: MaterialRequest, original_text: str) -> None:
    if len(request.target_properties) != 1:
        return
    prop = request.target_properties[0].model_copy(deep=True)
    compact = re.sub(r"\s+", "", original_text)
    if "形成能" in compact and prop.name == "band_gap":
        prop.name = "formation_energy"
        prop.unit = "eV/atom"
        request.parser_warnings.append("v62_explicit_property:formation_energy")
    operator_patterns = (
        (Comparator.GTE, r"(?:不低于|不少于|大于或等于|大于等于)[^，。；]{0,16}[+-]?(?:\d+(?:\.\d*)?|\.\d+)|[+-]?(?:\d+(?:\.\d*)?|\.\d+)[^，。；]{0,8}及以上"),
        (Comparator.LTE, r"(?:不高于|不超过|不大于|不超|小于或等于|小于等于)[^，。；]{0,16}[+-]?(?:\d+(?:\.\d*)?|\.\d+)"),
        (Comparator.GT, r"(?:严格大于|高于|超过|大于)[^，。；]{0,16}[+-]?(?:\d+(?:\.\d*)?|\.\d+)"),
        (Comparator.LT, r"(?:严格小于|低于|小于)[^，。；]{0,16}[+-]?(?:\d+(?:\.\d*)?|\.\d+)"),
    )
    matched_operator = next(
        (operator for operator, pattern in operator_patterns if re.search(pattern, compact)),
        None,
    )
    if matched_operator is not None and prop.value is not None and prop.operator != matched_operator:
        prop.operator = matched_operator
        request.parser_warnings.append(f"v62_explicit_text_operator:{matched_operator.value}")
    request.target_properties = [prop]


def _apply_explicit_family_alias(request: MaterialRequest, original_text: str) -> None:
    compact = re.sub(r"\s+", "", original_text)
    if re.search(r"(?:Fe[-–]?Co|铁钴)(?:基|系|体系|合金|材料)", compact, re.IGNORECASE):
        if request.material_family != "Fe-Co":
            request.material_family = "Fe-Co"
            request.parser_warnings.append("v62_explicit_family_alias:Fe-Co")


def apply_v62_semantic_overlay(
    finalized_request: MaterialRequest,
    original_text: str,
    *,
    property_linker: SemanticPropertyLinker,
    task_linker: SemanticTaskLinker,
) -> MaterialRequest:
    """Conservative overlay: never rebuild slots already finalized by V6.1.3."""
    result = finalized_request.model_copy(deep=True)
    result.original_text = original_text
    normalized = normalize_for_llm(original_text)

    task_match = task_linker.link(original_text)
    result.parser_warnings.append(
        f"v62_semantic_task_observed:{task_match.canonical_name or 'abstain'}:"
        f"score={task_match.score:.4f}:margin={task_match.margin:.4f}"
    )

    symbol = _explicit_symbol_operator(original_text)
    if symbol is not None and len(result.target_properties) == 1:
        prop = result.target_properties[0].model_copy(deep=True)
        if prop.value is not None and prop.operator != symbol:
            prop.operator = symbol
            result.target_properties = [prop]
            result.parser_warnings.append(f"v62_deterministic_symbol_operator:{symbol.value}")

    optimization = _explicit_optimization_operator(original_text)
    if not normalized.property_mentions and optimization is not None:
        match = property_linker.link(original_text)
        if match.accepted and match.canonical_name:
            result.target_properties = [
                PropertyConstraint(name=match.canonical_name, operator=optimization)
            ]
            result.parser_warnings.append(
                f"v62_semantic_property:{match.canonical_name}:"
                f"score={match.score:.4f}:margin={match.margin:.4f}"
            )
        else:
            result.parser_warnings.append(
                f"v62_semantic_property_abstained:{match.reason}:"
                f"score={match.score:.4f}:margin={match.margin:.4f}"
            )

    family = _explicit_family(original_text)
    if family is not None and result.material_family != family:
        result.material_family = family
        result.parser_warnings.append("v62_deterministic_material_family")
    task = _explicit_task_v62(original_text)
    if task is not None and result.task != task:
        result.task = task
        result.parser_warnings.append(f"v62_explicit_task:{task.value}")
    _apply_explicit_measurement_repairs(result, original_text)
    _apply_explicit_family_alias(result, original_text)
    _apply_explicit_composition(result, original_text)
    _apply_explicit_low_cost(result, original_text)
    result.parser_warnings = list(dict.fromkeys(result.parser_warnings))
    return result


def finalize_v62_semantic_slots(
    request: MaterialRequest,
    original_text: str,
    *,
    property_linker: SemanticPropertyLinker,
    task_linker: SemanticTaskLinker,
) -> MaterialRequest:
    return apply_v62_semantic_overlay(
        finalize_v61_slots(request, original_text),
        original_text,
        property_linker=property_linker,
        task_linker=task_linker,
    )


@dataclass
class SemanticSlotVerifiedInterpreter:
    base: object
    property_linker: SemanticPropertyLinker
    task_linker: SemanticTaskLinker

    def interpret(self, text: str) -> MaterialRequest:
        try:
            request = self.base.interpret(text)  # type: ignore[attr-defined]
        except Exception as exc:
            request = recover_v612_slots(text)
            request.parser_warnings.append(f"v62_interpreter_recovery:{type(exc).__name__}")
        return finalize_v62_semantic_slots(request, text, property_linker=self.property_linker, task_linker=self.task_linker)
