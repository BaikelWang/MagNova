from __future__ import annotations

import re
from dataclasses import dataclass

from .interpreter import apply_critical_guardrails, parse_elements
from .models import Comparator, MaterialRequest, PropertyConstraint, TaskType
from .normalization import normalize_for_llm


NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
UNIT_PATTERN = re.compile(
    r"μB/Å³|玻尔磁子[·/]立方埃(?:的倒数)?|"
    r"eV/atom|eV/原子|eV每原子|电子伏特每原子|"
    r"(?<![A-Za-z])eV(?!\s*(?:/|每)\s*(?:atom|原子))|电子伏特|"
    r"(?<![A-Za-z])K(?![A-Za-z])|开尔文",
    re.IGNORECASE,
)
CLAUSE_BOUNDARY = re.compile(r"[，；。、\n]")

OPTIMIZATION_HIGH_MARKERS = (
    "高值优先",
    "从高到低",
    "选最高",
    "越高越好",
    "越大越好",
    "尽可能高",
    "尽量提高",
    "最大化",
    "推到更大",
    "提高到最大",
    "优先增大",
)
OPTIMIZATION_LOW_MARKERS = (
    "低值优先",
    "从低到高",
    "选最低",
    "越低越好",
    "越小越好",
    "尽可能低",
    "尽量降低",
    "最小化",
    "压到更小",
    "降低到最小",
    "优先减小",
)


@dataclass(frozen=True)
class ExplicitMeasurement:
    name: str
    operator: Comparator
    value: float | None
    lower_bound: float | None
    upper_bound: float | None
    unit: str
    clause: str


@dataclass(frozen=True)
class ExplicitOptimization:
    name: str
    operator: Comparator
    raw_name: str
    context: str


def _unit_identity(raw: str) -> tuple[str, str]:
    compact = raw.replace(" ", "").casefold()
    if "μb" in compact or "玻尔磁子" in compact:
        return "magnetic_moment_density", "μB/Å³"
    if "/atom" in compact or "/原子" in compact or "每原子" in compact:
        return "formation_energy", "eV/atom"
    if compact == "k" or "开尔文" in compact:
        return "curie_temperature", "K"
    return "band_gap", "eV"


def _detect_comparator(clause: str, number_count: int) -> Comparator:
    lowered = clause.casefold()
    if number_count >= 2 and any(
        marker in lowered
        for marker in ("区间", "窗口", "含两端", "含端点", " 至 ", "—", "~", "～", "[")
    ):
        return Comparator.RANGE
    if any(marker in lowered for marker in ("不少于", "不低于", "至少", "大于等于", ">=", "≥", "开始均合格", "起算", "最低接受线", "达到也算合格")):
        return Comparator.GTE
    if any(marker in lowered for marker in ("不多于", "不高于", "不超过", "至多", "小于等于", "<=", "≤", "截至", "为止", "最高允许到", "等于仍可接收")):
        return Comparator.LTE
    if any(marker in lowered for marker in ("必须跨过", "高出", "越过", "超过", "高于", "大于", "严格高过", "等号仍不合格")):
        return Comparator.GT
    if any(marker in lowered for marker in ("必须落到", "低出", "小过", "低于", "小于", "以下", "严格低于", "等于上限不接收")):
        return Comparator.LT
    if any(marker in lowered for marker in ("正好", "相同", "目标点", "目标锁定为", "精确等于", "只接受与", "只接受这个点", "等于", "=")):
        return Comparator.EQ
    return Comparator.EQ


def extract_explicit_measurements(text: str) -> list[ExplicitMeasurement]:
    normalized = normalize_for_llm(text).model_text
    measurements: list[ExplicitMeasurement] = []
    for unit_match in UNIT_PATTERN.finditer(normalized):
        boundaries = list(CLAUSE_BOUNDARY.finditer(normalized, 0, unit_match.start()))
        clause_start = boundaries[-1].end() if boundaries else 0
        next_boundary = CLAUSE_BOUNDARY.search(normalized, unit_match.end())
        clause_end = next_boundary.start() if next_boundary else len(normalized)
        clause = normalized[clause_start:clause_end]
        relative_unit_start = unit_match.start() - clause_start
        number_matches = list(NUMBER_PATTERN.finditer(clause[:relative_unit_start]))
        if not number_matches:
            continue
        values = [float(match.group(0)) for match in number_matches[-2:]]
        name, canonical_unit = _unit_identity(unit_match.group(0))
        operator = _detect_comparator(clause, len(values))
        if operator == Comparator.RANGE and len(values) >= 2:
            value = None
            lower_bound, upper_bound = values[-2], values[-1]
        else:
            value = values[-1]
            lower_bound = upper_bound = None
        measurements.append(
            ExplicitMeasurement(
                name=name,
                operator=operator,
                value=value,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                unit=canonical_unit,
                clause=clause,
            )
        )
    return measurements


def extract_explicit_optimizations(text: str) -> list[ExplicitOptimization]:
    """Bind maximize/minimize phrases to property names explicitly present in text."""
    normalization = normalize_for_llm(text)
    mentions = normalization.property_mentions
    optimizations: dict[str, ExplicitOptimization] = {}
    for index, mention in enumerate(mentions):
        next_start = mentions[index + 1].start if index + 1 < len(mentions) else len(text)
        context = text[mention.start:next_start]
        high_requested = any(marker in context for marker in OPTIMIZATION_HIGH_MARKERS)
        low_requested = any(marker in context for marker in OPTIMIZATION_LOW_MARKERS)
        if high_requested == low_requested:
            continue
        operator = Comparator.MAXIMIZE if high_requested else Comparator.MINIMIZE
        optimizations[mention.canonical_name] = ExplicitOptimization(
            name=mention.canonical_name,
            operator=operator,
            raw_name=mention.raw_name,
            context=context,
        )
    return list(optimizations.values())


def _explicit_task(text: str) -> TaskType | None:
    lowered = text.casefold()
    no_candidates = (
        "零候选",
        "没有任何现成",
        "没有任何候选",
        "池中没有条目",
        "列表是空",
        "候选池为空",
        "筛选无对象",
    )
    existing_candidates = (
        "列表已有",
        "候选计算已收尾",
        "池中已有条目",
        "候选已经生成",
        "已有候选",
        "现有材料池",
    )
    if any(marker in lowered for marker in no_candidates):
        return TaskType.GENERATE
    if any(marker in lowered for marker in existing_candidates):
        return TaskType.SCREEN
    return None


def _explicit_rare_earth(text: str) -> bool | None:
    allowed_markers = (
        "无需排除稀土",
        "可以使用稀土",
        "稀土元素不作额外限制",
        "无需限制稀土",
    )
    forbidden_markers = (
        "不得进入组成",
        "不允许含有任一稀土",
        "必须完全排除稀土",
        "不含稀土",
        "无稀土",
    )
    if any(marker in text for marker in allowed_markers):
        return False
    if any(marker in text for marker in forbidden_markers):
        return True
    return None


def recover_v612_slots(original_text: str) -> MaterialRequest:
    """Build a conservative request when MiniMind emits invalid JSON or Schema."""
    seed = MaterialRequest(
        original_text=original_text,
        parser_warnings=["v612_slot_recovery:model_output_unusable"],
    )
    return finalize_v61_slots(seed, original_text)


def finalize_v61_slots(request: MaterialRequest, original_text: str) -> MaterialRequest:
    """Repair only facts explicitly recoverable from the user's original text."""
    result = request.model_copy(deep=True)
    result.original_text = original_text
    normalization = normalize_for_llm(original_text)
    measurements = extract_explicit_measurements(original_text)
    optimizations = extract_explicit_optimizations(original_text)
    measurement_by_name = {item.name: item for item in measurements}
    optimization_by_name = {item.name: item for item in optimizations}
    predicted_by_name = {item.name: item for item in result.target_properties}

    if optimization_by_name:
        predicted_by_name = {
            name: item
            for name, item in predicted_by_name.items()
            if item.operator not in {Comparator.MAXIMIZE, Comparator.MINIMIZE}
        }
        for name, optimization in optimization_by_name.items():
            predicted_by_name[name] = PropertyConstraint(
                name=name,
                raw_name=optimization.raw_name,
                operator=optimization.operator,
            )
            result.parser_warnings.append(
                f"v612_slot_verified:optimization:{name}:{optimization.operator.value}"
            )

    for name, measurement in measurement_by_name.items():
        previous = predicted_by_name.get(name)
        predicted_by_name[name] = PropertyConstraint(
            name=name,
            raw_name=(
                previous.raw_name
                if previous is not None and previous.raw_name
                else normalization.raw_name_for(name)
            ),
            operator=measurement.operator,
            value=measurement.value,
            lower_bound=measurement.lower_bound,
            upper_bound=measurement.upper_bound,
            unit=measurement.unit,
        )
        result.parser_warnings.append(f"v61_slot_verified:{name}")

    explicit_names = set(measurement_by_name) | set(optimization_by_name)
    if explicit_names:
        result.target_properties = sorted(
            (predicted_by_name[name] for name in explicit_names),
            key=lambda item: item.name,
        )
    else:
        result.target_properties = sorted(
            predicted_by_name.values(), key=lambda item: item.name
        )

    task = _explicit_task(original_text)
    if task is not None and result.task != task:
        result.task = task
        result.parser_warnings.append(f"v61_slot_verified:task:{task.value}")

    required = parse_elements(original_text, r"(?:必须含|必须包含|需要含|需要包含|含有)")
    excluded = parse_elements(original_text, r"(?:不含|不得含|不能含|排除|避免使用)")
    if result.composition.required_elements != required:
        result.composition.required_elements = required
        result.parser_warnings.append("v61_slot_verified:required_elements")
    if result.composition.excluded_elements != excluded:
        result.composition.excluded_elements = excluded
        result.parser_warnings.append("v61_slot_verified:excluded_elements")

    family_match = re.search(
        r"体系为\s*([A-Z][a-z]?(?:[-–][A-Z][a-z]?){1,4})",
        original_text,
    )
    if family_match:
        result.material_family = family_match.group(1).replace("–", "-")
        result.parser_warnings.append("v61_slot_verified:material_family")

    if (
        not family_match
        and result.material_family
        and result.material_family not in original_text
    ):
        result.material_family = None
        result.parser_warnings.append("v613_slot_verified:cleared_unlinked_family")

    allowed_constraints = ("室温稳定", "低成本", "无毒", "可合成")
    result.free_text_constraints = [
        item for item in allowed_constraints if item in normalization.model_text
    ]
    result = apply_critical_guardrails(result, original_text)
    explicit_rare_earth = _explicit_rare_earth(original_text)
    if (
        explicit_rare_earth is not None
        and result.composition.exclude_rare_earth != explicit_rare_earth
    ):
        result.composition.exclude_rare_earth = explicit_rare_earth
        result.parser_warnings.append("v61_slot_verified:exclude_rare_earth")
    result.parser_warnings = list(dict.fromkeys(result.parser_warnings))
    return result

@dataclass
class SlotVerifiedInterpreter:
    """Compose any requirement interpreter with deterministic slot verification."""

    base: object

    def interpret(self, text: str) -> MaterialRequest:
        try:
            request = self.base.interpret(text)  # type: ignore[attr-defined]
        except Exception as exc:
            recovered = recover_v612_slots(text)
            recovered.parser_warnings.append(
                f"slot_verified_interpreter_recovery:{type(exc).__name__}"
            )
            return recovered
        return finalize_v61_slots(request, text)
