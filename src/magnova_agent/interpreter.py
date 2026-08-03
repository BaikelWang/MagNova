from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

import requests
from pydantic import ValidationError

from .models import (
    Comparator,
    CompositionConstraint,
    MaterialRequest,
    PropertyConstraint,
    TaskType,
)
from .normalization import (
    explicit_task_hint,
    has_rare_earth_exclusion,
    normalize_for_llm,
)


class RequirementInterpreter(Protocol):
    def interpret(self, text: str) -> MaterialRequest:
        """Convert user text into the stable domain Schema."""


class InterpreterError(RuntimeError):
    """Raised when an LLM response cannot be converted into MaterialRequest."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from free-form model output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)

    start = cleaned.find("{")
    if start < 0:
        raise InterpreterError("MiniMind did not return a JSON object")
    try:
        payload, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise InterpreterError("MiniMind returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise InterpreterError("MiniMind JSON output must be an object")
    return payload


SYSTEM_PROMPT = """你是 MagNova 材料需求解析器。用户输入既可能是原始自然语言，也可能含有 <canonical_name> 规范标记；没有标记时也必须依据完整语义识别需求。把需求转换成一个合法 JSON 对象，不解释、不使用 Markdown。
只输出 task、material_family、target_properties、composition、free_text_constraints。
task 只能是 generate_material、screen_candidates 或 predict_properties。判断 task 时看当前候选状态和用户最终动作，不要只看某个关键词：当前没有候选、需要提出新组成或从零设计时用 generate_material；已经有候选、列表、计算结果或材料池，并且只要求保留、淘汰、过滤或核验时用 screen_candidates。
target_properties 每项只能包含 name、operator、value、lower_bound、upper_bound、unit，并按 name 字母顺序排列；不要输出 raw_name。operator 只能是 gt、gte、lt、lte、eq、range、maximize、minimize。
规范性质名只有 magnetic_moment_density、curie_temperature、formation_energy、band_gap。不能猜测用户未给出的数值或单位；缺失值用 null，缺失列表用 []。
composition 只能包含 required_elements、excluded_elements、exclude_rare_earth。
free_text_constraints 只保留规范短语：室温稳定、低成本、无毒、可合成。
不要输出 original_text、parser_warnings、raw_name 或任何其他字段。"""


@dataclass
class MiniMindInterpreter:
    """Use the internal MiniMind OpenAI-compatible API for structured parsing."""

    endpoint: str = "http://127.0.0.1:8503/v1/chat/completions"
    timeout_seconds: float = 60.0
    repair_attempts: int = 1
    fallback: RequirementInterpreter | None = None
    normalize_input: bool = True
    system_prompt: str = SYSTEM_PROMPT

    def _complete(self, messages: list[dict[str, str]]) -> str:
        response = requests.post(
            self.endpoint,
            json={
                "model": "minimind",
                "messages": messages,
                "temperature": 0.1,
                "top_p": 0.8,
                "do_sample": False,
                "max_tokens": 512,
                "stream": False,
                "open_thinking": False,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InterpreterError("MiniMind API returned an unexpected response") from exc

    @staticmethod
    def _validate(text: str, original_text: str) -> MaterialRequest:
        payload = extract_json_object(text)
        payload["original_text"] = original_text
        warnings = payload.get("parser_warnings")
        if not isinstance(warnings, list):
            payload["parser_warnings"] = []
        return MaterialRequest.model_validate(payload)

    def interpret(self, text: str) -> MaterialRequest:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            raise ValueError("user text must not be empty")

        normalization = normalize_for_llm(cleaned)
        model_text = normalization.model_text if self.normalize_input else cleaned
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": model_text},
        ]
        last_error: Exception | None = None
        for attempt in range(self.repair_attempts + 1):
            raw_output = ""
            try:
                raw_output = self._complete(messages)
                request = self._validate(raw_output, cleaned)
                if self.normalize_input:
                    recognized_names = {
                        mention.canonical_name
                        for mention in normalization.property_mentions
                    }
                    unlinked = [
                        prop.name
                        for prop in request.target_properties
                        if prop.name not in recognized_names
                    ]
                    if unlinked:
                        # MaterialRequest already limits names to the closed
                        # ontology. The normalizer lexicon is intentionally
                        # incomplete, so preserve new paraphrases and surface
                        # uncertainty instead of deleting valid properties.
                        request.parser_warnings.append(
                            "entity_gate_unverified_unlinked_properties:"
                            + ",".join(sorted(set(unlinked)))
                        )
                    for prop in request.target_properties:
                        if prop.raw_name is None:
                            prop.raw_name = normalization.raw_name_for(prop.name)
                    request.parser_warnings.extend(
                        f"normalized:{item}" for item in normalization.trace
                    )
                return apply_critical_guardrails(request, cleaned)
            except (requests.RequestException, ValueError, ValidationError, InterpreterError) as exc:
                last_error = exc
                if isinstance(exc, requests.RequestException) or attempt >= self.repair_attempts:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": raw_output},
                        {
                            "role": "user",
                            "content": (
                                "上一个输出不符合 Schema。错误是："
                                f"{exc}。请修复并只返回完整合法 JSON。"
                            ),
                        },
                    ]
                )

        if self.fallback is not None:
            request = self.fallback.interpret(cleaned)
            request.parser_warnings.append(
                f"minimind_fallback:{type(last_error).__name__ if last_error else 'unknown'}"
            )
            return request
        raise InterpreterError(f"MiniMind interpretation failed: {last_error}") from last_error


ELEMENT_EXCLUSION_PREFIX = r"(?:不含|不得含|不能含|成分中不要|排除|避免使用)"
CHINESE_ELEMENT_ALIASES = {
    "铅": "Pb",
    "汞": "Hg",
    "镉": "Cd",
    "砷": "As",
}


def apply_critical_guardrails(
    request: MaterialRequest,
    original_text: str,
) -> MaterialRequest:
    """Correct explicit safety/composition facts after probabilistic decoding."""
    expected_rare_earth = has_rare_earth_exclusion(original_text)
    if request.composition.exclude_rare_earth != expected_rare_earth:
        request.composition.exclude_rare_earth = expected_rare_earth
        request.parser_warnings.append("guardrail_corrected:exclude_rare_earth")

    explicit_excluded = parse_elements(original_text, ELEMENT_EXCLUSION_PREFIX)
    for chinese_name, symbol in CHINESE_ELEMENT_ALIASES.items():
        if re.search(
            ELEMENT_EXCLUSION_PREFIX + r"\s*(?:元素)?\s*" + chinese_name,
            original_text,
        ):
            explicit_excluded.append(symbol)
    for symbol in explicit_excluded:
        if symbol not in request.composition.excluded_elements:
            request.composition.excluded_elements.append(symbol)
            request.parser_warnings.append(
                f"guardrail_added:excluded_element:{symbol}"
            )

    task_hint = explicit_task_hint(original_text)
    if task_hint is not None and request.task.value != task_hint:
        request.task = TaskType(task_hint)
        request.parser_warnings.append(f"guardrail_corrected:task:{task_hint}")
    return request


@dataclass(frozen=True)
class PropertySpec:
    canonical_name: str
    aliases: tuple[str, ...]
    canonical_unit: str


PROPERTY_SPECS = (
    PropertySpec("magnetic_moment_density", ("磁矩密度", "磁化密度", "magnetic moment density"), "μB/Å³"),
    PropertySpec("curie_temperature", ("居里温度", "curie temperature", "tc"), "K"),
    PropertySpec("formation_energy", ("形成能", "formation energy"), "eV/atom"),
    PropertySpec("band_gap", ("带隙", "band gap"), "eV"),
)

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
UNIT = r"μB/Å³|μB/Å3|uB/A\^?3|eV/atom|eV|K"

UNIT_ALIASES = {
    "μb/å³": "μB/Å³",
    "μb/å3": "μB/Å³",
    "ub/a3": "μB/Å³",
    "ub/a^3": "μB/Å³",
    "k": "K",
    "ev/atom": "eV/atom",
    "ev": "eV",
}

COMPARATOR_PATTERNS = (
    (Comparator.GTE, ("不低于", "至少", "大于等于", ">=" , "≥")),
    (Comparator.LTE, ("不高于", "至多", "不超过", "小于等于", "<=", "≤")),
    (Comparator.GT, ("大于", "超过", "高于", ">")),
    (Comparator.LT, ("小于", "低于", "低于", "<")),
    (Comparator.EQ, ("等于", "为", "=", "约为", "接近")),
)


def normalize_unit(raw: str | None) -> str | None:
    if raw is None:
        return None
    compact = raw.strip().replace(" ", "")
    return UNIT_ALIASES.get(compact.lower(), compact)


def detect_comparator(context: str, has_value: bool) -> Comparator:
    lowered = context.lower()
    for comparator, markers in COMPARATOR_PATTERNS:
        if any(marker.lower() in lowered for marker in markers):
            return comparator
    if any(marker in lowered for marker in ("最大化", "尽可能高", "越高越好", "较高", "高")):
        return Comparator.MAXIMIZE
    if any(marker in lowered for marker in ("最小化", "尽可能低", "越低越好", "较低", "低")):
        return Comparator.MINIMIZE
    return Comparator.EQ if has_value else Comparator.EQ


def detect_task(text: str) -> TaskType:
    lowered = text.lower()
    if any(word in lowered for word in ("筛选", "过滤", "screen")):
        return TaskType.SCREEN
    if any(word in lowered for word in ("预测", "predict")):
        return TaskType.PREDICT
    return TaskType.GENERATE


def parse_elements(text: str, prefix_pattern: str) -> list[str]:
    match = re.search(
        prefix_pattern + r"\s*(?:元素)?\s*([A-Z][a-z]?(?:\s*[,，、和及/-]\s*[A-Z][a-z]?)*)",
        text,
    )
    if not match:
        return []
    return list(dict.fromkeys(re.findall(r"[A-Z][a-z]?", match.group(1))))


class RuleBasedInterpreter:
    """Transparent baseline used before MiniMind structured-output training."""

    def interpret(self, text: str) -> MaterialRequest:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            raise ValueError("user text must not be empty")

        properties: list[PropertyConstraint] = []
        occupied_names: set[str] = set()
        lowered = cleaned.lower()

        for spec in PROPERTY_SPECS:
            best_match: re.Match[str] | None = None
            best_alias: str | None = None
            for alias in sorted(spec.aliases, key=len, reverse=True):
                match = re.search(re.escape(alias.lower()), lowered)
                if match and (best_match is None or match.start() < best_match.start()):
                    best_match = match
                    best_alias = cleaned[match.start() : match.end()]
            if best_match is None or spec.canonical_name in occupied_names:
                continue

            before = cleaned[max(0, best_match.start() - 10) : best_match.start()]
            after = cleaned[best_match.end() : best_match.end() + 40]
            range_match = re.search(
                rf"(?:在|介于)?\s*({NUMBER})\s*(?:到|至|~|～)\s*({NUMBER})\s*({UNIT})?",
                after,
                re.IGNORECASE,
            )
            if range_match:
                prop = PropertyConstraint(
                    name=spec.canonical_name,
                    raw_name=best_alias,
                    operator=Comparator.RANGE,
                    lower_bound=float(range_match.group(1)),
                    upper_bound=float(range_match.group(2)),
                    unit=normalize_unit(range_match.group(3)),
                )
            else:
                value_match = re.search(rf"({NUMBER})\s*({UNIT})?", after, re.IGNORECASE)
                value = float(value_match.group(1)) if value_match else None
                unit = normalize_unit(value_match.group(2)) if value_match else None
                context_end = value_match.start() if value_match else min(len(after), 12)
                context = before + after[:context_end]
                operator = detect_comparator(context, value is not None)
                prop = PropertyConstraint(
                    name=spec.canonical_name,
                    raw_name=best_alias,
                    operator=operator,
                    value=value,
                    unit=unit,
                )
            properties.append(prop)
            occupied_names.add(spec.canonical_name)

        exclude_rare_earth = bool(re.search(r"(?:不含|排除|避免使用|无)\s*稀土", cleaned))
        excluded_elements = parse_elements(cleaned, r"(?:不含|排除|避免使用)")
        required_elements = parse_elements(cleaned, r"(?:必须含|包含|含有)")

        family_match = re.search(r"\b[A-Z][a-z]?(?:[-–][A-Z][a-z]?){1,4}\b", cleaned)
        material_family = family_match.group(0).replace("–", "-") if family_match else None

        free_constraints = [
            phrase
            for phrase in ("室温稳定", "低成本", "无毒", "可合成")
            if phrase in cleaned
        ]
        warnings: list[str] = []
        if not properties:
            warnings.append("rule_baseline_found_no_supported_property")

        return MaterialRequest(
            task=detect_task(cleaned),
            material_family=material_family,
            target_properties=properties,
            composition=CompositionConstraint(
                required_elements=required_elements,
                excluded_elements=excluded_elements,
                exclude_rare_earth=exclude_rare_earth,
            ),
            free_text_constraints=free_constraints,
            original_text=cleaned,
            parser_warnings=warnings,
        )
