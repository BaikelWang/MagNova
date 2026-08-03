from __future__ import annotations

import re
from dataclasses import dataclass, field

from .agent import MagNovaAgent
from .interpreter import apply_critical_guardrails
from .models import AgentOutcome, AgentStatus, Comparator, MaterialRequest
from .normalization import explicit_task_hint, normalize_for_llm


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
UNIT_PATTERN = re.compile(r"μB/Å³|eV/atom|eV|K", re.IGNORECASE)
RANGE_PATTERN = re.compile(
    rf"({NUMBER})\s*(?:到|至|和|~|～|-)\s*({NUMBER})"
)
VALUE_PATTERN = re.compile(NUMBER)


@dataclass
class ConversationState:
    pending_request: MaterialRequest | None = None
    active_task_id: str | None = None
    last_completed_task_id: str | None = None
    turn_count: int = 0
    history: list[dict[str, str]] = field(default_factory=list)


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}

    def get(self, session_id: str) -> ConversationState:
        return self._states.setdefault(session_id, ConversationState())

    def reset(self, session_id: str) -> None:
        self._states.pop(session_id, None)


def _select_properties(
    request: MaterialRequest,
    mentioned_names: set[str],
    issue_suffix: str,
) -> list:
    if mentioned_names:
        return [
            prop
            for prop in request.target_properties
            if prop.name in mentioned_names
        ]
    candidates = []
    for index, prop in enumerate(request.target_properties):
        if any(
            issue == f"target_properties[{index}].{issue_suffix}"
            for issue in request.readiness_issues()
        ):
            candidates.append(prop)
    return candidates if len(candidates) == 1 else []


def merge_follow_up(
    pending: MaterialRequest,
    follow_up: str,
) -> MaterialRequest:
    request = pending.model_copy(deep=True)
    cleaned = " ".join(follow_up.strip().split())
    normalization = normalize_for_llm(cleaned)
    mentioned_names = {
        mention.canonical_name for mention in normalization.property_mentions
    }
    model_text = normalization.model_text

    unit_match = UNIT_PATTERN.search(model_text)
    if unit_match:
        unit = unit_match.group(0)
        if unit.casefold() == "k":
            unit = "K"
        targets = _select_properties(request, mentioned_names, "unit")
        if len(targets) == 1:
            targets[0].unit = unit
            request.parser_warnings.append(
                f"multi_turn_merged:unit:{targets[0].name}:{unit}"
            )

    range_match = RANGE_PATTERN.search(model_text)
    range_targets = [
        prop
        for prop in request.target_properties
        if prop.operator == Comparator.RANGE
        and (not mentioned_names or prop.name in mentioned_names)
        and (prop.lower_bound is None or prop.upper_bound is None)
    ]
    if range_match and len(range_targets) == 1:
        lower, upper = float(range_match.group(1)), float(range_match.group(2))
        range_targets[0].lower_bound = lower
        range_targets[0].upper_bound = upper
        request.parser_warnings.append(
            f"multi_turn_merged:range:{range_targets[0].name}:{lower:g}:{upper:g}"
        )

    value_match = VALUE_PATTERN.search(model_text)
    if value_match and not range_match:
        targets = _select_properties(request, mentioned_names, "value")
        if len(targets) == 1:
            value = float(value_match.group(0))
            targets[0].value = value
            request.parser_warnings.append(
                f"multi_turn_merged:value:{targets[0].name}:{value:g}"
            )

    task_hint = explicit_task_hint(cleaned)
    if task_hint is not None and request.task.value != task_hint:
        request.task = type(request.task)(task_hint)
        request.parser_warnings.append(
            f"multi_turn_merged:task:{task_hint}"
        )

    for constraint in ("室温稳定", "低成本", "无毒", "可合成"):
        if (
            constraint in model_text
            and constraint not in request.free_text_constraints
        ):
            request.free_text_constraints.append(constraint)
            request.parser_warnings.append(
                f"multi_turn_merged:constraint:{constraint}"
            )

    request.original_text = f"{pending.original_text}；补充：{cleaned}"
    request = apply_critical_guardrails(request, request.original_text)
    return request


class MultiTurnMagNovaAgent:
    def __init__(
        self,
        agent: MagNovaAgent,
        store: InMemoryConversationStore | None = None,
    ) -> None:
        self.agent = agent
        self.store = store or InMemoryConversationStore()

    def _advance(
        self,
        user_text: str,
        session_id: str,
        *,
        execute: bool,
    ) -> AgentOutcome:
        cleaned = " ".join(user_text.strip().split())
        if not cleaned:
            raise ValueError("user text must not be empty")
        state = self.store.get(session_id)
        state.turn_count += 1
        state.history.append({"role": "user", "content": cleaned})

        if state.pending_request is None:
            outcome = (
                self.agent.run(cleaned)
                if execute
                else self.agent.prepare(cleaned)
            )
        else:
            request = merge_follow_up(state.pending_request, cleaned)
            outcome = (
                self.agent.execute(request)
                if execute
                else self.agent.prepare_request(request)
            )

        state.history.append(
            {"role": "assistant", "content": outcome.status.value}
        )
        if outcome.status == AgentStatus.NEEDS_CLARIFICATION:
            state.pending_request = outcome.request.model_copy(deep=True)
        else:
            state.pending_request = None
        return outcome

    def run(
        self,
        user_text: str,
        session_id: str = "default",
    ) -> AgentOutcome:
        return self._advance(user_text, session_id, execute=True)

    def prepare(
        self,
        user_text: str,
        session_id: str = "default",
    ) -> AgentOutcome:
        return self._advance(user_text, session_id, execute=False)

    def track_task(self, task_id: str, session_id: str = "default") -> None:
        self.store.get(session_id).active_task_id = task_id

    def complete_task(
        self,
        task_id: str,
        session_id: str = "default",
    ) -> None:
        state = self.store.get(session_id)
        if state.active_task_id == task_id:
            state.active_task_id = None
        state.last_completed_task_id = task_id

    def reset(self, session_id: str = "default") -> None:
        self.store.reset(session_id)

    def state(self, session_id: str = "default") -> ConversationState:
        return self.store.get(session_id)
