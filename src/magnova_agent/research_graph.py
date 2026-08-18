from __future__ import annotations

import operator
import re
from typing import Annotated, Any, Literal, Protocol

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from .agent import MagNovaAgent
from .conversation import merge_follow_up
from .models import AgentOutcome, AgentStatus, MaterialRequest


class EvidenceItem(TypedDict, total=False):
    content: str
    source: str
    page: int
    chunk_id: str
    score: float


class ResearchGraphState(TypedDict, total=False):
    session_id: str
    user_text: str
    messages: Annotated[list[dict[str, str]], operator.add]
    research_summary: str
    retrieved_evidence: list[EvidenceItem]
    retrieval_query: str
    phase: str
    response: str
    pending_request: dict[str, Any] | None
    material_request: dict[str, Any] | None
    agent_outcome: dict[str, Any] | None
    agent_status: str | None
    confirmation_required: bool
    confirmed: bool
    last_node: str


class ResearchResponder(Protocol):
    def answer(
        self,
        question: str,
        *,
        messages: list[dict[str, str]],
        summary: str,
        evidence: list[EvidenceItem],
    ) -> str:
        """Answer a research question using conversation and retrieved evidence."""


class EvidenceRetriever(Protocol):
    def retrieve(self, query: str, k: int = 4) -> list[EvidenceItem]:
        """Return source-aware evidence ordered by relevance."""


class QueryCondenser(Protocol):
    def condense(
        self,
        question: str,
        *,
        messages: list[dict[str, str]],
        summary: str,
    ) -> str:
        """Rewrite a contextual follow-up into a standalone retrieval query."""


class UnconfiguredResearchResponder:
    """Safe placeholder until the MiniMind + LangChain RAG node is connected."""

    def answer(
        self,
        question: str,
        *,
        messages: list[dict[str, str]],
        summary: str,
        evidence: list[EvidenceItem],
    ) -> str:
        del question, messages, summary, evidence
        return (
            "当前科研问答模型与论文检索器尚未配置。你仍可明确提出材料"
            "生成、设计或筛选需求；系统会进入结构化任务流程。"
        )


class ExplicitIntentRouter:
    """Conservative router: discussion does not become a task without an action."""

    _question_markers = re.compile(r"(?:为什么|是什么|有哪些|如何|怎么|讨论|解释|原理|理论|\?|？)")
    _explicit_action = re.compile(
        r"(?:请|帮我|现在|开始|根据(?:刚才|上述|以上)[^，。；]{0,20})"
        r"[^，。；]{0,12}(?:生成|设计|筛选|预测)"
    )

    def route(self, text: str, *, pending_request: bool) -> Literal["research", "task"]:
        if pending_request:
            return "task"
        cleaned = " ".join(text.strip().split())
        if self._explicit_action.search(cleaned):
            return "task"
        if self._question_markers.search(cleaned):
            return "research"
        return "research"


class RuleBasedQueryCondenser:
    """Conservative, auditable baseline for contextual research follow-ups."""

    _topic_patterns = (
        re.compile(r"\bMatterGen\b", re.IGNORECASE),
        re.compile(r"\bCGCNN\b", re.IGNORECASE),
        re.compile(r"\bMPNN\b", re.IGNORECASE),
        re.compile(r"\bSACGT\b", re.IGNORECASE),
        re.compile(r"晶体图卷积神经网络"),
        re.compile(r"消息传递神经网络"),
        re.compile(r"扩散模型"),
        re.compile(r"形成能"),
        re.compile(r"带隙"),
        re.compile(r"磁矩密度"),
        re.compile(r"居里温度"),
    )
    _context_reference = re.compile(
        r"(?:它|该模型|这个模型|这种模型|该方法|这个方法|这种方法|"
        r"上述模型|上述方法|前者|后者|那它|那么它|其中的)"
    )
    _elliptic_follow_up = re.compile(r"(?:呢|又如何|又是怎样|具体呢)[？?]?$" )

    @classmethod
    def _find_topic(cls, text: str) -> str | None:
        for pattern in cls._topic_patterns:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None

    def condense(
        self,
        question: str,
        *,
        messages: list[dict[str, str]],
        summary: str,
    ) -> str:
        cleaned = " ".join(question.strip().split())
        if not cleaned:
            raise ValueError("question must not be empty")
        if self._find_topic(cleaned):
            return cleaned
        if not (
            self._context_reference.search(cleaned)
            or self._elliptic_follow_up.search(cleaned)
        ):
            return cleaned

        history = list(messages)
        if (
            history
            and history[-1].get("role") == "user"
            and " ".join(history[-1].get("content", "").strip().split()) == cleaned
        ):
            history = history[:-1]

        search_spaces = [
            item.get("content", "")
            for item in reversed(history)
            if item.get("role") == "user"
        ]
        if summary.strip():
            search_spaces.append(summary)
        search_spaces.extend(
            item.get("content", "")
            for item in reversed(history)
            if item.get("role") == "assistant"
        )
        for text in search_spaces:
            topic = self._find_topic(text)
            if topic:
                return f"{topic}：{cleaned}"
        return cleaned


class MagNovaResearchGraph:
    """Minimal stateful LangGraph for research chat and guarded tool execution."""

    def __init__(
        self,
        agent: MagNovaAgent,
        *,
        research_responder: ResearchResponder | None = None,
        evidence_retriever: EvidenceRetriever | None = None,
        query_condenser: QueryCondenser | None = None,
        intent_router: ExplicitIntentRouter | None = None,
        checkpointer=None,
    ) -> None:
        self.agent = agent
        self.research_responder = research_responder or UnconfiguredResearchResponder()
        self.evidence_retriever = evidence_retriever
        self.query_condenser = query_condenser or RuleBasedQueryCondenser()
        self.intent_router = intent_router or ExplicitIntentRouter()
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(ResearchGraphState)
        builder.add_node("classify_intent", self._classify_intent)
        builder.add_node("retrieve_evidence", self._retrieve_evidence)
        builder.add_node("research_chat", self._research_chat)
        builder.add_node("requirement_draft", self._requirement_draft)
        builder.add_node("clarification", self._clarification)
        builder.add_node("confirmation", self._confirmation)
        builder.add_node("rejected", self._rejected)
        builder.add_node("tool_execution", self._tool_execution)
        builder.add_node("result_explanation", self._result_explanation)

        builder.add_edge(START, "classify_intent")
        builder.add_conditional_edges(
            "classify_intent",
            self._route_after_intent,
            {
                "retrieve_evidence": "retrieve_evidence",
                "requirement_draft": "requirement_draft",
            },
        )
        builder.add_edge("retrieve_evidence", "research_chat")
        builder.add_edge("research_chat", END)
        builder.add_conditional_edges(
            "requirement_draft",
            self._route_after_requirement,
            {"clarification": "clarification", "confirmation": "confirmation"},
        )
        builder.add_edge("clarification", END)
        builder.add_conditional_edges(
            "confirmation",
            self._route_after_confirmation,
            {"tool_execution": "tool_execution", "rejected": "rejected"},
        )
        builder.add_edge("rejected", END)
        builder.add_edge("tool_execution", "result_explanation")
        builder.add_edge("result_explanation", END)
        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _clean_text(state: ResearchGraphState) -> str:
        cleaned = " ".join(state.get("user_text", "").strip().split())
        if not cleaned:
            raise ValueError("user_text must not be empty")
        return cleaned

    def _classify_intent(self, state: ResearchGraphState) -> dict[str, Any]:
        text = self._clean_text(state)
        route = self.intent_router.route(
            text,
            pending_request=state.get("pending_request") is not None,
        )
        return {
            "phase": "requirement_draft" if route == "task" else "research_chat",
            "last_node": "classify_intent",
            "messages": [{"role": "user", "content": text}],
        }

    @staticmethod
    def _route_after_intent(
        state: ResearchGraphState,
    ) -> Literal["retrieve_evidence", "requirement_draft"]:
        return (
            "requirement_draft"
            if state.get("phase") == "requirement_draft"
            else "retrieve_evidence"
        )

    def _retrieve_evidence(self, state: ResearchGraphState) -> dict[str, Any]:
        retrieval_query = self.query_condenser.condense(
            self._clean_text(state),
            messages=state.get("messages", []),
            summary=state.get("research_summary", ""),
        )
        if self.evidence_retriever is None:
            evidence = state.get("retrieved_evidence", [])
        else:
            evidence = self.evidence_retriever.retrieve(retrieval_query, k=4)
        return {
            "retrieved_evidence": evidence,
            "retrieval_query": retrieval_query,
            "last_node": "retrieve_evidence",
        }

    def _research_chat(self, state: ResearchGraphState) -> dict[str, Any]:
        answer = self.research_responder.answer(
            self._clean_text(state),
            messages=state.get("messages", []),
            summary=state.get("research_summary", ""),
            evidence=state.get("retrieved_evidence", []),
        )
        return {
            "phase": "research_chat",
            "response": answer,
            "last_node": "research_chat",
            "messages": [{"role": "assistant", "content": answer}],
        }

    def _requirement_draft(self, state: ResearchGraphState) -> dict[str, Any]:
        text = self._clean_text(state)
        pending_payload = state.get("pending_request")
        if pending_payload is None:
            outcome = self.agent.prepare(text)
        else:
            pending = MaterialRequest.model_validate(pending_payload)
            request = merge_follow_up(pending, text)
            outcome = self.agent.prepare_request(request)

        request_payload = outcome.request.model_dump(mode="json")
        pending_request = (
            request_payload
            if outcome.status == AgentStatus.NEEDS_CLARIFICATION
            else None
        )
        return {
            "phase": "requirement_draft",
            "material_request": request_payload,
            "pending_request": pending_request,
            "agent_outcome": outcome.model_dump(mode="json"),
            "agent_status": outcome.status.value,
            "confirmation_required": outcome.status == AgentStatus.READY,
            "confirmed": False,
            "last_node": "requirement_draft",
        }

    @staticmethod
    def _route_after_requirement(
        state: ResearchGraphState,
    ) -> Literal["clarification", "confirmation"]:
        return (
            "confirmation"
            if state.get("agent_status") == AgentStatus.READY.value
            else "clarification"
        )

    @staticmethod
    def _clarification(state: ResearchGraphState) -> dict[str, Any]:
        outcome = AgentOutcome.model_validate(state["agent_outcome"])
        response = "\n".join(outcome.clarification_questions)
        return {
            "phase": "clarification",
            "response": response,
            "last_node": "clarification",
            "messages": [{"role": "assistant", "content": response}],
        }

    @staticmethod
    def _confirmation(state: ResearchGraphState) -> dict[str, Any]:
        decision = interrupt(
            {
                "type": "magnova_task_confirmation",
                "message": "需求已通过校验，是否提交 MagNova？",
                "material_request": state["material_request"],
            }
        )
        approved = decision is True or decision == "approve"
        if isinstance(decision, dict):
            approved = decision.get("type") == "approve"
        return {
            "phase": "confirmation",
            "confirmed": approved,
            "last_node": "confirmation",
        }

    @staticmethod
    def _route_after_confirmation(
        state: ResearchGraphState,
    ) -> Literal["tool_execution", "rejected"]:
        return "tool_execution" if state.get("confirmed") else "rejected"

    @staticmethod
    def _rejected(state: ResearchGraphState) -> dict[str, Any]:
        del state
        response = "已取消本次 MagNova 任务，未调用任何材料计算工具。"
        return {
            "phase": "rejected",
            "response": response,
            "confirmation_required": False,
            "last_node": "rejected",
            "messages": [{"role": "assistant", "content": response}],
        }

    def _tool_execution(self, state: ResearchGraphState) -> dict[str, Any]:
        request = MaterialRequest.model_validate(state["material_request"])
        outcome = self.agent.execute(request)
        return {
            "phase": "tool_execution",
            "agent_outcome": outcome.model_dump(mode="json"),
            "agent_status": outcome.status.value,
            "confirmation_required": False,
            "last_node": "tool_execution",
        }

    @staticmethod
    def _result_explanation(state: ResearchGraphState) -> dict[str, Any]:
        outcome = AgentOutcome.model_validate(state["agent_outcome"])
        response = outcome.message
        return {
            "phase": "result_explanation",
            "response": response,
            "pending_request": None,
            "last_node": "result_explanation",
            "messages": [{"role": "assistant", "content": response}],
        }

    @staticmethod
    def config(session_id: str) -> dict[str, dict[str, str]]:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        return {"configurable": {"thread_id": session_id}}
