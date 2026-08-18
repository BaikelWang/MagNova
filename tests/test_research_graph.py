from __future__ import annotations

from langgraph.types import Command

from magnova_agent import MagNovaAgent, RuleBasedInterpreter
from magnova_agent.models import AgentStatus
from magnova_agent.research_graph import MagNovaResearchGraph, RuleBasedQueryCondenser


class StubResearchResponder:
    def answer(self, question, *, messages, summary, evidence):
        sources = ",".join(item.get("source", "") for item in evidence)
        return f"research:{question}|summary:{summary}|sources:{sources}"


class StubEvidenceRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, k=4):
        self.queries.append((query, k))
        return [{"source": "retrieved-paper.md", "content": "retrieved evidence"}]


def build_graph() -> MagNovaResearchGraph:
    return MagNovaResearchGraph(
        MagNovaAgent(interpreter=RuleBasedInterpreter()),
        research_responder=StubResearchResponder(),
    )


def test_research_question_does_not_call_tool() -> None:
    workflow = build_graph()
    result = workflow.graph.invoke(
        {
            "session_id": "research",
            "user_text": "无稀土磁性材料有哪些研究方向？",
            "research_summary": "关注低成本永磁体",
            "retrieved_evidence": [{"source": "paper-a", "content": "evidence"}],
            "messages": [],
        },
        config=workflow.config("research"),
    )
    assert result["phase"] == "research_chat"
    assert result["last_node"] == "research_chat"
    assert "paper-a" in result["response"]
    assert result.get("agent_outcome") is None


def test_research_route_retrieves_evidence_before_answering() -> None:
    retriever = StubEvidenceRetriever()
    workflow = MagNovaResearchGraph(
        MagNovaAgent(interpreter=RuleBasedInterpreter()),
        research_responder=StubResearchResponder(),
        evidence_retriever=retriever,
    )
    result = workflow.graph.invoke(
        {"user_text": "How does crystal graph convolution work?", "messages": []},
        config=workflow.config("retrieval"),
    )
    assert retriever.queries == [("How does crystal graph convolution work?", 4)]
    assert result["retrieved_evidence"][0]["source"] == "retrieved-paper.md"
    assert "retrieved-paper.md" in result["response"]


def test_query_condenser_inherits_topic_for_pronoun_follow_up() -> None:
    condenser = RuleBasedQueryCondenser()
    query = condenser.condense(
        "那它如何处理周期边界？",
        messages=[
            {"role": "user", "content": "MatterGen 如何生成材料？"},
            {"role": "assistant", "content": "它使用联合扩散。"},
            {"role": "user", "content": "那它如何处理周期边界？"},
        ],
        summary="",
    )
    assert query == "MatterGen：那它如何处理周期边界？"


def test_explicit_new_topic_is_not_overwritten_by_history() -> None:
    condenser = RuleBasedQueryCondenser()
    query = condenser.condense(
        "CGCNN 呢？",
        messages=[{"role": "user", "content": "MatterGen 如何生成材料？"}],
        summary="",
    )
    assert query == "CGCNN 呢？"


def test_graph_condenses_second_research_turn_before_retrieval() -> None:
    retriever = StubEvidenceRetriever()
    workflow = MagNovaResearchGraph(
        MagNovaAgent(interpreter=RuleBasedInterpreter()),
        research_responder=StubResearchResponder(),
        evidence_retriever=retriever,
    )
    config = workflow.config("contextual-retrieval")
    workflow.graph.invoke(
        {"user_text": "MatterGen 如何生成材料？", "messages": []},
        config=config,
    )
    second = workflow.graph.invoke(
        {"user_text": "那它如何处理周期边界？"},
        config=config,
    )
    assert retriever.queries[0] == ("MatterGen 如何生成材料？", 4)
    assert retriever.queries[1] == ("MatterGen：那它如何处理周期边界？", 4)
    assert second["retrieval_query"] == "MatterGen：那它如何处理周期边界？"


def test_follow_up_without_resolvable_topic_stays_unchanged() -> None:
    condenser = RuleBasedQueryCondenser()
    query = condenser.condense(
        "那它有什么局限？",
        messages=[],
        summary="",
    )
    assert query == "那它有什么局限？"


def test_explicit_design_pauses_for_confirmation_then_executes() -> None:
    workflow = build_graph()
    config = workflow.config("approval")
    paused = workflow.graph.invoke(
        {
            "session_id": "approval",
            "user_text": "请帮我设计带隙大于 1.5 eV 的材料",
            "messages": [],
        },
        config=config,
    )
    assert paused["agent_status"] == AgentStatus.READY.value
    assert paused["__interrupt__"][0].value["type"] == "magnova_task_confirmation"

    completed = workflow.graph.invoke(Command(resume=True), config=config)
    assert completed["agent_status"] == AgentStatus.SUBMITTED.value
    assert completed["phase"] == "result_explanation"
    assert completed["confirmed"] is True
    assert completed["agent_outcome"]["task_id"]


def test_rejection_never_executes_tool() -> None:
    workflow = build_graph()
    config = workflow.config("reject")
    workflow.graph.invoke(
        {
            "session_id": "reject",
            "user_text": "现在请生成居里温度高于 500 K 的材料",
            "messages": [],
        },
        config=config,
    )
    rejected = workflow.graph.invoke(Command(resume=False), config=config)
    assert rejected["phase"] == "rejected"
    assert rejected["confirmed"] is False
    assert rejected["agent_status"] == AgentStatus.READY.value
    assert "未调用" in rejected["response"]


def test_clarification_survives_across_graph_turns() -> None:
    workflow = build_graph()
    config = workflow.config("clarify")
    first = workflow.graph.invoke(
        {
            "session_id": "clarify",
            "user_text": "请帮我设计磁矩密度大于 0.2 的材料",
            "messages": [],
        },
        config=config,
    )
    assert first["phase"] == "clarification"
    assert first["pending_request"] is not None
    assert "单位" in first["response"]

    second = workflow.graph.invoke(
        {"user_text": "单位是 μB/Å³"},
        config=config,
    )
    assert second["agent_status"] == AgentStatus.READY.value
    assert second["__interrupt__"]
    final = workflow.graph.invoke(Command(resume=True), config=config)
    assert final["agent_status"] == AgentStatus.SUBMITTED.value
    assert final["pending_request"] is None


def test_threads_keep_independent_state() -> None:
    workflow = build_graph()
    workflow.graph.invoke(
        {
            "session_id": "one",
            "user_text": "请设计带隙大于 1.5 的材料",
            "messages": [],
        },
        config=workflow.config("one"),
    )
    other = workflow.graph.invoke(
        {
            "session_id": "two",
            "user_text": "形成能为什么通常要求较低？",
            "messages": [],
        },
        config=workflow.config("two"),
    )
    assert workflow.graph.get_state(workflow.config("one")).values["pending_request"]
    assert other.get("pending_request") is None
    assert other["phase"] == "research_chat"
