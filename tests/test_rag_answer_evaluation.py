from __future__ import annotations

import json

import pytest

from magnova_agent.rag_answer_evaluation import (
    RAGAnswerEvaluationCase,
    RAGAnswerEvaluator,
    extract_cited_sources,
    infer_response_mode,
    load_rag_answer_evaluation_cases,
)


class StubRetriever:
    def __init__(self, evidence):
        self.evidence = evidence

    def retrieve(self, query, k=4):
        del query
        return self.evidence[:k]


class StubResponder:
    def __init__(self, answer):
        self.result = answer

    def answer(self, question, *, messages, summary, evidence):
        del question, messages, summary, evidence
        return self.result


class GuardedStubResponder(StubResponder):
    def __init__(self, answer, raw_answer, quality_reasons=()):
        super().__init__(answer)
        self.raw_answer = raw_answer
        self.quality_reasons = list(quality_reasons)
        self.last_raw_answer = None
        self.last_quality_reasons = []

    def answer(self, question, *, messages, summary, evidence):
        result = super().answer(
            question,
            messages=messages,
            summary=summary,
            evidence=evidence,
        )
        self.last_raw_answer = self.raw_answer
        self.last_quality_reasons = self.quality_reasons
        return result


def test_grounded_answer_metrics() -> None:
    case = RAGAnswerEvaluationCase(
        id="grounded",
        question="MatterGen 生成什么？",
        relevant_sources=("mattergen.md",),
        should_answer=True,
        expected_facts=(("原子类型",), ("坐标",), ("晶格",)),
    )
    evaluator = RAGAnswerEvaluator(
        StubRetriever([{"source": "mattergen.md", "content": "evidence"}]),
        StubResponder("联合生成原子类型、坐标和晶格。[mattergen.md]"),
    )
    report = evaluator.evaluate([case])
    metrics = report["metrics"]
    assert metrics["macro_citation_precision"] == 1
    assert metrics["macro_citation_recall"] == 1
    assert metrics["macro_fact_recall"] == 1
    assert metrics["macro_evidence_fact_recall"] == 0
    assert metrics["grounded_proxy_pass_rate"] == 1
    assert metrics["model_direct_answer_rate"] == 1
    assert metrics["raw_macro_fact_recall"] == 1
    assert metrics["raw_grounded_proxy_pass_rate"] == 1


def test_raw_metrics_are_not_inflated_by_safe_fallback() -> None:
    case = RAGAnswerEvaluationCase(
        id="guarded",
        question="MatterSim 的误差是多少？",
        relevant_sources=("paper.pdf",),
        should_answer=True,
        expected_facts=(("15 meV/atom",),),
    )
    responder = GuardedStubResponder(
        "MiniMind 本次生成未通过证据引用或重复性质量检查。"
        "证据给出 15 meV/atom。[paper.pdf p.50]",
        "该模型效果很好。[paper.pdf p.50]",
        ("low_lexical_grounding",),
    )
    report = RAGAnswerEvaluator(
        StubRetriever([{"source": "paper.pdf", "content": "15 meV/atom"}]),
        responder,
    ).evaluate([case])
    metrics = report["metrics"]
    assert metrics["macro_fact_recall"] == 1
    assert metrics["raw_macro_fact_recall"] == 0
    assert metrics["grounded_proxy_pass_rate"] == 1
    assert metrics["raw_grounded_proxy_pass_rate"] == 0


def test_wrong_citation_and_forbidden_claim_are_exposed() -> None:
    case = RAGAnswerEvaluationCase(
        id="hallucination",
        question="question",
        relevant_sources=("mattergen.md",),
        should_answer=True,
        expected_facts=(("坐标",),),
        forbidden_claims=("保证实验成功",),
    )
    evaluator = RAGAnswerEvaluator(
        StubRetriever([{"source": "mattergen.md", "content": "evidence"}]),
        StubResponder("坐标，因此保证实验成功。[invented.md]"),
    )
    result = evaluator.evaluate([case])["cases"][0]
    assert result["citation_precision"] == 0
    assert result["citation_recall"] == 0
    assert result["forbidden_hits"] == ["保证实验成功"]
    assert not result["grounded_proxy_passed"]


def test_negated_forbidden_claim_is_not_a_false_positive() -> None:
    case = RAGAnswerEvaluationCase(
        id="negated_claim",
        question="是否要重新训练？",
        relevant_sources=("mattergen.md",),
        should_answer=True,
        forbidden_claims=("每个任务从头训练",),
    )
    evaluator = RAGAnswerEvaluator(
        StubRetriever([{"source": "mattergen.md", "content": "evidence"}]),
        StubResponder("使用 adapter，因此不必为每个任务从头训练。[mattergen.md]"),
    )
    result = evaluator.evaluate([case])["cases"][0]
    assert result["forbidden_hits"] == []


def test_no_answer_refusal_and_safe_fallback_are_measured() -> None:
    refusal = RAGAnswerEvaluationCase(
        id="refusal",
        question="李白是谁？",
        relevant_sources=(),
        should_answer=False,
    )
    report = RAGAnswerEvaluator(
        StubRetriever([]),
        StubResponder("当前知识库没有检索到足够相关的证据，暂不根据记忆猜测。"),
    ).evaluate([refusal])
    assert report["metrics"]["answer_decision_accuracy"] == 1
    assert report["metrics"]["no_answer_refusal_accuracy"] == 1
    assert report["metrics"]["raw_answer_decision_accuracy"] == 1
    assert report["metrics"]["raw_no_answer_refusal_accuracy"] == 1
    assert report["metrics"]["raw_answerable_response_rate"] is None

    answerable = RAGAnswerEvaluationCase(
        id="fallback",
        question="question",
        relevant_sources=("source.md",),
        should_answer=True,
    )
    result = RAGAnswerEvaluator(
        StubRetriever([{"source": "source.md"}]),
        StubResponder(
            "MiniMind 本次生成未通过证据引用或重复性质量检查。[source.md]"
        ),
    ).evaluate([answerable])
    assert result["metrics"]["safe_fallback_rate"] == 1
    assert result["metrics"]["model_direct_answer_rate"] == 0
    assert result["metrics"]["raw_answerable_response_rate"] == 1


def test_training_refusal_wording_is_recognized() -> None:
    refusal = RAGAnswerEvaluationCase(
        id="trained_refusal",
        question="证据没有答案的问题",
        relevant_sources=(),
        should_answer=False,
    )
    report = RAGAnswerEvaluator(
        StubRetriever([{"source": "distractor.pdf", "content": "irrelevant"}]),
        StubResponder("根据给定检索证据，无法确定这个问题的答案。"),
    ).evaluate([refusal])
    metrics = report["metrics"]
    assert metrics["no_answer_refusal_accuracy"] == 1
    assert metrics["raw_no_answer_refusal_accuracy"] == 1



def test_three_way_response_mode_distinguishes_grounded_negative() -> None:
    case = RAGAnswerEvaluationCase(
        id="grounded-negative",
        question="MagNova 是否保证候选一定可以合成？",
        relevant_sources=("magnova-plan.md",),
        should_answer=False,
        response_mode="grounded_negative",
        acceptable_negative_fragments=("不保证", "不能保证"),
    )
    report = RAGAnswerEvaluator(
        StubRetriever([{"source": "magnova-plan.md", "content": "候选仍需实验验证"}]),
        StubResponder("MagNova 不保证候选一定可以实验合成。[magnova-plan.md]"),
    ).evaluate([case])
    result = report["cases"][0]
    assert result["predicted_response_mode"] == "grounded_negative"
    assert result["response_mode_passed"]
    assert result["negative_fragment_hit"]
    assert report["metrics"]["grounded_negative_accuracy"] == 1
    assert report["metrics"]["response_mode_accuracy"] == 1


def test_three_way_response_mode_keeps_refusal_separate() -> None:
    answer = "根据给定检索证据，无法确定这个问题的答案。"
    assert infer_response_mode(answer) == "insufficient_evidence"
    assert infer_response_mode(
        "系统不能保证该候选可合成。[plan.md]", cited_sources=("plan.md",)
    ) == "grounded_negative"


def test_citation_extraction_deduplicates_sources() -> None:
    assert extract_cited_sources(
        "[paper.md p.2] text [paper.md] [notes.txt]"
    ) == ["paper.md", "notes.txt"]


def test_jsonl_loader_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "answer-cases.jsonl"
    payload = {
        "id": "duplicate",
        "question": "q",
        "relevant_sources": [],
        "should_answer": False,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False)
        + "\n"
        + json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_rag_answer_evaluation_cases(path)
