from __future__ import annotations

import json

import pytest

from magnova_agent.rag_evaluation import (
    RAGEvaluationCase,
    RAGEvaluator,
    load_rag_evaluation_cases,
)


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def retrieve(self, query, k=4):
        self.queries.append((query, k))
        return self.results.get(query, [])[:k]


def test_metrics_use_unique_source_ranking() -> None:
    retriever = StubRetriever(
        {
            "question": [
                {"source": "noise.md", "score": 0.9},
                {"source": "noise.md", "score": 0.8},
                {"source": "answer.md", "score": 0.7},
            ]
        }
    )
    case = RAGEvaluationCase(
        id="rank",
        question="question",
        relevant_sources=("answer.md",),
        should_answer=True,
    )
    report = RAGEvaluator(retriever).evaluate([case])
    metrics = report["metrics"]
    assert metrics["hit_rate_at_1"] == 0
    assert metrics["recall_at_3"] == 1
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg_at_3"] == pytest.approx(1 / 1.5849625)


def test_no_answer_and_multiturn_decisions_are_measured() -> None:
    retriever = StubRetriever(
        {
            "MatterGen：那它如何处理周期边界？": [
                {"source": "mattergen.md", "score": 0.8}
            ],
            "李白是谁？": [{"source": "noise.md", "score": 0.30}],
        }
    )
    cases = [
        RAGEvaluationCase(
            id="follow-up",
            question="那它如何处理周期边界？",
            history=({"role": "user", "content": "MatterGen 如何生成材料？"},),
            expected_query_contains=("MatterGen",),
            relevant_sources=("mattergen.md",),
            should_answer=True,
            tags=("multiturn",),
        ),
        RAGEvaluationCase(
            id="no-answer",
            question="李白是谁？",
            relevant_sources=(),
            should_answer=False,
        ),
    ]
    metrics = RAGEvaluator(retriever).evaluate(cases)["metrics"]
    assert metrics["decision_accuracy"] == 1
    assert metrics["no_answer_accuracy"] == 1
    assert metrics["false_acceptance_rate"] == 0
    assert metrics["query_condensation_accuracy"] == 1


def test_jsonl_loader_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
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
        load_rag_evaluation_cases(path)
