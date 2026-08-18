from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable, Protocol

from .rag import EvidenceAnswerabilityGate, MaterialScienceQueryGuard
from .research_graph import RuleBasedQueryCondenser


class EvaluationRetriever(Protocol):
    def retrieve(self, query: str, k: int = 4) -> list[dict]: ...


@dataclass(frozen=True)
class RAGEvaluationCase:
    id: str
    question: str
    relevant_sources: tuple[str, ...]
    should_answer: bool
    history: tuple[dict[str, str], ...] = ()
    expected_query_contains: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict) -> "RAGEvaluationCase":
        return cls(
            id=payload["id"],
            question=payload["question"],
            relevant_sources=tuple(payload.get("relevant_sources", [])),
            should_answer=bool(payload["should_answer"]),
            history=tuple(payload.get("history", [])),
            expected_query_contains=tuple(payload.get("expected_query_contains", [])),
            tags=tuple(payload.get("tags", [])),
        )


def load_rag_evaluation_cases(path: str | Path) -> list[RAGEvaluationCase]:
    cases = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            cases.append(RAGEvaluationCase.from_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid evaluation case at line {line_number}") from exc
    if not cases:
        raise ValueError("evaluation dataset must not be empty")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("evaluation case ids must be unique")
    return cases


def _unique_ranked_sources(evidence: Iterable[dict]) -> list[str]:
    ranked = []
    seen = set()
    for item in evidence:
        source = item.get("source")
        if source and source not in seen:
            seen.add(source)
            ranked.append(source)
    return ranked


def _dcg(relevances: list[int]) -> float:
    return sum(
        relevance / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances)
    )


@dataclass
class RAGEvaluator:
    retriever: EvaluationRetriever
    condenser: RuleBasedQueryCondenser = field(
        default_factory=RuleBasedQueryCondenser
    )
    k_values: tuple[int, ...] = (1, 3, 4)
    decision_threshold: float = 0.50
    scope_guard: MaterialScienceQueryGuard = field(
        default_factory=MaterialScienceQueryGuard
    )
    answerability_gate: EvidenceAnswerabilityGate | None = None

    def evaluate_case(self, case: RAGEvaluationCase) -> dict:
        messages = list(case.history) + [
            {"role": "user", "content": case.question}
        ]
        retrieval_query = self.condenser.condense(
            case.question,
            messages=messages,
            summary="",
        )
        evidence = self.retriever.retrieve(
            retrieval_query,
            k=max(self.k_values),
        )
        ranked_sources = _unique_ranked_sources(evidence)
        relevant = set(case.relevant_sources)
        first_relevant_rank = next(
            (
                rank
                for rank, source in enumerate(ranked_sources, start=1)
                if source in relevant
            ),
            None,
        )
        top_score = float(evidence[0].get("score", 0.0)) if evidence else None
        in_scope = self.scope_guard.is_in_scope(retrieval_query)
        gate_result = (
            self.answerability_gate.assess(retrieval_query, evidence)
            if self.answerability_gate
            else None
        )
        result = {
            "id": case.id,
            "question": case.question,
            "retrieval_query": retrieval_query,
            "should_answer": case.should_answer,
            "in_scope": in_scope,
            "predicted_should_answer": (
                in_scope
                and (
                    gate_result["can_answer"]
                    if gate_result is not None
                    else top_score is not None and top_score >= self.decision_threshold
                )
            ),
            "answerability_reasons": gate_result["reasons"] if gate_result else [],
            "relevant_sources": list(case.relevant_sources),
            "ranked_sources": ranked_sources,
            "top_score": top_score,
            "reciprocal_rank": (
                1.0 / first_relevant_rank if first_relevant_rank else 0.0
            ),
            "query_condensation_passed": all(
                expected.lower() in retrieval_query.lower()
                for expected in case.expected_query_contains
            ),
            "tags": list(case.tags),
        }
        for k in self.k_values:
            top_k = ranked_sources[:k]
            retrieved_relevant = relevant.intersection(top_k)
            result[f"hit_at_{k}"] = bool(retrieved_relevant)
            result[f"recall_at_{k}"] = (
                len(retrieved_relevant) / len(relevant) if relevant else None
            )
            relevances = [1 if source in relevant else 0 for source in top_k]
            ideal = [1] * min(len(relevant), k)
            ideal_dcg = _dcg(ideal)
            result[f"ndcg_at_{k}"] = (
                _dcg(relevances) / ideal_dcg if ideal_dcg else None
            )
        return result

    def evaluate(self, cases: Iterable[RAGEvaluationCase]) -> dict:
        case_results = [self.evaluate_case(case) for case in cases]
        if not case_results:
            raise ValueError("at least one evaluation case is required")
        answerable = [item for item in case_results if item["should_answer"]]
        no_answer = [item for item in case_results if not item["should_answer"]]
        contextual = [
            item
            for item in case_results
            if any(tag == "multiturn" for tag in item["tags"])
        ]

        metrics = {
            "num_cases": len(case_results),
            "num_answerable": len(answerable),
            "num_no_answer": len(no_answer),
            "decision_threshold": self.decision_threshold,
            "decision_accuracy": mean(
                item["predicted_should_answer"] == item["should_answer"]
                for item in case_results
            ),
            "scope_accuracy": mean(
                item["in_scope"] == item["should_answer"]
                for item in case_results
            ),
            "no_answer_accuracy": (
                mean(not item["predicted_should_answer"] for item in no_answer)
                if no_answer
                else None
            ),
            "false_acceptance_rate": (
                mean(item["predicted_should_answer"] for item in no_answer)
                if no_answer
                else None
            ),
            "mrr": mean(item["reciprocal_rank"] for item in answerable),
            "query_condensation_accuracy": (
                mean(item["query_condensation_passed"] for item in contextual)
                if contextual
                else None
            ),
            "mean_top_score_answerable": (
                mean(
                    item["top_score"]
                    for item in answerable
                    if item["top_score"] is not None
                )
                if any(item["top_score"] is not None for item in answerable)
                else None
            ),
            "mean_top_score_no_answer": (
                mean(
                    item["top_score"]
                    for item in no_answer
                    if item["top_score"] is not None
                )
                if any(item["top_score"] is not None for item in no_answer)
                else None
            ),
        }
        for k in self.k_values:
            metrics[f"hit_rate_at_{k}"] = mean(
                item[f"hit_at_{k}"] for item in answerable
            )
            metrics[f"recall_at_{k}"] = mean(
                item[f"recall_at_{k}"] for item in answerable
            )
            metrics[f"ndcg_at_{k}"] = mean(
                item[f"ndcg_at_{k}"] for item in answerable
            )

        failures = [
            item
            for item in case_results
            if (
                item["predicted_should_answer"] != item["should_answer"]
                or (
                    item["should_answer"]
                    and not item[f"hit_at_{max(self.k_values)}"]
                )
                or not item["query_condensation_passed"]
            )
        ]
        return {"metrics": metrics, "failures": failures, "cases": case_results}
