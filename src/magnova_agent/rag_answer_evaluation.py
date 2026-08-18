from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable, Protocol

from .research_graph import RuleBasedQueryCondenser


class AnswerEvaluationRetriever(Protocol):
    def retrieve(self, query: str, k: int = 4) -> list[dict]: ...


class AnswerEvaluationResponder(Protocol):
    def answer(
        self,
        question: str,
        *,
        messages: list[dict[str, str]],
        summary: str,
        evidence: list[dict],
    ) -> str: ...


@dataclass(frozen=True)
class RAGAnswerEvaluationCase:
    id: str
    question: str
    relevant_sources: tuple[str, ...]
    should_answer: bool
    expected_facts: tuple[tuple[str, ...], ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    history: tuple[dict[str, str], ...] = ()
    expected_query_contains: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    response_mode: str | None = None
    acceptable_negative_fragments: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict) -> "RAGAnswerEvaluationCase":
        expected_facts = tuple(
            (fact,) if isinstance(fact, str) else tuple(fact)
            for fact in payload.get("expected_facts", [])
        )
        if any(not aliases for aliases in expected_facts):
            raise ValueError("expected fact alias groups must not be empty")
        response_mode = payload.get("response_mode")
        valid_modes = {None, "positive_answer", "grounded_negative", "insufficient_evidence"}
        if response_mode not in valid_modes:
            raise ValueError(f"invalid response mode: {response_mode}")
        return cls(
            id=payload["id"],
            question=payload["question"],
            relevant_sources=tuple(payload.get("relevant_sources", [])),
            should_answer=bool(payload["should_answer"]),
            expected_facts=expected_facts,
            forbidden_claims=tuple(payload.get("forbidden_claims", [])),
            history=tuple(payload.get("history", [])),
            expected_query_contains=tuple(payload.get("expected_query_contains", [])),
            tags=tuple(payload.get("tags", [])),
            response_mode=response_mode,
            acceptable_negative_fragments=tuple(payload.get("acceptable_negative_fragments", [])),
        )

    @property
    def expected_response_mode(self) -> str:
        """Return the three-way label while keeping legacy JSONL compatible."""
        if self.response_mode is not None:
            return self.response_mode
        return "positive_answer" if self.should_answer else "insufficient_evidence"


def load_rag_answer_evaluation_cases(
    path: str | Path,
) -> list[RAGAnswerEvaluationCase]:
    cases: list[RAGAnswerEvaluationCase] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            cases.append(RAGAnswerEvaluationCase.from_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(
                f"invalid answer evaluation case at line {line_number}"
            ) from exc
    if not cases:
        raise ValueError("answer evaluation dataset must not be empty")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("answer evaluation case ids must be unique")
    return cases


_CITATION_PATTERN = re.compile(
    r"\[([^\]\s/\\]+\.(?:md|txt|pdf))(?:\s+p\.\d+)?\]",
    re.IGNORECASE,
)
_REFUSAL_MARKERS = (
    "没有检索到足够相关",
    "没有足够相关",
    "资料不足",
    "证据不足",
    "无法根据现有",
    "根据给定检索证据，无法确定",
    "根据检索证据，无法确定",
    "暂不根据",
    "不能根据现有",
)
_FALLBACK_MARKER = "未通过证据引用或重复性质量检查"
_GROUNDED_NEGATIVE_PATTERNS = (
    "不保证", "不能保证", "无法保证", "未证明", "不能证明", "没有证据表明", "证据不支持",
)


def extract_cited_sources(answer: str) -> list[str]:
    """Extract de-duplicated source filenames from the project's citation format."""

    return list(dict.fromkeys(match.group(1) for match in _CITATION_PATTERN.finditer(answer)))


def infer_response_mode(answer: str, *, cited_sources: Iterable[str] = ()) -> str:
    """Distinguish a cited negative answer from an insufficient-evidence refusal."""
    if any(marker in answer for marker in _REFUSAL_MARKERS):
        return "insufficient_evidence"
    if list(cited_sources) and any(marker in answer for marker in _GROUNDED_NEGATIVE_PATTERNS):
        return "grounded_negative"
    return "positive_answer"


def _contains(text: str, phrase: str) -> bool:
    return phrase.casefold() in text.casefold()


_NEGATION_PREFIXES = (
    "不",
    "不必",
    "不会",
    "不能",
    "不应",
    "不是",
    "并非",
    "并不",
    "无须",
    "无需",
    "未",
    "没有",
)


def _contains_affirmed_claim(text: str, phrase: str) -> bool:
    """Return true only when a forbidden phrase is not locally negated.

    The benchmark stores compact forbidden propositions such as
    ``每个任务从头训练``. A plain substring match would incorrectly flag the
    grounded sentence ``不必为每个任务从头训练``. This remains a lightweight
    lexical metric rather than a general Chinese entailment classifier.
    """

    folded_text = text.casefold()
    folded_phrase = phrase.casefold()
    start = 0
    while True:
        index = folded_text.find(folded_phrase, start)
        if index < 0:
            return False
        prefix = re.sub(r"[\s，。；：、,.!?！？]", "", folded_text[max(0, index - 10) : index])
        if not any(marker in prefix for marker in _NEGATION_PREFIXES):
            return True
        start = index + len(folded_phrase)


def _percentile_nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int((percentile * len(ordered)) + 0.999999999))
    return ordered[min(rank - 1, len(ordered) - 1)]


@dataclass
class RAGAnswerEvaluator:
    retriever: AnswerEvaluationRetriever
    responder: AnswerEvaluationResponder
    condenser: RuleBasedQueryCondenser = field(default_factory=RuleBasedQueryCondenser)
    retrieval_k: int = 4

    def evaluate_case(self, case: RAGAnswerEvaluationCase) -> dict:
        messages = list(case.history) + [
            {"role": "user", "content": case.question}
        ]
        retrieval_query = self.condenser.condense(
            case.question,
            messages=messages,
            summary="",
        )
        started_at = time.perf_counter()
        error = None
        evidence: list[dict] = []
        try:
            evidence = self.retriever.retrieve(retrieval_query, k=self.retrieval_k)
            answer = self.responder.answer(
                case.question,
                messages=messages,
                summary="",
                evidence=evidence,
            )
        except Exception as exc:  # Keep batch evaluation inspectable.
            error = f"{type(exc).__name__}: {exc}"
            answer = ""
        latency_seconds = time.perf_counter() - started_at

        cited_sources = extract_cited_sources(answer)
        relevant = set(case.relevant_sources)
        cited_relevant = relevant.intersection(cited_sources)
        fact_hits = [
            any(_contains(answer, alias) for alias in aliases)
            for aliases in case.expected_facts
        ]
        evidence_text = "\n".join(
            str(item.get("content", "")) for item in evidence
        )
        evidence_fact_hits = [
            any(_contains(evidence_text, alias) for alias in aliases)
            for aliases in case.expected_facts
        ]
        forbidden_hits = [
            claim
            for claim in case.forbidden_claims
            if _contains_affirmed_claim(answer, claim)
        ]
        is_refusal = any(marker in answer for marker in _REFUSAL_MARKERS)
        used_safe_fallback = _FALLBACK_MARKER in answer
        raw_sentinel = object()
        raw_value = getattr(self.responder, "last_raw_answer", raw_sentinel)
        # Lightweight responders used outside the guarded runtime return their
        # model text directly. In that case the final answer is also the raw
        # answer. The guarded responder explicitly exposes None when the
        # pre-generation gate did not call the model.
        raw_answer = answer if raw_value is raw_sentinel else raw_value
        quality_reasons = list(
            getattr(self.responder, "last_quality_reasons", [])
        )

        raw_text = raw_answer or ""
        raw_cited_sources = extract_cited_sources(raw_text)
        raw_cited_relevant = relevant.intersection(raw_cited_sources)
        raw_fact_hits = [
            any(_contains(raw_text, alias) for alias in aliases)
            for aliases in case.expected_facts
        ]
        raw_forbidden_hits = [
            claim
            for claim in case.forbidden_claims
            if _contains_affirmed_claim(raw_text, claim)
        ]
        raw_is_refusal = any(marker in raw_text for marker in _REFUSAL_MARKERS)
        expected_response_mode = case.expected_response_mode
        predicted_response_mode = infer_response_mode(answer, cited_sources=cited_sources)
        raw_predicted_response_mode = infer_response_mode(raw_text, cited_sources=raw_cited_sources)
        negative_fragment_hit = (
            any(_contains(answer, fragment) for fragment in case.acceptable_negative_fragments)
            if case.acceptable_negative_fragments else None
        )
        raw_citation_precision = (
            len(raw_cited_relevant) / len(raw_cited_sources)
            if raw_cited_sources
            else 0.0
        )
        raw_citation_recall = (
            len(raw_cited_relevant) / len(relevant) if relevant else None
        )
        raw_fact_recall = (
            sum(raw_fact_hits) / len(raw_fact_hits) if raw_fact_hits else None
        )
        raw_grounded_proxy_passed = (
            case.should_answer
            and bool(raw_text)
            and not raw_is_refusal
            and raw_citation_precision == 1.0
            and raw_citation_recall == 1.0
            and raw_fact_recall == 1.0
            and not raw_forbidden_hits
        )

        citation_precision = (
            len(cited_relevant) / len(cited_sources) if cited_sources else 0.0
        )
        citation_recall = (
            len(cited_relevant) / len(relevant) if relevant else None
        )
        fact_recall = (
            sum(fact_hits) / len(fact_hits) if fact_hits else None
        )
        evidence_fact_recall = (
            sum(evidence_fact_hits) / len(evidence_fact_hits)
            if evidence_fact_hits
            else None
        )
        grounded_proxy_passed = (
            case.should_answer
            and bool(answer)
            and not is_refusal
            and citation_precision == 1.0
            and citation_recall == 1.0
            and fact_recall == 1.0
            and not forbidden_hits
        )
        return {
            "id": case.id,
            "question": case.question,
            "retrieval_query": retrieval_query,
            "should_answer": case.should_answer,
            "predicted_should_answer": bool(answer) and not is_refusal,
            "expected_response_mode": expected_response_mode,
            "predicted_response_mode": predicted_response_mode,
            "response_mode_passed": predicted_response_mode == expected_response_mode,
            "negative_fragment_hit": negative_fragment_hit,
            "answer": answer,
            "relevant_sources": list(case.relevant_sources),
            "evidence_sources": list(
                dict.fromkeys(
                    item.get("source") for item in evidence if item.get("source")
                )
            ),
            "cited_sources": cited_sources,
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "fact_hits": fact_hits,
            "fact_recall": fact_recall,
            "evidence_fact_hits": evidence_fact_hits,
            "evidence_fact_recall": evidence_fact_recall,
            "forbidden_hits": forbidden_hits,
            "is_refusal": is_refusal,
            "used_safe_fallback": used_safe_fallback,
            "raw_answer": raw_answer,
            "raw_model_attempted": raw_answer is not None,
            "raw_predicted_should_answer": bool(raw_text) and not raw_is_refusal,
            "raw_predicted_response_mode": raw_predicted_response_mode,
            "raw_response_mode_passed": raw_predicted_response_mode == expected_response_mode,
            "raw_cited_sources": raw_cited_sources,
            "raw_citation_precision": raw_citation_precision,
            "raw_citation_recall": raw_citation_recall,
            "raw_fact_hits": raw_fact_hits,
            "raw_fact_recall": raw_fact_recall,
            "raw_forbidden_hits": raw_forbidden_hits,
            "raw_is_refusal": raw_is_refusal,
            "raw_grounded_proxy_passed": raw_grounded_proxy_passed,
            "quality_reasons": quality_reasons,
            "grounded_proxy_passed": grounded_proxy_passed,
            "query_condensation_passed": all(
                expected.casefold() in retrieval_query.casefold()
                for expected in case.expected_query_contains
            ),
            "latency_seconds": latency_seconds,
            "error": error,
            "tags": list(case.tags),
        }

    def evaluate(self, cases: Iterable[RAGAnswerEvaluationCase]) -> dict:
        case_results = [self.evaluate_case(case) for case in cases]
        if not case_results:
            raise ValueError("at least one answer evaluation case is required")
        answerable = [item for item in case_results if item["should_answer"]]
        no_answer = [item for item in case_results if not item["should_answer"]]
        grounded_negative = [item for item in case_results if item["expected_response_mode"] == "grounded_negative"]
        insufficient_evidence = [item for item in case_results if item["expected_response_mode"] == "insufficient_evidence"]
        contextual = [
            item for item in case_results if "multiturn" in item["tags"]
        ]
        latencies = [item["latency_seconds"] for item in case_results]

        metrics = {
            "num_cases": len(case_results),
            "num_answerable": len(answerable),
            "num_no_answer": len(no_answer),
            "num_grounded_negative": len(grounded_negative),
            "num_insufficient_evidence": len(insufficient_evidence),
            "response_mode_accuracy": mean(item["response_mode_passed"] for item in case_results),
            "raw_response_mode_accuracy": mean(item["raw_response_mode_passed"] for item in case_results),
            "grounded_negative_accuracy": (
                mean(item["response_mode_passed"] for item in grounded_negative) if grounded_negative else None
            ),
            "insufficient_evidence_accuracy": (
                mean(item["response_mode_passed"] for item in insufficient_evidence) if insufficient_evidence else None
            ),
            "answer_decision_accuracy": mean(
                item["predicted_should_answer"] == item["should_answer"]
                for item in case_results
            ),
            "no_answer_refusal_accuracy": (
                mean(item["is_refusal"] for item in no_answer)
                if no_answer
                else None
            ),
            "raw_answer_decision_accuracy": mean(
                item["raw_predicted_should_answer"] == item["should_answer"]
                for item in case_results
            ),
            "raw_no_answer_refusal_accuracy": (
                mean(item["raw_is_refusal"] for item in no_answer)
                if no_answer
                else None
            ),
            "raw_answerable_response_rate": (
                mean(item["raw_predicted_should_answer"] for item in answerable)
                if answerable
                else None
            ),
            "macro_citation_precision": (
                mean(item["citation_precision"] for item in answerable)
                if answerable
                else None
            ),
            "macro_citation_recall": (
                mean(item["citation_recall"] for item in answerable)
                if answerable
                else None
            ),
            "macro_fact_recall": (
                mean(
                    item["fact_recall"]
                    for item in answerable
                    if item["fact_recall"] is not None
                )
                if any(item["fact_recall"] is not None for item in answerable)
                else None
            ),
            "macro_evidence_fact_recall": (
                mean(
                    item["evidence_fact_recall"]
                    for item in answerable
                    if item["evidence_fact_recall"] is not None
                )
                if any(
                    item["evidence_fact_recall"] is not None
                    for item in answerable
                )
                else None
            ),
            "forbidden_claim_rate": mean(
                bool(item["forbidden_hits"]) for item in case_results
            ),
            "grounded_proxy_pass_rate": (
                mean(item["grounded_proxy_passed"] for item in answerable)
                if answerable
                else None
            ),
            "model_direct_answer_rate": (
                mean(
                    item["predicted_should_answer"]
                    and not item["used_safe_fallback"]
                    for item in answerable
                )
                if answerable
                else None
            ),
            "safe_fallback_rate": (
                mean(item["used_safe_fallback"] for item in answerable)
                if answerable
                else None
            ),
            "raw_model_attempt_rate": (
                mean(item["raw_model_attempted"] for item in answerable)
                if answerable
                else None
            ),
            "raw_macro_citation_precision": (
                mean(item["raw_citation_precision"] for item in answerable)
                if answerable
                else None
            ),
            "raw_macro_citation_recall": (
                mean(item["raw_citation_recall"] for item in answerable)
                if answerable
                else None
            ),
            "raw_macro_fact_recall": (
                mean(
                    item["raw_fact_recall"]
                    for item in answerable
                    if item["raw_fact_recall"] is not None
                )
                if any(
                    item["raw_fact_recall"] is not None
                    for item in answerable
                )
                else None
            ),
            "raw_forbidden_claim_rate": (
                mean(bool(item["raw_forbidden_hits"]) for item in answerable)
                if answerable
                else None
            ),
            "raw_grounded_proxy_pass_rate": (
                mean(
                    item["raw_grounded_proxy_passed"]
                    for item in answerable
                )
                if answerable
                else None
            ),
            "query_condensation_accuracy": (
                mean(item["query_condensation_passed"] for item in contextual)
                if contextual
                else None
            ),
            "error_rate": mean(bool(item["error"]) for item in case_results),
            "mean_latency_seconds": mean(latencies),
            "p95_latency_seconds": _percentile_nearest_rank(latencies, 0.95),
        }
        failures = [
            item
            for item in case_results
            if item["error"]
            or item["predicted_should_answer"] != item["should_answer"]
            or (item["should_answer"] and not item["grounded_proxy_passed"])
            or item["forbidden_hits"]
            or not item["query_condensation_passed"]
        ]
        response_mode_failures = [item for item in case_results if not item["response_mode_passed"]]
        return {
            "metrics": metrics,
            "failures": failures,
            "response_mode_failures": response_mode_failures,
            "cases": case_results,
        }
