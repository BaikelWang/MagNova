from __future__ import annotations

import re
import time
from dataclasses import dataclass
from statistics import mean
from typing import Protocol

import requests

from .claim_evidence import (
    CLAIM_EVIDENCE_SYSTEM_PROMPT,
    Answerability,
    ClaimEvidenceAnswer,
    ClaimEvidenceValidator,
    format_claim_evidence_user_prompt,
    parse_claim_evidence_response,
)


class ClaimResponder(Protocol):
    def generate(self, question: str, evidence: list[dict]) -> str: ...


class HTTPClaimResponder:
    def __init__(self, endpoint: str, *, timeout_seconds: float = 60.0) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def generate(self, question: str, evidence: list[dict]) -> str:
        response = requests.post(
            self.endpoint,
            json={
                "model": "minimind",
                "messages": [
                    {"role": "system", "content": CLAIM_EVIDENCE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": format_claim_evidence_user_prompt(question, evidence),
                    },
                ],
                "temperature": 0.0,
                "do_sample": False,
                "max_tokens": 384,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "stream": False,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


def _normalise(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())


def _bigrams(text: str) -> set[str]:
    cleaned = _normalise(text)
    return {cleaned[index : index + 2] for index in range(max(0, len(cleaned) - 1))}


def claim_coverage(expected: str, predicted: str) -> float:
    expected_terms = _bigrams(expected)
    return (
        len(expected_terms.intersection(_bigrams(predicted))) / len(expected_terms)
        if expected_terms
        else 0.0
    )


def match_claims(
    expected: ClaimEvidenceAnswer,
    predicted: ClaimEvidenceAnswer,
    *,
    threshold: float = 0.55,
) -> list[tuple[int, int]]:
    candidates = sorted(
        (
            (claim_coverage(expected_claim.text, predicted_claim.text), expected_index, predicted_index)
            for expected_index, expected_claim in enumerate(expected.claims)
            for predicted_index, predicted_claim in enumerate(predicted.claims)
        ),
        reverse=True,
    )
    matches: list[tuple[int, int]] = []
    used_expected: set[int] = set()
    used_predicted: set[int] = set()
    for score, expected_index, predicted_index in candidates:
        if score < threshold:
            break
        if expected_index in used_expected or predicted_index in used_predicted:
            continue
        used_expected.add(expected_index)
        used_predicted.add(predicted_index)
        matches.append((expected_index, predicted_index))
    return matches


@dataclass
class ClaimEvidenceEvaluator:
    responder: ClaimResponder
    match_threshold: float = 0.55

    def evaluate(self, cases: list[dict]) -> dict:
        results: list[dict] = []
        for case in cases:
            started = time.perf_counter()
            error = None
            raw = ""
            predicted = None
            try:
                raw = self.responder.generate(case["question"], case["evidence"])
                predicted = parse_claim_evidence_response(raw)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - started
            expected = ClaimEvidenceAnswer.model_validate(case["expected"])
            schema_valid = predicted is not None
            answerability_correct = bool(
                predicted and predicted.answerability == expected.answerability
            )
            matches = (
                match_claims(expected, predicted, threshold=self.match_threshold)
                if predicted and expected.answerability == Answerability.ANSWER
                else []
            )
            expected_claims = len(expected.claims)
            predicted_claims = len(predicted.claims) if predicted else 0
            evidence_true_positive = 0
            expected_evidence_total = 0
            predicted_evidence_total = 0
            if predicted:
                for expected_index, predicted_index in matches:
                    expected_ids = set(expected.claims[expected_index].evidence_ids)
                    predicted_ids = set(predicted.claims[predicted_index].evidence_ids)
                    evidence_true_positive += len(expected_ids.intersection(predicted_ids))
                    expected_evidence_total += len(expected_ids)
                    predicted_evidence_total += len(predicted_ids)
            validation = (
                ClaimEvidenceValidator().validate(predicted, case["evidence"])
                if predicted
                else None
            )
            results.append(
                {
                    "id": case["id"],
                    "expected_answerability": expected.answerability.value,
                    "predicted_answerability": predicted.answerability.value if predicted else None,
                    "schema_valid": schema_valid,
                    "answerability_correct": answerability_correct,
                    "expected_claims": expected_claims,
                    "predicted_claims": predicted_claims,
                    "matched_claims": len(matches),
                    "evidence_true_positive": evidence_true_positive,
                    "expected_evidence_total": expected_evidence_total,
                    "predicted_evidence_total": predicted_evidence_total,
                    "faithfulness_valid": bool(validation and validation.valid),
                    "faithfulness_issues": (
                        [issue.model_dump(mode="json") for issue in validation.issues]
                        if validation
                        else []
                    ),
                    "raw": raw,
                    "latency_seconds": latency,
                    "error": error,
                }
            )

        num_cases = len(results)
        answer_cases = [
            item for item in results if item["expected_answerability"] == "answer"
        ]
        refusal_cases = [
            item
            for item in results
            if item["expected_answerability"] == "insufficient_evidence"
        ]
        total_expected_claims = sum(item["expected_claims"] for item in answer_cases)
        total_matched_claims = sum(item["matched_claims"] for item in answer_cases)
        evidence_tp = sum(item["evidence_true_positive"] for item in results)
        expected_evidence = sum(item["expected_evidence_total"] for item in results)
        predicted_evidence = sum(item["predicted_evidence_total"] for item in results)
        latencies = [item["latency_seconds"] for item in results]
        ordered_latency = sorted(latencies)
        p95_index = max(0, min(len(ordered_latency) - 1, int(0.95 * len(ordered_latency) + 0.9999) - 1))
        metrics = {
            "num_cases": num_cases,
            "schema_valid_rate": mean(item["schema_valid"] for item in results),
            "answerability_accuracy": mean(item["answerability_correct"] for item in results),
            "no_answer_accuracy": (
                mean(item["answerability_correct"] for item in refusal_cases)
                if refusal_cases
                else None
            ),
            "claim_recall": (
                total_matched_claims / total_expected_claims if total_expected_claims else None
            ),
            "evidence_id_precision": (
                evidence_tp / predicted_evidence if predicted_evidence else 0.0
            ),
            "evidence_id_recall": (
                evidence_tp / expected_evidence if expected_evidence else 0.0
            ),
            "faithfulness_valid_rate": mean(
                item["faithfulness_valid"] for item in results
            ),
            "error_rate": mean(bool(item["error"]) for item in results),
            "mean_latency_seconds": mean(latencies),
            "p95_latency_seconds": ordered_latency[p95_index],
        }
        return {"metrics": metrics, "cases": results}
