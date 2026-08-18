from __future__ import annotations

import json

from magnova_agent.claim_evidence_evaluation import (
    ClaimEvidenceEvaluator,
    claim_coverage,
    match_claims,
)
from magnova_agent import ClaimEvidenceAnswer


class StubResponder:
    def __init__(self, payload: dict | None) -> None:
        self.payload = payload

    def generate(self, question: str, evidence: list[dict]) -> str:
        del question, evidence
        return (
            json.dumps(self.payload, ensure_ascii=False)
            if self.payload is not None
            else "不是 JSON"
        )


def test_claim_matching_uses_expected_fact_coverage() -> None:
    expected = ClaimEvidenceAnswer.model_validate(
        {
            "answerability": "answer",
            "claims": [{"claim_id": "C1", "text": "坐标扩散使用 wrapped Normal 处理周期边界。", "evidence_ids": ["E1"]}],
        }
    )
    predicted = ClaimEvidenceAnswer.model_validate(
        {
            "answerability": "answer",
            "claims": [{"claim_id": "C9", "text": "MatterGen 使用 wrapped Normal 处理周期坐标边界。", "evidence_ids": ["E1"]}],
        }
    )
    assert claim_coverage(expected.claims[0].text, predicted.claims[0].text) > 0.55
    assert match_claims(expected, predicted) == [(0, 0)]


def test_evaluator_reports_schema_answerability_claim_and_evidence_metrics() -> None:
    case = {
        "id": "case",
        "question": "如何处理周期坐标？",
        "evidence": [{"evidence_id": "E1", "source": "paper.md", "content": "坐标扩散使用 wrapped Normal 处理周期边界。"}],
        "expected": {
            "answerability": "answer",
            "claims": [{"claim_id": "C1", "text": "坐标扩散使用 wrapped Normal 处理周期边界。", "evidence_ids": ["E1"]}],
        },
    }
    report = ClaimEvidenceEvaluator(StubResponder(case["expected"])).evaluate([case])
    assert report["metrics"]["schema_valid_rate"] == 1
    assert report["metrics"]["answerability_accuracy"] == 1
    assert report["metrics"]["claim_recall"] == 1
    assert report["metrics"]["evidence_id_precision"] == 1
    assert report["metrics"]["faithfulness_valid_rate"] == 1


def test_invalid_json_counts_as_schema_and_answerability_failure() -> None:
    case = {
        "id": "invalid",
        "question": "q",
        "evidence": [{"evidence_id": "E1", "content": "足够长的证据文本内容。"}],
        "expected": {
            "answerability": "insufficient_evidence",
            "claims": [],
            "answerability_reason": "证据不足",
        },
    }
    report = ClaimEvidenceEvaluator(StubResponder(None)).evaluate([case])
    assert report["metrics"]["schema_valid_rate"] == 0
    assert report["metrics"]["answerability_accuracy"] == 0
    assert report["metrics"]["error_rate"] == 1
