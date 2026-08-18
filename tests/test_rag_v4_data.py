from __future__ import annotations

import pytest
from pydantic import ValidationError

from magnova_agent import Answerability, ClaimEvidenceAnswer, EvidenceClaim
from magnova_agent.rag_v4_data import (
    ClaimEvidenceCandidate,
    EvidenceRecord,
    ReviewStatus,
    audit_candidate_pool,
)


def candidate(**overrides) -> ClaimEvidenceCandidate:
    payload = {
        "sample_id": "sample-1",
        "question": "MatterGen 如何处理周期坐标？",
        "evidence": [
            EvidenceRecord(
                evidence_id="E1",
                source="mattergen.md",
                content="坐标扩散使用 wrapped Normal 处理周期边界。",
                source_sha256="a" * 64,
            )
        ],
        "target": ClaimEvidenceAnswer(
            answerability=Answerability.ANSWER,
            claims=[
                EvidenceClaim(
                    claim_id="C1",
                    text="坐标扩散使用 wrapped Normal。",
                    evidence_ids=["E1"],
                )
            ],
        ),
        "leakage_group": "mattergen_periodic",
        "origin_split": "train",
    }
    payload.update(overrides)
    return ClaimEvidenceCandidate(**payload)


def test_candidate_rejects_unknown_evidence_reference() -> None:
    target = ClaimEvidenceAnswer(
        answerability=Answerability.ANSWER,
        claims=[EvidenceClaim(claim_id="C1", text="结论", evidence_ids=["E2"])],
    )
    with pytest.raises(ValidationError, match="unknown evidence"):
        candidate(target=target)


def test_approval_requires_clean_issues_reviewer_and_hashes() -> None:
    with pytest.raises(ValidationError, match="automatic issues"):
        candidate(
            review_status=ReviewStatus.APPROVED,
            reviewer="human",
            auto_issues=["low_alignment"],
        )
    approved = candidate(review_status=ReviewStatus.APPROVED, reviewer="human")
    assert approved.review_status == ReviewStatus.APPROVED
    with pytest.raises(ValidationError, match="requires reviewer"):
        candidate(review_status=ReviewStatus.REVIEWED)


def test_pool_audit_detects_duplicates() -> None:
    first = candidate()
    second = candidate(sample_id="sample-2")
    assert audit_candidate_pool([first, second]) == ["duplicate_question"]
