from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator, model_validator

from .claim_evidence import ClaimEvidenceAnswer
from .models import StrictModel


class ReviewStatus(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


class EvidenceRecord(StrictModel):
    evidence_id: str = Field(pattern=r"^E[1-9]\d*$")
    source: str = Field(min_length=1)
    content: str = Field(min_length=10)
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    page: int | None = Field(default=None, ge=1)
    chunk_id: str | None = None


class ClaimEvidenceCandidate(StrictModel):
    sample_id: str = Field(min_length=1)
    question: str = Field(min_length=2)
    evidence: list[EvidenceRecord] = Field(min_length=1)
    target: ClaimEvidenceAnswer
    leakage_group: str = Field(min_length=1)
    origin_split: str
    tags: list[str] = Field(default_factory=list)
    auto_issues: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.DRAFT
    reviewer: str | None = None
    review_notes: list[str] = Field(default_factory=list)

    @field_validator("tags", "auto_issues", "review_notes")
    @classmethod
    def unique_nonempty_strings(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def validate_candidate_contract(self) -> "ClaimEvidenceCandidate":
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence ids must be unique")
        known = set(evidence_ids)
        referenced = {
            evidence_id
            for claim in self.target.claims
            for evidence_id in claim.evidence_ids
        }
        if not referenced.issubset(known):
            raise ValueError("target references unknown evidence ids")
        if self.review_status in {ReviewStatus.REVIEWED, ReviewStatus.APPROVED}:
            if not self.reviewer:
                raise ValueError("reviewed candidate requires reviewer")
        if self.review_status == ReviewStatus.APPROVED:
            if self.auto_issues:
                raise ValueError("candidate with automatic issues cannot be approved")
            if any(item.source_sha256 is None for item in self.evidence):
                raise ValueError("approved candidate requires source hashes")
        return self


def audit_candidate_pool(candidates: list[ClaimEvidenceCandidate]) -> list[str]:
    issues: list[str] = []
    sample_ids = [item.sample_id for item in candidates]
    if len(sample_ids) != len(set(sample_ids)):
        issues.append("duplicate_sample_id")
    questions = [item.question.casefold().strip() for item in candidates]
    if len(questions) != len(set(questions)):
        issues.append("duplicate_question")
    approved = [item for item in candidates if item.review_status == ReviewStatus.APPROVED]
    groups_by_split: dict[str, set[str]] = {}
    for item in approved:
        groups_by_split.setdefault(item.origin_split, set()).add(item.leakage_group)
    split_names = sorted(groups_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = groups_by_split[left].intersection(groups_by_split[right])
            if overlap:
                issues.append(f"leakage_group_overlap:{left}:{right}:{','.join(sorted(overlap))}")
    return issues
