from __future__ import annotations

import json
import re
from enum import Enum

from pydantic import Field, field_validator, model_validator

from .models import StrictModel


CLAIM_EVIDENCE_SYSTEM_PROMPT = (
    "你是 MagNova 的材料科研助手。只能依据本轮 E1、E2 等检索证据回答，"
    "不得使用外部记忆补充事实。只输出符合约定 Schema 的 JSON："
    "answerability 为 answer 或 insufficient_evidence；answer 时把每个独立结论写成"
    "带 claim_id、text、evidence_ids 的 claims；证据不足时 claims 必须为空并填写"
    "answerability_reason。不要输出 Markdown，不要执行材料生成流程。"
)


class Answerability(str, Enum):
    ANSWER = "answer"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EvidenceClaim(StrictModel):
    claim_id: str = Field(pattern=r"^C[1-9]\d*$")
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, values: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(value.strip().upper() for value in values))
        if any(not re.fullmatch(r"E[1-9]\d*", value) for value in cleaned):
            raise ValueError("evidence ids must use E1, E2, ... format")
        return cleaned


class ClaimEvidenceAnswer(StrictModel):
    answerability: Answerability
    claims: list[EvidenceClaim] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    answerability_reason: str | None = None

    @field_validator("uncertainties")
    @classmethod
    def clean_uncertainties(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def validate_answerability_contract(self) -> "ClaimEvidenceAnswer":
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim ids must be unique")
        if self.answerability == Answerability.ANSWER and not self.claims:
            raise ValueError("answerable output must contain at least one claim")
        if self.answerability == Answerability.INSUFFICIENT_EVIDENCE:
            if self.claims:
                raise ValueError("insufficient-evidence output must not contain claims")
            if not self.answerability_reason:
                raise ValueError("insufficient-evidence output requires a reason")
        return self


class ClaimValidationCode(str, Enum):
    UNKNOWN_EVIDENCE_ID = "unknown_evidence_id"
    UNSUPPORTED_NUMBER = "unsupported_number"
    LOW_LEXICAL_GROUNDING = "low_lexical_grounding"


class ClaimValidationIssue(StrictModel):
    claim_id: str
    code: ClaimValidationCode
    detail: str


class ClaimValidationReport(StrictModel):
    valid: bool
    issues: list[ClaimValidationIssue] = Field(default_factory=list)


def index_evidence(evidence: list[dict]) -> list[dict]:
    """Attach stable prompt-local E1/E2 identifiers without mutating retrieval data."""

    return [
        {**item, "evidence_id": f"E{index}"}
        for index, item in enumerate(evidence, start=1)
    ]


def format_claim_evidence_user_prompt(question: str, evidence: list[dict]) -> str:
    evidence_text = "\n\n".join(
        f"[{item['evidence_id']}] 来源：{item.get('source', 'unknown')}\n{item.get('content', '')}"
        for item in evidence
    )
    return f"检索证据：\n{evidence_text}\n\n当前问题：{question}"


class ClaimEvidenceValidator:
    _number_pattern = re.compile(r"\d+(?:[.,]\d+)*(?:%|Å|eV|K)?", re.IGNORECASE)

    def __init__(self, *, min_bigram_coverage: float = 0.25) -> None:
        if not 0 <= min_bigram_coverage <= 1:
            raise ValueError("min_bigram_coverage must be in [0, 1]")
        self.min_bigram_coverage = min_bigram_coverage

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())

    @classmethod
    def _bigrams(cls, text: str) -> set[str]:
        normalised = cls._normalise(text)
        return {
            normalised[index : index + 2]
            for index in range(max(0, len(normalised) - 1))
        }

    def validate(
        self,
        answer: ClaimEvidenceAnswer,
        evidence: list[dict],
    ) -> ClaimValidationReport:
        if answer.answerability == Answerability.INSUFFICIENT_EVIDENCE:
            return ClaimValidationReport(valid=True)

        evidence_by_id = {
            str(item.get("evidence_id", "")).upper(): item
            for item in evidence
            if item.get("evidence_id")
        }
        issues: list[ClaimValidationIssue] = []
        for claim in answer.claims:
            missing_ids = [
                evidence_id
                for evidence_id in claim.evidence_ids
                if evidence_id not in evidence_by_id
            ]
            if missing_ids:
                issues.append(
                    ClaimValidationIssue(
                        claim_id=claim.claim_id,
                        code=ClaimValidationCode.UNKNOWN_EVIDENCE_ID,
                        detail=",".join(missing_ids),
                    )
                )
                continue

            cited_text = "\n".join(
                str(evidence_by_id[evidence_id].get("content", ""))
                for evidence_id in claim.evidence_ids
            )
            unsupported_numbers = [
                number
                for number in self._number_pattern.findall(claim.text)
                if self._normalise(number) not in self._normalise(cited_text)
            ]
            if unsupported_numbers:
                issues.append(
                    ClaimValidationIssue(
                        claim_id=claim.claim_id,
                        code=ClaimValidationCode.UNSUPPORTED_NUMBER,
                        detail=",".join(unsupported_numbers),
                    )
                )

            claim_bigrams = self._bigrams(claim.text)
            evidence_bigrams = self._bigrams(cited_text)
            coverage = (
                len(claim_bigrams.intersection(evidence_bigrams)) / len(claim_bigrams)
                if claim_bigrams
                else 0.0
            )
            if coverage < self.min_bigram_coverage:
                issues.append(
                    ClaimValidationIssue(
                        claim_id=claim.claim_id,
                        code=ClaimValidationCode.LOW_LEXICAL_GROUNDING,
                        detail=f"coverage={coverage:.4f}",
                    )
                )
        return ClaimValidationReport(valid=not issues, issues=issues)


def parse_claim_evidence_response(text: str) -> ClaimEvidenceAnswer:
    """Parse strict JSON, allowing only an optional surrounding Markdown fence."""

    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("claim-evidence response must be valid JSON") from exc
    return ClaimEvidenceAnswer.model_validate(payload)


def render_claim_evidence_answer(
    answer: ClaimEvidenceAnswer,
    evidence: list[dict],
) -> str:
    """Render validated internal claims as readable text with source citations."""

    if answer.answerability == Answerability.INSUFFICIENT_EVIDENCE:
        return answer.answerability_reason or "现有证据不足，无法确定答案。"
    evidence_by_id = {
        str(item.get("evidence_id", "")).upper(): item for item in evidence
    }
    lines: list[str] = []
    for claim in answer.claims:
        sources = list(
            dict.fromkeys(
                str(evidence_by_id[evidence_id].get("source", "unknown"))
                for evidence_id in claim.evidence_ids
                if evidence_id in evidence_by_id
            )
        )
        citations = "".join(f"[{source}]" for source in sources)
        lines.append(f"{claim.text}{citations}")
    if answer.uncertainties:
        lines.append("不确定性：" + "；".join(answer.uncertainties))
    return "\n".join(lines)
