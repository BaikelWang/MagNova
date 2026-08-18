from __future__ import annotations

from .claim_evidence import (
    Answerability,
    ClaimEvidenceValidator,
    index_evidence,
    parse_claim_evidence_response,
    render_claim_evidence_answer,
)
from .claim_evidence_evaluation import ClaimResponder, HTTPClaimResponder
from .rag import MiniMindRAGResponder


class MiniMindClaimEvidenceResponder:
    """Structured MiniMind responder compatible with MagNovaResearchGraph."""

    def __init__(
        self,
        endpoint: str,
        *,
        generator: ClaimResponder | None = None,
        validator: ClaimEvidenceValidator | None = None,
        max_evidence_items: int = 8,
        max_evidence_chars: int = 320,
    ) -> None:
        self.generator = generator or HTTPClaimResponder(endpoint)
        self.validator = validator or ClaimEvidenceValidator()
        self.fallback = MiniMindRAGResponder(
            endpoint,
            max_evidence_items=max_evidence_items,
            max_evidence_chars=max_evidence_chars,
        )
        self.last_raw_answer: str | None = None
        self.last_quality_reasons: list[str] = []
        self.last_used_fallback = False

    def answer(self, question, *, messages, summary, evidence) -> str:
        del messages, summary
        self.last_raw_answer = None
        self.last_quality_reasons = []
        self.last_used_fallback = False
        if not evidence:
            self.last_quality_reasons = ["no_evidence"]
            return (
                "当前论文知识库中没有检索到足够相关的证据。"
                "我暂不根据记忆猜测；请补充论文、关键词或限定研究范围。"
            )

        prepared = self.fallback.prepare_evidence(evidence)
        indexed = index_evidence(prepared)
        try:
            raw = self.generator.generate(question, indexed)
            self.last_raw_answer = raw
            parsed = parse_claim_evidence_response(raw)
        except Exception as exc:
            self.last_quality_reasons = [f"invalid_claim_json:{type(exc).__name__}"]
            self.last_used_fallback = True
            return self.fallback._extractive_fallback(prepared)

        validation = self.validator.validate(parsed, indexed)
        if not validation.valid:
            self.last_quality_reasons = [
                f"{issue.claim_id}:{issue.code.value}" for issue in validation.issues
            ]
            self.last_used_fallback = True
            return self.fallback._extractive_fallback(prepared)
        if parsed.answerability == Answerability.INSUFFICIENT_EVIDENCE:
            reason = parsed.answerability_reason or "无法根据现有证据确定答案。"
            return f"现有证据不足，{reason}"
        return render_claim_evidence_answer(parsed, indexed)
