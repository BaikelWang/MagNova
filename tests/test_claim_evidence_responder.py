from __future__ import annotations

import json

from magnova_agent.claim_evidence_responder import MiniMindClaimEvidenceResponder


class StubGenerator:
    def __init__(self, output) -> None:
        self.output = output

    def generate(self, question, evidence) -> str:
        del question, evidence
        return self.output if isinstance(self.output, str) else json.dumps(self.output, ensure_ascii=False)


EVIDENCE = [{"source": "paper.md", "content": "坐标扩散使用 wrapped Normal 处理周期边界。", "score": 0.9}]


def test_structured_responder_renders_claim_source_citation() -> None:
    responder = MiniMindClaimEvidenceResponder(
        "http://unused",
        generator=StubGenerator(
            {
                "answerability": "answer",
                "claims": [
                    {
                        "claim_id": "C1",
                        "text": "坐标扩散使用 wrapped Normal 处理周期边界。",
                        "evidence_ids": ["E1"],
                    }
                ],
            }
        ),
    )
    answer = responder.answer("问题", messages=[], summary="", evidence=EVIDENCE)
    assert "[paper.md]" in answer
    assert not responder.last_used_fallback


def test_structured_responder_falls_back_on_invalid_json() -> None:
    responder = MiniMindClaimEvidenceResponder(
        "http://unused", generator=StubGenerator("not-json")
    )
    answer = responder.answer("问题", messages=[], summary="", evidence=EVIDENCE)
    assert "原始证据摘要" in answer
    assert responder.last_used_fallback
    assert responder.last_quality_reasons[0].startswith("invalid_claim_json")


def test_structured_responder_preserves_explicit_refusal() -> None:
    responder = MiniMindClaimEvidenceResponder(
        "http://unused",
        generator=StubGenerator(
            {
                "answerability": "insufficient_evidence",
                "claims": [],
                "answerability_reason": "无法根据现有证据确定学习率。",
            }
        ),
    )
    answer = responder.answer("问题", messages=[], summary="", evidence=EVIDENCE)
    assert "证据不足" in answer
    assert "无法根据" in answer
