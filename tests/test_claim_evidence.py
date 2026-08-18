from __future__ import annotations

import pytest
from pydantic import ValidationError

from magnova_agent.claim_evidence import (
    Answerability,
    ClaimEvidenceAnswer,
    ClaimEvidenceValidator,
    ClaimValidationCode,
    EvidenceClaim,
    index_evidence,
    parse_claim_evidence_response,
    render_claim_evidence_answer,
)


def test_answer_requires_at_least_one_claim() -> None:
    with pytest.raises(ValidationError, match="at least one claim"):
        ClaimEvidenceAnswer(answerability=Answerability.ANSWER)


def test_insufficient_evidence_has_no_claims_and_requires_reason() -> None:
    with pytest.raises(ValidationError, match="must not contain claims"):
        ClaimEvidenceAnswer(
            answerability=Answerability.INSUFFICIENT_EVIDENCE,
            answerability_reason="证据不足",
            claims=[EvidenceClaim(claim_id="C1", text="结论", evidence_ids=["E1"])],
        )
    with pytest.raises(ValidationError, match="requires a reason"):
        ClaimEvidenceAnswer(answerability=Answerability.INSUFFICIENT_EVIDENCE)


def test_claim_and_evidence_ids_are_unique_and_strict() -> None:
    claim = EvidenceClaim(claim_id="C1", text="结论", evidence_ids=["e1", "E1"])
    assert claim.evidence_ids == ["E1"]
    with pytest.raises(ValidationError, match="E1"):
        EvidenceClaim(claim_id="C2", text="结论", evidence_ids=["paper.md"])


def test_index_evidence_is_stable_and_does_not_mutate_input() -> None:
    original = [{"source": "paper.md", "content": "证据一"}, {"content": "证据二"}]
    indexed = index_evidence(original)
    assert [item["evidence_id"] for item in indexed] == ["E1", "E2"]
    assert "evidence_id" not in original[0]


def test_validator_accepts_individually_grounded_claim() -> None:
    answer = ClaimEvidenceAnswer(
        answerability=Answerability.ANSWER,
        claims=[
            EvidenceClaim(
                claim_id="C1",
                text="带隙阈值为 1.5 eV。",
                evidence_ids=["E1"],
            )
        ],
    )
    report = ClaimEvidenceValidator().validate(
        answer,
        [{"evidence_id": "E1", "content": "筛选条件要求带隙阈值为 1.5 eV。"}],
    )
    assert report.valid
    assert report.issues == []


def test_validator_exposes_unknown_id_and_unsupported_number() -> None:
    answer = ClaimEvidenceAnswer(
        answerability=Answerability.ANSWER,
        claims=[
            EvidenceClaim(claim_id="C1", text="形成能为 2.5 eV。", evidence_ids=["E1"]),
            EvidenceClaim(claim_id="C2", text="另一结论", evidence_ids=["E9"]),
        ],
    )
    report = ClaimEvidenceValidator(min_bigram_coverage=0).validate(
        answer,
        [{"evidence_id": "E1", "content": "形成能为 1.0 eV。"}],
    )
    assert not report.valid
    assert {issue.code for issue in report.issues} == {
        ClaimValidationCode.UNSUPPORTED_NUMBER,
        ClaimValidationCode.UNKNOWN_EVIDENCE_ID,
    }


def test_parser_accepts_json_fence_but_rejects_prose_wrappers() -> None:
    parsed = parse_claim_evidence_response(
        '```json\n{"answerability":"answer","claims":['
        '{"claim_id":"C1","text":"结论","evidence_ids":["E1"]}]}'
        '\n```'
    )
    assert parsed.claims[0].claim_id == "C1"
    with pytest.raises(ValueError, match="valid JSON"):
        parse_claim_evidence_response("下面是答案：{\"answerability\":\"answer\"}")


def test_renderer_converts_internal_ids_to_source_citations() -> None:
    answer = ClaimEvidenceAnswer(
        answerability=Answerability.ANSWER,
        claims=[
            EvidenceClaim(
                claim_id="C1",
                text="MatterGen 联合生成晶体状态。",
                evidence_ids=["E1", "E2"],
            )
        ],
        uncertainties=["实验可合成性仍需验证"],
    )
    rendered = render_claim_evidence_answer(
        answer,
        [
            {"evidence_id": "E1", "source": "mattergen.md"},
            {"evidence_id": "E2", "source": "mattergen.md"},
        ],
    )
    assert rendered.count("[mattergen.md]") == 1
    assert "不确定性" in rendered
