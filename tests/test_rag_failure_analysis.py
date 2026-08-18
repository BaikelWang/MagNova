from __future__ import annotations

from magnova_agent.rag_failure_analysis import (
    analyse_failure_reports,
    render_failure_analysis_markdown,
)


def _report(case: dict) -> dict:
    return {"cases": [case]}


def test_fact_flow_separates_retrieval_packing_and_extraction_gaps() -> None:
    packed = _report(
        {
            "id": "case-1",
            "should_answer": True,
            "predicted_should_answer": True,
            "fact_hits": [False, False, True, True],
            "evidence_fact_hits": [False, True, False, True],
            "citation_precision": 1,
            "citation_recall": 1,
            "forbidden_hits": [],
            "quality_reasons": [],
            "used_safe_fallback": False,
            "error": None,
        }
    )
    raw = _report(
        {
            "id": "case-1",
            "evidence_fact_hits": [False, True, True, True],
        }
    )
    analysis = analyse_failure_reports(packed, raw_report=raw)
    assert analysis["fact_flow"] == {
        "expected_facts": 4,
        "chunk_retrieval_misses": 1,
        "raw_evidence_hits": 3,
        "packed_evidence_hits": 2,
        "available_but_omitted": 1,
        "answer_hits": 2,
        "facts_lost_by_packing": 1,
        "answer_hits_without_packed_evidence": 1,
    }
    assert set(analysis["cases"][0]["buckets"]) == {
        "chunk_retrieval_gap",
        "sentence_packing_loss",
        "model_extraction_gap",
        "answer_without_packed_evidence",
    }


def test_decision_and_quality_failures_are_counted() -> None:
    packed = _report(
        {
            "id": "case-2",
            "should_answer": True,
            "predicted_should_answer": False,
            "fact_hits": [],
            "evidence_fact_hits": [],
            "citation_precision": 0,
            "citation_recall": 0,
            "forbidden_hits": [],
            "quality_reasons": ["repetitive_or_empty"],
            "used_safe_fallback": True,
            "error": None,
        }
    )
    analysis = analyse_failure_reports(packed)
    assert analysis["bucket_counts"] == {
        "answerability_decision_error": 1,
        "citation_failure": 1,
        "generation_quality_rejected": 1,
        "safe_fallback": 1,
    }
    markdown = render_failure_analysis_markdown(analysis)
    assert "禁止把隐藏集" in markdown
    assert "answerability_decision_error" in markdown
