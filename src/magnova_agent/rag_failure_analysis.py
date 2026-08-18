from __future__ import annotations

from collections import Counter
from typing import Any


def _cases_by_id(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not report:
        return {}
    cases = report.get("cases", report.get("items", []))
    return {str(item["id"]): item for item in cases}


def _hit_at(values: list[bool], index: int) -> bool | None:
    return bool(values[index]) if index < len(values) else None


def analyse_failure_reports(
    packed_report: dict[str, Any],
    *,
    raw_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify failure mechanisms without copying hidden answers into training data."""

    packed_cases = _cases_by_id(packed_report)
    raw_cases = _cases_by_id(raw_report)
    bucket_counts: Counter[str] = Counter()
    bucket_case_ids: dict[str, list[str]] = {}
    fact_flow: Counter[str] = Counter()
    case_results: list[dict[str, Any]] = []

    for case_id, item in packed_cases.items():
        raw_item = raw_cases.get(case_id, {})
        answer_hits = list(item.get("fact_hits") or [])
        packed_hits = list(item.get("evidence_fact_hits") or [])
        raw_hits = list(raw_item.get("evidence_fact_hits") or [])
        fact_count = max(len(answer_hits), len(packed_hits), len(raw_hits))
        buckets: set[str] = set()

        if item.get("error"):
            buckets.add("runtime_error")
        if item.get("should_answer") != item.get("predicted_should_answer"):
            buckets.add("answerability_decision_error")
        elif not item.get("should_answer") and not item.get("predicted_should_answer"):
            buckets.add("correct_refusal")
        if item.get("forbidden_hits"):
            buckets.add("forbidden_claim")
        if item.get("quality_reasons"):
            buckets.add("generation_quality_rejected")
        if item.get("used_safe_fallback"):
            buckets.add("safe_fallback")
        if item.get("should_answer") and (
            (item.get("citation_precision") or 0) < 1
            or (item.get("citation_recall") or 0) < 1
        ):
            buckets.add("citation_failure")

        for index in range(fact_count):
            raw_hit = _hit_at(raw_hits, index)
            packed_hit = _hit_at(packed_hits, index)
            answer_hit = _hit_at(answer_hits, index)
            fact_flow["expected_facts"] += 1
            if raw_hit is True:
                fact_flow["raw_evidence_hits"] += 1
            if packed_hit is True:
                fact_flow["packed_evidence_hits"] += 1
            if answer_hit is True:
                fact_flow["answer_hits"] += 1

            if raw_hit is False:
                buckets.add("chunk_retrieval_gap")
                fact_flow["chunk_retrieval_misses"] += 1
            elif raw_hit is True and packed_hit is False:
                buckets.add("sentence_packing_loss")
                fact_flow["facts_lost_by_packing"] += 1
            elif raw_hit is None and packed_hit is False:
                buckets.add("packed_evidence_gap")

            if packed_hit is True and answer_hit is False:
                buckets.add("model_extraction_gap")
                fact_flow["available_but_omitted"] += 1
            elif packed_hit is False and answer_hit is True:
                buckets.add("answer_without_packed_evidence")
                fact_flow["answer_hits_without_packed_evidence"] += 1

        if item.get("should_answer") and fact_count and all(answer_hits):
            buckets.add("complete_fact_coverage")
        if not buckets:
            buckets.add("no_detected_failure")

        ordered_buckets = sorted(buckets)
        for bucket in ordered_buckets:
            bucket_counts[bucket] += 1
            bucket_case_ids.setdefault(bucket, []).append(case_id)
        case_results.append(
            {
                "id": case_id,
                "buckets": ordered_buckets,
                "fact_count": fact_count,
                "raw_evidence_hits": sum(value is True for value in raw_hits),
                "packed_evidence_hits": sum(value is True for value in packed_hits),
                "answer_hits": sum(value is True for value in answer_hits),
            }
        )

    return {
        "notice": (
            "Diagnostic output only. Hidden case text and answers must not be copied "
            "into training or development data."
        ),
        "num_cases": len(case_results),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "bucket_case_ids": dict(sorted(bucket_case_ids.items())),
        "fact_flow": dict(fact_flow),
        "cases": case_results,
    }


def render_failure_analysis_markdown(analysis: dict[str, Any]) -> str:
    fact_flow = analysis["fact_flow"]
    expected = fact_flow.get("expected_facts", 0)

    def percentage(key: str) -> str:
        value = fact_flow.get(key, 0)
        return f"{value / expected:.2%}" if expected else "N/A"

    lines = [
        "# MagNova RAG 冻结集失败分桶",
        "",
        "> 该报告只用于诊断系统机制。禁止把隐藏集问题、答案或逐例标签复制进训练/开发数据。",
        "",
        "## 事实流",
        "",
        "| 阶段 | 命中数 | 占全部期望事实 |",
        "|---|---:|---:|",
        f"| 原始 Chunk 证据 | {fact_flow.get('raw_evidence_hits', 0)} | {percentage('raw_evidence_hits')} |",
        f"| 句子打包证据 | {fact_flow.get('packed_evidence_hits', 0)} | {percentage('packed_evidence_hits')} |",
        f"| 最终答案 | {fact_flow.get('answer_hits', 0)} | {percentage('answer_hits')} |",
        "",
        "## 机制分桶",
        "",
        "| 分桶 | 样例数 |",
        "|---|---:|",
    ]
    for bucket, count in analysis["bucket_counts"].items():
        lines.append(f"| `{bucket}` | {count} |")
    lines.extend(
        [
            "",
            "## 事实损失位置",
            "",
            f"- 原始 Chunk 未覆盖：{fact_flow.get('chunk_retrieval_misses', 0)} 个事实。",
            f"- Sentence Packing 丢失：{fact_flow.get('facts_lost_by_packing', 0)} 个事实。",
            f"- 已进入打包证据但模型漏答：{fact_flow.get('available_but_omitted', 0)} 个事实。",
            f"- 答案命中但打包证据未命中：{fact_flow.get('answer_hits_without_packed_evidence', 0)} 个事实。",
            "",
            "## 使用规则",
            "",
            "隐藏集只决定方向和最终是否发布。训练数据必须从原论文另行构造同类型问题，不能复制本报告中的隐藏样例。",
            "",
        ]
    )
    return "\n".join(lines)
