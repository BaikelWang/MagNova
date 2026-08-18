"""Artifact visualisation panels for the MagNova Streamlit demo.

Frontend only: reads local read-only demo artifacts and renders charts.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from hashlib import md5
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from web_ui import esc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path(
    os.getenv("MAGNOVA_DEMO_ARTIFACTS", PROJECT_ROOT / "examples" / "demo_artifacts")
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
THDMOL_JS = STATIC_DIR / "3Dmol-min.js"
STRUCTURE_VIEWER_HEIGHT = 420

ARTIFACT_VIEWS = ("筛选漏斗", "可信度矩阵", "晶体结构")

PALETTE = {
    "green": "#0c8068",
    "mint": "#3fbf9f",
    "amber": "#d97706",
    "slate": "#94a3b8",
}


def _chart_ink() -> str:
    return "#c9ded8" if (
        st.session_state.get("mn_pres_mode")
        or st.session_state.get("mn_dark_mode")
    ) else "#334155"


def _chart_bg() -> str:
    return "#132923" if (
        st.session_state.get("mn_pres_mode")
        or st.session_state.get("mn_dark_mode")
    ) else "#ffffff"


def _themed(chart: alt.Chart) -> alt.Chart:
    ink = _chart_ink()
    return (
        chart.configure_view(strokeWidth=0, fill=_chart_bg())
        .configure(background=_chart_bg())
        .configure_axis(
            labelColor=ink,
            titleColor=ink,
            gridColor="rgba(148,163,184,.16)",
            domainColor="rgba(148,163,184,.28)",
            tickColor="rgba(148,163,184,.28)",
        )
        .configure_legend(labelColor=ink, titleColor=ink)
    )


@st.cache_data(show_spinner=False)
def load_demo_artifacts(root: str | Path = DEFAULT_ARTIFACT_ROOT) -> dict[str, Any]:
    base = Path(root)
    payload: dict[str, Any] = {"root": str(base), "available": base.exists()}
    if not base.exists():
        return payload
    funnel_path = base / "funnel.json"
    if funnel_path.exists():
        payload["funnel"] = json.loads(funnel_path.read_text(encoding="utf-8"))
    candidates_path = base / "per_candidate_results.json"
    if candidates_path.exists():
        payload["candidates"] = json.loads(candidates_path.read_text(encoding="utf-8"))
    return payload


def infer_stage(
    last_result: dict[str, Any] | None,
    chat_history: list[dict[str, Any]],
) -> int:
    """Furthest workflow stage reached, as an index into WORKFLOW_STEPS."""
    stage = 0
    for message in chat_history:
        if message.get("evidence"):
            stage = max(stage, 1)
        if message.get("request"):
            stage = max(stage, 2)
        if message.get("outcome"):
            stage = max(stage, 4)

    if not last_result:
        return stage
    if last_result.get("__interrupt__"):
        return max(stage, 3)

    phase = last_result.get("phase") or ""
    node = last_result.get("last_node") or ""
    agent_status = last_result.get("agent_status") or ""

    if phase in {"tool_execution", "result_explanation"} or node in {
        "tool_execution",
        "result_explanation",
    }:
        return max(stage, 4)
    if phase == "confirmation" or node == "confirmation":
        return max(stage, 3)
    if phase in {"requirement_draft", "clarification"} or agent_status in {
        "needs_clarification",
        "ready",
    }:
        return max(stage, 2)
    if node == "retrieve_evidence":
        return max(stage, 1)
    return stage


def _funnel_rows(funnel: dict[str, Any]) -> list[dict[str, Any]]:
    generated = int(funnel.get("prefilter_pass", 0)) + int(funnel.get("prefilter_fail", 0))
    if generated <= 0:
        generated = int(funnel.get("RECOVERED_RELAXED_COUNT", 28)) + int(
            funnel.get("prefilter_fail", 4)
        )
    prefilter_pass = int(funnel.get("prefilter_pass", funnel.get("relax_success", 0)))
    relax_success = int(funnel.get("relax_success", prefilter_pass))
    inferred = int(funnel.get("inferred", relax_success))
    exploitation = int(funnel.get("exploitation", 0))
    exploration = int(funnel.get("exploration", inferred - exploitation))
    prefilter_fail = int(funnel.get("prefilter_fail", max(generated - prefilter_pass, 0)))

    return [
        {"stage": "MatterGen 生成池", "count": generated, "drop": 0, "reason": "条件生成完成", "order": 1},
        {"stage": "结构合法性预筛", "count": prefilter_pass, "drop": prefilter_fail, "reason": "CIF/3D/磁性元素", "order": 2},
        {"stage": "MatterSim 预弛豫", "count": relax_success, "drop": max(prefilter_pass - relax_success, 0), "reason": "弛豫失败", "order": 3},
        {"stage": "SACGT 性质预测", "count": inferred, "drop": max(relax_success - inferred, 0), "reason": "推理失败", "order": 4},
        {"stage": "可信门控 · 主队列", "count": exploitation, "drop": exploration, "reason": "LCB/OOD/漂移", "order": 5},
        {"stage": "探索队列候选", "count": exploration, "drop": 0, "reason": "待 DFT/实验", "order": 6},
    ]


def render_funnel_panel(funnel: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    generated = int(funnel.get("prefilter_pass", 0)) + int(funnel.get("prefilter_fail", 0))
    c1.metric("初始生成", generated or 32)
    c2.metric("合法性通过", int(funnel.get("prefilter_pass", 0)))
    c3.metric("SACGT 推理", int(funnel.get("inferred", 0)))
    c4.metric("探索队列", int(funnel.get("exploration", 0)))

    exploitation = int(funnel.get("exploitation", 0))
    exploration = int(funnel.get("exploration", 0))
    if exploitation == 0 and exploration > 0:
        st.markdown(
            f'<div class="mn-funnel-note">'
            f"可信门控 · 主队列为 <b>0</b>，表示 LCB / OOD / 漂移门控较保守，"
            f"并非流水线失败。<b>{exploration}</b> 个候选进入探索队列，待 DFT / 实验验证。"
            f"</div>",
            unsafe_allow_html=True,
        )

    frame = pd.DataFrame(_funnel_rows(funnel))
    base = alt.Chart(frame).encode(
        y=alt.Y("stage:N", sort=alt.EncodingSortField(field="order"), title=None)
    )
    bars = base.mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, height=26).encode(
        x=alt.X("count:Q", title="候选数量"),
        color=alt.condition(
            alt.datum.order >= 5, alt.value(PALETTE["amber"]), alt.value(PALETTE["green"])
        ),
        tooltip=["stage", "count", "drop", "reason"],
    )
    labels = base.mark_text(align="left", baseline="middle", dx=8, color=_chart_ink()).encode(
        x="count:Q", text="count:Q"
    )
    st.altair_chart(_themed((bars + labels).properties(height=320)), width="stretch")
    drops = [row for row in _funnel_rows(funnel) if int(row.get("drop") or 0) > 0]
    if drops:
        drop_text = " · ".join(
            f"{row['stage']} −{row['drop']}（{row['reason']}）" for row in drops
        )
        st.caption(f"各阶段淘汰：{drop_text}")
    st.caption("数据来源：G1×32 恢复产物 funnel.json（本地只读）")


@lru_cache(maxsize=1)
def _load_3dmol_js() -> str:
    if THDMOL_JS.exists():
        return THDMOL_JS.read_text(encoding="utf-8", errors="ignore")
    return ""


def _cif_viewer_html(cif_text: str, viewer_id: str, height: int) -> str:
    inline_js = _load_3dmol_js()
    if inline_js:
        library = f"<script>{inline_js}</script>"
    else:
        library = '<script src="https://3Dmol.org/build/3Dmol-min.js"></script>'
    return f"""
<div id="{viewer_id}" style="width:100%;height:{height}px;border-radius:12px;overflow:hidden;background:#0f172a"></div>
{library}
<script>
(function() {{
  const viewer = $3Dmol.createViewer(document.getElementById("{viewer_id}"), {{backgroundColor: "#0f172a"}});
  viewer.addModel({json.dumps(cif_text)}, "cif");
  viewer.setStyle({{}}, {{sphere: {{radius: 0.42, colorscheme: "Jmol"}}, stick: {{radius: 0.14, colorscheme: "Jmol"}}}});
  viewer.addUnitCell(); viewer.zoomTo(); viewer.render();
  viewer.spin("y", 0.6);
}})();
</script>
"""


def render_structure_viewer(cif_path: str | None, *, title: str, height: int = STRUCTURE_VIEWER_HEIGHT) -> None:
    st.caption(title)
    if not cif_path or not Path(cif_path).exists():
        st.warning("未找到对应 CIF 文件。")
        return
    cif_text = Path(cif_path).read_text(encoding="utf-8", errors="ignore")
    st.iframe(
        _cif_viewer_html(
            cif_text,
            viewer_id=f"viewer-{md5(f'{title}{cif_path}'.encode()).hexdigest()[:10]}",
            height=height,
        ),
        height=height + 12,
    )


def render_structure_hero(candidates: list[dict[str, Any]]) -> None:
    """Large auto-spin preview of the top-ranked artifact candidate."""
    if not candidates:
        return
    top = max(candidates, key=lambda item: float(item.get("score") or 0.0))
    formula = top.get("formula", "?")
    score = float(top.get("score") or 0.0)
    cid = top.get("candidate_id", "—")
    path = top.get("structure_path_relaxed") or top.get("structure_path_raw")
    st.markdown(
        f"""
<div class="mn-hero-3d">
  <div class="mn-hero-3d-title">Top 候选 · 3D 结构预览</div>
  <p class="mn-hero-3d-sub">{esc(cid)} · {esc(formula)} · 综合分数 {score:.3f}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    render_structure_viewer(path, title="弛豫结构（自动预览）", height=360)


def render_structure_panel(candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        st.info("暂无候选结构。")
        return
    top = sorted(candidates, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    labels = [
        f"{item['candidate_id']} · {item.get('formula', '?')} · {float(item.get('score') or 0):.3f}"
        for item in top[:12]
    ]
    pick = st.selectbox("选择候选查看 3D 结构", labels, key="mn_structure_pick")
    chosen = top[labels.index(pick)]
    st.caption(
        f"漂移 {float(chosen.get('relax_probability_drift') or 0):.3f} · "
        f"OOD {chosen.get('ood_tier_relaxed', 'unknown')}"
    )
    view = st.segmented_control(
        "结构版本",
        ("生成 Raw", "弛豫 Relaxed"),
        default="生成 Raw",
        key="mn_structure_mode",
        label_visibility="collapsed",
    ) or "生成 Raw"
    cif_path = (
        chosen.get("structure_path_raw")
        if view == "生成 Raw"
        else chosen.get("structure_path_relaxed")
    )
    render_structure_viewer(cif_path, title=view, height=STRUCTURE_VIEWER_HEIGHT)


def _candidate_frame(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in candidates:
        performance = float(
            item.get("score")
            or item.get("p_fm_ground_state_relaxed")
            or item.get("p_fm_ground_state_raw")
            or 0.0
        )
        std = float(
            item.get("ensemble_std_fm_relaxed") or item.get("ensemble_std_fm_raw") or 0.15
        )
        drift = float(item.get("relax_probability_drift") or 0.0)
        ood_tier = str(item.get("ood_tier_relaxed") or item.get("ood_tier_raw") or "medium")
        penalty = {"low": 0.05, "medium": 0.25, "high": 0.45}.get(ood_tier, 0.25)
        trust = max(0.0, min(1.0, 1.0 - std * 2.5 - drift - penalty))
        if performance >= 0.35 and trust >= 0.55:
            quadrant = "优先验证"
        elif performance >= 0.35:
            quadrant = "探索性候选"
        elif trust >= 0.55:
            quadrant = "补充计算"
        else:
            quadrant = "暂缓推荐"
        rows.append(
            {
                **item,
                "candidate_id": item.get("candidate_id"),
                "formula": item.get("formula") or "UNKNOWN",
                "performance": performance,
                "trust": trust,
                "queue": item.get("queue") or "exploration",
                "quadrant": quadrant,
                "drift": drift,
                "ood_tier": ood_tier,
                "structure_path_raw": item.get("structure_path_raw"),
                "structure_path_relaxed": item.get("structure_path_relaxed"),
            }
        )
    return pd.DataFrame(rows)


def render_trust_panel(candidates: list[dict[str, Any]]) -> str | None:
    frame = _candidate_frame(candidates)
    if frame.empty:
        st.info("暂无可视化候选数据。")
        return None

    scatter = (
        alt.Chart(frame)
        .mark_circle(size=170, opacity=0.9)
        .encode(
            x=alt.X("performance:Q", title="预测性能", scale=alt.Scale(zero=False)),
            y=alt.Y("trust:Q", title="预测可信度", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "quadrant:N",
                title=None,
                scale=alt.Scale(
                    domain=["优先验证", "探索性候选", "补充计算", "暂缓推荐"],
                    range=[PALETTE["green"], "#0284c7", PALETTE["amber"], PALETTE["slate"]],
                ),
            ),
            tooltip=["candidate_id", "formula", "performance", "trust", "queue", "ood_tier", "drift"],
        )
        .properties(height=340)
    )
    rules = alt.Chart(pd.DataFrame({"x": [0.35], "y": [0.55]})).mark_rule(
        color="rgba(148,163,184,.5)", strokeDash=[6, 4]
    )
    st.altair_chart(
        _themed(scatter + rules.encode(x="x:Q") + rules.encode(y="y:Q")),
        width="stretch",
    )

    selected = st.selectbox(
        "选择候选查看详情",
        frame["candidate_id"].tolist(),
        format_func=lambda cid: f"{cid} · {frame.loc[frame['candidate_id'] == cid, 'formula'].iloc[0]}",
        key="mn_trust_pick",
    )
    detail = frame.loc[frame["candidate_id"] == selected].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("分层", detail["quadrant"])
    c2.metric("性能", f"{detail['performance']:.3f}")
    c3.metric("可信度", f"{detail['trust']:.3f}")
    c4.metric("漂移", f"{detail['drift']:.3f}")
    return selected


def render_artifact_panels() -> None:
    """G1x32 historical demo artifacts, with a self-contained view switcher."""
    artifacts = load_demo_artifacts()
    if not artifacts.get("available"):
        st.warning("未找到 G1×32 演示产物目录。")
        return

    view = st.segmented_control(
        "视图",
        ARTIFACT_VIEWS,
        default=ARTIFACT_VIEWS[0],
        key="mn_artifact_view",
        label_visibility="collapsed",
    ) or ARTIFACT_VIEWS[0]

    funnel = artifacts.get("funnel") or {}
    candidates = artifacts.get("candidates") or []

    if view == "筛选漏斗":
        render_funnel_panel(funnel)
    elif view == "可信度矩阵":
        selected = render_trust_panel(candidates)
        if selected:
            chosen = next(item for item in candidates if item["candidate_id"] == selected)
            with st.expander("SACGT 代理概率"):
                st.json(
                    {
                        key: chosen.get(key)
                        for key in (
                            "p_is_stable_relaxed",
                            "p_bandgap_gt02_relaxed",
                            "p_fm_ground_state_relaxed",
                            "ensemble_std_fm_relaxed",
                            "relax_probability_drift",
                            "queue",
                            "score",
                        )
                    }
                )
    else:
        render_structure_panel(candidates)


def render_provenance_page(
    last_result: dict[str, Any] | None,
    chat_history: list[dict[str, Any]],
) -> None:
    requests = [message["request"] for message in chat_history if message.get("request")]
    outcomes = [message["outcome"] for message in chat_history if message.get("outcome")]
    latest_request = requests[-1] if requests else (last_result or {}).get("material_request")
    latest_outcome = outcomes[-1] if outcomes else None

    c1, c2, c3 = st.columns(3)
    c1.metric("会话轮次", len(chat_history))
    c2.metric("LangGraph 阶段", (last_result or {}).get("phase", "—"))
    c3.metric("最近节点", (last_result or {}).get("last_node", "—"))

    st.markdown("**MaterialRequest 契约**")
    if latest_request:
        st.json(latest_request)
    else:
        st.caption("尚未生成结构化材料需求，请先在「对话」页发送材料设计需求。")

    st.markdown("**Agent / 工具返回**")
    if latest_outcome:
        st.json(latest_outcome)
    else:
        st.caption("尚未调用 MagNova 工具。")

    with st.expander("完整 last_result 快照"):
        st.json(last_result or {})


def render_system_page(
    *,
    health: dict[str, bool],
    backend_label: str,
    backend_mode: str,
    thread_id: str,
    on_reset,
) -> None:
    st.markdown("**服务健康检查**")
    for label, ok in health.items():
        st.markdown(f"- {'🟢' if ok else '🔴'} {label}")
    st.markdown(f"- 🟢 材料流程 · {backend_label}")

    st.markdown("**组件与端口**")
    st.markdown(
        "- `8506` MiniMind 需求解析\n"
        "- `8508` BGE Embedding\n"
        "- `8512` MiniMind RAG-SFT V2\n"
        "- `8510` 原版演示 APP\n"
        "- `8511` 可视化增强版 APP"
    )

    st.markdown("**模型与语料**")
    st.code(
        "\n".join(
            [
                f"MAGNOVA_RAG_CORPUS={os.getenv('MAGNOVA_RAG_CORPUS', 'rag-corpus-v5')}",
                f"MINIMIND_TASK_API={os.getenv('MINIMIND_TASK_API_URL', '8506')}",
                f"MINIMIND_RESEARCH_API={os.getenv('MINIMIND_RESEARCH_API_URL', '8512')}",
                f"MAGNOVA_BACKEND_MODE={backend_mode}",
                f"MAGNOVA_DEMO_ARTIFACTS={DEFAULT_ARTIFACT_ROOT}",
            ]
        )
    )

    if backend_mode == "mock":
        st.info("Mock 模式：候选仅用于接口演示，不代表科研结果。")
    else:
        st.warning("Real 模式：候选来自 SACGT AI 筛选，仍非 DFT / 实验验证。")

    st.caption(f"当前会话：{thread_id[:16]}…")
    if st.button("新建会话"):
        on_reset()
        st.rerun()
