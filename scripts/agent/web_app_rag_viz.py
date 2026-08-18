from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st
from langgraph.types import Command

sys.path.insert(0, os.path.dirname(__file__))
from web_ui import (
    EXAMPLES,
    NAV_ITEMS,
    PAGES,
    advance_demo,
    confirmation_dialog,
    inject_css,
    inject_css_footer,
    is_dark_mode,
    next_demo_action,
    render_candidates,
    render_evidence,
    render_export_button,
    render_gate_banner,
    render_header,
    render_metrics_page,
    render_progress,
    render_request_summary,
    render_welcome,
    run_with_pipeline_status,
    stream_text_chunks,
)
from web_viz import (
    infer_stage,
    load_demo_artifacts,
    render_artifact_panels,
    render_provenance_page,
    render_structure_hero,
    render_system_page,
)

from magnova_agent import (
    CliMagNovaBackend,
    GuardedEvidenceRetriever,
    HTTPBGEEmbeddings,
    HTTPMagNovaBackend,
    LangChainPaperRetriever,
    MagNovaAgent,
    MagNovaResearchGraph,
    MagNovaToolAdapter,
    MiniMindInterpreter,
    MiniMindRAGResponder,
    MockPipelineMagNovaBackend,
    PollingPolicy,
    RuleBasedInterpreter,
    SentenceEvidencePacker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_MODE = os.getenv("MAGNOVA_BACKEND_MODE", "mock").strip().lower()
BACKEND_LABEL = {
    "mock": "Mock（演示，非科研结果）",
    "http": "真实 HTTP MagNova 后端",
    "cli": "真实 CLI→SACGT MagNova 后端",
    "real": "真实 CLI→SACGT MagNova 后端",
}.get(BACKEND_MODE, BACKEND_MODE)
SERVICES = {
    "需求理解": "http://127.0.0.1:8506/docs",
    "论文检索": "http://127.0.0.1:8508/health",
    "科研回答": "http://127.0.0.1:8512/docs",
}
EXAMPLE_LABELS = [label for label, _ in EXAMPLES]
EXAMPLE_PROMPTS = dict(EXAMPLES)

WIDGET_KEYS = (
    "mn_artifact_view",
    "mn_example_pick",
    "mn_trust_pick",
    "mn_structure_pick",
    "mn_structure_mode",
)

st.set_page_config(
    page_title="MagNova · 材料科研智能体",
    page_icon="🧲",
    layout="centered",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# session state
# --------------------------------------------------------------------------
MAX_CONVERSATIONS = 20


def new_conversation() -> dict:
    """A conversation record. thread_id is also the LangGraph checkpoint key."""
    return {
        "id": uuid4().hex,
        "thread_id": uuid4().hex,
        "title": "新对话",
        "chat_history": [],
        "last_result": None,
    }


def init_state() -> None:
    if "mn_convs" not in st.session_state:
        record = new_conversation()
        st.session_state.mn_convs = [record]
        st.session_state.mn_conv_id = record["id"]
    defaults = {
        "mn_pres_mode": False,
        "mn_dark_mode": False,
        "mn_page": PAGES[0],
        "pending_prompt": None,
        "mn_last_example": None,
        "mn_gate_open": False,
        "mn_gate_decision": None,
        "mn_demo_on": False,
        "mn_demo_step": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Migrate legacy widget keys from earlier builds.
    if st.session_state.get("mn_presentation") or st.session_state.get("mn_presentation_on"):
        st.session_state.mn_pres_mode = True
    if st.session_state.get("mn_dark"):
        st.session_state.mn_dark_mode = True


def conv() -> dict:
    """The active conversation record. Mutating it persists to history."""
    records = st.session_state.mn_convs
    for record in records:
        if record["id"] == st.session_state.mn_conv_id:
            return record
    st.session_state.mn_conv_id = records[0]["id"]
    return records[0]


def activate_conversation(conv_id: str) -> None:
    st.session_state.mn_conv_id = conv_id
    st.session_state.pending_prompt = None
    st.session_state.mn_last_example = None
    st.session_state.mn_gate_open = False
    st.session_state.mn_gate_decision = None
    st.session_state.mn_demo_on = False
    st.session_state.mn_demo_step = 0
    st.session_state["_mn_clear_widgets"] = True


def start_new_conversation() -> None:
    """Archive the current conversation and switch to a fresh one."""
    if not conv()["chat_history"]:
        return
    record = new_conversation()
    st.session_state.mn_convs.insert(0, record)
    del st.session_state.mn_convs[MAX_CONVERSATIONS:]
    activate_conversation(record["id"])


def apply_deferred_state() -> None:
    """Must run at the very top, before any widget is instantiated."""
    if st.session_state.pop("_mn_clear_widgets", False):
        for key in WIDGET_KEYS:
            st.session_state.pop(key, None)
    if st.session_state.pop("_mn_reset_artifact_view", False):
        st.session_state.pop("mn_artifact_view", None)
    pending_view = st.session_state.pop("_mn_artifact_view", None)
    if pending_view:
        st.session_state.mn_artifact_view = pending_view


def go_to_candidates_with_structure() -> None:
    go_to("候选")
    st.session_state["_mn_reset_artifact_view"] = True
    st.session_state["_mn_artifact_view"] = "晶体结构"


def go_to(page: str) -> None:
    if page in PAGES:
        st.session_state.mn_page = page


# --------------------------------------------------------------------------
# backend wiring (unchanged)
# --------------------------------------------------------------------------
def build_backend():
    if BACKEND_MODE == "mock":
        return MockPipelineMagNovaBackend()
    if BACKEND_MODE == "http":
        base_url = os.getenv("MAGNOVA_BACKEND_URL", "").strip()
        if not base_url:
            raise RuntimeError("MAGNOVA_BACKEND_MODE=http 时必须配置 MAGNOVA_BACKEND_URL")
        return HTTPMagNovaBackend(
            base_url, api_key=os.getenv("MAGNOVA_BACKEND_API_KEY") or None
        )
    if BACKEND_MODE in {"cli", "real"}:
        return CliMagNovaBackend(
            sacgt_python=os.getenv("SACGT_PYTHON", "python"),
            cli_script=os.getenv(
                "MAGNOVA_SACGT_CLI", "run_magnova_request.py"
            ),
            output_root=os.getenv(
                "MAGNOVA_OUTPUT_ROOT", str(PROJECT_ROOT / "runtime" / "magnova_runs")
            ),
            mode=os.getenv("MAGNOVA_CLI_MODE", "from_artifacts"),
            request_timeout_seconds=float(os.getenv("MAGNOVA_BACKEND_TIMEOUT", "86400")),
            working_directory=os.getenv("MAGNOVA_SACGT_WORKDIR") or None,
        )
    raise RuntimeError("MAGNOVA_BACKEND_MODE 仅支持 mock | http | cli | real")


@st.cache_resource
def build_workflow() -> MagNovaResearchGraph:
    embeddings = HTTPBGEEmbeddings(
        os.getenv("BGE_API_URL", "http://127.0.0.1:8508/v1/embeddings")
    )
    base = LangChainPaperRetriever.from_directory(
        os.getenv("MAGNOVA_RAG_CORPUS", str(PROJECT_ROOT / "rag-corpus-v5")),
        embeddings=embeddings,
        score_threshold=float(os.getenv("MAGNOVA_RAG_THRESHOLD", "0.0")),
    )
    retriever = GuardedEvidenceRetriever(SentenceEvidencePacker(base, embeddings))
    responder = MiniMindRAGResponder(
        os.getenv(
            "MINIMIND_RESEARCH_API_URL", "http://127.0.0.1:8512/v1/chat/completions"
        )
    )
    interpreter = MiniMindInterpreter(
        endpoint=os.getenv(
            "MINIMIND_TASK_API_URL", "http://127.0.0.1:8506/v1/chat/completions"
        ),
        fallback=RuleBasedInterpreter(),
        model_version="magnova-sft-v51-v54-runtime",
    )
    tool = MagNovaToolAdapter(
        build_backend(),
        policy=PollingPolicy(
            timeout_seconds=float(os.getenv("MAGNOVA_BACKEND_TIMEOUT", "300"))
        ),
    )
    return MagNovaResearchGraph(
        MagNovaAgent(interpreter=interpreter, tool=tool),
        research_responder=responder,
        evidence_retriever=retriever,
    )


@st.cache_data(ttl=15)
def service_health() -> dict[str, bool]:
    status = {}
    for label, url in SERVICES.items():
        try:
            status[label] = requests.get(url, timeout=2).ok
        except requests.RequestException:
            status[label] = False
    return status


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------
def format_constraint(item: dict) -> str:
    operators = {
        "gt": ">",
        "gte": "≥",
        "lt": "<",
        "lte": "≤",
        "eq": "=",
        "range": "范围",
        "maximize": "尽可能高",
        "minimize": "尽可能低",
    }
    operator = operators.get(item.get("operator"), item.get("operator", ""))
    if item.get("operator") == "range":
        value = f"{item.get('lower_bound')}–{item.get('upper_bound')}"
    else:
        value = item.get("value", "")
    return " ".join(
        str(part)
        for part in (item.get("name", "目标性质"), operator, value, item.get("unit"))
        if part not in (None, "")
    )


# --------------------------------------------------------------------------
# conversation flow
# --------------------------------------------------------------------------
def has_candidates(result: dict) -> bool:
    """`requirement_draft` also emits agent_outcome, but without any candidates."""
    outcome = result.get("agent_outcome") or {}
    return bool(outcome.get("candidates"))


def append_result(result: dict, stream_text: str | None = None) -> None:
    history = conv()["chat_history"]
    if result.get("__interrupt__"):
        history.append(
            {
                "role": "assistant",
                "content": (
                    "需求已经结构化并通过校验。请检查参数，确认后才会调用"
                    f" {BACKEND_LABEL}。"
                ),
                "request": result.get("material_request", {}),
            }
        )
        return

    entry = {
        "role": "assistant",
        "content": stream_text or result.get("response") or "工作流已完成。",
        "evidence": result.get("retrieved_evidence", []),
        "retrieval_query": result.get("retrieval_query"),
        "stream_pending": bool(stream_text),
    }
    if result.get("material_request"):
        entry["request"] = result["material_request"]
    if has_candidates(result):
        entry["outcome"] = result["agent_outcome"]
        go_to_candidates_with_structure()
    history.append(entry)


def append_error(exc: Exception) -> None:
    conv()["chat_history"].append(
        {
            "role": "assistant",
            "content": (
                "当前请求没有完成。请稍后重试；如果问题持续存在，请到「系统」页检查服务状态。"
                f"\n\n技术信息：`{type(exc).__name__}: {str(exc)[:240]}`"
            ),
        }
    )


def submit_prompt(workflow: MagNovaResearchGraph, config: dict, prompt: str) -> None:
    record = conv()
    if not record["chat_history"]:
        record["title"] = prompt[:16] + ("…" if len(prompt) > 16 else "")
    record["chat_history"].append({"role": "user", "content": prompt})
    try:
        with st.spinner("MiniMind 正在理解并检索…"):
            result = workflow.graph.invoke(
                {
                    "session_id": record["thread_id"],
                    "user_text": prompt,
                    "messages": [],
                },
                config=config,
            )
        record["last_result"] = result
        response = result.get("response") or (
            f"需求已经结构化并通过校验。请检查参数，确认后才会调用 {BACKEND_LABEL}。"
            if result.get("__interrupt__")
            else "工作流已完成。"
        )
        append_result(result, stream_text=response)
    except Exception as exc:
        record["last_result"] = None
        append_error(exc)


def resume_workflow(
    workflow: MagNovaResearchGraph, config: dict, *, approve: bool
) -> None:
    record = conv()
    try:
        if approve:
            result = run_with_pipeline_status(
                lambda: workflow.graph.invoke(Command(resume=True), config=config),
                backend_mode=BACKEND_MODE,
            )
        else:
            result = workflow.graph.invoke(Command(resume=False), config=config)
        record["last_result"] = result
        append_result(result)
    except Exception as exc:
        record["last_result"] = None
        append_error(exc)


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------
def render_chat_page(workflow: MagNovaResearchGraph, config: dict) -> None:
    record = conv()
    history = record["chat_history"]

    if not history:
        render_welcome()
        st.caption("点选一个示例问题即可发送；随时可用侧边栏的「＋ 新对话」回到这里。")
        picked = st.pills(
            "示例",
            EXAMPLE_LABELS,
            key="mn_example_pick",
            label_visibility="collapsed",
        )
        if picked and picked != st.session_state.get("mn_last_example"):
            st.session_state.mn_last_example = picked
            st.session_state.pending_prompt = EXAMPLE_PROMPTS[picked]

    for index, message in enumerate(history):
        with st.chat_message(message["role"]):
            if message.get("stream_pending"):
                st.write_stream(stream_text_chunks(message["content"]))
                history[index]["stream_pending"] = False
            else:
                st.markdown(message["content"])
            if message.get("retrieval_query"):
                st.caption(f"检索查询：{message['retrieval_query']}")
            render_evidence(message.get("evidence") or [])
            if message.get("request"):
                with st.expander("模型理解的材料需求", expanded=True):
                    render_request_summary(message["request"], format_constraint)
                    st.json(message["request"], expanded=False)
            if message.get("outcome"):
                render_outcome(message["outcome"])

    last_result = record["last_result"]
    if last_result and last_result.get("__interrupt__"):
        decision = st.session_state.pop("mn_gate_decision", None)
        if decision:
            resume_workflow(workflow, config, approve=decision == "approve")
            if st.session_state.get("mn_demo_on"):
                advance_demo()
            st.rerun()
        if st.session_state.get("mn_gate_open"):
            confirmation_dialog(
                last_result.get("material_request", {}),
                backend_mode=BACKEND_MODE,
                backend_label=BACKEND_LABEL,
                format_constraint=format_constraint,
            )
        else:
            render_gate_banner()
            if st.button("打开确认闸门", type="primary"):
                st.session_state.mn_gate_open = True
                st.rerun()

    typed = st.chat_input("询问论文、讨论方法，或提出材料设计需求…")
    prompt = st.session_state.pending_prompt or typed

    action = next_demo_action()
    if action and not prompt:
        kind, value = action
        if kind == "prompt":
            prompt = value
        elif last_result and last_result.get("__interrupt__"):
            st.session_state.mn_gate_decision = value
            advance_demo()
            st.rerun()

    if prompt:
        st.session_state.pending_prompt = None
        submit_prompt(workflow, config, prompt)
        if st.session_state.get("mn_demo_on"):
            advance_demo()
        st.rerun()


def render_outcome(outcome: dict) -> None:
    candidates = outcome.get("candidates") or []
    task_id = outcome.get("task_id")
    c1, c2, c3 = st.columns(3)
    c1.metric("执行状态", outcome.get("status", "unknown"))
    c2.metric("候选数量", len(candidates))
    c3.metric("任务编号", task_id[-10:] if task_id else "—")
    render_candidates(candidates, backend_mode=BACKEND_MODE)
    if candidates:
        st.caption("完整漏斗与 3D 结构见「候选」页。")
    with st.expander("完整 Agent / 工具返回"):
        st.json(outcome)


def render_candidate_page() -> None:
    record = conv()
    outcomes = [
        message["outcome"]
        for message in record["chat_history"]
        if message.get("outcome")
    ]
    if outcomes:
        latest = outcomes[-1]
        st.markdown("**本次对话返回的候选**")
        render_candidates(latest.get("candidates") or [], backend_mode=BACKEND_MODE)
        render_export_button(
            latest, backend_mode=BACKEND_MODE, thread_id=record["thread_id"]
        )
        st.divider()

    artifacts = load_demo_artifacts()
    if artifacts.get("available") and artifacts.get("candidates"):
        render_structure_hero(artifacts["candidates"])
        st.divider()
        st.markdown("**G1×32 历史演示产物**")
    render_artifact_panels()


def render_conversation_list() -> None:
    records = st.session_state.mn_convs
    st.markdown(
        f'<div class="mn-nav-label">历史对话 · {len(records)}</div>',
        unsafe_allow_html=True,
    )
    active_id = st.session_state.mn_conv_id
    for record in records:
        active = record["id"] == active_id
        turns = sum(1 for message in record["chat_history"] if message["role"] == "user")
        title = record["title"] if record["chat_history"] else "新对话（空）"
        state = "on" if active else "off"
        with st.container(key=f"mn-conv-{state}-{record['id'][:8]}"):
            if st.button(
                f"{title}　{turns} 问" if turns else title,
                key=f"mn_conv_btn_{record['id']}",
                width="stretch",
                help=f"会话 {record['thread_id'][:8]}…",
            ):
                if not active:
                    activate_conversation(record["id"])
                    go_to("对话")
                    st.rerun()


def render_sidebar(health: dict[str, bool]) -> str:
    current = st.session_state.get("mn_page", PAGES[0])
    with st.sidebar:
        st.markdown(
            '<div class="mn-brand">'
            '<div class="mn-brand-name">🧲 MagNova</div>'
            '<div class="mn-brand-sub">材料科研智能体</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        with st.container(key="mn-new-chat"):
            if st.button("＋ 新对话", width="stretch"):
                start_new_conversation()
                go_to("对话")
                st.rerun()

        st.markdown('<div class="mn-nav-label">工作区</div>', unsafe_allow_html=True)
        for index, (label, icon) in enumerate(NAV_ITEMS):
            active = label == current
            state = "on" if active else "off"
            with st.container(key=f"mn-nav-{state}-{index}"):
                if st.button(f"{icon}  {label}", key=f"mn_nav_btn_{label}", width="stretch"):
                    if not active:
                        go_to(label)
                        st.rerun()

        render_conversation_list()
        st.divider()

        with st.container(key="mn-theme-controls"):
            if st.session_state.get("mn_pres_mode"):
                st.session_state._ui_pres = True

            def _sync_pres_mode() -> None:
                st.session_state.mn_pres_mode = bool(st.session_state._ui_pres)

            if "_ui_pres" not in st.session_state:
                st.session_state._ui_pres = st.session_state.mn_pres_mode
            st.toggle(
                "答辩模式",
                key="_ui_pres",
                on_change=_sync_pres_mode,
                help="深色主题 + 隐藏步骤说明，适合投影演示",
            )

            if st.session_state.mn_pres_mode:
                st.caption("已启用深色主题（答辩）")
            else:
                if "_ui_dark" not in st.session_state:
                    st.session_state._ui_dark = st.session_state.mn_dark_mode

                def _sync_dark_mode() -> None:
                    st.session_state.mn_dark_mode = bool(st.session_state._ui_dark)

                st.toggle("深色模式", key="_ui_dark", on_change=_sync_dark_mode)

        if st.session_state.get("mn_demo_on"):
            st.caption(f"演示进行中 · 第 {st.session_state.get('mn_demo_step', 0) + 1} 步")
            if st.button("停止演示", width="stretch"):
                st.session_state.mn_demo_on = False
                st.session_state.mn_demo_step = 0
                st.rerun()
        elif st.button("▶ 一键演示", width="stretch"):
            st.session_state.mn_demo_on = True
            st.session_state.mn_demo_step = 0
            go_to("对话")
            st.rerun()

        if not all(health.values()):
            st.error("部分服务不可用，详见「系统」页。")
        st.caption(f"会话 {conv()['thread_id'][:10]}…")
    return current


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------
init_state()
apply_deferred_state()

# Theme reads the canonical mn_pres_mode / mn_dark_mode flags, not widget return
# values, so it can be injected before the sidebar renders — no unstyled flash.
inject_css(is_dark_mode())

health = service_health()
page = render_sidebar(health)

active = conv()

render_header(backend_mode=BACKEND_MODE)
stage = infer_stage(active["last_result"], active["chat_history"])
if page == "对话":
    render_progress(stage)
    if stage == 0 and not st.session_state.get("mn_pres_mode"):
        st.caption(
            "完整链路：点示例「材料设计」→ 确认闸门 → 候选与 3D 结构。"
            " 步骤 3–5 需明确材料任务才会点亮。"
        )

try:
    workflow = build_workflow()
except Exception as exc:
    st.error("Agent 初始化失败。请确认模型、Embedding 服务和论文语料均已启动。")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

config = workflow.config(active["thread_id"])

if page == "对话":
    render_chat_page(workflow, config)
elif page == "候选":
    render_candidate_page()
elif page == "溯源":
    render_provenance_page(active["last_result"], active["chat_history"])
elif page == "指标":
    render_metrics_page()
else:
    render_system_page(
        health=health,
        backend_label=BACKEND_LABEL,
        backend_mode=BACKEND_MODE,
        thread_id=active["thread_id"],
        on_reset=start_new_conversation,
    )

if page in {"候选", "溯源"}:
    st.markdown(
        '<div class="mn-foot">研究原型：候选需经 DFT / 实验验证，Mock 与 Real 结果禁止混用。</div>',
        unsafe_allow_html=True,
    )

inject_css_footer(is_dark_mode())
