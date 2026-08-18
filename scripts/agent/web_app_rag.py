from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st
from langgraph.types import Command

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
EXAMPLES = (
    ("🔬 论文事实问答", "单层 CrI3 的居里温度是多少，与块体相比如何？", "检验论文检索、引用与安全回答"),
    ("🧠 方法原理讨论", "MatterGen 如何生成材料？", "检验材料生成知识与论文 RAG"),
    ("🔗 多轮上下文", "那它如何处理周期边界？", "请先询问 MatterGen，再用本例检验指代"),
    ("🛡️ 知识边界", "唐代诗人李白的生平是什么？", "检验证据不足时是否拒绝编造"),
    (
        "⚗️ 材料设计",
        "请设计带隙大于 1.5 eV 且不含稀土元素的材料",
        "检验 MaterialRequest 与人工确认节点",
    ),
    ("💬 需求澄清", "请帮我设计高磁矩密度材料", "检验缺少数值或单位时的主动追问"),
)

st.set_page_config(
    page_title="MagNova · 材料科研智能体",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
<style>
:root{--ink:#10231f;--muted:#5f746e;--green:#0c8068;--line:rgba(19,93,78,.15)}
.stApp{background:radial-gradient(circle at 85% 4%,rgba(94,210,184,.18),transparent 24rem),linear-gradient(180deg,#f8fcfb 0%,#f4f8f7 100%)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#102a25 0%,#153b34 100%)}
[data-testid="stSidebar"] *{color:#edf9f5}
[data-testid="stSidebar"] code{color:#bff0df;background:rgba(255,255,255,.08)}
.mn-hero{padding:2.1rem 2.3rem;border:1px solid var(--line);border-radius:24px;background:linear-gradient(125deg,rgba(255,255,255,.96),rgba(222,246,239,.86));box-shadow:0 18px 55px rgba(15,80,67,.09);margin:.2rem 0 1.2rem}
.mn-eyebrow{color:var(--green);font-size:.78rem;font-weight:800;letter-spacing:.16em;margin-bottom:.65rem}
.mn-hero h1{color:var(--ink);font-size:clamp(2rem,4vw,3.35rem);line-height:1.08;margin:0 0 .75rem;letter-spacing:-.04em}
.mn-hero p{color:var(--muted);font-size:1.03rem;line-height:1.75;max-width:54rem;margin:0}
.mn-badge{display:inline-block;padding:.35rem .7rem;margin:1rem .35rem 0 0;border:1px solid rgba(12,128,104,.2);border-radius:999px;background:rgba(255,255,255,.72);color:#176c5b;font-size:.78rem;font-weight:700}
.mn-title{color:var(--ink);font-weight:800;font-size:1.12rem;margin:1.35rem 0 .3rem}
.mn-note{color:var(--muted);font-size:.9rem;margin-bottom:.8rem}
.mn-flow{padding:.9rem 1rem;border-radius:14px;border:1px solid var(--line);background:rgba(255,255,255,.75);color:#315c53;text-align:center;font-size:.88rem;font-weight:700;min-height:4.2rem}
.mn-service{padding:.5rem .7rem;margin:.35rem 0;border-radius:10px;background:rgba(255,255,255,.07);font-size:.86rem}
.ok{color:#6ee7b7}.bad{color:#fda4af}
div[data-testid="stChatMessage"]{border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.79);box-shadow:0 6px 20px rgba(19,70,60,.04);padding:.3rem .55rem}
div[data-testid="stButton"]>button{border-radius:12px;border-color:rgba(12,128,104,.22);min-height:2.65rem}
div[data-testid="stButton"]>button:hover{border-color:#0c8068;color:#086452}
[data-testid="stChatInput"]{border-color:rgba(12,128,104,.25);box-shadow:0 8px 30px rgba(19,70,60,.08)}
.mn-footer{color:#71827e;font-size:.78rem;line-height:1.65;padding:.8rem 0 1.5rem}
</style>
""",
    unsafe_allow_html=True,
)


def build_backend():
    if BACKEND_MODE == "mock":
        return MockPipelineMagNovaBackend()
    if BACKEND_MODE == "http":
        base_url = os.getenv("MAGNOVA_BACKEND_URL", "").strip()
        if not base_url:
            raise RuntimeError(
                "MAGNOVA_BACKEND_MODE=http 时必须配置 MAGNOVA_BACKEND_URL"
            )
        return HTTPMagNovaBackend(
            base_url, api_key=os.getenv("MAGNOVA_BACKEND_API_KEY") or None
        )
    if BACKEND_MODE in {"cli", "real"}:
        # Real path: subprocess into a separately configured scientific backend.
        return CliMagNovaBackend(
            sacgt_python=os.getenv("SACGT_PYTHON", "python"),
            cli_script=os.getenv("MAGNOVA_SACGT_CLI", "run_magnova_request.py"),
            output_root=os.getenv("MAGNOVA_OUTPUT_ROOT", str(PROJECT_ROOT / "runtime" / "magnova_runs")),
            mode=os.getenv("MAGNOVA_CLI_MODE", "from_artifacts"),
            request_timeout_seconds=float(
                os.getenv("MAGNOVA_BACKEND_TIMEOUT", "86400")
            ),
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
    retriever = GuardedEvidenceRetriever(
        SentenceEvidencePacker(base, embeddings)
    )
    responder = MiniMindRAGResponder(
        os.getenv(
            "MINIMIND_RESEARCH_API_URL",
            "http://127.0.0.1:8512/v1/chat/completions",
        )
    )
    interpreter = MiniMindInterpreter(
        endpoint=os.getenv(
            "MINIMIND_TASK_API_URL",
            "http://127.0.0.1:8506/v1/chat/completions",
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
    result = {}
    for label, url in SERVICES.items():
        try:
            result[label] = requests.get(url, timeout=2).ok
        except requests.RequestException:
            result[label] = False
    return result


def ensure_session() -> None:
    defaults = {
        "thread_id": uuid4().hex,
        "chat_history": [],
        "last_result": None,
        "pending_prompt": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    st.session_state.thread_id = uuid4().hex
    st.session_state.chat_history = []
    st.session_state.last_result = None
    st.session_state.pending_prompt = None


def evidence_caption(item: dict) -> str:
    page = f" · p.{item['page']}" if item.get("page") else ""
    rank = (
        f" · 检索排序 {item['dense_rank']}"
        if item.get("dense_rank") is not None
        else ""
    )
    hint = " · 命中来源提示" if item.get("source_hint_match") else ""
    return (
        f"{item.get('source', 'unknown')}{page} · "
        f"相关度 {item.get('score', 0):.4f}{rank}{hint}"
    )


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
        for part in (
            item.get("name", "目标性质"),
            operator,
            value,
            item.get("unit"),
        )
        if part not in (None, "")
    )


def render_request(request: dict) -> None:
    st.markdown("**模型理解的材料需求**")
    for item in request.get("target_properties", []):
        st.markdown(f"- `{format_constraint(item)}`")
    composition = request.get("composition", {})
    details = []
    if composition.get("required_elements"):
        details.append("必须包含：" + "、".join(composition["required_elements"]))
    if composition.get("excluded_elements"):
        details.append("排除元素：" + "、".join(composition["excluded_elements"]))
    if composition.get("exclude_rare_earth"):
        details.append("不含稀土元素")
    if request.get("material_family"):
        details.append(f"材料类别：{request['material_family']}")
    if details:
        st.caption(" · ".join(details))
    with st.expander("查看完整 MaterialRequest JSON"):
        st.json(request)


def render_outcome(outcome: dict) -> None:
    candidates = outcome.get("candidates", [])
    task_id = outcome.get("task_id")
    cols = st.columns(3)
    cols[0].metric("执行状态", outcome.get("status", "unknown"))
    cols[1].metric("候选数量", len(candidates))
    cols[2].metric("任务编号", task_id[-12:] if task_id else "—")
    if candidates:
        rows = [
            {
                "候选材料": item.get("formula"),
                "综合分数": item.get("score"),
                "预测性质": "；".join(
                    f"{name}={value}"
                    for name, value in item.get(
                        "predicted_properties", {}
                    ).items()
                ),
                "候选 ID": item.get("candidate_id"),
            }
            for item in candidates
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        if BACKEND_MODE == "mock":
            st.info(
                "以上候选来自 Mock Pipeline，仅用于验证接口与交互，"
                "不代表真实科研预测。"
            )
        else:
            st.warning(
                "以上候选来自真实 MagNova/SACGT AI 筛选闭环，"
                "仍不是 DFT 或实验验证结果；不得与 Mock 候选混用。"
            )
    with st.expander("查看完整 Agent / 工具返回"):
        st.json(outcome)


def append_result(result: dict) -> None:
    if result.get("__interrupt__"):
        st.session_state.chat_history.append(
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
        "content": result.get("response") or "工作流已完成。",
        "evidence": result.get("retrieved_evidence", []),
        "retrieval_query": result.get("retrieval_query"),
    }
    if result.get("agent_outcome"):
        entry["outcome"] = result["agent_outcome"]
    st.session_state.chat_history.append(entry)


def append_error(exc: Exception) -> None:
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": (
                "当前请求没有完成。请稍后重试；如果问题持续存在，请检查侧边栏"
                f"的服务状态。技术信息：`{type(exc).__name__}: {str(exc)[:240]}`"
            ),
        }
    )


ensure_session()
health = service_health()

st.markdown(
    """
<div class="mn-hero">
  <div class="mn-eyebrow">MAGNOVA · MATERIAL DISCOVERY COPILOT</div>
  <h1>从科研讨论，到可执行的材料需求</h1>
  <p>基于 MiniMind、论文 RAG 与 LangGraph 的材料科研智能体原型。你可以先讨论理论和论文，待需求明确后，再让 Agent 生成结构化指令并交给 MagNova 流程。</p>
  <span class="mn-badge">MiniMind V2 基线</span>
  <span class="mn-badge">论文证据约束</span>
  <span class="mn-badge">多轮上下文</span>
  <span class="mn-badge">人工确认后调用工具</span>
</div>
""",
    unsafe_allow_html=True,
)

flow_cols = st.columns(5)
for column, text in zip(
    flow_cols,
    (
        "01 讨论科研问题",
        "02 检索论文证据",
        "03 明确材料需求",
        "04 确认结构化参数",
        "05 调用 MagNova",
    ),
):
    column.markdown(
        f'<div class="mn-flow">{text}</div>', unsafe_allow_html=True
    )

with st.sidebar:
    st.markdown("## 🧲 MagNova Agent")
    st.caption("V2 Production Baseline · Demo")
    st.markdown("### 服务状态")
    for label, ok in health.items():
        css_class = "ok" if ok else "bad"
        state = "正常" if ok else "不可用"
        st.markdown(
            f'<div class="mn-service"><span class="{css_class}">●</span> '
            f'{label}<span style="float:right">{state}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="mn-service"><span class="ok">●</span> '
        f'材料流程<span style="float:right">{BACKEND_LABEL}</span></div>',
        unsafe_allow_html=True,
    )
    if not all(health.values()):
        st.error("部分服务不可用，请运行 `scripts/start_demo_stack.sh`。")
    st.markdown("### 本次演示边界")
    if BACKEND_MODE == "mock":
        st.info(
            "当前为 Mock 模式：Agent/RAG 真实运行，但生成-筛选-排序返回演示候选，"
            "不得当作科研结果。切换 Real：`MAGNOVA_BACKEND_MODE=cli`。"
        )
    else:
        st.warning(
            f"当前为 Real 模式（{BACKEND_LABEL}）：候选来自 SACGT MagNova AI 闭环，"
            "仍非 DFT/实验验证；失败时不会回退 Mock。"
        )
    if st.button("＋ 新建会话", use_container_width=True):
        reset_session()
        st.rerun()
    st.caption(f"会话：`{st.session_state.thread_id[:12]}…`")
    with st.expander("系统组件与端口"):
        st.markdown(
            "- `8506`：材料需求解析\n"
            "- `8508`：BGE Embedding\n"
            "- `8512`：MiniMind RAG-SFT V2\n"
            "- `8510`：当前演示 APP"
        )

st.markdown('<div class="mn-title">快速体验</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="mn-note">点击示例即可发送。多轮示例需要在同一会话中按顺序体验。</div>',
    unsafe_allow_html=True,
)
example_cols = st.columns(3)
for index, (title, prompt, description) in enumerate(EXAMPLES):
    with example_cols[index % 3]:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)
            if st.button(
                "发送这个示例",
                key=f"example-{index}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = prompt

try:
    workflow = build_workflow()
except Exception as exc:
    st.error("Agent 初始化失败。请确认模型、Embedding 服务和论文语料均已启动。")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

config = workflow.config(st.session_state.thread_id)
st.markdown('<div class="mn-title">科研对话</div>', unsafe_allow_html=True)
if not st.session_state.chat_history:
    st.caption("从上方选择示例，或者直接在下方输入科研问题与材料需求。")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("retrieval_query"):
            st.caption(f"独立检索查询：`{message['retrieval_query']}`")
        evidence = message.get("evidence", [])
        if evidence:
            with st.expander(f"查看论文证据（{len(evidence)} 条）"):
                for index, item in enumerate(evidence, start=1):
                    st.markdown(f"**{index}. {evidence_caption(item)}**")
                    st.write(item.get("content", "")[:1000])
        if message.get("request"):
            render_request(message["request"])
        if message.get("outcome"):
            render_outcome(message["outcome"])

last_result = st.session_state.last_result
if last_result and last_result.get("__interrupt__"):
    st.warning("需求已经准备好。只有你明确确认后，系统才会调用材料计算工具。")
    approve_col, reject_col = st.columns(2)
    if approve_col.button(
        f"确认调用 {'Mock ' if BACKEND_MODE == 'mock' else '真实 '}MagNova",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("正在执行材料生成、预测、筛选与排序流程…"):
                completed = workflow.graph.invoke(
                    Command(resume=True), config=config
                )
            st.session_state.last_result = completed
            append_result(completed)
        except Exception as exc:
            st.session_state.last_result = None
            append_error(exc)
        st.rerun()
    if reject_col.button("取消本次任务", use_container_width=True):
        try:
            rejected = workflow.graph.invoke(
                Command(resume=False), config=config
            )
            st.session_state.last_result = rejected
            append_result(rejected)
        except Exception as exc:
            st.session_state.last_result = None
            append_error(exc)
        st.rerun()

typed_prompt = st.chat_input("询问论文、讨论理论，或提出材料生成与设计需求…")
prompt = st.session_state.pending_prompt or typed_prompt
if prompt:
    st.session_state.pending_prompt = None
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    try:
        with st.spinner("MiniMind 正在理解需求并检索论文证据…"):
            result = workflow.graph.invoke(
                {
                    "session_id": st.session_state.thread_id,
                    "user_text": prompt,
                    "messages": [],
                },
                config=config,
            )
        st.session_state.last_result = result
        append_result(result)
    except Exception as exc:
        st.session_state.last_result = None
        append_error(exc)
    st.rerun()

st.divider()
st.markdown(
    """
<div class="mn-footer">
研究原型声明：MiniMind 的回答受当前论文知识库范围与 64M 模型容量限制；科研结论应回到原论文核验。
Mock 模式候选仅用于接口演示；Real/CLI 模式候选为 SACGT AI 筛选，仍不是 DFT 或实验依据。两种模式禁止混用结果。
</div>
""",
    unsafe_allow_html=True,
)
