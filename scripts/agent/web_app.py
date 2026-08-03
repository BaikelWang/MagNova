from __future__ import annotations

import os
import re
from typing import Any
from uuid import uuid4

import streamlit as st

from magnova_agent import (
    HTTPMagNovaBackend,
    MagNovaAgent,
    MagNovaTaskManager,
    MagNovaToolAdapter,
    MiniMindInterpreter,
    MockPipelineMagNovaBackend,
    MultiTurnMagNovaAgent,
    PollingPolicy,
    RuleBasedInterpreter,
    SQLiteTaskStore,
)
from magnova_agent.models import AgentStatus, JobStatus, WorkflowStage


st.set_page_config(page_title="MagNova Agent MVP", page_icon="🧲", layout="wide")


def get_session_id() -> str:
    raw = st.query_params.get("session")
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, str) and re.fullmatch(r"[0-9a-f]{16}", raw):
        return raw
    session_id = uuid4().hex[:16]
    st.query_params["session"] = session_id
    return session_id


@st.cache_resource
def get_task_runtime() -> tuple[MagNovaToolAdapter, MagNovaTaskManager, str]:
    base_url = os.getenv("MAGNOVA_BASE_URL", "").strip()
    if base_url:
        backend = HTTPMagNovaBackend(
            base_url,
            api_key=os.getenv("MAGNOVA_API_KEY"),
            request_timeout_seconds=float(
                os.getenv("MAGNOVA_REQUEST_TIMEOUT", "30")
            ),
        )
        backend_label = f"HTTP MagNova：{base_url}"
    else:
        backend = MockPipelineMagNovaBackend()
        backend_label = "三阶段 Mock Pipeline（生成 → 预测 → 筛选）"

    adapter = MagNovaToolAdapter(
        backend,
        policy=PollingPolicy(
            timeout_seconds=float(os.getenv("MAGNOVA_JOB_TIMEOUT", "300")),
            initial_interval_seconds=float(
                os.getenv("MAGNOVA_POLL_INTERVAL", "1")
            ),
            max_interval_seconds=float(
                os.getenv("MAGNOVA_MAX_POLL_INTERVAL", "10")
            ),
            backoff_multiplier=1.5,
            max_transient_errors=3,
        ),
    )
    store = SQLiteTaskStore(
        os.getenv(
            "MAGNOVA_TASK_DB",
            "runtime/agent_tasks.sqlite3",
        )
    )
    return adapter, MagNovaTaskManager(adapter, store), backend_label


@st.cache_resource
def get_agent(mode: str) -> MagNovaAgent:
    adapter, _, _ = get_task_runtime()
    if mode == "规则 Baseline":
        return MagNovaAgent(
            interpreter=RuleBasedInterpreter(),
            tool=adapter,
        )

    fallback = RuleBasedInterpreter() if mode == "MiniMind（安全降级）" else None
    interpreter = MiniMindInterpreter(
        endpoint=os.getenv(
            "MINIMIND_API_URL",
            "http://127.0.0.1:8504/v1/chat/completions",
        ),
        fallback=fallback,
    )
    return MagNovaAgent(interpreter=interpreter, tool=adapter)


def render_candidates(candidates: list[dict[str, Any]]) -> None:
    st.dataframe(
        [
            {
                "候选 ID": item["candidate_id"],
                "化学式": item["formula"],
                "综合评分": item["score"],
                **item["predicted_properties"],
            }
            for item in candidates
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "未配置 MAGNOVA_BASE_URL 时，结构、性质和评分来自 Mock Pipeline，"
        "仅用于验证工作流，不是科学预测。"
    )


def render_outcome(payload: dict[str, Any]) -> None:
    status = payload["status"]
    if status == "needs_clarification":
        st.warning(payload["message"])
        for question in payload["clarification_questions"]:
            st.markdown(f"- {question}")
    elif status == "submitted":
        st.info(payload["message"])
        if payload.get("task_id"):
            st.caption(
                f"任务 ID：`{payload['task_id']}` · "
                f"提交状态：`{payload.get('backend_status')}`"
            )
    elif status == "ready":
        st.info(payload["message"])
    else:
        st.error(payload["message"])

    warnings = payload.get("request", {}).get("parser_warnings", [])
    if any(
        str(item).startswith("minimind_fallback:")
        for item in warnings
    ):
        st.warning(
            "本次 MiniMind 输出未通过 Schema 校验或服务不可用，"
            "已自动切换到规则 Baseline；这不是 MiniMind 的成功结果。"
        )

    left, right = st.columns(2)
    with left:
        with st.expander("结构化 MaterialRequest", expanded=True):
            st.json(payload["request"])
    with right:
        with st.expander("工具调用", expanded=True):
            if payload["tool_call"] is None:
                st.info("未调用工具：请求仍需澄清或处理失败。")
            else:
                st.json(payload["tool_call"])


STAGE_LABELS = {
    WorkflowStage.QUEUED: "等待调度",
    WorkflowStage.GENERATE: "材料生成",
    WorkflowStage.PREDICT: "性质预测",
    WorkflowStage.SCREEN: "条件筛选",
    WorkflowStage.COMPLETED: "工作流完成",
}


def render_workflow_trace(record) -> None:
    st.markdown("##### 工作流轨迹")
    if not record.workflow_trace:
        st.caption("尚未产生阶段事件。")
        return
    st.dataframe(
        [
            {
                "阶段": STAGE_LABELS[event.stage],
                "事件": event.message,
                "产物数量": event.artifact_count,
            }
            for event in record.workflow_trace
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_task(record) -> None:
    st.markdown(f"#### MagNova 任务 `{record.task_id}`")
    st.caption(
        f"会话：`{record.session_id}` · 状态：`{record.status.value}` · "
        f"当前阶段：`{record.stage.value}`（{STAGE_LABELS[record.stage]}）"
    )
    st.progress(
        int(record.progress * 100),
        text=f"{STAGE_LABELS[record.stage]} · {record.progress:.0%}",
    )

    if record.status == JobStatus.QUEUED:
        st.info("任务已经写入 SQLite，等待工作流开始。")
    elif record.status == JobStatus.RUNNING:
        messages = {
            WorkflowStage.GENERATE: "正在生成候选材料结构。",
            WorkflowStage.PREDICT: "正在预测候选材料的目标性质和不确定性。",
            WorkflowStage.SCREEN: "正在执行确定性条件筛选与排序。",
        }
        st.info(messages.get(record.stage, "MagNova 工作流正在运行。"))
    elif record.status == JobStatus.SUCCEEDED:
        st.success(f"三阶段工作流完成，保留 {len(record.candidates)} 个候选材料。")
        if record.candidates:
            render_candidates(
                [
                    item.model_dump(mode="json")
                    for item in record.candidates
                ]
            )
    else:
        st.error(f"任务失败：{record.error_message or '未知错误'}")

    render_workflow_trace(record)
    with st.expander("持久化任务记录"):
        st.json(record.model_dump(mode="json"))


session_id = get_session_id()
adapter, task_manager, backend_label = get_task_runtime()

st.title("🧲 MagNova Agent · MiniMind")

with st.sidebar:
    st.header("需求解释器")
    interpreter_mode = st.radio(
        "选择自然语言解析方式",
        ("MiniMind（安全降级）", "MiniMind（严格模式）", "规则 Baseline"),
        help="严格模式可真实观察模型失败；安全模式失败时回退规则解析，并明确显示警告。",
    )
    st.caption("模型权重不包含在本仓库中；通过 MINIMIND_API_URL 连接本地推理服务。")
    st.caption(f"会话 ID：`{session_id}`")
    st.divider()
    st.header("测试建议")
    st.markdown(
        """
**完整请求**

`请生成磁矩密度大于 0.2 μB/Å³ 且不含稀土元素的材料`

**多属性**

`生成带隙大于 1.2 eV 且居里温度高于 500 K 的材料`

**缺少单位**

`设计磁矩密度大于 0.2 的材料`
        """
    )
    st.divider()
    st.caption(
        "流程：需求理解 → Schema → task_id → SQLite → "
        "材料生成 → 性质预测 → 条件筛选 → 候选排序"
    )
    if st.button("新建会话", use_container_width=True):
        st.session_state.messages = []
        if "conversation_agent" in st.session_state:
            st.session_state.conversation_agent.reset(session_id)
        st.query_params["session"] = uuid4().hex[:16]
        st.rerun()

st.info(
    f"当前模式：{interpreter_mode}。MiniMind 负责需求理解；"
    "数值筛选由确定性代码完成，不交给 LLM 判断。"
)
st.caption(f"当前工具后端：{backend_label}")

if "messages" not in st.session_state:
    st.session_state.messages = []
if st.session_state.get("conversation_agent_mode") != interpreter_mode:
    st.session_state.conversation_agent = MultiTurnMagNovaAgent(
        get_agent(interpreter_mode)
    )
    st.session_state.conversation_agent_mode = interpreter_mode
    st.session_state.messages = []

conversation_agent = st.session_state.conversation_agent
pending = conversation_agent.state(session_id).pending_request
if pending is not None:
    missing = "、".join(pending.readiness_issues())
    st.warning(
        "当前会话正在等待补充信息。你可以只回答缺失内容，"
        f"无需重述完整需求。待补充：{missing}"
    )

latest = task_manager.latest_for_session(session_id)
task_active = latest is not None and not latest.terminal


@st.fragment(run_every=1.0)
def live_task_panel() -> None:
    record = task_manager.latest_for_session(session_id)
    if record is None:
        return
    was_terminal = record.terminal
    if not record.terminal:
        record = task_manager.refresh(record.task_id, session_id)
    render_task(record)
    if record.terminal:
        conversation_agent.complete_task(record.task_id, session_id)
    if not was_terminal and record.terminal:
        st.rerun()


live_task_panel()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_outcome(message["outcome"])

prompt = st.chat_input(
    "描述你需要的材料性质、数值、单位和约束……",
    disabled=task_active,
)
if task_active:
    st.caption("当前工作流仍在执行。完成后输入框会自动恢复。")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        prepared = conversation_agent.prepare(
            prompt,
            session_id=session_id,
        )
        outcome = prepared.model_dump(mode="json")
        if prepared.status == AgentStatus.READY:
            record = task_manager.submit(prepared.request, session_id)
            conversation_agent.track_task(record.task_id, session_id)
            outcome["status"] = AgentStatus.SUBMITTED.value
            outcome["task_id"] = record.task_id
            outcome["backend_status"] = record.status.value
            outcome["message"] = (
                f"三阶段 MagNova 工作流已提交，task_id={record.task_id}。"
                "生成、预测和筛选会在任务区依次更新。"
            )
    except Exception as exc:
        outcome = {
            "status": "failed",
            "message": f"需求处理失败：{exc}",
            "request": {"original_text": prompt},
            "clarification_questions": [],
            "tool_call": None,
            "task_id": None,
            "backend_status": None,
            "candidates": [],
        }

    st.session_state.messages.append(
        {"role": "assistant", "outcome": outcome}
    )
    st.rerun()
