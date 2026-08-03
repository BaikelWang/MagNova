# Agent 工程：非阻塞任务与 SQLite 持久化

## 1. 为什么上一版还不是真正的异步

上一版后端采用 task_id 和轮询协议，但 Agent 调用的是：

```python
result = adapter.submit(request)
```

这个兼容方法内部会一直轮询，直到任务完成才返回。因此“后端是异步的”，
但“用户请求仍然是阻塞的”。真实 MagNova 如果需要十分钟，Streamlit
请求也会等待十分钟。

本轮把流程拆成：

```text
准备需求 → 提交一次 → 立即返回 task_id
                        ↓
                 SQLite 保存任务
                        ↓
              页面局部定时查询状态
```

## 2. prepare 和 execute 的区别

`MagNovaAgent.prepare()` 只完成：

1. 自然语言解析；
2. Pydantic Schema 校验；
3. 缺失信息澄清；
4. 构造计划中的 ToolCall。

它不会调用 MagNova。

```python
prepared = agent.prepare(user_text)
```

准备完成时状态是 `ready`。任务管理器随后独立提交：

```python
record = task_manager.submit(prepared.request, session_id)
```

这样 LLM 推理、需求校验和长耗时计算被清楚地分开。

## 3. SQLite 保存了什么

`agent_tasks` 表保存：

- `task_id`：后端任务唯一标识；
- `session_id`：任务属于哪个用户会话；
- `request_json`：完整 MaterialRequest；
- `status`：queued、running、succeeded 或 failed；
- `candidates_json`：候选材料；
- `error_message`：后端或轮询错误；
- `transient_error_count`：连续瞬时错误次数；
- `created_at`、`updated_at`：超时和审计依据。

Pydantic 对象写入前序列化成 JSON，读取后再次执行 Schema 校验。数据库
中的数据也被视为边界输入，而不是无条件信任。

## 4. 为什么使用 WAL

SQLite 初始化时开启：

```sql
PRAGMA journal_mode=WAL;
```

WAL（Write-Ahead Logging）允许读操作和写操作更好地并行，适合当前
单机、轻并发的 MVP。它不是无限扩展方案；多实例部署后应迁移到
PostgreSQL 或 Redis + PostgreSQL。

## 5. 会话隔离

读取任务时同时检查：

```text
task_id + session_id
```

即使另一个会话知道 task_id，也不能通过当前任务管理接口读取该记录。
当前页面用随机 16 位十六进制会话 ID，并把它保存在 URL 查询参数中，
因此刷新页面后仍能找到同一会话的最近任务。

这只是 MVP 隔离，不等于生产级鉴权。真实系统仍需登录用户 ID、权限校验
和服务端会话。

## 6. 页面为什么使用局部刷新

Streamlit 1.60 支持：

```python
@st.fragment(run_every=1.0)
def live_task_panel():
    ...
```

只有任务状态区域每秒重新执行，不需要整页反复运行 MiniMind，也不会
重复提交任务。状态流转为：

```text
queued → running → succeeded
```

任务进入终态后停止后端查询，并重新启用聊天输入框。

## 7. 失败与恢复语义

- 瞬时错误：保存错误次数，下一次局部刷新继续查询；
- 超过重试预算：标记为 failed；
- 永久错误：立即标记为 failed；
- 超过任务 deadline：标记为 failed；
- 页面刷新：从 SQLite 读取最近任务；
- Agent 进程重启：任务记录仍存在。

需要注意：当前 InMemory 模拟后端本身不具备跨进程恢复能力。真实 HTTP
MagNova 后端保存自己的任务状态，因此 Agent 重启后可继续用 task_id
查询。模拟后端只用于验证接口和页面状态机。

## 8. 面试表达

> 我把 LLM 的需求理解阶段和耗时工具执行阶段拆开。Agent 的 prepare
> 阶段只产出经过 Schema 校验的 MaterialRequest，任务管理器提交后立即
> 返回 task_id，并把请求、状态和结果持久化到 SQLite。前端使用局部定时
> 刷新查询状态，瞬时错误、重试预算和超时也都持久化。这样页面刷新不会
> 丢任务，也不会因为长耗时材料计算阻塞一次 HTTP 请求。

## 9. 本轮验证

- 自动测试：47 项全部通过；
- 同请求、不同会话生成不同 task_id；
- 错误会话无法读取他人任务；
- 新建 SQLite Store 实例可以恢复任务；
- 真实 MiniMind 两轮解析：
  - 第一轮：`needs_clarification`；
  - 第二轮：`ready`；
  - 提交耗时约数毫秒，立即得到 `queued`；
  - 重开数据库仍为 `queued`；
  - 后续状态为 `running → succeeded`；
  - 返回 3 个模拟候选。
