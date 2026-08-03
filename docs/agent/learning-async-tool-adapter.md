# Agent 工程：异步 Tool Adapter

## 1. 本轮解决了什么问题

旧接口是：

```python
candidates = tool.submit(request)
```

它假设 MagNova 能立刻返回候选。真实材料生成、性质预测和筛选可能需要
几秒到几小时，因此真实接口更接近：

```text
提交请求 → 获得 task_id → 查询状态 → 获得结果
```

本轮将工具边界拆成：

```text
MagNovaAgent
  → MagNovaToolAdapter
      → MagNovaBackend.submit_job
      → MagNovaBackend.get_job
```

Agent 只关心统一的 `ToolResult`，不知道后面连接的是内存模拟器、HTTP
服务还是未来的 Python SDK。这就是 Adapter（适配器）的作用。

## 2. 为什么需要 task_id

`task_id` 是一次后端任务的唯一标识。它让客户端可以：

- 在后续请求中查询同一个任务；
- 页面刷新后恢复任务；
- 记录日志并定位失败；
- 避免把不同用户的结果混在一起；
- 将 Agent 服务和计算服务解耦。

## 3. 状态机

当前后端任务只允许四种状态：

```text
queued → running → succeeded
                   ↘ failed
```

- `queued`：任务已接受，正在排队；
- `running`：后端正在计算；
- `succeeded`：终态，结果可以读取；
- `failed`：终态，必须返回错误原因。

Schema 使用枚举限制状态值，可以尽早发现后端拼写错误或协议漂移。

## 4. 轮询、退避与超时

轮询是客户端定期调用 `get_job(task_id)`。如果固定高频查询，会给后端
造成压力，因此使用退避策略：

```text
1s → 1.5s → 2.25s → … → 最大 10s
```

同时必须设置总超时。超时不一定代表后端任务失败，只代表本次等待已经
超过客户端可接受的截止时间。生产系统通常会保存 task_id，稍后继续查，
而不是丢弃任务。

## 5. 瞬时错误和永久错误

需要区分：

- 瞬时错误：网络闪断、HTTP 429、HTTP 5xx，可以有限重试；
- 永久错误：HTTP 4xx、响应不符合 Schema、未知 task_id，不应盲目重试；
- 任务失败：后端明确返回 `failed`；
- 客户端超时：等待超过配置的 deadline。

`POST /v1/tasks` 当前不会自动重试，因为重试提交可能生成两个重复任务。
真实后端如果需要安全重试，应支持 `Idempotency-Key`。

## 6. 为什么暂时还保留同步 submit

底层已经是异步任务协议，但当前 Streamlit Agent 仍调用：

```python
result = adapter.submit(request)
```

这个兼容方法内部完成“提交 + 等待”。它让我们在不重写整个 UI 的情况
下先验证协议。下一步可以把 UI 改成真正的非阻塞形式：

```text
用户提交 → 页面立即显示 task_id
后台计算 → 页面周期刷新状态
完成后 → 展示候选
```

## 7. 面试表达

可以这样回答：

> 材料生成是长耗时任务，所以我没有把工具设计成一次 HTTP 请求立即返回。
> 我定义了 submit/poll 异步任务协议，用 Pydantic 校验任务快照，通过
> Adapter 统一处理状态机、指数退避、瞬时错误重试、超时和失败。Agent
> 只依赖抽象接口，因此模拟后端和真实 HTTP 后端可以无缝替换。为了避免
> 重复创建任务，提交请求不会在缺少幂等键时盲目重试。

## 8. 当前验证结果

- 完整自动测试：40 项全部通过；
- 真实 MiniMind 两轮测试：
  - 第一轮缺少单位，返回 `needs_clarification`；
  - 第二轮补充单位，合并到结构化请求；
  - 生成 `mock-async-*` task_id；
  - 状态到达 `succeeded`；
  - 返回 3 个模拟候选。
