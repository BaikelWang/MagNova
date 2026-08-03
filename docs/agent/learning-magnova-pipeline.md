# Agent 工程：生成—预测—筛选工作流

## 1. 本轮的目标

之前的 Mock 后端只是在任务完成时直接构造候选材料，看不出 MagNova
内部究竟做了什么。本轮把它拆成三个独立、可替换、可测试的工具：

```text
MaterialGenerator
  → PropertyPredictor
  → CandidateScreener
```

LLM 不负责执行材料计算，也不负责比较浮点数。MiniMind 只把用户语言
转换成 `MaterialRequest`，后续全部使用结构化数据和确定性程序。

## 2. 材料生成

接口：

```python
generate(request: MaterialRequest) -> list[GeneratedMaterial]
```

生成结果包含：

- `structure_id`；
- 化学式；
- 生成分数；
- 生成器元数据。

当前 `MockMaterialGenerator` 会执行 required/excluded elements 检查，避免
明显违反成分约束。它只是接口模拟器，不是扩散模型或晶体生成模型。

## 3. 性质预测

接口：

```python
predict(
    materials: list[GeneratedMaterial],
    constraints: list[PropertyConstraint],
) -> list[PredictedMaterial]
```

每个结果包含预测值和不确定性。真实系统中可以分别接入：

- 图神经网络；
- 等变神经网络；
- DFT 计算服务；
- MagNova 自己的性质预测模型。

预测器并不淘汰候选，它只负责产生数值和置信信息。

## 4. 条件筛选

接口：

```python
screen(
    materials: list[PredictedMaterial],
    request: MaterialRequest,
) -> list[CandidateMaterial]
```

筛选器实现 `gt/gte/lt/lte/eq/range`。这是确定性的数值逻辑，不应该让
LLM 判断。例如：

```python
value > threshold
lower_bound <= value <= upper_bound
```

只有满足所有硬约束的材料才能留下，再综合生成分数和预测不确定性排序。

## 5. 为什么要分成三个 Tool

如果把所有逻辑放进一个大函数：

- 无法判断失败发生在哪一阶段；
- 无法单独更换生成模型或预测模型；
- 无法分别评测召回率、预测误差和筛选准确率；
- 难以缓存中间结果；
- 很难进行失败重试和资源调度。

拆分后，每个节点都有明确的输入、输出和责任。

## 6. 工作流状态

任务现在同时有两类状态：

```text
任务状态：queued / running / succeeded / failed
工作流阶段：queued / generation / prediction / screening / completed
```

任务状态回答“任务是否结束”，工作流阶段回答“现在执行到哪里”。二者
不能混为一谈。

每次阶段完成都会追加 `WorkflowEvent`：

```json
{
  "stage": "prediction",
  "message": "性质预测阶段完成",
  "artifact_count": 6
}
```

事件、进度和候选都会持久化到 SQLite，因此刷新页面后仍能恢复工作流
轨迹。

## 7. SQLite 增量迁移

旧数据库没有 `stage`、`progress` 和 `workflow_trace_json`。本轮没有
删除数据库重新建表，而是检查 `PRAGMA table_info` 后执行 additive
migration：

```sql
ALTER TABLE agent_tasks ADD COLUMN stage ...
ALTER TABLE agent_tasks ADD COLUMN progress ...
ALTER TABLE agent_tasks ADD COLUMN workflow_trace_json ...
```

这是向后兼容的数据迁移。生产环境通常使用 Alembic 等迁移工具，但原理
相同：数据库 Schema 也需要版本化演进。

## 8. 面试表达

> 我把 MagNova 后端拆成材料生成、性质预测和确定性筛选三个节点。每个
> 节点使用 Pydantic Schema 定义输入输出，可以独立替换模型和测试。
> MiniMind 只负责需求理解，浮点比较和硬约束筛选由程序执行。工作流同时
> 记录任务状态、当前阶段、进度和 artifact 数量，并通过 SQLite 增量
> 迁移实现刷新恢复与可观测性。

## 9. 当前边界

- 结构、预测值和评分仍是 Mock；
- 工作流推进依赖页面轮询；
- 尚未接入真实晶体文件、模型权重或 DFT 服务；
- 筛选器尚未处理单位换算、容差策略和 Pareto 多目标优化。

下一步可以从“结果解释与证据”或“真实 MagNova API 契约”中选择一个
方向继续。
