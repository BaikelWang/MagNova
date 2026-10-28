# 本地体验与复现范围

本仓库提供精选公开工程基线与竞赛项目展示。本地示例不需要 GPU、私有模型或外部付费 API，也不会复现完整竞赛系统的科学计算结果。

## 安装

```bash
git clone https://github.com/BaikelWang/MagNova.git
cd MagNova
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,ui]"
pytest
```

Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1` 激活环境。项目支持 Python 3.10 及以上，CI 配置覆盖 Python 3.10 与 3.12。

## 只解析，不执行计算

```python
from magnova_agent import MagNovaAgent

outcome = MagNovaAgent().prepare(
    "请生成带隙严格大于 0.2 eV 且不含稀土元素的材料"
)
print(outcome.model_dump_json(indent=2))
```

默认使用规则解释器。`prepare()` 只形成需求契约及就绪结果，不提交任务、不加载 LLM，也不生成晶体。这是接口与校验演示，不是 MiniMind 模型性能评测。

## 运行 Mock 界面

```bash
streamlit run scripts/agent/web_app.py
```

没有模型服务时选择「规则 Baseline」。内置后端是用于验证工程流程的 Mock，不把模拟结构、性质与排序用于科学结论。任务数据库默认位于 `runtime/agent_tasks.sqlite3`，不应提交到 Git，可用 `MAGNOVA_TASK_DB` 指定本地路径。

## 可选：连接你自己的服务

MiniMind 解释器使用 OpenAI 兼容接口。单独启动模型服务后，再配置本地地址：

```bash
export MINIMIND_API_URL=http://127.0.0.1:8504/v1/chat/completions
```

科学后端通过异步任务协议接入：

```text
POST /v1/tasks           → task_id + queued
GET  /v1/tasks/{task_id}  → queued | running | succeeded | failed
```

配置项见 [`.env.example`](../.env.example)，适配器见 [`tooling.py`](../src/magnova_agent/tooling.py)。令牌只能通过环境变量注入，不要把真实凭据写入示例。

## 哪些内容不随安装提供

- 截图中的完整竞赛部署界面与私有服务。
- SACGT 私有实现、模型权重、完整论文库及服务数据库。
- 完整训练语料、开发与内部测试集。

MatterGen / MatterSim 集成实验需要另行安装上游项目并合法获取模型与数据，参见[集成说明](../integrations/mattergen/fm_semiconductor_screening/README.md)。

公开基线的 Schema 曾包含比实际科学后端更广的字段。能表达某个字段，不意味着真实生成、预测和验证链路支持该性质；例如，不据此承诺居里温度目标可以直接计算。当前仓库测试与完整部署系统的历史验收按[不同口径](validation.md)报告。
