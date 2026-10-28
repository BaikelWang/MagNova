<p align="center">
  <img src="docs/assets/hero.svg" alt="MagNova 磁星智研 · 2026 华为杯国家一等奖、全国 20 强" width="100%" />
</p>

<p align="center"><strong>让材料研究从一句需求，走向有证据、可执行、可追溯的科学工作流。</strong></p>

<p align="center">
  <a href="README.en.md">English</a> · <a href="#产品展示">产品展示</a> · <a href="docs/showcase-case.md">真实案例</a> · <a href="#本地体验">本地体验</a> · <a href="docs/public-release-scope.md">公开范围</a>
</p>

<p align="center">
  <a href="https://github.com/BaikelWang/MagNova/actions/workflows/ci.yml"><img src="https://github.com/BaikelWang/MagNova/actions/workflows/ci.yml/badge.svg" alt="Public code CI" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-324B62?style=flat-square" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/AI4Science-Materials%20Discovery-416C60?style=flat-square" alt="AI for materials discovery" />
</p>

## MagNova｜磁星智研

MagNova 是面向磁性材料探索的科研 Agent，连接自然语言需求、论文证据与材料计算。研究人员可以从讨论问题开始，逐步明确约束、确认任务，再查看候选晶体、代理预测与后续验证路径。

项目获得 **2026「华为杯」中国研究生人工智能创新大赛国家一等奖，进入全国 20 强**。

**MiniMind 理解需求，Harness 控制执行，科学模型计算候选，原始记录约束解释。** 这是 MagNova 的核心协作方式。

> 本页展示已完成的竞赛系统。本仓库保留可运行的公开工程基线与精选展示资料，未整体开放终版部署、私有科学模型和完整数据。截图中的真实任务与本地 Mock 演示分别标注。[公开边界 →](docs/public-release-scope.md)

## 产品展示

### 结构、性质与来源，在同一任务中对应

![历史真实任务的晶体结构、代理预测与 CIF 下载界面](docs/assets/crystal-results.png)

可交互的 3D 晶体、候选标识、原始预测与 CIF 联动展示。图中是历史真实任务的探索候选：代理概率不是已验证物性，Top1 不意味着已经满足全部目标。

### 每条文献证据，都能回到原文

![论文标题、年份、PDF 页码与原文摘录](docs/assets/literature-retrieval.png)

研究讨论可以追溯到论文标题、年份、页码与摘录。归档版本展示证据检索与原文阅读，不将摘录伪装成生成的科研结论。[论文出处及图片说明 →](docs/assets/README.md)

### 从模糊意图，走向明确的执行参数

![带隙严格大于 0.2 eV、稀土排除与用户确认](docs/assets/requirement-confirmation.png)

数值、单位、比较关系与成分约束形成 `MaterialRequest`。缺项先澄清，有冲突先核对；**严格 `>` 不混成 `>=`，确认之前不启动科学计算**。

<details>
<summary><strong>展开完整案例：研究讨论与结果分析</strong></summary>

![研究方向讨论与需求逐步形成](docs/assets/research-discussion.png)

![依据当前候选原始记录分析预测、风险及验证限制](docs/assets/result-analysis.png)

展示串联「讨论 → 文献 → 澄清 → 确认 → 结构 → 分析」，选自同一真实任务，包含修复后分段恢复的记录，不是未剪辑、从不中断的全自动会话。[案例说明 →](docs/showcase-case.md)

</details>

## 系统如何协作

![领域模型、确定性 Harness、科学计算与可追溯输出的职责架构](docs/assets/architecture.svg)

- **MiniMind：需求理解。** 领域适配、需求结构化与多轮澄清协作，不承担晶体生成或物性计算。
- **Harness：可靠执行。** 参数校验、人工确认、状态持久化、账号隔离、任务恢复与结果溯源。结果未知时查询原任务，不盲目重算。
- **科学工具：候选计算。** MatterGen 提出结构，MatterSim 参与弛豫，SACGT 提供代理预测与风险信号，结果绑定当前任务与结构。

文献支持方法选择，代理预测用于预筛，DFT 与实验承担进一步的科学验证。[系统架构 →](docs/system-architecture.md)

## 真实案例与工程证据

一项「探索不含稀土、目标带隙严格大于 0.2 eV 的候选」任务，归档阶段汇总为：**40 个生成结构、35 个预筛通过、34 个弛豫／预测结果，默认返回 10 个候选**。公开候选仍处于探索阶段，不据此宣称发现或验证了高性能新材料。[数量口径与验证状态 →](docs/showcase-case.md)

2026 年 10 月 22 日的完整部署系统归档记录包含 **1182 项工程回归通过、1 项跳过，以及 10 项会话浏览器检查通过**，并复核了登录、检索、历史案例、3D、CIF 下载与退出。这些是历史工程验收，**不是模型准确率，也不是当前公开代码的测试数量**。[验收说明 →](docs/validation.md)

## 本地体验

公开工程基线默认使用规则解析与 Mock 后端，无需 GPU 或模型权重。

```bash
git clone https://github.com/BaikelWang/MagNova.git
cd MagNova
python -m venv .venv
source .venv/bin/activate  # Windows：.venv\Scripts\activate
python -m pip install -e ".[dev,ui]"
pytest
```

先只形成需求契约，不执行科学计算：

```python
from magnova_agent import MagNovaAgent

outcome = MagNovaAgent().prepare(
    "请生成带隙严格大于 0.2 eV 且不含稀土元素的材料"
)
print(outcome.model_dump_json(indent=2))
```

```bash
streamlit run scripts/agent/web_app.py
```

未配置模型服务时选择「规则 Baseline」。该界面是公开基线，**不是截图中的完整竞赛部署**；真实科学链路需要另行接入后端与合法获取的资产。[完整安装与配置 →](docs/getting-started.md)

## 阅读与复现

- [`src/magnova_agent/`](src/magnova_agent/)：公开 Agent、契约、编排与任务管理。
- [`training/minimind/`](training/minimind/)：领域训练、检查与导出工具，不含权重和完整语料。
- [`integrations/mattergen/`](integrations/mattergen/)：MatterGen / MatterSim 集成实验。
- [`reports/`](reports/)：历史聚合评测，口径与应用验收分开。
- [`docs/`](docs/)：架构、案例、验收、公开范围与展示图片。

## 致谢与许可

感谢 [MiniMind](https://github.com/jingyaogong/minimind)、[MatterGen](https://github.com/microsoft/mattergen)、[MatterSim](https://github.com/microsoft/mattersim)、Pydantic 与 Streamlit 的原作者。MagNova 聚焦领域适配、科研流程协作、可靠执行与可追溯交互，上游模型与工具的贡献属于其原作者。

本仓库不分发权重、私有 SACGT 实现、完整论文库、用户数据库或内部评测集。原创代码沿用现有展示许可边界，**未新增开源许可证授权**；第三方内容遵循各自许可。[NOTICE](NOTICE.md) · [图片与文献署名](docs/assets/README.md) · [安全说明](SECURITY.md)
