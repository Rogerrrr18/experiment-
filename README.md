# 动态意图驱动回放评测 — 实验复现

基于 [历史数据回放评测方法调研](https://bytedance.feishu.cn/docx/STwed3Abjo0NdqxYX4zcjcgfnfh) 中「方法二：动态意图驱动的回放评测」的算法实现与实验验证。

## 方法概要

从历史 session 中提取意图序列作为评测核心，将 user query 抽象为意图列表。评测时 SimUser 按意图顺序与 Agent 交互，但每个意图的达成判定是动态的：
- Agent 回复覆盖当前意图 → 自动切换到下一意图
- Agent 未覆盖 → 换方式追问（最多 N 次）
- Agent 完全偏离 → 触发偏离检测

### 评分公式

```
总分 = 意图达成率 × 0.5 + 追问效率 × 0.2 + (1 - 偏离率) × 0.2 + 轮次效率 × 0.1
```

## 5 组实验

| # | 实验 | 类型 | 核心问题 |
|---|------|------|----------|
| 1 | 意图提取质量消融 | 消融 | LLM/模板对意图提取精度的影响 |
| 2 | 回放策略对比 | 对比 | 动态意图 vs 固定回放 vs 自由 SimUser |
| 3 | 追问策略消融 | 消融 | 追问次数 N 对评测公平性的影响 |
| 4 | 评分权重敏感性 | 敏感性 | 权重配置对评分一致性的影响 |
| 5 | 跨数据集泛化 | 泛化 | 意图提取器在未见领域的迁移能力 |

## 推荐数据集

| 优先级 | 数据集 | 语言 | 对话数 | 平均轮次 | 下载方式 |
|--------|--------|------|--------|----------|----------|
| 1 | MultiWOZ 2.2 | EN | 10,438 | 13.7 | `git clone` |
| 2 | ABCD | EN | 10,042 | 17-22 | `git clone` |
| 3 | SGD | EN | 22,825 | 20 | `git clone` |
| 4 | KdConv | ZH | 4,500 | 19 | `git clone` |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据集
bash data/download_datasets.sh

# 3. 运行实验
python src/run_experiment.py --exp exp1  # 单实验
python src/run_experiment.py --all       # 全部实验
```

## 项目结构

```
├── src/
│   ├── intent_extractor.py   # 意图序列提取
│   ├── sim_user.py           # 动态 SimUser
│   ├── judge.py              # Rubric 裁判
│   ├── replay_evaluator.py   # 回放评测编排
│   └── run_experiment.py     # 实验运行入口
├── experiments/              # 实验设计文档
├── data/                     # 数据下载脚本
├── docs/                     # 数据集报告
└── results/                  # 实验结果（运行后生成）
```

## 可复现性保证

- 所有 LLM 调用固定 `temperature=0.3`，`seed=42`
- 意图提取使用固定 prompt 模板
- SimUser 仅在模板内变化，不自由发挥
- 裁判使用结构化 rubric（覆盖/部分覆盖/未覆盖 三档）
- 偏离检测使用 embedding 相似度 + 关键词匹配（确定性判断）
