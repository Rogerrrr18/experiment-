# 实验一：意图提取质量消融

## 目标
对比 LLM 提取 vs 模板提取的意图序列质量，评估不同提取方式对后续回放评测的影响。

## 假设
- H1: LLM 提取的意图序列更准确（与人工标注一致度 > 80%）
- H2: LLM 提取导致更高的评测分数（因意图粒度更合理）
- H3: 模板提取虽然粗糙，但因其确定性而更适合基线对比

## 变量
| 变量 | 取值 | 说明 |
|------|------|------|
| 提取模式 | `template`, `llm` | 模板规则 vs GPT-4/DeepSeek |
| LLM 模型 | `gpt-4`, `deepseek-v3` | 仅在 mode=llm 时 |
| Prompt 模板 | `v1`, `v2`, `v3` | 不同 prompt 设计 |

## 指标
| 指标 | 计算方式 | 目标 |
|------|----------|------|
| 意图数量一致性 | `abs(extracted_intents - human_annotated_intents) / human_annotated_intents` | < 20% |
| 意图语义覆盖率 | 人工标注意图中被 LLM 提取覆盖的比例 | > 80% |
| 评测分数差异 | 同一 session 在两种提取方式下的 total_score 差值 | — |
| 提取时间 | 单 session 提取耗时 | < 5s |

## 执行步骤

```bash
# 1. 准备人工标注（50 个 session）
# 可以从 MultiWOZ 的 dialogue acts 中提取 ground-truth intents

# 2. 运行实验
python src/run_experiment.py --exp exp1 --dataset multiwoz --sessions 50

# 3. 对比结果
python -c "
import json
with open('results/exp1_multiwoz_*.json') as f:
    data = json.load(f)
for r in data['results']:
    print(f'{r[\"mode\"]}: avg_intents={r[\"avg_intents_extracted\"]:.1f}, avg_score={r[\"avg_score\"]:.3f}')
"
```

## 预期结果
- LLM 模式提取意图数 ≈ 人工标注数（误差 < 15%）
- 模板模式提取意图数 = 用户发言轮次（偏高）
- LLM 模式评测分数更稳定（方差更小）
