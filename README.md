# 基于意图指针的动态评测效果实验（Pilot）

这是一个按 **新方案 v1.0** 重构后的仓库：

> **单臂动态回放 + 原始 session 基线 B 对比 + locked schema assets + refillables 注入**

2026-05-15 起，已吸收外部参考项 `eval-test-v1` 的关键设计：把历史 session 从「固定对白样本」进一步升级为 **可执行的意图状态机 + 全 session 可回填事实包**。

当前仓库不再以旧的 `exp1~exp5` 研究矩阵为主线，而是先聚焦一个更可落地的工程化 pilot：

- 从 CSV / MultiWOZ session 读入历史对话
- 一次抽取出 `intent_sequence + refillables`
- `intent_sequence` 优先抽象为稳定意图指针，例如 `restaurant:find_restaurant` / `hotel:book_hotel` / `taxi:find_taxi`，不再机械按每轮 user 拆分
- `refillables` 从完整 session 抽取可复用事实，不再只截取本轮 answer
- 允许人工修订并锁版为 `*_locked.json`
- 只消费 locked assets 跑动态评测
- 产出：
  - 原始 session 基线 **B**
  - 动态回放结果 **D**
  - `run_log.jsonl`
  - `metrics_summary.json`
  - `radar.svg`

---

## 1. 当前实验主线

### 目标
验证以下闭环能否稳定工作：

```text
CSV / Session
→ schema 化抽取
→ locked asset
→ 动态回放（意图指针 + 预算 + refillables）
→ B / D 指标对比
→ 雷达图与日志审计
```

### 首期冻结决策
- **单臂动态回放**，不再把 fixed/free 作为首要主线
- 对照使用 **原始 session 基线 B**，不是多 replay 机制大乱斗
- Judge 为三分类：
  - `SATISFIED`
  - `NOT_SATISFIED`
  - `DEVIATION`
- 允许先抽取、再人工修订、再锁版
- 优先支持本地可复现；外部 LLM 调用为可选增强

---

## 2. 目录结构

```text
├── extract_session.schema.json          # 锁定 Schema
├── src/
│   ├── pilot_types.py                   # Pilot 核心数据结构
│   ├── session_asset_extractor.py       # session → draft/locked asset
│   ├── pilot_runner.py                  # B/D 评测、日志、雷达输出
│   ├── run_experiment.py                # 新版主入口
│   ├── intent_extractor.py              # legacy
│   ├── replay_evaluator.py              # legacy
│   ├── sim_user.py                      # legacy
│   └── judge.py                         # legacy
├── experiments/
│   ├── pilot_dynamic_intent.md          # 新主方案说明
│   ├── exp1_intent_extraction.md        # legacy 归档说明
│   ├── exp2_replay_comparison.md        # legacy 归档说明
│   ├── exp3_reask_strategy.md           # legacy 归档说明
│   ├── exp4_scoring_sensitivity.md      # legacy 归档说明
│   └── exp5_cross_dataset.md            # legacy 归档说明
└── results/
```

---

## 3. 输入数据

当前默认支持：

### A. turn-level CSV
建议最小列集：
- `session_id` / `dialogue_id`
- `turn_index` / `turn_num`
- `role` / `speaker`
- `content` / `utterance`
- `timestamp`（可选）

如果输入包含 MultiWOZ 风格辅助列：

- `services`
- `active_intents`
- `slot_values`

抽取器会把它们作为 `CSV_AUX` 辅助信息，用于生成更稳定的 `intent_sequence` 与 `refillables`。这些辅助列只用于抽取，不作为原始 baseline B 的直接判分输入。

### B. MultiWOZ JSON/JSONL
如果 `data/multiwoz/` 下存在原始 MultiWOZ 文件，也会自动读取。

---

## 4. 运行方式

### 4.0 打开可视化 Demo 页面

最常用的打开方式：

```bash
cd /Users/rogeryang/.openclaw/workspace/experiment-
python3 src/demo_server.py
```

然后在浏览器打开：

```text
http://127.0.0.1:8765
```

页面里可以：

- 选择 session
- 设置预算参数 `alpha / b_min / global_cap`
- 选择困难模式
- 对比弱基线 / 中等基线 / 真实模型
- 查看意图指针、refillables、动态评测轨迹、Judge 诊断和报告链接

如果要接真实模型，在项目根目录 `.env` 中配置：

```bash
ZEVAL_JUDGE_BASE_URL="你的 OpenAI-compatible base url"
ZEVAL_JUDGE_MODEL="评审/抽取模型名"
ZEVAL_TEST_AGENT_MODEL="真实待测 Agent 模型名"
ZEVAL_INTENT_EXPERIMENT_API_KEY="你的 API Key"
```

停止服务：回到终端按 `Ctrl+C`。

### 4.1 完整跑通 pilot

```bash
python src/run_experiment.py \
  --dataset multiwoz \
  --data-dir . \
  --sessions 10 \
  --output-dir results/pilot
```

### 4.2 只做 schema 抽取与锁版草稿

```bash
python src/run_experiment.py \
  --dataset multiwoz \
  --data-dir . \
  --sessions 10 \
  --step extract \
  --output-dir results/pilot
```

---

## 5. 输出内容

运行后会生成：

- `assets/{session_id}.draft.json`
- `assets/{session_id}_locked.json`
- `draft_assets_*.json`
- `run_log_*.jsonl`
- `metrics_summary_*.json`
- `radar_*.svg`

其中：

### `draft.json`
首轮自动抽取结果，允许人工修订。

### `*_locked.json`
锁版后的正式资产。后续动态评测只消费这个文件。

### `metrics_summary`
包含：
- per-session baseline B metrics
- per-session dynamic D metrics
- B vs D 的 summary 与 delta

### `radar.svg`
将以下维度映射到同一雷达图：
- Intent Completion
- Low Followup
- Low Deviation
- Turn Efficiency
- Composite

---

## 6. 环境变量（可选）

如果你要接 SiliconFlow / OpenAI 兼容网关，可设置：

```bash
export ZEVAL_JUDGE_BASE_URL="https://api.siliconflow.cn/v1"
export ZEVAL_JUDGE_MODEL="Qwen/Qwen3.5-27B"
export ZEVAL_JUDGE_ENABLE_THINKING="false"
export ZEVAL_INTENT_EXPERIMENT_API_KEY="..."
```

回退顺序：
- `ZEVAL_INTENT_EXPERIMENT_API_KEY`
- `ZEVAL_JUDGE_API_KEY`
- `OPENAI_API_KEY`

未配置时，仓库会走 **本地 heuristic fallback**，仍可跑通 pipeline。

---

## 7. 当前方法论边界

这版仓库刻意偏向 **pilot / 工程试跑**，不是最终研究版：

- 先验证 pipeline、schema、locked asset、日志审计、B/D 对比是否成立
- 暂不把跨数据集、复杂消融、显著性检验作为主线
- 旧 `exp1~exp5` 保留为 **legacy 归档**，避免混淆当前方向

如果后续要继续升级，推荐顺序是：

1. 人工修订一批 locked assets，重点检查意图指针是否过粗/过细、依赖关系是否正确
2. 补强全 session refillables：确认号、状态、电话、实体名称、日期时间人数、工具结果等都应沉淀为内部事实
3. 提升 Judge 与 extractor 的人工一致性
4. 接真实待测 Agent 通道
5. 再重建更严谨的 method-eval / product-eval

---

## 8. 2026-05-15 优化点

本轮优化主要迁移 `eval-test-v1` 的优势：

- **抽象意图指针**：MultiWOZ fallback 会根据 `active_intents`、用户话术和上下文，把连续追问合并为稳定指针，如 `restaurant:book_restaurant`，而不是每个 user turn 一个意图。
- **全 session 回填项**：从完整对话和 `slot_values` 中抽取电话、确认号、实体名、预订条件、车辆信息等事实，生成 `injection_text`。
- **依赖关系**：`book_*` 自动依赖同 domain 的 `find_*`；`taxi` 会依赖之前的 restaurant / hotel 相关意图。
- **内部事实注入**：动态评测时会把相关事实包成「用户不可见」的 system prefix，并要求被测 Agent 自然作答，不暴露抽取/评测/JSON/锁版等机制。
- **首轮原文锚定**：每个意图的第一轮 user 消息严格使用历史 session 原文，不调用 SimUser 改写；SimUser 只用于后续未满足/偏航时的追问、纠偏或困难模式施压。
- **验证门禁**：`validate_asset` 会检查空意图、重复意图序号、非抽象意图指针、重复回填 key，以及回填文案是否泄漏评测词。

冒烟验证命令：

```bash
python3 -m compileall src
python3 src/run_experiment.py --dataset multiwoz --data-dir . --sessions 10 --output-dir results/pilot_optimized_full10
```

最近一次本地验证输出：

- `results/pilot_optimized_full10/metrics_summary_20260515_093827.json`
- `results/pilot_optimized_full10/run_log_20260515_093827.jsonl`
- `results/pilot_optimized_full10/report_20260515_093827.html`
- `results/pilot_optimized_full10/radar_20260515_093827.svg`

---

## 9. 一句话定位

这个仓库现在的定位不是：

> “证明动态意图回放在研究上全面优于所有方法”

而是：

> **“先把意图指针动态评测这条工程链路做成一个可信、可审计、可解释的 pilot。”**
