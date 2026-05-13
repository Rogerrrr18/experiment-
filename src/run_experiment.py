"""
实验运行入口 — 支持 5 组实验的独立运行和批量执行
用法:
    python src/run_experiment.py --exp exp1          # 单个实验
    python src/run_experiment.py --all               # 全部实验
    python src/run_experiment.py --exp exp1 --dataset multiwoz --sessions 50
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# 确保 src 目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intent_extractor import IntentExtractor
from src.sim_user import SimUser
from src.judge import RubricJudge
from src.replay_evaluator import ReplayEvaluator, EvalConfig, MockAgent


# ── 数据集加载器 ─────────────────────────────────────────────

def load_multiwoz(data_dir: str, max_sessions: int = None) -> list[dict]:
    """加载 MultiWOZ 数据集"""
    sessions = []
    data_path = Path(data_dir) / "data" / "multiwoz"

    # MultiWOZ 2.2 格式
    for split in ["train", "dev", "test"]:
        json_files = list(data_path.glob(f"*{split}*.json")) + list(data_path.glob(f"*{split}*.jsonl"))
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f) if jf.suffix == ".json" else [json.loads(l) for l in f if l.strip()]
            if isinstance(data, list):
                for item in data:
                    sessions.append(_normalize_multiwoz(item))
            elif isinstance(data, dict):
                for k, v in data.items():
                    sessions.append(_normalize_multiwoz(v, sid=k))

    if max_sessions:
        sessions = sessions[:max_sessions]
    return sessions


def _normalize_multiwoz(item: dict, sid: str = None) -> dict:
    """将 MultiWOZ 格式标准化为统一格式"""
    turns = []
    dialogue = item.get("turns", item.get("dialogue", []))
    for t in dialogue:
        if isinstance(t, dict):
            turns.append(t)
    return {
        "id": sid or item.get("dialogue_id", item.get("id", "unknown")),
        "turns": turns,
    }


def load_abcd(data_dir: str, max_sessions: int = None) -> list[dict]:
    """加载 ABCD 数据集"""
    sessions = []
    data_path = Path(data_dir) / "data" / "abcd"

    for jf in data_path.glob("*.json"):
        with open(jf) as f:
            data = json.load(f)
        if isinstance(data, list):
            sessions.extend(data)
        elif isinstance(data, dict):
            sessions.extend(data.values())

    if max_sessions:
        sessions = sessions[:max_sessions]
    return sessions


def load_dataset(name: str, data_dir: str, max_sessions: int = 50) -> list[dict]:
    loaders = {
        "multiwoz": load_multiwoz,
        "abcd": load_abcd,
    }
    loader = loaders.get(name.lower())
    if loader is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    return loader(data_dir, max_sessions)


# ── 实验函数 ──────────────────────────────────────────────────

def run_experiment(
    exp_name: str,
    dataset: str = "multiwoz",
    data_dir: str = "./data",
    sessions: int = 50,
    output_dir: str = "./results",
) -> dict:
    """运行单个实验"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Dataset: {dataset}, Sessions: {sessions}")
    print(f"{'='*60}\n")

    # 加载数据
    session_data = load_dataset(dataset, data_dir, max_sessions=sessions)
    print(f"Loaded {len(session_data)} sessions")

    if not session_data:
        # 无真实数据时使用模拟数据
        print("No real data found, using synthetic sessions")
        session_data = _generate_synthetic_sessions(sessions)

    # 准备组件
    extractor = IntentExtractor(mode="template")  # 默认用模板保证可复现
    judge = RubricJudge(num_samples=1)             # 1 次采样加速

    results = []

    if exp_name == "exp1":
        results = _run_exp1_intent_extraction(session_data, extractor, judge)
    elif exp_name == "exp2":
        results = _run_exp2_replay_comparison(session_data, extractor, judge)
    elif exp_name == "exp3":
        results = _run_exp3_reask_strategy(session_data, extractor, judge)
    elif exp_name == "exp4":
        results = _run_exp4_scoring_sensitivity(session_data, extractor, judge)
    elif exp_name == "exp5":
        results = _run_exp5_cross_dataset(session_data, extractor, judge)
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = Path(output_dir) / f"{exp_name}_{dataset}_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {result_path}")

    return results


def _run_exp1_intent_extraction(sessions, extractor, judge):
    """实验1: 意图提取质量消融"""
    print("Exp1: 对比 LLM vs Template 意图提取")
    config = EvalConfig(max_reasks=3)
    agent = MockAgent(mode="helpful")

    modes = ["template", "llm"]  # 如有 API key 会尝试 LLM
    results = []

    for mode in modes:
        try:
            ext = IntentExtractor(mode=mode)
            evaluator = ReplayEvaluator(
                agent_fn=agent.reply,
                extractor=ext,
                judge=judge,
                config=config,
            )
            mode_results = []
            for s in sessions[:min(30, len(sessions))]:
                result = evaluator.evaluate(s)
                mode_results.append({
                    "session_id": result.session_id,
                    "total_intents": result.total_intents,
                    "achieved": result.achieved_intents,
                    "score": result.total_score,
                })

            avg_intents = sum(r["total_intents"] for r in mode_results) / len(mode_results)
            avg_score = sum(r["score"] for r in mode_results) / len(mode_results)

            results.append({
                "mode": mode,
                "sessions": len(mode_results),
                "avg_intents_extracted": avg_intents,
                "avg_score": avg_score,
                "details": mode_results[:5],
            })
            print(f"  {mode}: avg_intents={avg_intents:.1f}, avg_score={avg_score:.3f}")
        except Exception as e:
            print(f"  {mode}: FAILED — {e}")

    return {"experiment": "exp1_intent_extraction", "results": results}


def _run_exp2_replay_comparison(sessions, extractor, judge):
    """实验2: 回放策略对比"""
    print("Exp2: 对比 Dynamic Intent vs Fixed Replay vs Free SimUser")
    config = EvalConfig(max_reasks=3)
    results = []

    # 只测 helpful agent
    agent = MockAgent(mode="helpful")
    evaluator = ReplayEvaluator(
        agent_fn=agent.reply,
        extractor=extractor,
        judge=judge,
        config=config,
    )

    eval_results = []
    for s in sessions[:min(30, len(sessions))]:
        result = evaluator.evaluate(s)
        eval_results.append(result)

    avg_score = sum(r.total_score for r in eval_results) / len(eval_results)
    avg_achievement = sum(r.intent_achievement_rate for r in eval_results) / len(eval_results)
    avg_reask = sum(r.reask_efficiency for r in eval_results) / len(eval_results)
    avg_deviation = sum(r.deviation_rate for r in eval_results) / len(eval_results)

    results.append({
        "method": "dynamic_intent",
        "sessions": len(eval_results),
        "avg_total_score": avg_score,
        "avg_intent_achievement": avg_achievement,
        "avg_reask_efficiency": avg_reask,
        "avg_deviation_rate": avg_deviation,
    })
    print(f"  dynamic_intent: score={avg_score:.3f}, achievement={avg_achievement:.3f}")

    return {"experiment": "exp2_replay_comparison", "results": results}


def _run_exp3_reask_strategy(sessions, extractor, judge):
    """实验3: 追问策略消融"""
    print("Exp3: 追问次数 N = {1, 3, 5, unlimited}")
    agent = MockAgent(mode="helpful")
    results = []

    for max_reasks in [1, 3, 5, 10]:
        config = EvalConfig(max_reasks=max_reasks)
        evaluator = ReplayEvaluator(
            agent_fn=agent.reply,
            extractor=extractor,
            judge=judge,
            config=config,
        )

        eval_results = []
        for s in sessions[:min(20, len(sessions))]:
            result = evaluator.evaluate(s)
            eval_results.append(result)

        avg_score = sum(r.total_score for r in eval_results) / len(eval_results)
        avg_reask = sum(r.reask_efficiency for r in eval_results) / len(eval_results)
        avg_turns = sum(r.total_turns for r in eval_results) / len(eval_results)

        results.append({
            "max_reasks": max_reasks,
            "avg_score": avg_score,
            "avg_reask_efficiency": avg_reask,
            "avg_turns": avg_turns,
        })
        print(f"  N={max_reasks}: score={avg_score:.3f}, reask_eff={avg_reask:.3f}, turns={avg_turns:.1f}")

    return {"experiment": "exp3_reask_strategy", "results": results}


def _run_exp4_scoring_sensitivity(sessions, extractor, judge):
    """实验4: 评分权重敏感性"""
    print("Exp4: 不同权重配置")
    agent = MockAgent(mode="helpful")
    results = []

    weight_configs = [
        ("default", {"intent_achievement": 0.5, "reask_efficiency": 0.2, "deviation": 0.2, "turn_efficiency": 0.1}),
        ("equal", {"intent_achievement": 0.25, "reask_efficiency": 0.25, "deviation": 0.25, "turn_efficiency": 0.25}),
        ("intent_heavy", {"intent_achievement": 0.7, "reask_efficiency": 0.1, "deviation": 0.15, "turn_efficiency": 0.05}),
        ("efficiency_heavy", {"intent_achievement": 0.3, "reask_efficiency": 0.3, "deviation": 0.3, "turn_efficiency": 0.1}),
    ]

    for wname, weights in weight_configs:
        config = EvalConfig(max_reasks=3, weights=weights)
        evaluator = ReplayEvaluator(
            agent_fn=agent.reply,
            extractor=extractor,
            judge=judge,
            config=config,
        )

        eval_results = []
        for s in sessions[:min(20, len(sessions))]:
            result = evaluator.evaluate(s)
            eval_results.append(result)

        avg_score = sum(r.total_score for r in eval_results) / len(eval_results)

        results.append({
            "weight_config": wname,
            "weights": weights,
            "avg_score": avg_score,
        })
        print(f"  {wname}: avg_score={avg_score:.3f}")

    return {"experiment": "exp4_scoring_sensitivity", "results": results}


def _run_exp5_cross_dataset(sessions, extractor, judge):
    """实验5: 跨数据集泛化"""
    print("Exp5: 跨数据集泛化（当前使用同一数据集的不同子集模拟）")
    config = EvalConfig(max_reasks=3)
    agent = MockAgent(mode="helpful")

    # 分割数据模拟 train/test 来自不同分布
    split = len(sessions) // 2
    train_sessions = sessions[:split]
    test_sessions = sessions[split:]

    evaluator_train = ReplayEvaluator(
        agent_fn=agent.reply,
        extractor=extractor,
        judge=judge,
        config=config,
    )

    results = []
    # 在 train 上运行
    train_results = []
    for s in train_sessions[:min(10, len(train_sessions))]:
        result = evaluator_train.evaluate(s)
        train_results.append(result.total_score)
    results.append({"set": "train", "avg_score": sum(train_results) / len(train_results)})

    # 在 test 上运行
    test_results = []
    for s in test_sessions[:min(10, len(test_sessions))]:
        result = evaluator_train.evaluate(s)
        test_results.append(result.total_score)
    results.append({"set": "test", "avg_score": sum(test_results) / len(test_results)})

    print(f"  Train avg_score: {results[0]['avg_score']:.3f}")
    print(f"  Test avg_score: {results[1]['avg_score']:.3f}")

    return {"experiment": "exp5_cross_dataset", "results": results}


def _generate_synthetic_sessions(n: int = 10) -> list[dict]:
    """生成模拟对话数据"""
    templates = [
        [
            {"role": "user", "text": "你好，我想咨询一下我的订单状态"},
            {"role": "agent", "text": "好的，请提供您的订单号"},
            {"role": "user", "text": "订单号是 ORD-2024-8891"},
            {"role": "agent", "text": "查到了，您的订单目前正在配送中，预计明天到达"},
            {"role": "user", "text": "能帮我改一下收货地址吗"},
            {"role": "agent", "text": "可以的，请告诉我新的收货地址"},
            {"role": "user", "text": "北京市朝阳区望京SOHO T1 15层"},
            {"role": "agent", "text": "已为您更新收货地址，订单将在新地址配送"},
            {"role": "user", "text": "大概什么时候能到"},
            {"role": "agent", "text": "预计后天下午 3 点前到达"},
        ],
        [
            {"role": "user", "text": "我想退一个商品"},
            {"role": "agent", "text": "请问是什么商品呢？方便提供订单号吗"},
            {"role": "user", "text": "上周买的耳机，订单号 ORD-2024-7732"},
            {"role": "agent", "text": "好的，请问退货原因是什么？"},
            {"role": "user", "text": "音质不太满意"},
            {"role": "agent", "text": "了解了，您的商品还在7天无理由退货期内，可以退货。我帮您生成退货单号"},
            {"role": "user", "text": "运费谁出"},
            {"role": "agent", "text": "7天无理由退货需要您承担寄回运费，但我们可以补贴10元运费"},
            {"role": "user", "text": "好的，那帮我办吧"},
            {"role": "agent", "text": "已生成退货单号 RTN-2024-5512，请将商品原包装寄回"},
        ],
    ]

    sessions = []
    for i in range(n):
        template = templates[i % len(templates)]
        sessions.append({"id": f"synth_{i+1}", "turns": template})
    return sessions


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run replay evaluation experiments")
    parser.add_argument("--exp", type=str, help="Experiment name: exp1-exp5")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--dataset", type=str, default="multiwoz", help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--sessions", type=int, default=20, help="Number of sessions")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    args = parser.parse_args()

    experiments = ["exp1", "exp2", "exp3", "exp4", "exp5"] if args.all else [args.exp]

    for exp in experiments:
        run_experiment(
            exp_name=exp,
            dataset=args.dataset,
            data_dir=args.data_dir,
            sessions=args.sessions,
            output_dir=args.output_dir,
        )
