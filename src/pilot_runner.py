from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Callable

from src.pilot_types import EvalTurn, IntentItem, JudgeDecision, LockedSessionAsset, SessionMetrics


class PilotJudge:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("ZEVAL_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self._client = None

    def _get_client(self):
        api_key = (
            os.environ.get("ZEVAL_INTENT_EXPERIMENT_API_KEY")
            or os.environ.get("ZEVAL_JUDGE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        base_url = os.environ.get("ZEVAL_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            return None
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def decide(self, intent: IntentItem, user_text: str, assistant_text: str) -> JudgeDecision:
        client = self._get_client()
        if client is None:
            return self._heuristic_decide(intent, user_text, assistant_text)
        try:
            return self._llm_decide(intent, user_text, assistant_text, client)
        except Exception:
            return self._heuristic_decide(intent, user_text, assistant_text)

    def _llm_decide(self, intent: IntentItem, user_text: str, assistant_text: str, client) -> JudgeDecision:
        prompt = f"""你是一个严格的对话评审器。只输出 JSON。
标签只能是 SATISFIED / NOT_SATISFIED / DEVIATION。

当前意图：{intent.intent_text}
成功标准：{intent.success_criteria}
用户本轮：{user_text}
助手回复：{assistant_text}

规则：
- SATISFIED：已经正面完成或明确回答意图
- NOT_SATISFIED：相关但未完成/仍需追问
- DEVIATION：明显跑题或误解意图

输出：{{"label":"...","rationale":"一句话","evidence_quote":"从助手回复复制的短句"}}"""
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            seed=42,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return JudgeDecision(
            label=data.get("label", "NOT_SATISFIED"),
            rationale=data.get("rationale", ""),
            evidence_quote=data.get("evidence_quote", assistant_text[:80]),
        )

    def _heuristic_decide(self, intent: IntentItem, user_text: str, assistant_text: str) -> JudgeDecision:
        reply = (assistant_text or "").strip()
        if not reply:
            return JudgeDecision("DEVIATION", "空回复，视为无效。", "")
        bad_markers = ["不知道", "没理解", "再说一遍", "无法", "抱歉"]
        if any(x in reply for x in bad_markers):
            return JudgeDecision("NOT_SATISFIED", "回复承认未理解或无法处理。", self._quote(reply))

        intent_tokens = _tokenize(intent.intent_text + " " + " ".join(intent.example_user_queries))
        reply_tokens = _tokenize(reply)
        overlap = len(intent_tokens & reply_tokens) / max(len(intent_tokens), 1)

        fact_like = any(ch.isdigit() for ch in reply) or any(x in reply for x in ["已", "可以", "成功", "状态", "地址", "电话", "时间", "reference"])
        if overlap >= 0.3 or fact_like:
            return JudgeDecision("SATISFIED", f"回复与意图相关，overlap={overlap:.2f}。", self._quote(reply))
        if overlap >= 0.12:
            return JudgeDecision("NOT_SATISFIED", f"回复部分相关，但未满足成功标准，overlap={overlap:.2f}。", self._quote(reply))
        return JudgeDecision("DEVIATION", f"回复与意图词重合较低，overlap={overlap:.2f}。", self._quote(reply))

    def _quote(self, text: str, limit: int = 80) -> str:
        return text if len(text) <= limit else text[: limit - 1] + "…"


class DynamicSimUser:
    def __init__(self, asset: LockedSessionAsset, alpha: float = 2.0, b_min: int = 3, global_cap: int = 40, seed: int = 42):
        self.asset = asset
        self.alpha = alpha
        self.b_min = b_min
        self.global_cap = global_cap
        self.rng = random.Random(seed)
        self.intent_pos = 0
        self.intent_cycles = 0
        self.global_cycles = 0

    @property
    def done(self) -> bool:
        return self.intent_pos >= len(self.asset.intent_sequence) or self.global_cycles >= self.global_cap

    @property
    def current_intent(self) -> IntentItem | None:
        if self.intent_pos >= len(self.asset.intent_sequence):
            return None
        return self.asset.intent_sequence[self.intent_pos]

    def current_budget(self) -> int:
        intent = self.current_intent
        if intent is None:
            return 0
        n_i = max(intent.turn_span_user_turns, 0)
        if n_i == 0:
            return self.b_min
        return min(self.global_cap, max(self.b_min, math.ceil(self.alpha * n_i)))

    def next_user_text(self) -> str | None:
        intent = self.current_intent
        if intent is None:
            return None
        templates_initial = [
            "我想处理一下：{text}",
            "请帮我看一下：{text}",
            "关于这个需求，想请你直接处理：{text}",
        ]
        templates_followup = [
            "你刚才没有解决我的重点，我真正想要的是：{text}",
            "换个说法，我关心的是：{text}",
            "还是没回答到位，请直接处理：{text}",
        ]
        pool = templates_initial if self.intent_cycles == 0 else templates_followup
        template = self.rng.choice(pool)
        text = intent.example_user_queries[0] if intent.example_user_queries else intent.intent_text
        return template.format(text=text)

    def consume(self, label: str) -> str:
        budget = self.current_budget()
        self.intent_cycles += 1
        self.global_cycles += 1
        if label == "SATISFIED":
            self.intent_pos += 1
            self.intent_cycles = 0
            return "intent_satisfied"
        if self.intent_cycles >= budget:
            self.intent_pos += 1
            self.intent_cycles = 0
            return "intent_failed_budget"
        return "continue"


def default_agent(user_text: str, system_prefix: str = "", context: str = "") -> str:
    fact = ""
    if system_prefix:
        fact = system_prefix.splitlines()[0][:80]
    return f"已收到。基于当前需求和已知事实，我的处理是：{user_text[:60]}。{fact}"


class PilotExperimentRunner:
    def __init__(self, agent_fn: Callable[[str, str, str], str] | None = None, alpha: float = 2.0, b_min: int = 3, global_cap: int = 40):
        self.agent_fn = agent_fn or default_agent
        self.alpha = alpha
        self.b_min = b_min
        self.global_cap = global_cap
        self.judge = PilotJudge()

    def run_dynamic(self, asset: LockedSessionAsset) -> tuple[SessionMetrics, list[EvalTurn]]:
        sim = DynamicSimUser(asset, alpha=self.alpha, b_min=self.b_min, global_cap=self.global_cap)
        turns: list[EvalTurn] = []
        satisfied = 0
        failed = 0
        deviations = 0
        followups = 0
        context = ""

        while not sim.done:
            intent = sim.current_intent
            if intent is None:
                break
            user_text = sim.next_user_text()
            if user_text is None:
                break
            system_prefix = self._build_refill_prefix(asset, intent.intent_index)
            assistant_text = self.agent_fn(user_text, system_prefix, context)
            decision = self.judge.decide(intent, user_text, assistant_text)
            event = sim.consume(decision.label)
            if decision.label == "DEVIATION":
                deviations += 1
            if sim.global_cycles > 1 and sim.intent_cycles > 0:
                followups += 1
            if event == "intent_satisfied":
                satisfied += 1
            elif event == "intent_failed_budget":
                failed += 1

            turns.append(
                EvalTurn(
                    session_id=asset.session_id,
                    intent_index=intent.intent_index,
                    cycle_index=sim.global_cycles,
                    budget=max(sim.current_budget(), self.b_min),
                    user_text=user_text,
                    assistant_text=assistant_text,
                    judge_label=decision.label,
                    rationale=decision.rationale,
                    evidence_quote=decision.evidence_quote,
                    event=event,
                )
            )
            context += f"\nUSER: {user_text}\nASSISTANT: {assistant_text}"

        metrics = self._build_metrics(asset, satisfied, failed, deviations, followups, len(turns))
        return metrics, turns

    def run_baseline(self, asset: LockedSessionAsset, session_data: dict) -> SessionMetrics:
        turns = session_data.get("turns", [])
        satisfied = 0
        deviations = 0
        followups = 0
        total_turns = 0
        for intent in asset.intent_sequence:
            assistant_bundle = self._assistant_text_in_span(turns, intent)
            user_bundle = (intent.example_user_queries[0] if intent.example_user_queries else intent.intent_text)
            decision = self.judge.decide(intent, user_bundle, assistant_bundle)
            total_turns += max(intent.turn_span_user_turns, 1)
            if decision.label == "SATISFIED":
                satisfied += 1
            elif decision.label == "DEVIATION":
                deviations += 1
            if max(intent.turn_span_user_turns, 1) > 1:
                followups += max(intent.turn_span_user_turns - 1, 0)
        failed = max(len(asset.intent_sequence) - satisfied, 0)
        return self._build_metrics(asset, satisfied, failed, deviations, followups, total_turns)

    def _assistant_text_in_span(self, turns: list[dict], intent: IntentItem) -> str:
        if intent.historical_span is None:
            return ""
        chunks = []
        for t in turns:
            turn_num = int(t.get("turn_num", 0))
            if intent.historical_span.start_turn_index <= turn_num <= intent.historical_span.end_turn_index and t.get("role") == "agent":
                chunks.append(t.get("text", ""))
        return " ".join(chunks).strip()

    def _build_refill_prefix(self, asset: LockedSessionAsset, intent_index: int) -> str:
        selected = []
        for item in asset.refillables:
            if item.bind_intent_index is None or item.bind_intent_index == intent_index:
                selected.append(item.injection_text or f"【会话已知事实】{item.refill_reference}")
        return "\n".join(selected)

    def _build_metrics(self, asset: LockedSessionAsset, satisfied: int, failed: int, deviations: int, followups: int, total_turns: int) -> SessionMetrics:
        total_intents = max(len(asset.intent_sequence), 1)
        hist_turns = sum(max(i.turn_span_user_turns, 1) for i in asset.intent_sequence)
        completion = satisfied / total_intents
        followup_per_intent = followups / total_intents
        deviation_rate = deviations / max(total_turns, 1)
        turn_efficiency = hist_turns / max(total_turns, 1)
        turn_efficiency = min(turn_efficiency, 1.5)
        composite = (completion * 0.5) + ((1 / (1 + followup_per_intent)) * 0.2) + ((1 - deviation_rate) * 0.2) + (min(turn_efficiency, 1.0) * 0.1)
        return SessionMetrics(
            session_id=asset.session_id,
            total_intents=total_intents,
            satisfied_intents=satisfied,
            failed_intents=failed,
            total_turns=total_turns,
            deviation_turns=deviations,
            followup_turns=followups,
            intent_completion_rate=round(completion, 4),
            followup_per_intent=round(followup_per_intent, 4),
            deviation_rate=round(deviation_rate, 4),
            turn_efficiency=round(turn_efficiency, 4),
            composite_score=round(composite, 4),
        )


def summarize_results(baseline_metrics: list[SessionMetrics], dynamic_metrics: list[SessionMetrics]) -> dict:
    def avg(items: list[SessionMetrics], field: str) -> float:
        vals = [getattr(x, field) for x in items]
        return round(sum(vals) / max(len(vals), 1), 4)

    fields = ["intent_completion_rate", "followup_per_intent", "deviation_rate", "turn_efficiency", "composite_score"]
    baseline = {f: avg(baseline_metrics, f) for f in fields}
    dynamic = {f: avg(dynamic_metrics, f) for f in fields}
    delta = {f: round(dynamic[f] - baseline[f], 4) for f in fields}
    return {
        "baseline": baseline,
        "dynamic": dynamic,
        "delta": delta,
        "sessions": len(dynamic_metrics),
    }


def generate_radar_svg(summary: dict, out_path: Path):
    # normalize followup/deviation by converting lower-is-better to higher-is-better
    labels = [
        ("Intent Completion", "intent_completion_rate", False),
        ("Low Followup", "followup_per_intent", True),
        ("Low Deviation", "deviation_rate", True),
        ("Turn Efficiency", "turn_efficiency", False),
        ("Composite", "composite_score", False),
    ]

    def score(side: str, key: str, invert: bool):
        value = summary[side][key]
        if invert:
            return max(0.0, min(1.0, 1 / (1 + value)))
        return max(0.0, min(1.0, value))

    cx, cy, radius = 220, 220, 140
    circles = []
    for r in [0.25, 0.5, 0.75, 1.0]:
        circles.append(f'<circle cx="{cx}" cy="{cy}" r="{radius*r}" fill="none" stroke="#dbe3ee" stroke-dasharray="3 3"/>')

    axes = []
    baseline_points = []
    dynamic_points = []
    label_nodes = []
    for idx, (label, key, invert) in enumerate(labels):
        angle = -math.pi / 2 + idx * (2 * math.pi / len(labels))
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        axes.append(f'<line x1="{cx}" y1="{cy}" x2="{x}" y2="{y}" stroke="#cbd5e1"/>')
        label_nodes.append(f'<text x="{cx + (radius+22)*math.cos(angle):.1f}" y="{cy + (radius+22)*math.sin(angle):.1f}" font-size="12" text-anchor="middle" fill="#334155">{label}</text>')
        for side, collector in [("baseline", baseline_points), ("dynamic", dynamic_points)]:
            s = score(side, key, invert)
            px = cx + radius * s * math.cos(angle)
            py = cy + radius * s * math.sin(angle)
            collector.append(f"{px:.1f},{py:.1f}")

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="520" height="520" viewBox="0 0 520 520">
  <rect width="520" height="520" fill="#ffffff" rx="18"/>
  {''.join(circles)}
  {''.join(axes)}
  <polygon points="{' '.join(baseline_points)}" fill="rgba(59,130,246,0.18)" stroke="#2563eb" stroke-width="2"/>
  <polygon points="{' '.join(dynamic_points)}" fill="rgba(239,68,68,0.18)" stroke="#ef4444" stroke-width="2"/>
  {''.join(label_nodes)}
  <text x="28" y="28" font-size="18" fill="#111827">B vs D Radar</text>
  <rect x="28" y="42" width="12" height="12" fill="rgba(59,130,246,0.18)" stroke="#2563eb"/>
  <text x="48" y="52" font-size="12" fill="#334155">B = Original session baseline</text>
  <rect x="28" y="62" width="12" height="12" fill="rgba(239,68,68,0.18)" stroke="#ef4444"/>
  <text x="48" y="72" font-size="12" fill="#334155">D = Dynamic replay</text>
</svg>'''
    out_path.write_text(svg, encoding="utf-8")


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tokenize(text: str) -> set[str]:
    text = (text or "").lower()
    tokens = []
    current = []
    for ch in text:
        if ch.isalnum() or '\u4e00' <= ch <= '\u9fff':
            current.append(ch)
        else:
            if current:
                tokens.append(''.join(current))
                current = []
    if current:
        tokens.append(''.join(current))
    return {t for t in tokens if len(t) > 1}
