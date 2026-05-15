from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.pilot_types import HistoricalSpan, IntentItem, LockedSessionAsset, RefillableItem

SCHEMA_VERSION = "2026-02-intent-refill-v1"
PROMPT_VERSION = "pilot-v1"

EXTRACTION_PROMPT = """你是一个会话整理器。请把给定 transcript 整理为固定 JSON Schema，输出字段必须严格匹配：
- schema_version
- schema_lock_revision
- prompt_version
- session_id
- intent_sequence
- refillables

要求：
1. 只提取真实 transcript 中能支持的信息
2. intent_sequence 必须抽象成稳定的「意图指针」，优先使用 domain:action 形式，例如 restaurant:find_restaurant / hotel:book_hotel / taxi:book_taxi；不要把每一轮 user 话术机械拆成一个意图
3. turn_span_user_turns 是该意图在历史中对应的用户发起轮数；同一意图连续追问、补充条件应合并
4. refillables 必须从整个 session 抽取，而不是只看某一轮 answer；优先抽取可复用的已闭环业务事实：实体名称、电话、地址、价格/区域、预订确认号、车牌/车型、出发地/目的地、日期时间人数、状态结论等
5. refillables 的 injection_text 应写成自然的「内部已知事实」，可直接注入给被测 Agent；不要只填原始 answer，也不要暴露评测机制
6. 可利用 transcript 后的行级 metadata（services / active_intents / slot_values）辅助抽取，但不要虚构 transcript 中没有支持的事实
7. 只输出 JSON
"""


class SessionAssetExtractor:
    def __init__(self, model: str | None = None, temperature: float = 0.1):
        self.model = model or os.environ.get("ZEVAL_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.temperature = temperature
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
            try:
                from openai import OpenAI
            except ModuleNotFoundError:
                return None
            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def extract(self, session: dict) -> LockedSessionAsset:
        client = self._get_client()
        if client is None:
            return self._heuristic_extract(session)
        try:
            return self._llm_extract(session, client)
        except Exception:
            return self._heuristic_extract(session)

    def _llm_extract(self, session: dict, client) -> LockedSessionAsset:
        transcript = self._format_session(session)
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            seed=42,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": transcript},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return self._from_dict(data, session)

    def _heuristic_extract(self, session: dict) -> LockedSessionAsset:
        if self._has_multiwoz_metadata(session):
            return self._heuristic_extract_multiwoz(session)

        turns = session.get("turns", [])
        user_turns = [t for t in turns if t.get("role") == "user"]
        intent_sequence: list[IntentItem] = []
        refillables: list[RefillableItem] = []

        for idx, turn in enumerate(user_turns, start=1):
            turn_num = int(turn.get("turn_num", idx))
            next_user_turn_num = self._next_user_turn_num(user_turns, idx)
            hist_end = next_user_turn_num - 1 if next_user_turn_num is not None else int(turns[-1].get("turn_num", turn_num))
            example = turn.get("text", "").strip()
            success = self._infer_success_criteria(turn)
            span_user_turns = self._count_user_turns_in_span(turns, turn_num, hist_end)
            intent_sequence.append(
                IntentItem(
                    intent_index=idx,
                    intent_text=self._summarize_text(example),
                    turn_span_user_turns=max(span_user_turns, 1),
                    example_user_queries=[example] if example else [],
                    success_criteria=success,
                    depends_on=[idx - 1] if idx > 1 else [],
                    historical_span=HistoricalSpan(start_turn_index=turn_num, end_turn_index=hist_end),
                )
            )
            assistant_reference = self._find_last_assistant_text(turns, turn_num, hist_end)
            if assistant_reference:
                refillables.append(
                    RefillableItem(
                        refill_index=len(refillables) + 1,
                        trigger_condition=f"进入意图 {idx}: {self._summarize_text(example, 24)}",
                        refill_reference=self._summarize_text(assistant_reference, 80),
                        key=f"intent_{idx}_known_fact",
                        source_turn_index=hist_end,
                        confidence="medium",
                        injection_text=(
                            f"【会话已知事实】关于意图 {idx}，历史对话已出现如下事实：{self._summarize_text(assistant_reference, 120)}。"
                            "请在不重复调用外部系统的前提下，优先基于该事实作答。"
                        ),
                        bind_intent_index=idx,
                    )
                )

        return LockedSessionAsset(
            schema_version=SCHEMA_VERSION,
            schema_lock_revision=1,
            prompt_version=PROMPT_VERSION,
            session_id=session.get("id", "unknown"),
            intent_sequence=intent_sequence,
            refillables=refillables,
            source_file=session.get("source_file", ""),
        )

    def _heuristic_extract_multiwoz(self, session: dict) -> LockedSessionAsset:
        """MultiWOZ fallback: build abstract intent pointers and full-session refillables.

        This is intentionally stronger than the generic fallback: it uses active_intents /
        slot_values as auxiliary hints to group user turns into domain:action pointers, then
        extracts reusable closed facts from the whole session.
        """
        turns = session.get("turns", [])
        user_turns = [t for t in turns if t.get("role") == "user"]
        groups: list[dict[str, Any]] = []

        for turn in user_turns:
            previous_pointer = groups[-1]["pointer"] if groups else ""
            pointer = self._dominant_multiwoz_intent(turn, previous_pointer=previous_pointer)
            if not pointer:
                pointer = "conversation:close" if self._is_close_turn(turn.get("text", "")) else f"user:intent_{len(groups) + 1}"
            if groups and groups[-1]["pointer"] == pointer:
                groups[-1]["user_turns"].append(turn)
                groups[-1]["end"] = int(turn.get("turn_num", groups[-1]["end"]))
            else:
                groups.append({
                    "pointer": pointer,
                    "user_turns": [turn],
                    "start": int(turn.get("turn_num", 0)),
                    "end": int(turn.get("turn_num", 0)),
                })

        # Historical span ends at the turn before the next intent starts; this captures the
        # assistant result that closes the current intent.
        for idx, group in enumerate(groups):
            if idx + 1 < len(groups):
                group["span_end"] = max(group["end"], int(groups[idx + 1]["start"]) - 1)
            else:
                group["span_end"] = int(turns[-1].get("turn_num", group["end"])) if turns else group["end"]

        intent_sequence: list[IntentItem] = []
        pointer_to_index: dict[str, int] = {}
        for idx, group in enumerate(groups, start=1):
            pointer = group["pointer"]
            pointer_to_index[pointer] = idx
            examples = [t.get("text", "").strip() for t in group["user_turns"] if t.get("text", "").strip()]
            depends_on = self._infer_dependencies(pointer, pointer_to_index)
            intent_sequence.append(
                IntentItem(
                    intent_index=idx,
                    intent_text=pointer,
                    turn_span_user_turns=max(len(group["user_turns"]), 1),
                    example_user_queries=examples[:3] or [pointer],
                    success_criteria=self._multiwoz_success_criteria(pointer, examples),
                    depends_on=depends_on,
                    historical_span=HistoricalSpan(start_turn_index=group["start"], end_turn_index=group["span_end"]),
                )
            )

        refillables = self._extract_full_session_refillables(turns, intent_sequence)
        return LockedSessionAsset(
            schema_version=SCHEMA_VERSION,
            schema_lock_revision=1,
            prompt_version=PROMPT_VERSION,
            session_id=session.get("id", "unknown"),
            intent_sequence=intent_sequence,
            refillables=refillables,
            source_file=session.get("source_file", ""),
        )

    def _from_dict(self, data: dict, session: dict) -> LockedSessionAsset:
        intents = []
        for item in data.get("intent_sequence", []):
            span = item.get("historical_span") or {}
            intents.append(
                IntentItem(
                    intent_index=int(item.get("intent_index", len(intents) + 1)),
                    intent_text=item.get("intent_text") or item.get("description") or "",
                    turn_span_user_turns=int(item.get("turn_span_user_turns", 1) or 1),
                    example_user_queries=item.get("example_user_queries", []),
                    success_criteria=item.get("success_criteria", ""),
                    depends_on=[int(x) for x in item.get("depends_on", [])],
                    historical_span=HistoricalSpan(
                        start_turn_index=int(span.get("start_turn_index", 0)),
                        end_turn_index=int(span.get("end_turn_index", 0)),
                    ) if span else None,
                    n_i_conflict=bool(item.get("n_i_conflict", False)),
                    n_i_heuristic=bool(item.get("n_i_heuristic", False)),
                )
            )
        refillables = []
        for item in data.get("refillables", []):
            refillables.append(
                RefillableItem(
                    refill_index=int(item.get("refill_index", len(refillables) + 1)),
                    trigger_condition=item.get("trigger_condition", ""),
                    refill_reference=item.get("refill_reference", ""),
                    key=item.get("key", ""),
                    source_turn_index=item.get("source_turn_index"),
                    confidence=item.get("confidence", "medium"),
                    injection_text=item.get("injection_text", ""),
                    bind_intent_index=item.get("bind_intent_index"),
                )
            )
        return LockedSessionAsset(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            schema_lock_revision=int(data.get("schema_lock_revision", 1)),
            prompt_version=data.get("prompt_version", PROMPT_VERSION),
            session_id=data.get("session_id", session.get("id", "unknown")),
            intent_sequence=intents,
            refillables=refillables,
            source_file=session.get("source_file", ""),
        )

    @staticmethod
    def validate_asset(asset: LockedSessionAsset) -> list[str]:
        errors = []
        if not asset.intent_sequence:
            errors.append("intent_sequence is empty")
        seen = set()
        for item in asset.intent_sequence:
            if item.intent_index in seen:
                errors.append(f"duplicate intent_index: {item.intent_index}")
            seen.add(item.intent_index)
            if not item.intent_text.strip():
                errors.append(f"intent {item.intent_index} missing intent_text")
            if item.turn_span_user_turns < 1:
                errors.append(f"intent {item.intent_index} turn_span_user_turns < 1")
            if ":" not in item.intent_text and item.intent_text != "conversation:close":
                errors.append(f"intent {item.intent_index} is not an abstract intent pointer: {item.intent_text}")
        refill_keys = set()
        for item in asset.refillables:
            if item.key and item.key in refill_keys:
                errors.append(f"duplicate refillable key: {item.key}")
            if item.key:
                refill_keys.add(item.key)
            if item.injection_text and any(x in item.injection_text for x in ["提示词", "评测机制", "JSON Schema"]):
                errors.append(f"refillable {item.refill_index} injection_text leaks eval wording")
        return errors

    @staticmethod
    def dump_asset(asset: LockedSessionAsset, path: Path):
        path.write_text(json.dumps(asset.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load_asset(path: Path) -> LockedSessionAsset:
        data = json.loads(path.read_text(encoding="utf-8"))
        extractor = SessionAssetExtractor()
        return extractor._from_dict(data, {"id": data.get("session_id", "unknown")})

    @staticmethod
    def ensure_locked_asset(asset: LockedSessionAsset, draft_path: Path, locked_path: Path):
        SessionAssetExtractor.dump_asset(asset, draft_path)
        if not locked_path.exists():
            SessionAssetExtractor.dump_asset(asset, locked_path)

    def _format_session(self, session: dict) -> str:
        parts = [f"session_id={session.get('id', 'unknown')}"]
        for turn in session.get("turns", []):
            parts.append(f"[{turn.get('turn_num', '')}] {turn.get('role', 'user').upper()}: {turn.get('text', '')}")
        if self._has_multiwoz_metadata(session):
            parts.append("\n--- CSV_AUX_MULTIWOZ_V1（仅作意图/槽位抽取辅助，非对白正文）---")
            for turn in session.get("turns", []):
                parts.append(
                    f"[{turn.get('turn_num', '')}] services={turn.get('services', '')} | "
                    f"active_intents={turn.get('active_intents', '')} | slot_values={turn.get('slot_values', '')}"
                )
            parts.append("--- END_CSV_AUX ---")
        return "\n".join(parts)

    def _has_multiwoz_metadata(self, session: dict) -> bool:
        return any(t.get("active_intents") or t.get("slot_values") or t.get("services") for t in session.get("turns", []))

    def _dominant_multiwoz_intent(self, turn: dict, previous_pointer: str = "") -> str:
        active = [x.strip() for x in str(turn.get("active_intents", "")).split("|") if x.strip()]
        if not active:
            return ""
        text = (turn.get("text", "") or "").lower()
        prev_domain = previous_pointer.split(":", 1)[0] if ":" in previous_pointer else ""
        if prev_domain:
            same_domain = [a for a in active if a.startswith(prev_domain + ":")]
            other_domain_keywords = {
                "restaurant": ["restaurant", "food", "cuisine", "table"],
                "hotel": ["hotel", "place to stay", "wifi", "parking", "guesthouse", "room", "nights"],
                "taxi": ["taxi", "cab", "commute", "arrive", "pick"],
                "train": ["train", "rail", "station"],
                "bus": ["bus", "coach"],
                "attraction": ["attraction", "theatre", "museum", "college", "park"],
            }
            explicit_other_domain = any(
                domain != prev_domain and any(k in text for k in keywords)
                for domain, keywords in other_domain_keywords.items()
            )
            continuation_markers = ["it", "that", "them", "yes", "also", "great", "can you", "please", "instead", "改成", "可以"]
            if same_domain and not explicit_other_domain and any(m in text for m in continuation_markers):
                if any(x in text for x in ["book", "reservation", "reserve", "订", "预订"]):
                    book_matches = [a for a in same_domain if ":book_" in a]
                    if book_matches:
                        return book_matches[0]
                return same_domain[0]
        keyword_priority = [
            ("taxi", ["taxi", "cab", "car", "commute", "pick", "arrive", "leave"]),
            ("train", ["train", "rail", "station"]),
            ("bus", ["bus", "coach"]),
            ("restaurant:book", ["book", "table", "reservation", "reserve"]),
            ("hotel:book", ["book", "stay", "nights", "people", "room", "reservation"]),
            ("restaurant", ["restaurant", "food", "price", "phone", "number", "cuisine"]),
            ("hotel", ["hotel", "place to stay", "wifi", "parking", "guesthouse"]),
            ("attraction", ["attraction", "theatre", "museum", "college", "park", "information"]),
        ]
        for domain_or_action, keywords in keyword_priority:
            if not any(k in text for k in keywords):
                continue
            matches = [a for a in active if a.startswith(domain_or_action)]
            if matches:
                return matches[0]
            domain = domain_or_action.split(":", 1)[0]
            matches = [a for a in active if a.startswith(domain + ":")]
            if matches:
                return matches[0]
        return active[0]

    def _infer_dependencies(self, pointer: str, pointer_to_index: dict[str, int]) -> list[int]:
        domain, _, action = pointer.partition(":")
        deps: list[int] = []
        if action.startswith("book"):
            find_key = f"{domain}:find_{domain}"
            if find_key in pointer_to_index:
                deps.append(pointer_to_index[find_key])
        if domain == "taxi":
            deps.extend(i for p, i in pointer_to_index.items() if p.startswith(("restaurant:", "hotel:")) and i not in deps)
        return sorted(x for x in deps if x < pointer_to_index.get(pointer, 10**9))

    def _multiwoz_success_criteria(self, pointer: str, examples: list[str]) -> str:
        domain, _, action = pointer.partition(":")
        if pointer == "conversation:close":
            return "助手自然结束对话，不引入新任务。"
        if action.startswith("find"):
            return f"推荐或确认符合用户约束的 {domain}，并给出用户要求的关键信息。"
        if action.startswith("book"):
            return f"成功完成 {domain} 预订，并返回确认号/车辆/时间等可核验结果；如失败需明确失败并引导可替代方案。"
        return "助手需要正面完成该意图，给出明确结果或下一步。"

    def _extract_full_session_refillables(self, turns: list[dict], intents: list[IntentItem]) -> list[RefillableItem]:
        facts: list[tuple[str, str, int, str, int | None]] = []
        key_seen: set[str] = set()

        def add(key: str, reference: str, source_turn: int, trigger: str, bind: int | None = None):
            key_norm = re.sub(r"[^a-zA-Z0-9_.-]+", "_", key).strip("_")
            if not key_norm or key_norm in key_seen or not str(reference).strip():
                return
            key_seen.add(key_norm)
            facts.append((key_norm, str(reference).strip(), int(source_turn), trigger, bind))

        # Slot values provide durable constraints and entity names accumulated across the session.
        latest_slots: dict[str, tuple[str, int]] = {}
        for turn in turns:
            for raw in str(turn.get("slot_values", "") or "").split("|"):
                if "=" not in raw:
                    continue
                k, v = raw.split("=", 1)
                k, v = k.strip(), v.strip()
                if k and v and v.lower() not in {"not mentioned", "none", "?"}:
                    latest_slots[k] = (v, int(turn.get("turn_num", 0)))
        for k, (v, source_turn) in sorted(latest_slots.items()):
            domain = k.split(".", 1)[0]
            bind = self._first_intent_index_for_domain(intents, domain)
            add(k, f"{k} = {v}", source_turn, f"用户后续问题涉及 {domain} 已知约束或实体", bind)

        # Assistant utterances contain closed operational results unavailable to a tool-less test agent.
        for turn in turns:
            if turn.get("role") != "agent":
                continue
            text = turn.get("text", "") or ""
            source_turn = int(turn.get("turn_num", 0))
            lower = text.lower()
            bind = self._intent_index_covering_turn(intents, source_turn)
            for phone in re.findall(r"\b0\d{7,12}\b", text):
                add(f"phone_{phone}", f"联系电话：{phone}", source_turn, "用户询问联系方式或相关预订信息", bind)
            for ref in re.findall(r"\b[A-Z0-9]{6,10}\b", text):
                if ref.upper() in {"SYSTEM", "USER"}:
                    continue
                add(f"reference_{ref}", f"确认号 / reference number：{ref}", source_turn, "用户询问预订结果、订单状态或确认号", bind)
            vehicle = re.search(r"\b(?:grey|gray|black|white|blue|red|silver)\s+[a-z]+\b", lower)
            if vehicle:
                add(f"taxi_vehicle_{source_turn}", f"出租车车辆信息：{vehicle.group(0)}", source_turn, "用户询问出租车安排", bind)
            if any(x in lower for x in ["booked", "booking was successful", "booking is complete", "reference number"]):
                add(f"booking_result_{source_turn}", self._summarize_text(text, 180), source_turn, "用户询问预订是否成功或后续安排", bind)
            elif any(x in lower for x in ["phone number", "located", "postcode", "address", "expensive", "free wifi", "parking"]):
                add(f"info_result_{source_turn}", self._summarize_text(text, 180), source_turn, "用户询问已推荐实体的详情", bind)

        refillables: list[RefillableItem] = []
        for idx, (key, reference, source_turn, trigger, bind) in enumerate(facts, start=1):
            refillables.append(
                RefillableItem(
                    refill_index=idx,
                    trigger_condition=trigger,
                    refill_reference=reference,
                    key=key,
                    source_turn_index=source_turn,
                    confidence="high",
                    injection_text=f"【内部已知事实】{reference}。若用户问题涉及该事实，请直接自然作答，不要重复调用外部系统。",
                    bind_intent_index=bind,
                )
            )
        return refillables

    def _first_intent_index_for_domain(self, intents: list[IntentItem], domain: str) -> int | None:
        for intent in intents:
            if intent.intent_text.startswith(domain + ":"):
                return intent.intent_index
        return None

    def _intent_index_covering_turn(self, intents: list[IntentItem], turn_num: int) -> int | None:
        for intent in intents:
            if intent.historical_span and intent.historical_span.start_turn_index <= turn_num <= intent.historical_span.end_turn_index:
                return intent.intent_index
        return None

    def _is_close_turn(self, text: str) -> bool:
        t = (text or "").lower()
        return any(x in t for x in ["thank", "goodbye", "that's it", "that will be all", "no, thanks", "谢谢", "再见"])

    def _next_user_turn_num(self, user_turns: list[dict], idx: int):
        if idx >= len(user_turns):
            return None
        return int(user_turns[idx].get("turn_num", idx + 1))

    def _count_user_turns_in_span(self, turns: list[dict], start: int, end: int) -> int:
        return sum(1 for t in turns if start <= int(t.get("turn_num", 0)) <= end and t.get("role") == "user")

    def _find_last_assistant_text(self, turns: list[dict], start: int, end: int) -> str:
        text = ""
        for t in turns:
            turn_num = int(t.get("turn_num", 0))
            if start <= turn_num <= end and t.get("role") == "agent":
                text = t.get("text", "")
        return text.strip()

    def _infer_success_criteria(self, turn: dict) -> str:
        utterance = turn.get("text", "").strip()
        if not utterance:
            return "助手正面回应并解决该意图。"
        if utterance.endswith("?") or utterance.endswith("？"):
            return "助手需要直接回答这个问题或给出明确下一步。"
        return "助手需要确认理解并给出有效结果或明确处理动作。"

    def _summarize_text(self, text: str, limit: int = 36) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"
