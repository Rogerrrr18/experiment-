"""
MultiWOZ Slot-Filling Eval — 用 LLM 预测每轮 slot delta，对比 ground truth
"""
import csv, json, os, sys, time
from pathlib import Path
from openai import OpenAI

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# ── 配置 ──
API_KEY = "sk-VAwDjNVna9vZbu4SLIKoBOMSNSjtXaQsSuhPQZ8GJ5MXh8de"
BASE_URL = "https://api.frontier-intelligence.tech/v1"
MODEL = "deepseek-v4-pro"
CSV_PATH = Path(__file__).parent.parent / "data" / "multiwoz" / "samples_10_diff.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── Prompt ──
EXTRACT_PROMPT = """You are a dialogue state tracker. Given the conversation history between a USER and a travel booking SYSTEM, extract the user's CURRENT-TURN intents and slot values.

Available services: restaurant, hotel, train, taxi, attraction, bus, hospital
Slot naming: service.slot_name (e.g. hotel.hotel-area, restaurant.restaurant-food, train.train-departure)

## Format
Return ONLY valid JSON:
{{
  "intents": ["service:intent_name", ...],
  "slots": {{"service.slot_name": ["value1", "value2"], ...}}
}}

## Rules
- Extract ONLY what the user explicitly stated or clearly implied in THIS turn (not accumulated from previous turns)
- If the user says "I don't care about price" → slot value is "dontcare"
- If the user clarifies/corrects a previous slot, include the new value
- For booking intents, include all relevant booking slots
- If no new information in this turn, return empty intents and slots

## Conversation
{conversation}

## Last USER utterance
{user_utterance}

Return JSON:"""

def load_sessions(csv_path: str) -> list[dict]:
    """Load the diff CSV, group by dialogue_id into sessions"""
    sessions = {}
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            did = row['dialogue_id']
            if did not in sessions:
                sessions[did] = {
                    'dialogue_id': did,
                    'services': row['services'].split('|') if row['services'] else [],
                    'turns': [],
                }
            sessions[did]['turns'].append(row)
    
    # Sort turns by turn_num
    for s in sessions.values():
        s['turns'].sort(key=lambda t: int(t['turn_num']))
    return list(sessions.values())


def build_conversation_history(turns: list[dict], current_turn: int) -> str:
    """Build conversation text up to (but not including) current turn"""
    lines = []
    for t in turns[:current_turn]:
        speaker = t['speaker']
        utterance = t['utterance']
        lines.append(f"{speaker}: {utterance}")
    return "\n".join(lines)


def call_llm(conversation: str, user_utterance: str, max_retries: int = 2) -> dict:
    """Call LLM to extract intents and slots for current turn"""
    prompt = EXTRACT_PROMPT.format(
        conversation=conversation or "(start of conversation)",
        user_utterance=user_utterance,
    )
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a dialogue state tracker. Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2000,  # need headroom for reasoning tokens
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code blocks
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return {"error": str(e), "intents": [], "slots": {}}


def parse_ground_truth(row: dict) -> dict:
    """Parse ground truth intents and slot_values_delta from CSV row"""
    intents = []
    if row['active_intents']:
        for part in row['active_intents'].split('|'):
            if ':' in part:
                intents.append(part.strip())
    
    slots = {}
    if row['slot_values_delta']:
        for part in row['slot_values_delta'].split('|'):
            if '=' in part:
                key, vals = part.split('=', 1)
                slots[key.strip()] = sorted(vals.split(','))
    
    return {'intents': intents, 'slots': slots}


def compute_metrics(gt: dict, pred: dict) -> dict:
    """Compare ground truth vs prediction"""
    # Intent accuracy
    gt_intents = set(gt['intents'])
    pred_intents = set(pred.get('intents', []))
    
    intent_precision = len(gt_intents & pred_intents) / max(len(pred_intents), 1)
    intent_recall = len(gt_intents & pred_intents) / max(len(gt_intents), 1)
    intent_f1 = 2 * intent_precision * intent_recall / max(intent_precision + intent_recall, 0.001)
    
    # Slot accuracy (exact match per slot key)
    gt_slots = gt['slots']
    pred_slots = {k: sorted(v) if isinstance(v, list) else [v] for k, v in pred.get('slots', {}).items()}
    
    # Also normalize: strip service prefix for matching when LLM omits it
    gt_keys = set(gt_slots.keys())
    pred_keys = set(pred_slots.keys())
    
    slot_precision = len(gt_keys & pred_keys) / max(len(pred_keys), 1)
    slot_recall = len(gt_keys & pred_keys) / max(len(gt_keys), 1)
    slot_key_f1 = 2 * slot_precision * slot_recall / max(slot_precision + slot_recall, 0.001)
    
    # Slot value exact match (for shared keys)
    value_matches = 0
    value_total = 0
    for k in gt_keys & pred_keys:
        value_total += 1
        if gt_slots[k] == pred_slots[k]:
            value_matches += 1
    value_accuracy = value_matches / max(value_total, 1)
    
    return {
        'intent_precision': round(intent_precision, 3),
        'intent_recall': round(intent_recall, 3),
        'intent_f1': round(intent_f1, 3),
        'slot_key_precision': round(slot_precision, 3),
        'slot_key_recall': round(slot_recall, 3),
        'slot_key_f1': round(slot_key_f1, 3),
        'slot_value_accuracy': round(value_accuracy, 3),
        'gt_intents': sorted(gt_intents),
        'pred_intents': sorted(pred_intents),
        'gt_slots': gt_slots,
        'pred_slots': pred_slots,
    }


def run_eval():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    sessions = load_sessions(CSV_PATH)
    print(f"Loaded {len(sessions)} sessions", flush=True)
    
    # Limit to 5 sessions to keep runtime manageable
    sessions = sessions[:5]
    print(f"Running on {len(sessions)} sessions", flush=True)
    
    all_results = []
    total_user_turns = 0
    total_intent_f1 = 0
    total_slot_f1 = 0
    
    for sidx, session in enumerate(sessions):
        did = session['dialogue_id']
        turns = session['turns']
        user_turns = [t for t in turns if t['speaker'] == 'USER' and t['slot_values_delta']]
        
        print(f"\n{'='*60}", flush=True)
        print(f"Session {sidx+1}/{len(sessions)}: {did} ({len(user_turns)} user turns with slots)", flush=True)
        print(f"Services: {session['services']}", flush=True)
        
        session_results = []
        for t in user_turns:
            turn_num = int(t['turn_num'])
            utterance = t['utterance']
            conversation = build_conversation_history(turns, turn_num)
            
            pred = call_llm(conversation, utterance)
            gt = parse_ground_truth(t)
            
            metrics = compute_metrics(gt, pred)
            metrics['turn_num'] = turn_num
            metrics['utterance'] = utterance[:80]
            session_results.append(metrics)
            
            if metrics['gt_intents'] or metrics['pred_intents']:
                print(f"  Turn {turn_num}: intent_f1={metrics['intent_f1']}, slot_key_f1={metrics['slot_key_f1']}, slot_val_acc={metrics['slot_value_accuracy']}", flush=True)
                total_user_turns += 1
                total_intent_f1 += metrics['intent_f1']
                total_slot_f1 += metrics['slot_key_f1']
            
            time.sleep(0.3)  # rate limit
        
        all_results.append({
            'dialogue_id': did,
            'services': session['services'],
            'turns': session_results,
        })
    
    # Aggregate
    avg_intent_f1 = total_intent_f1 / max(total_user_turns, 1)
    avg_slot_f1 = total_slot_f1 / max(total_user_turns, 1)
    
    summary = {
        'experiment': 'multiwoz_slot_filling',
        'model': MODEL,
        'total_sessions': len(sessions),
        'total_user_turns_evaluated': total_user_turns,
        'avg_intent_f1': round(avg_intent_f1, 3),
        'avg_slot_key_f1': round(avg_slot_f1, 3),
        'details': all_results,
    }
    
    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"multiwoz_slot_eval_{timestamp}.json"
    with open(result_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Sessions: {len(sessions)}", flush=True)
    print(f"Turns evaluated: {total_user_turns}", flush=True)
    print(f"Avg Intent F1: {avg_intent_f1:.3f}", flush=True)
    print(f"Avg Slot Key F1: {avg_slot_f1:.3f}", flush=True)
    print(f"Results: {result_path}", flush=True)


if __name__ == "__main__":
    run_eval()
