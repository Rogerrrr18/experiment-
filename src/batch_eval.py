"""Batch MultiWOZ eval — 3 sessions per run, saves intermediate results"""
import csv, json, os, sys, time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from openai import OpenAI

API_KEY = "sk-VAwDjNVna9vZbu4SLIKoBOMSNSjtXaQsSuhPQZ8GJ5MXh8de"
BASE_URL = "https://api.frontier-intelligence.tech/v1"
MODEL = "deepseek-v4-pro"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

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

CSV_PATH = Path(__file__).parent.parent / "data" / "multiwoz" / "samples_10_diff.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load sessions
sessions = {}
with open(CSV_PATH, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        did = row['dialogue_id']
        if did not in sessions:
            sessions[did] = {'dialogue_id': did, 'services': row['services'].split('|') if row['services'] else [], 'turns': []}
        sessions[did]['turns'].append(row)

for s in sessions.values():
    s['turns'].sort(key=lambda t: int(t['turn_num']))

session_list = list(sessions.values())

# Read batch info from args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=True, help='Batch number (1-based)')
parser.add_argument('--batch-size', type=int, default=3)
args = parser.parse_args()

start = (args.batch - 1) * args.batch_size
batch = session_list[start:start + args.batch_size]

print(f"Batch {args.batch}: sessions {start+1}-{min(start+len(batch), len(session_list))} of {len(session_list)}", flush=True)

total_turns = 0
total_intent_f1 = 0
total_slot_f1 = 0
all_results = []

for sidx, session in enumerate(batch):
    did = session['dialogue_id']
    turns = session['turns']
    user_turns = [t for t in turns if t['speaker'] == 'USER' and t['slot_values_delta']]
    
    print(f"\n[Session] {did} ({len(user_turns)} turns)", flush=True)
    
    session_results = []
    for t in user_turns:
        turn_num = int(t['turn_num'])
        utterance = t['utterance']
        
        hist_lines = []
        for ht in turns[:turn_num]:
            hist_lines.append(f"{ht['speaker']}: {ht['utterance']}")
        conversation = "\n".join(hist_lines) or "(start of conversation)"
        
        prompt = EXTRACT_PROMPT.format(conversation=conversation, user_utterance=utterance)
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {'role': 'system', 'content': 'You are a dialogue state tracker. Respond with valid JSON only.'},
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            raw = r.choices[0].message.content.strip()
            if raw.startswith('```'):
                raw = raw.split('```')[1]
                if raw.startswith('json'):
                    raw = raw[4:]
            pred = json.loads(raw.strip())
        except Exception as e:
            pred = {"error": str(e), "intents": [], "slots": {}}
        
        elapsed = time.time() - t0
        
        gt_intents = [p.strip() for p in t['active_intents'].split('|') if ':' in p] if t['active_intents'] else []
        gt_slots = {}
        if t['slot_values_delta']:
            for part in t['slot_values_delta'].split('|'):
                if '=' in part:
                    k, v = part.split('=', 1)
                    gt_slots[k.strip()] = sorted(v.split(','))
        
        pred_intents = pred.get('intents', [])
        pred_slots = {k: sorted(v) if isinstance(v, list) else [v] for k, v in pred.get('slots', {}).items()}
        
        gt_iset = set(gt_intents)
        pred_iset = set(pred_intents)
        i_f1 = 0.0
        if gt_iset or pred_iset:
            i_prec = len(gt_iset & pred_iset) / max(len(pred_iset), 1)
            i_rec = len(gt_iset & pred_iset) / max(len(gt_iset), 1)
            i_f1 = 2 * i_prec * i_rec / max(i_prec + i_rec, 0.001)
        
        gt_kset = set(gt_slots.keys())
        pred_kset = set(pred_slots.keys())
        s_f1 = 0.0
        if gt_kset or pred_kset:
            s_prec = len(gt_kset & pred_kset) / max(len(pred_kset), 1)
            s_rec = len(gt_kset & pred_kset) / max(len(gt_kset), 1)
            s_f1 = 2 * s_prec * s_rec / max(s_prec + s_rec, 0.001)
        
        total_turns += 1
        total_intent_f1 += i_f1
        total_slot_f1 += s_f1
        
        print(f"  T{turn_num} [{elapsed:.1f}s] iF1={i_f1:.2f} sF1={s_f1:.2f} | gt_i={gt_intents} pred_i={pred_intents}", flush=True)
        
        session_results.append({
            'turn': turn_num, 'intent_f1': round(i_f1, 3), 'slot_f1': round(s_f1, 3),
            'gt_intents': gt_intents, 'pred_intents': pred_intents,
            'gt_slots': gt_slots, 'pred_slots': pred_slots,
        })
        
        time.sleep(0.1)
    
    all_results.append({'dialogue_id': did, 'services': session['services'], 'turns': session_results})

avg_i_f1 = total_intent_f1 / max(total_turns, 1)
avg_s_f1 = total_slot_f1 / max(total_turns, 1)

out_path = RESULTS_DIR / f"batch_{args.batch:02d}.json"
with open(out_path, 'w') as f:
    json.dump({
        'batch': args.batch,
        'model': MODEL,
        'sessions': len(batch),
        'turns': total_turns,
        'avg_intent_f1': round(avg_i_f1, 3),
        'avg_slot_key_f1': round(avg_s_f1, 3),
        'details': all_results,
    }, f, indent=2, ensure_ascii=False)

print(f"\nBatch {args.batch} done: iF1={avg_i_f1:.3f} sF1={avg_s_f1:.3f}", flush=True)
print(f"Saved: {out_path}", flush=True)
