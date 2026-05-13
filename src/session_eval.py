"""Run single MultiWOZ session eval"""
import csv, json, os, sys, time
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
from openai import OpenAI

c = OpenAI(api_key='sk-VAwDjNVna9vZbu4SLIKoBOMSNSjtXaQsSuhPQZ8GJ5MXh8de', base_url='https://api.frontier-intelligence.tech/v1')

PROMPT_T = """You are a dialogue state tracker. Given the conversation history between a USER and a travel booking SYSTEM, extract the user's CURRENT-TURN intents and slot values.
Available services: restaurant, hotel, train, taxi, attraction, bus, hospital
Slot naming: service.slot_name (e.g. hotel.hotel-area, restaurant.restaurant-food)
## Format - Return ONLY valid JSON: {{"intents": ["s:intent"], "slots": {{"s.slot": ["v"]}} }}
## Conversation
{conversation}
## Last USER utterance
{user_utterance}
Return JSON:"""

session_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 2
csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "multiwoz", "samples_10_diff.csv")
sessions = {}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        did = row['dialogue_id']
        if did not in sessions:
            sessions[did] = {'dialogue_id': did, 'services': row['services'].split('|') if row['services'] else [], 'turns': []}
        sessions[did]['turns'].append(row)
for s in sessions.values():
    s['turns'].sort(key=lambda t: int(t['turn_num']))
sl = list(sessions.values())
session = sl[session_idx]
did = session['dialogue_id']
turns = session['turns']
ut = [t for t in turns if t['speaker'] == 'USER' and t['slot_values_delta']]

print(f"Session {session_idx}: {did} ({len(ut)} turns) {session['services']}", flush=True)
results = []
for t in ut:
    tn = int(t['turn_num'])
    hist = "\n".join(f"{ht['speaker']}: {ht['utterance']}" for ht in turns[:tn]) or "(start)"
    prompt = PROMPT_T.format(conversation=hist, user_utterance=t['utterance'])
    t0 = time.time()
    r = c.chat.completions.create(model='deepseek-v4-pro', messages=[
        {'role':'system','content':'You are a dialogue state tracker. Respond with valid JSON only.'},
        {'role':'user','content':prompt},
    ], temperature=0.1, max_tokens=2000)
    raw = r.choices[0].message.content.strip()
    if raw.startswith('```'): raw = raw.split('```')[1]
    if raw.startswith('json'): raw = raw[4:]
    pred = json.loads(raw.strip())
    elapsed = time.time()-t0
    gt_i = [p.strip() for p in t['active_intents'].split('|') if ':' in p] if t['active_intents'] else []
    gt_s = {}
    if t['slot_values_delta']:
        for part in t['slot_values_delta'].split('|'):
            if '=' in part:
                k,v = part.split('=',1)
                gt_s[k.strip()] = sorted(v.split(','))
    pred_i = pred.get('intents',[])
    pred_s = {k: sorted(v) if isinstance(v,list) else [v] for k,v in pred.get('slots',{}).items()}
    gi,pi = set(gt_i), set(pred_i)
    i_f1 = 2*len(gi&pi)/(len(pi)+len(gi)) if (gi or pi) else 0
    gk,pk = set(gt_s.keys()), set(pred_s.keys())
    s_f1 = 2*len(gk&pk)/(len(pk)+len(gk)) if (gk or pk) else 0
    print(f"  T{tn} [{elapsed:.0f}s] iF1={i_f1:.2f} sF1={s_f1:.2f} gt={gt_i} pred={pred_i}", flush=True)
    results.append({'turn':tn,'intent_f1':round(i_f1,3),'slot_f1':round(s_f1,3),'gt_intents':gt_i,'pred_intents':pred_i,'gt_slots':gt_s,'pred_slots':pred_s})
    time.sleep(0.1)

out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, f"session_{session_idx:02d}.json")
with open(out, 'w') as f: json.dump({'dialogue_id':did,'services':session['services'],'turns':results}, f, indent=2, ensure_ascii=False)
print(f"DONE -> {out}", flush=True)
