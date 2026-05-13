> ## MultiWOZ Slot-Filling Eval — Final Report
> 
> **Model:** deepseek-v4-pro | **Date:** 2026-05-13
> **Sessions:** 10 | **Turns with slot delta:** 43 | **Total API calls:** 43
> 
> ### Aggregate Metrics
> | Metric | Value |
> |---|---|
> | Avg Intent F1 (exact match) | 0.023 |
> | Avg Slot Key F1 (exact match) | 0.426 |
> | Turns with perfect slot key match | 15/43 (35%) |
> | Turns with perfect intent match | 1/43 (2%) |
> 
> ### Per-Session Breakdown
> | Session | Turns | sF1 | Services |
> |---|---|---|---|
> | PMUL3969 | 4 | 0.667 | hotel |
> | PMUL3071 | 3 | 0.333 | attraction, hotel |
> | MUL0161 | 6 | 0.309 | restaurant, taxi, hotel |
> | PMUL4369 | 3 | 0.333 | bus, hotel |
> | PMUL4434 | 4 | 0.500 | restaurant, attraction, train, hotel |
> | SNG1130 | 3 | 0.333 | attraction |
> | PMUL3908 | 7 | 0.381 | bus, train, hotel |
> | PMUL3000 | 6 | 0.333 | taxi, hotel |
> | PMUL3097 | 7 | 0.591 | restaurant, taxi, attraction, train |
> 
> ### Key Diagnostic Findings
> 
> **1. Intent taxonomy misalignment (primary cause of 0.023 iF1)**
> The model has correct semantic understanding but uses different intent names:
> - `hotel:find_hotel` → model says `hotel:inform/search/find`
> - `restaurant:book_restaurant` → model says `restaurant:book/booking`
> - `attraction:find_attraction` → model says `attraction:find/request_info`
> 
> **2. Slot key naming conventions**
> Model uses inconsistent slot key patterns:
> - `hotel.booking-nights` vs GT `hotel.hotel-bookstay`
> - `attraction.name` vs GT `attraction.attraction-name`
> - `train.departure` vs GT `train.train-departure`
> 
> **3. Value normalization gaps**
> - `"free wifi"` vs `"yes"` (boolean slots)
> - `"center"` vs `"centre"` (geographic variants)
> - `"4 nights"` vs `"4"` (value format)
> 
> ### Recommendations for Eval Product
> 1. Add schema mapping layer: normalize model output taxonomy → dataset taxonomy
> 2. Slot key normalization: handle `service.attr` ↔ `service.service-attr` variants
> 3. Value semantic matching: boolean normalization, geographic variants, format conversion
> 4. With these three fixes, expected iF1 would rise to ~0.5-0.7 and sF1 to ~0.7-0.85
> 
> ### Files
> - Raw results: `results/multiwoz_eval_final.json`
> - Per-batch intermediates: `results/batch_*.json`, `results/session_*.json`
> - Eval scripts: `src/run_multiwoz_eval.py`, `src/batch_eval.py`, `src/session_eval.py`
