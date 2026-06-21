---
name: project_round5_results
description: R5 catastrophic forgetting results and key findings that shaped R6 strategy
metadata: 
  node_type: memory
  type: project
  originSessionId: 76c864a0-4c6d-4e2a-9bcc-517a239c9ebe
---

R5: ko 80% + en 14% VoxPopuli + ja 6%, no layer freeze, max_epochs=10, DGX GB300.

**Results:**
- best val_wer=0.3590 (E7) — Korean KsponSpeech improved consistently
- fl_ko CER: 8.81 (E0-S500 best) → 9.72 (E8), baseline 10.25 — initial improvement then regression
- fl_en_raw WER: 26.07 (E0-S250) → 63.35 (E8-S250) — monotonic deterioration
- normalized WER probe: baseline 16.17 → R5-best(E4) 53.87 (+37.7pt)

**Why replay failed:**
1. VoxPopuli is European Parliament speech — wrong domain for conversational ASR
2. Replay 20% too large — diluted Korean learning without anchoring English
3. Early stopping monitored val_wer (Korean) only — could not catch English forgetting
4. 10 epochs excessive — fl_ko and fl_en both peaked at E0/E1

**Why:** VoxPopuli parliamentary speech register mismatch with base model's original training data. The decoder drifted despite large replay ratio because the replay data didn't match the base model's text style distribution.

**How to apply:** R6 uses FLEURS (39 languages, clean, diverse, properly cased+punct), 6.7% replay ratio, max_epochs=3. See [[project_round6_plan]].
