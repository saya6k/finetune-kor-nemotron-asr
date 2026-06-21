---
name: feedback_table_baseline_first
description: "When showing training progress or eval result tables, always put baseline row first"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 76c864a0-4c6d-4e2a-9bcc-517a239c9ebe
---

Always put the baseline as the first row in any training progress or eval results table.

**Why:** Without baseline at the top, improvement is hard to assess at a glance — the reader has to hunt for the reference point.

**How to apply:** Any time showing val_wer/CER/WER across checkpoints or epochs, add a baseline row (pretrained model score) at the top before listing fine-tuned results.
