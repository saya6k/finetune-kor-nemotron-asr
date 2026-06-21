---
name: feedback_metric_display
description: "Show CER for CJK languages, WER for non-CJK languages in eval result tables"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 76c864a0-4c6d-4e2a-9bcc-517a239c9ebe
---

In eval result tables, use the metric appropriate for each language:
- **CER**: Korean, Chinese, Japanese (CJK — character-based, no word boundaries)
- **WER**: all other languages (Russian, French, German, English, etc.)

**Why:** CER is the meaningful metric for CJK scripts where "words" aren't space-delimited. WER is meaningful for space-delimited scripts.

**How to apply:** Any time showing eval results across multiple languages, pick CER or WER per row based on the language, not a single metric for all rows.
