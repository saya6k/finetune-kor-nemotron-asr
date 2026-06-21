---
name: project_round6_plan
description: "R6 FLEURS replay strategy — ko 93.3% + 39-language replay 6.7%, max_epochs=3"
metadata: 
  node_type: memory
  type: project
  originSessionId: 76c864a0-4c6d-4e2a-9bcc-517a239c9ebe
---

R6: Korean ASR fine-tuning with FLEURS-based multilingual replay anchoring.

**Data:** ko 812,825 (93.3%) + FLEURS replay 58,500 (6.7%, 39 languages × 1,500 each) = 871,325 utterances.

**Strategy (NVIDIA HF blog 기반):**
- Replay "a slice" of base model's languages via FLEURS (clean, diverse)
- raw_transcription (cased+punct) — matches base model output style
- Proper target_lang tags on every clip for prompt mechanism
- 6.7% replay ratio — anchoring purpose only, not co-training
- Original .nemo restart — avoid R5 VoxPopuli contamination

**Training:** Full FT, max_epochs=3, lr=5e-5, patience=5, batch_duration=5000.

**Skipped languages:** es (es_es→es_419 fix needed), zh (cmn_hant_tw→cmn_hans_cn fix needed).

**Why:** R5 showed replay can work in principle but VoxPopuli was wrong domain. FLEURS matches base model's original training distribution. 3 epochs prevents overfitting that R5's 10 epochs caused.

**How to apply:** Manifest built at /workspace/data/processed/train_manifest_round6.jsonl (871,325 lines). Training command in CLAUDE.md. Start with `docker run --name nemotron-train-round6 ...` on DGX.
