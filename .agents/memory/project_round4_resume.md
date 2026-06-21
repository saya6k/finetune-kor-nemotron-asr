---
name: project-round4-resume
description: Round 4 resume 학습 설정 차이 — 전체 레이어 학습(동결 없음)으로 재개됨
metadata: 
  node_type: memory
  type: project
  originSessionId: 76c864a0-4c6d-4e2a-9bcc-517a239c9ebe
---

Round 4 resume 학습(2026-06-21)은 원래 학습과 레이어 동결 설정이 다름.

**원래 Round 4 학습**: `finetune_with_freeze.py` — 하위 8 인코더 레이어 동결 (436M trainable / 201M non-trainable)

**Resume 학습**: `speech_to_text_finetune.py` — 전체 레이어 학습 (637M trainable / 0 non-trainable)

Resume 시작 체크포인트: `val_wer=0.4757-epoch=1-last.ckpt` (E1 최종)

변경 파라미터: `patience=5 → 20`, `max_epochs=10 → 20`

**Why:** Early stopping이 S250(val_wer=0.4643) 이후 5회 연속 비개선으로 너무 일찍 발동. 실제 테스트 지표(KSpon CER, Zeroth CER, FL-ru WER)는 계속 개선 중이었음.

**How to apply:** Resume 이후 체크포인트는 "전체 레이어 fine-tuning" 구간으로 구분. 동결 학습 체크포인트(E0-E1)와 resume 이후 체크포인트(E1+) 비교 시 이 차이를 감안할 것.
