---
name: project_round3_plan
description: "Round 3 training plan — composition, changes from Round 2, rationale"
metadata: 
  node_type: memory
  type: project
  originSessionId: 76c864a0-4c6d-4e2a-9bcc-517a239c9ebe
---

Round 3 학습 계획 (Round 2 완료 후 진행).

**학습 데이터 구성 (승인됨):**
- ko: 812,825발화 (~85%) — Round 2와 동일
- en: 80,000발화 (~8%) — Round 2(24,879)에서 증가
- ja: 60,000발화 (~6%) — 신규 추가
- zh: 제거 (Round 2의 40,000발화 삭제)
- 합계: ~952,825발화

**데이터 소스 (DGX에 이미 ingest 완료):**
- en: `/workspace/data/processed/_temp_ingest_en.jsonl` (458,335발화)
- ja: `/workspace/data/processed/_temp_ingest_ja.jsonl` (144,538발화)
- 각각에서 샘플링 필요

**Round 2 대비 변경점:**
- zh 제거: 학습 텍스트에 불일치한 공백 형식(어절/글자 단위 혼재) → epoch 1부터 catastrophic forgetting 발생 (CER 23% → 62%)
- en 증가: 2.8% → ~8%
- ja 신규: ~6%
- patience: 10 → 5 (파인튜닝에 더 적합)

**Why:** zh 학습 데이터의 공백 형식 불일치가 epoch 1 이후 zh CER 폭등의 원인. ja 추가로 다국어 커버리지 향상. patience 5로 낮춰 오버피팅 조기 차단.

**How to apply:** Round 3 manifest 생성 시 en/ja 각각 랜덤 샘플링, zh 미포함. train_manifest_round3.jsonl로 저장.
