# TODO (v3 · Emilia-YODAS Korean): Nemotron 3.5 ASR 한국어 파인튜닝

> 모델: `nvidia/nemotron-3.5-asr-streaming-0.6b` / NeMo 26.06 / 데이터: Emilia-YODAS KO 7,300h (CC BY 4.0)

## Phase 0: 환경·체크포인트

- [ ] **Task 0**: NeMo 26.06 컨테이너 + `.nemo` 다운로드 (`HF_CKPT_PATH` 확정)

## Phase 1: 한글화

- [ ] **Task 1**: 마크다운 + 주석 한글 번역, 모델명/데이터셋명 정정
- [ ] **Task 2**: 의존성 셀 — NeMo 26.06 핀, `datasets`/`soundfile`/`librosa`/`tqdm`/`huggingface_hub` 추가

### ✅ Checkpoint 1
- [ ] JSON 유효성 / 마크다운 한국어 / 모델명·데이터셋명 정정

## Phase 2: 데이터 파이프라인

- [ ] **Task 3**: `DATA_DIR` 설정 + HF `.nemo` 다운로드 셀 (수동 경로 assert 포함)

- [ ] **Task 4** (L): Emilia-YODAS KO 전처리 — Task 3 이후 ⚠️핵심
  - HF datasets `streaming=True`, `Emilia-YODAS/KO/**/*.tar`
  - `dnsmos ≥ 3.0` 품질 필터
  - MP3 → 16kHz mono WAV (`librosa.resample`)
  - 매니페스트: `target_lang:"ko-KR"`, 빈 텍스트 제외
  - 스모크(`smoke_manifest.json`, N=5000) + 전체(`train_manifest.json`) 분리
  - eval(`eval_manifest.json`): hold-out 또는 FLEURS Korean
  - 검증: JSON 유효, `ko-KR` 확인, 빈 텍스트 없음, WAV 재생

- [ ] **Task 5**: 오디오 샘플 셀 — smoke_manifest 첫 항목 동적 참조, text/dnsmos/duration 출력

### ✅ Checkpoint 2
- [ ] cell-3~14 무오류
- [ ] 매니페스트 생성·ko-KR 확인
- [ ] `grep -i "an4\|en-US\|ksponspeech" *.ipynb` → 없음

## Phase 3: 학습·평가

- [ ] **Task 6**: 스모크 학습 셀 (smoke_manifest, limit=100 유지) — Task 4 이후
  - ⚠️ `limit_train_batches` 본학습 제거 경고 주석 (#15782)
  - `HF_CKPT_PATH` assert, `max_epochs=3`, `precision=bf16`

- [ ] **Task 7** (M): 트랙 B `train_emilia_ko.sh` + `RUN.md` — Task 4 이후
  - 1단계(~200h) → CER 검증 → 2단계(전체 7,500h)
  - `limit_train_batches` 제거, 멀티GPU, 체크포인트 재개
  - 디스크 요구사항·예상 시간·로컬/클라우드 안내

- [ ] **Task 8**: 평가 셀 — eval_manifest, **CER** 보고 주석 (베이스라인 7.12%) — Task 6 이후

- [ ] **Task 9**: 리소스/다음단계 섹션 한국어 + Emilia-YODAS·파인튜닝 블로그 링크 (독립)

### ✅ Checkpoint 3 (최종)
- [ ] `python -m json.tool *.ipynb > /dev/null`
- [ ] `grep -i "an4\|en-US\|ksponspeech\|an268" *.ipynb` → 없음
- [ ] 매니페스트 `target_lang` 전부 `ko-KR`
- [ ] 트랙 A + 트랙 B RUN.md 재현 가능

---

## ✅ 결정 완료
1. 데이터셋: **Emilia-YODAS Korean** (7,300h, CC BY 4.0)
2. 단계: 스모크(5,000샘플) → 1단계(~200h) → 전체(7,500h)
3. 전처리: MP3→16kHz WAV, dnsmos ≥ 3.0, UTF-8 그대로

## ⚠️ 남은 미결 (구현 중 결정)
- eval 데이터: Emilia KO hold-out vs FLEURS Korean
- WAV 저장 vs MP3 직접 읽기 (NeMo 26.06 데이터로더 지원 여부)
