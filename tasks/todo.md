# TODO: GB300 DGX Station Porting

> RunPod L40S 검증 완료 (2026-06-17). GB300 DGX Station 포팅 진행 중.

## Phase 0: 오프라인 준비 (DGX 불필요)

- [x] Task 0: `scripts/probe_dgx_environment.sh` 작성
- [x] Task 1: `scripts/train_pipeline.py` — aarch64 CUDA 경로 + Blackwell sm_100 skip
- [x] Task 2: `scripts/patch_numba_codegen.py` — sysconfig 동적 경로 + sm_100 skip
- [x] Task 3: `scripts/verify_on_dgx.sh` 작성
- [x] Task 4: `scripts/setup_dgx.sh` 스켈레톤 작성

### Checkpoint A 검증

```bash
bash -n scripts/probe_dgx_environment.sh
bash -n scripts/verify_on_dgx.sh
bash -n scripts/setup_dgx.sh
python3 scripts/patch_numba_codegen.py      # GPU 없이 exit 0
grep "aarch64-linux\|_sm_major >= 10" scripts/train_pipeline.py
```

---

## Phase 1: 온디바이스 탐색 (DGX 접속 후)

- [x] Task 5: DGX 환경 탐색 → SPEC.md §4 업데이트 완료 (2026-06-20)
  - `nvcr.io/nvidia/nemo:latest` 사용 불가 (lightning 미포함) → `pytorch:26.05-py3` + `Dockerfile.dgx`로 해결
  - `nemotron-asr-dgx:latest` 빌드 (24.1 GiB)
- [x] Task 6: `Dockerfile.dgx` 빌드 + `verify_on_dgx.sh` → **9/9 PASS** (2026-06-20)

### Checkpoint B 검증 결과

```
✅ PASS:9 FAIL:0 — verify_on_dgx.sh 9/9 통과 (2026-06-20)
⚠️  SMOKE_N=100 학습: OOM (DeepSeek-V4-Flash VLLM이 GPU 97% 점유)
   → Step 1-5 PASS, Step 6 (Fine-Tuning) FAIL (VLLM 때문, NeMo 문제 아님)
   → Issue #15799 및 WarpRNNT sm_103 커널 미검증
⏳ VLLM 서비스 중단 후 재시도 필요
```

**VLLM 서비스 정보**: `deepseek-v4-flash` 컨테이너 (docker-compose 관리), `docker compose -f <경로> down`으로 중단 확인됨.

---

## Phase 2: 성능 특성화 (Checkpoint B 완료 후 VLLM 중단 필요)

- [ ] Task 6.5: VLLM 서비스 중단 후 SMOKE_N=100 재실행
  - [ ] WarpRNNT sm_103 커널 컴파일 성공 확인
  - [ ] Issue #15799 검증: 체크포인트 저장 성공 여부
  - [ ] Step 6 (Fine-Tuning) 첫 스텝 NaN 없음 확인
- [ ] Task 7: batch_duration 탐색 (100s → 500s → 1000s → OOM 경계)
  - [ ] benchmark_gpu.py `GPU_HOURLY_RATE` DGX 기준으로 업데이트
  - [ ] SPEC.md §6 최적 batch_duration 기록

### Checkpoint C 검증

L40S 대비 throughput 비율 계산 및 기록.

---

## Phase 2.5: 다운로드 (학습 전 완료 필요)

- [ ] Task 6.7: Emilia-YODAS 로컬 다운로드
  - [x] KO: 208 TAR 파일 (~210 GB) — 진행 중
  - [x] EN: 30 샤드 (10% 비율 충족) — 진행 중
  - [x] JA: 15 샤드 (4% 비율 충족) — EN 완료 후 시작
  - [x] ZH: 15 샤드 (4% 비율 충족) — JA 완료 후 시작
  - [ ] FR: 5 샤드 (1% 비율) — **미시작** (EN/JA/ZH 이후 추가 필요)
  - [ ] DE: 5 샤드 (1% 비율) — **미시작**
  - `EMILIA_LOCAL_DIR=/workspace/emilia_local` 로 `train_pipeline.py` 사용
- [ ] Task 6.8: Eval 데이터셋 다운로드
  - [x] FLEURS ko-KR: 382개 완료
  - [ ] FLEURS fr-FR: 미완료
  - [ ] FLEURS de-DE: 미완료
  - [ ] FLEURS ru-RU: 미완료 (catastrophic forgetting 프로브)
  - [ ] Zeroth Korean: 진행 중

## Phase 3: 풀 학습 (Phase 2.5 완료 후)

- [ ] Task 8: 3 epoch 학습 + checkpoint sweep
  - [ ] val_wer 감소 추세 확인 (NaN 없음)
  - [ ] CER on FLEURS ko-KR ≤ 7.12
  - [ ] FLEURS fr-FR, de-DE WER 측정 (학습 언어 성능)
  - [ ] FLEURS ru-RU WER 측정 (catastrophic forgetting 확인)
  - [ ] RunPod 결과와 비교표 작성

---

## 파일 변경 요약

| 파일 | 상태 |
|------|------|
| `Dockerfile.dgx` | ✅ 신규 — `pytorch:26.05-py3` + NeMo main + stub (9/9 PASS) |
| `scripts/probe_dgx_environment.sh` | ✅ 신규 |
| `scripts/verify_on_dgx.sh` | ✅ 신규 (9/9 PASS 달성) |
| `scripts/setup_dgx.sh` | ✅ 신규 (Dockerfile로 대체됨, 참조용 유지) |
| `scripts/train_pipeline.py` | ✅ 수정 (aarch64 + Blackwell + SKIP_SETUP_INSTALL) |
| `scripts/patch_numba_codegen.py` | ✅ 수정 (sysconfig + sm_100) |
| `SPEC.md` §2,3,4,5,6,9 | ✅ 2026-06-20 실측 결과로 업데이트 |
| `scripts/benchmark_gpu.py` | ⏳ Task 7 시 수정 |
| `SPEC.md` §6 batch_duration | ⏳ Task 7 시 확정 |
