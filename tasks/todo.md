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

- [ ] Task 5: DGX에서 `bash scripts/probe_dgx_environment.sh` → SPEC.md §4 TBD 채우기
- [ ] Task 6: `bash scripts/setup_dgx.sh` + `bash scripts/verify_on_dgx.sh` → 7/7 PASS

### Checkpoint B 검증

```bash
# verify_on_dgx.sh 출력 마지막 줄:
# ✅ Ready for training (PASS:7 FAIL:0 WARN:N)
```

---

## Phase 2: 성능 특성화 (Checkpoint B 이후)

- [ ] Task 7: batch_duration 탐색 (100s → 500s → 1000s → OOM 경계)
  - [ ] benchmark_gpu.py `GPU_HOURLY_RATE` DGX 기준으로 업데이트
  - [ ] SPEC.md §6 최적 batch_duration 기록

### Checkpoint C 검증

L40S 대비 throughput 비율 계산 및 기록.

---

## Phase 3: 풀 학습 (Checkpoint C 이후)

- [ ] Task 8: 3 epoch 학습 + checkpoint sweep
  - [ ] val_wer 감소 추세 확인 (NaN 없음)
  - [ ] CER on FLEURS ko-KR ≤ 7.12
  - [ ] RunPod 결과와 비교표 작성

---

## 파일 변경 요약

| 파일 | 상태 |
|------|------|
| `scripts/probe_dgx_environment.sh` | ✅ 신규 |
| `scripts/verify_on_dgx.sh` | ✅ 신규 |
| `scripts/setup_dgx.sh` | ✅ 신규 (probe 후 확정) |
| `scripts/train_pipeline.py` | ✅ 수정 (aarch64 + Blackwell) |
| `scripts/patch_numba_codegen.py` | ✅ 수정 (sysconfig + sm_100) |
| `scripts/benchmark_gpu.py` | ⏳ Task 7 시 수정 |
| `SPEC.md` §4 | ⏳ Task 5 시 업데이트 |
| `SPEC.md` §6 | ⏳ Task 7 시 업데이트 |
