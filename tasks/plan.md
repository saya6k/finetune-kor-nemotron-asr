# Implementation Plan: GB300 DGX Station Porting (v1)

> RunPod L40S 파이프라인 검증 완료 (2026-06-17). 이 계획은 GB300 DGX Station 포팅을 다룬다.

## Architecture Decisions

- **RunPod 스크립트 수정 금지**: `setup_environment.sh`, `verify_on_pod.sh`는 건드리지 않음. DGX 전용 파일 별도 추가.
- **probe 먼저**: 패치 필요 여부를 추측하지 않고 `probe_dgx_environment.sh`로 실측.
- **최소 수정 원칙**: 기존 스크립트는 아키텍처 감지 로직만 추가.

## Dependency Graph

```
[Task 0] probe_dgx_environment.sh  ─┐
[Task 1] train_pipeline.py 패치     ├── 오프라인, 순서 무관
[Task 2] patch_numba_codegen.py 패치 ├──
[Task 3] verify_on_dgx.sh          ─┤
[Task 4] setup_dgx.sh 스켈레톤     ─┘
          │
          ▼ [Checkpoint A] — DGX 접속 필요
[Task 5] probe 실행 → SPEC.md §4 업데이트
          │
          ▼
[Task 6] setup_dgx.sh + verify_on_dgx.sh 통과
          │
          ▼ [Checkpoint B]
[Task 7] batch_duration 탐색 (100→500→1000→OOM)
          │
          ▼ [Checkpoint C]
[Task 8] 풀 학습 3 epoch + CER 평가
```

## Phase 0: 오프라인 준비

### Task 0: `probe_dgx_environment.sh`
**Status:** ✅ 완료

8개 항목 점검 (GPU/CUDA/Python/NeMo/nv_one_logger/Numba/datasets/PYTHONPATH).
각 항목에 NEEDED/NOT_NEEDED/UNKNOWN 판정 출력.
결과를 `${WORKSPACE}/probe_results.txt`에 저장.

**Verify:** `bash -n scripts/probe_dgx_environment.sh`

---

### Task 1: `train_pipeline.py` aarch64 + Blackwell 수정
**Status:** ✅ 완료

- Lines 178-185: `_sm_major >= 10`이면 Numba PTX 패치 skip
- Line 724: `x86_64-linux` → `_arch` (동적 `platform.machine()` 기반)

**Verify:** `grep "aarch64-linux\|_sm_major >= 10" scripts/train_pipeline.py`

---

### Task 2: `patch_numba_codegen.py` Blackwell 대응
**Status:** ✅ 완료

- sm_100 감지 시 exit 0 (패치 불필요)
- 하드코딩된 Python 3.12 경로 → `sysconfig.get_paths()["purelib"]` 동적 감지
- 이미 패치된 경우 건너뜀 (idempotent)

**Verify:** `python3 scripts/patch_numba_codegen.py` (GPU 없어도 exit 0)

---

### Task 3: `verify_on_dgx.sh`
**Status:** ✅ 완료

`verify_on_pod.sh` 기반 DGX 버전:
- Step 1: SM 10.0(Blackwell) 확인 추가
- Step 2: `setup_dgx.sh` 실행
- Step 5: `aarch64-linux`, `_sm_major >= 10` 파라미터 포함 확인

**Verify:** `bash -n scripts/verify_on_dgx.sh`

---

### Task 4: `setup_dgx.sh` 스켈레톤
**Status:** ✅ 완료

RunPod 9개 패치의 DGX 조건부 버전:

| # | 패치 | Skip 조건 |
|---|------|-----------|
| 1 | Numba PTX | sm_major >= 10 |
| 2 | nv_one_logger stub | `import nv_one_logger` 성공 시 |
| 3 | Prompt model 파일 | `import EncDecRNNTBPEModelWithPrompt` 성공 시 |
| 4 | NeMo main clone | (3번과 연동) |
| 5 | WarpRNNT GPU 체크 | 항상 실행 |
| 6 | torchcodec | 이미 설치된 경우 skip |
| 7 | datasets Audio | 항상 실행 |
| 8 | Critical imports | 항상 실행 |
| 9 | .nemo 모델 | 항상 확인 |

**Verify:** `bash -n scripts/setup_dgx.sh`

---

### Checkpoint A: Phase 0 완료 기준

```
[ ] bash -n scripts/probe_dgx_environment.sh
[ ] bash -n scripts/verify_on_dgx.sh
[ ] bash -n scripts/setup_dgx.sh
[ ] python3 scripts/patch_numba_codegen.py  → exit 0 (GPU 없이)
[ ] grep "aarch64-linux" scripts/train_pipeline.py
[ ] grep "_sm_major >= 10" scripts/train_pipeline.py
```

---

## Phase 1: 온디바이스 탐색 (DGX 접속 필요)

### Task 5: probe 실행 → SPEC.md §4 업데이트
**Status:** ⏳ 대기 (DGX 접속 필요)

```bash
bash scripts/probe_dgx_environment.sh
cat ${WORKSPACE}/probe_results.txt
```
결과로 SPEC.md §4 표의 TBD를 NEEDED/NOT_NEEDED로 채운다.

---

### Task 6: setup_dgx.sh + verify_on_dgx.sh 통과
**Status:** ⏳ 대기

```bash
bash scripts/setup_dgx.sh
bash scripts/verify_on_dgx.sh
```
verify 7/7 PASS + SMOKE_N=100 학습 3 스텝 완료.

---

### Checkpoint B: 학습 진입 기준

```
[ ] verify_on_dgx.sh 7/7 PASS
[ ] SMOKE_N=100 학습 3 스텝 완료 (NaN 없음)
[ ] WarpRNNT sm_100 커널 컴파일 성공
```

---

## Phase 2: 성능 특성화

### Task 7: batch_duration 최적값 탐색
**Status:** ⏳ 대기 (Checkpoint B 이후)

```
100s → 기준점
500s → 첫 시도
1000s → 성공 시
2000s → 성공 시
OOM → 직전 값이 최적
```

---

### Checkpoint C: 풀 학습 기준

```
[ ] 최적 batch_duration 확정
[ ] L40S 대비 throughput 비율 기록
[ ] SPEC.md §6 파라미터 업데이트
```

---

### Task 8: 풀 학습 3 epoch + 평가
**Status:** ⏳ 대기 (Checkpoint C 이후)

3 epoch 완료 → checkpoint sweep (6 datasets) → CER on FLEURS ko-KR ≤ 7.12.

---

## Risks

| 리스크 | 확률 | 대응 |
|--------|------|------|
| WarpRNNT sm_100 미지원 | 중 | NeMo 업그레이드; GitHub Issue |
| NGC aarch64 미지원 | 낮 | pytorch NGC + 수동 NeMo 설치 |
| EncDecRNNTBPEModelWithPrompt NGC 미포함 | 중 | PYTHONPATH + NeMo main (RunPod 방식) |
| batch_duration OOM 불확실 | 중 | 이진 탐색 |
