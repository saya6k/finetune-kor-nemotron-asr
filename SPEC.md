# SPEC: Korean ASR Fine-Tuning on GB300 DGX Station

> **전제**: RunPod L40S 파이프라인은 2026-06-17에 검증 완료됨 (CLAUDE.md 참조).  
> 이 SPEC은 동일 파이프라인을 GB300 DGX Station(aarch64 Grace CPU + sm_100 Blackwell GPU)으로 포팅하는 탐구 문서다.

---

## 1. Objective

`nemotron-3.5-asr-streaming-0.6b` 한국어 fine-tuning 파이프라인을 GB300 DGX Station에서 동작시킨다.

**핵심 탐구 질문** (실행 전 알 수 없는 것들):

| 질문 | 가설 | 검증 방법 |
|------|------|-----------|
| RunPod 패치 9개 중 DGX에서 여전히 필요한 것은? | Numba PTX 패치·nv_one_logger stub은 불필요, 나머지는 유지 | 각각 skip 후 오류 여부 |
| NGC NeMo 컨테이너가 `EncDecRNNTBPEModelWithPrompt`를 포함하는가? | 최신 NGC라면 포함 가능 | `from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt` |
| sm_100(Blackwell)에서 WarpRNNT CUDA 커널이 컴파일되는가? | CUDA 12.8+이면 가능 | 첫 학습 스텝 통과 |
| aarch64에서 Numba PTX 버전은? | sm_100 → PTX 9.x, 다운그레이드 불필요 | `numba --sysinfo` + 학습 스텝 |
| 192GB HBM3e에서 최적 `batch_duration`은? | L40S 100s → DGX 1000s+ 가능 | OOM 경계 이진 탐색 |

### Success Criteria

| 기준 | 목표 |
|------|------|
| 학습 시작 (첫 스텝 완료, NaN 없음) | 필수 |
| CER on FLEURS ko-KR | ≤ 7.12 (RunPod 결과와 동등) |
| Throughput | L40S 대비 2× 이상 (HBM3e 대역폭 이점) |
| 학습 완료 시간 (3 epoch, 7,300h 데이터) | L40S 대비 ≤ 50% 시간 |

---

## 2. Platform 비교

| 항목 | RunPod (검증됨) | GB300 DGX Station (확인됨) |
|------|----------------|--------------------------|
| CPU 아키텍처 | x86_64 | **aarch64** (NVIDIA Grace ARM) |
| GPU 아키텍처 | Ada Lovelace sm_89 (L40S) | **Blackwell sm_103** (실측) |
| VRAM | 48GB GDDR6 | **276.5 GiB** (nominal 284 GiB, 실측) |
| CPU-GPU 연결 | PCIe | **NVLink-C2C** (통합 메모리) |
| 컨테이너 | runpod/pytorch (직접 pip 설치) | **`nvcr.io/nvidia/pytorch:26.05-py3` + `Dockerfile.dgx`** |
| CUDA toolkit | 12.8 (driver 12.4와 불일치) | Driver 595.58.03, toolkit 12.9 |
| nv_one_logger | 미포함 → stub 필요 | **미포함** (Dockerfile에서 stub 처리) |
| GPU 수 | 1 | 1 (단일 GPU) |
| GPU 공유 서비스 | 없음 | **DeepSeek-V4-Flash VLLM 상시 실행** (학습 전 조율 필요) |

---

## 3. 컨테이너 선택 (확정됨)

### 결론: `nvcr.io/nvidia/pytorch:26.05-py3` + `Dockerfile.dgx`

`nvcr.io/nvidia/nemo:latest`는 aarch64에서 `lightning` (PyTorch Lightning 2.x) 미포함 — 학습용 컨테이너가 아닌 추론용. `nvcr.io/nvidia/pytorch:26.05-py3`를 base로 커스텀 Dockerfile을 사용함 (이미지 태그: `nemotron-asr-dgx:latest`, 24.1 GiB).

**NeMo git main (3.1.0+8044a39)** 사용 이유: NVIDIA 공식 블로그에서 "NeMo 26.06+"을 요구하는데 NGC 컨테이너가 아직 미출시(2026-06-20 기준); git main이 동등.

### 빌드된 이미지

```bash
# DGX에서 한 번만 실행 (24.1 GiB, ~10분 소요)
cd /root/nemotron-asr-finetune/finetune-kor-nemotron-asr
docker build -f Dockerfile.dgx -t nemotron-asr-dgx:latest .
```

### 학습 실행 명령 (확정됨)

```bash
docker run --rm --gpus all --shm-size=64g --ulimit memlock=-1 --ipc=host \
  -v /root/nemotron-asr-finetune:/workspace \
  -e WORKSPACE=/workspace \
  -e NEMO_DIR=/opt/NeMo \
  -e HF_CKPT=/workspace/data/base_model/nemotron-3.5-asr-streaming-0.6b.nemo \
  -e HF_TOKEN="hf_..." \
  -e DATASETS_AUDIO_BACKEND=soundfile \
  -e SKIP_SETUP_INSTALL=1 \
  nemotron-asr-dgx:latest \
  python3 /workspace/finetune-kor-nemotron-asr/scripts/train_pipeline.py
```

**VLLM 조율 필수**: DGX에 DeepSeek-V4-Flash VLLM 서비스(`deepseek-v4-flash` 컨테이너)가 상시 실행 중이며 GPU 97%를 점유. 학습 전 서비스 담당자와 일정 조율 후 `docker stop deepseek-v4-flash` 필요.

**컨테이너 내부 경로 구조:**
```
/opt/NeMo/              ← NeMo git main (이미지에 고정, PYTHONPATH=/opt/NeMo)
/workspace/             ← 호스트 /root/nemotron-asr-finetune 마운트
├── finetune-kor-nemotron-asr/    ← 이 repo
├── data/
│   ├── base_model/nemotron-3.5-asr-streaming-0.6b.nemo
│   ├── processed/       ← 변환된 WAV + manifest JSON
│   └── wav_cache/       ← MP3→WAV 캐시
└── checkpoints/         ← 학습 체크포인트
```

**핵심 환경 변수:**
- `NEMO_DIR=/opt/NeMo` — 이미지에 포함된 NeMo main 경로
- `SKIP_SETUP_INSTALL=1` — 이미지에서 pip install 생략 (nemo_toolkit 다운그레이드 방지)
- `DATASETS_AUDIO_BACKEND=soundfile` — torchcodec aarch64 미지원 우회
- `HOLD_OUT_N=10` — 스모크 테스트 시 필수 (SMOKE_N < HOLD_OUT_N 기본값 1000이면 val=0)

**verify 결과:** `bash scripts/verify_on_dgx.sh` → 9/9 PASS (2026-06-20)

---

## 4. 패치 재분류 (RunPod 9개 → DGX 실측)

`nemotron-asr-dgx:latest` 이미지 기준 실측 결과 (2026-06-20).

| # | 패치 | RunPod 필요 이유 | DGX 실측 결과 | 처리 방식 |
|---|------|-----------------|---------------|-----------|
| 1 | Numba PTX 다운그레이드 (8.7→8.4) | CUDA toolkit 12.8 vs driver 12.4 불일치 | **NOT_NEEDED** (SM 10.3 ≥ 10, skip 로직 작동) | `train_pipeline.py` `_sm_major >= 10` 조건 자동 skip |
| 2 | nv_one_logger stub | PyPI 미포함 | **NEEDED** (pytorch NGC에도 미포함) | `Dockerfile.dgx`에서 `one_logger_callback.py` stub으로 교체 |
| 3 | Prompt model 파일 복사 | NeMo 2.7.3 pip 미포함 | **NEEDED** (NGC NeMo 컨테이너 방식 포기; NeMo git main 사용) | `Dockerfile.dgx`에서 `/opt/NeMo`에 NeMo git main 클론 |
| 4 | NeMo main 클론 + PYTHONPATH | pip vs main 불일치 | **NEEDED** (동일) | `PYTHONPATH=/opt/NeMo`, `SKIP_SETUP_INSTALL=1`로 재클론 방지 |
| 5 | WarpRNNT GPU 체크 | sm_89 커널 컴파일 | **TBD** (VLLM 점유로 첫 학습 스텝 미달성) | 학습 재시작 후 확인 |
| 6 | torchcodec 설치 | CUDA 버전 커플링 방지 | **NOT_NEEDED** (aarch64 `libtorchcodec_core4.so` 로드 실패) | `Dockerfile.dgx`에서 제외; `DATASETS_AUDIO_BACKEND=soundfile` 사용 |
| 7 | datasets Audio monkey-patch | soundfile 폴백 | **NEEDED** (env var 방식으로 대체) | `DATASETS_AUDIO_BACKEND=soundfile` 환경 변수 |
| 8 | Critical import 검증 | 파이프라인 전 검증 | **PASS** (9/9 verify 통과) | `verify_on_dgx.sh` 9/9 PASS 확인 |
| 9 | .nemo 모델 존재 확인 | 환경 완결성 | **PASS** | `/workspace/data/base_model/` 확인 완료 |

**smoke test 진행 결과 (SMOKE_N=100, HOLD_OUT_N=10)**:
- Step 1-5 (Setup → Tokenizer): PASS
- Step 6 (Fine-Tuning): **FAIL** — `torch.OutOfMemoryError: CUDA out of memory` (DeepSeek-V4-Flash VLLM이 GPU 277 GiB 점유)
- Issue #15799 (checkpoint save failure): **미검증** (OOM으로 미달성)

> **WarpRNNT sm_103 커널 컴파일 여부**: 첫 학습 스텝 완료 후 확인 예정 (VLLM 서비스 중단 후 재시도).

---

## 5. 환경 셋업 전략 (확정됨)

### 접근 방식: Dockerfile 기반 immutable 이미지

NGC 컨테이너 대신 `Dockerfile.dgx`로 모든 패치를 이미지에 고정.
`setup_dgx.sh`/`probe_dgx_environment.sh`는 Dockerfile 빌드 과정에서 대체됨 (이미지가 진실).

```bash
# 1. 이미지 빌드 (한 번만)
cd /root/nemotron-asr-finetune/finetune-kor-nemotron-asr
docker build -f Dockerfile.dgx -t nemotron-asr-dgx:latest .

# 2. 베이스 모델 다운로드 (한 번만)
docker run --rm \
  -v /root/nemotron-asr-finetune:/workspace \
  -e HF_TOKEN="hf_..." \
  nemotron-asr-dgx:latest \
  bash -c "mkdir -p /workspace/data/base_model && \
    hf download nvidia/nemotron-3.5-asr-streaming-0.6b \
      nemotron-3.5-asr-streaming-0.6b.nemo \
      --local-dir /workspace/data/base_model/"

# 3. verify (9/9 PASS 확인 후 학습 진입)
docker run --rm --gpus all \
  -v /root/nemotron-asr-finetune:/workspace \
  -e WORKSPACE=/workspace -e NEMO_DIR=/opt/NeMo \
  -e HF_CKPT=/workspace/data/base_model/nemotron-3.5-asr-streaming-0.6b.nemo \
  nemotron-asr-dgx:latest \
  bash /workspace/finetune-kor-nemotron-asr/scripts/verify_on_dgx.sh
```

### setup_environment.sh 수정 전략
- 기존 `setup_environment.sh`/`verify_on_pod.sh` 건드리지 않음 (RunPod 재현성 보존)
- DGX는 `Dockerfile.dgx` + `verify_on_dgx.sh`로 독립 관리

---

## 6. 학습 설정 (RunPod → DGX 변경점)

### 변경 필요 파라미터

| 파라미터 | RunPod | DGX (초기값) | 근거 |
|----------|--------|-------------|------|
| `trainer.devices` | `1` | `1` | 단일 GPU 검증 우선 |
| `trainer.precision` | `bf16` | `bf16` | 유지 |
| `model.train_ds.batch_duration` | `100` | `500` (시작점) | 276.5 GiB VRAM, 단계적 증가 |
| `model.train_ds.max_duration` | `20` | `20` | 유지 |
| `exp_manager.exp_dir` | `/workspace/data/checkpoints` | `/workspace/checkpoints` (호스트: `/root/nemotron-asr-finetune/checkpoints`) |

> **VLLM 조율 전제**: DeepSeek-V4-Flash VLLM 서비스(`deepseek-v4-flash` 컨테이너, `docker-compose` 관리)가 GPU 97% 점유. batch_duration 탐색 전 서비스 담당자 확인 후 `docker compose -f <경로> down`으로 중단 필요.

### batch_duration 탐색 계획 (Task 7, VLLM 중단 후 실행)
```
100  → 기준점 (L40S 검증값, WarpRNNT sm_103 커널 확인도 겸함)
500  → 첫 시도
1000 → 성공 시
2000 → 성공 시
OOM → 이진 탐색으로 최대값 확정
```

### 변경 없는 파라미터 (RunPod 검증값 유지)
- `lr=1e-4`, `max_epochs=3`, `gradient_clip_val=1.0`, `seed_everything=42`
- `warmup_ratio=0.05`, `warmup_steps=null`, `d_model=1024`
- `optim.name=adamw`, `weight_decay=0.001`
- `save_top_k=3`, `monitor=val_wer`, `patience=5`

> **커뮤니티 검증**: 타 언어 fine-tune 사례(§12) 분석 결과, 위 파라미터 조합은 지역사회 모델들과 일치하거나 더 보수적임. d_model 명시, early stopping, 언어 믹스 전략 모두 외부적으로 검증됨.  
> **선택적 실험**: Noam schedule(lr_scale=0.1~0.3)이 NbAiLab·LokaalHub에서 좋은 결과를 냄 — 풀 학습 성공 후 A/B 테스트 가치 있음.

---

## 7. 신규 작성 필요 스크립트

### `scripts/probe_dgx_environment.sh` (탐구 Phase 0)
패치 없이 환경 상태를 프로파일링하는 스크립트. 다음 정보를 출력:

```
[GPU]       이름, SM 버전, VRAM 용량
[CUDA]      toolkit 버전, driver 버전
[Python]    버전, 아키텍처
[NeMo]      pip 버전, EncDecRNNTBPEModelWithPrompt 포함 여부
[nv_one_logger] import 성공 여부
[Numba]     PTX 버전, GPU compute capability
[datasets]  Audio backend (torchcodec vs soundfile)
[PYTHONPATH] NeMo main 필요 여부
```

### `scripts/setup_dgx.sh` (최소 패치 셋)
`probe_dgx_environment.sh` 결과를 반영한 DGX 전용 셋업. 탐구 완료 후 작성.

### `scripts/verify_on_dgx.sh`
`verify_on_pod.sh`의 DGX 버전. 주요 변경:
- Python binary 탐지 (aarch64 NGC 컨테이너 경로 반영)
- GPU 체크: SM 버전 10.0(Blackwell) 확인 추가
- sentinel 경로: `/workspace` → DGX 로컬 NVMe

---

## 8. 검증 단계 (Phase별)

### Phase 0: 환경 확인 (사전 조건)
```
[ ] NGC 컨테이너 aarch64 지원 확인
[ ] GPU: sm_100 + VRAM 192GB 인식
[ ] Python 3.x (aarch64) 동작
[ ] CUDA toolkit = driver 버전 일치
```

### Phase 1: 패치 분류
```
[ ] nv_one_logger import 성공 여부
[ ] EncDecRNNTBPEModelWithPrompt import (패치 없이)
[ ] Numba: PTX 버전 확인, WarpRNNT 컴파일 성공
[ ] datasets Audio: 기본 backend 확인
[ ] ko-KR prompt embedding 지원 여부 확인 (미지원 시 nearest locale 결정 필요)
[ ] → probe_dgx_environment.sh 실행 결과로 확정
```

### Phase 2: 파이프라인 스모크 테스트
```
[ ] setup_dgx.sh 완료 (오류 없음)
[ ] SMOKE_N=100 으로 data ingest + manifest 생성
[ ] 학습 3 스텝 완료 (NaN 없음, OOM 없음)
[ ] eval_direct.py 실행 성공
```

### Phase 3: 성능 기준선
```
[ ] batch_duration 최적값 확정 (OOM 경계)
[ ] 100 스텝 throughput 측정 (samples/sec)
[ ] L40S 대비 throughput 비교
[ ] 3 epoch 예상 완료 시간 계산
```

### Phase 4: 풀 학습 + 평가
```
[ ] 3 epoch 학습 완료
[ ] checkpoint sweep eval (6 datasets)
[ ] CER on FLEURS ko-KR ≤ 7.12
[ ] RunPod 결과와 수치 비교
```

---

## 9. 알려진 리스크

| 리스크 | 가능성 | 대응 |
|--------|--------|------|
| **DeepSeek-V4-Flash VLLM 서비스 스케줄 충돌** | **높음 (실제 발생)** | 서비스 담당자와 학습 일정 조율; `docker compose -f <경로> down` 후 실행 |
| WarpRNNT sm_103 미지원 | 중간 | NeMo 업스트림 이슈 확인; VLLM 중단 후 첫 스텝에서 확인 |
| Issue #15799 (checkpoint save failure) | 중간 | 첫 학습 완료 후 체크포인트 존재 여부 확인; 실패 시 NeMo 이슈 추적 |
| `EncDecRNNTBPEModelWithPrompt` NeMo 버전 변경 | 낮음 | `Dockerfile.dgx`에서 `--depth 1 --branch main` 고정; 재빌드로 업데이트 가능 |
| NeMo git main 불안정 (depth=1) | 낮음 | 특정 커밋 pin (`ARG NEMO_COMMIT=<sha>`)으로 고정 가능 |
| Numba aarch64 호환 문제 | **NOT_NEEDED** (SM 10.3 자동 skip) | 해결됨 |
| `ko-KR` prompt locale 미지원 | 낮음 | tokenizer 검증(Step 5) PASS — 지원 확인됨 |

---

## 10. 프로젝트 구조 변경

기존 구조 유지, DGX 전용 파일 추가:

```
scripts/
  setup_environment.sh        # RunPod용 (유지, 수정 금지)
  setup_dgx.sh                # DGX용 (Phase 1 결과로 신규 작성)
  probe_dgx_environment.sh    # Phase 0 탐구 (신규 작성)
  verify_on_pod.sh            # RunPod용 (유지)
  verify_on_dgx.sh            # DGX용 (신규 작성)
  train_pipeline.py           # 공통 (수정 최소화)
  eval_direct.py              # 공통 (수정 최소화)
```

> RunPod 검증 파일(`setup_environment.sh`, `verify_on_pod.sh`)은 건드리지 않음.  
> DGX 전용 파일을 별도로 추가.

---

## 11. Boundaries

### Always Do
- RunPod 검증 스크립트 보존 (DGX 파일과 공존)
- 각 Phase 결과를 이 SPEC에 기록 (TBD → NEEDED/NOT_NEEDED)
- `bf16` precision 유지
- 학습 파라미터 (lr, epochs, etc.) RunPod 검증값에서 시작

### Ask First
- 멀티-GPU (DDP) 전환 시점
- NGC 컨테이너 버전 고정 여부 (latest vs 특정 버전 pin)
- `/raid` vs 다른 스토리지 경로 선택
- Blackwell용 최적화 (TF32, CUDA graphs) 활성화 여부

### Never Do
- `setup_environment.sh`, `verify_on_pod.sh` 수정 (RunPod 재현성 보존)
- `limit_train_batches` 설정 (NeMo Issue #15782)
- `max_steps`와 `max_epochs` 동시 설정
- 학습 파라미터를 RunPod 검증값에서 이유 없이 변경

---

## 12. 타 언어 Fine-Tune 사례 시사점

2026-06-20 기준, 커뮤니티에서 공개된 nemotron-3.5-asr-streaming-0.6b fine-tune 모델 8개를 분석한 결과.

### 분석 대상 모델

| 모델 | 언어 | 데이터 | WER 개선 | 특이사항 |
|------|------|--------|----------|----------|
| NbAiLab/nb-sami-asr-north-nemotron35 | North Sami | 390h | val WER 28.88% | lr=0.3 Noam, 미지원 문자를 그리스 문자 ID에 remapping |
| LokaalHub/frisian-asr-streaming-0.6b | 서부 프리지안 | 40h | 82%→20% (−77%) | lr=0.1 Noam, Dutch 슬롯 사용, d_model=512 명시 |
| LokaalHub/nemotron-3.5-sv-SE | 스웨덴어 | 26h | 40%→18% (−54%) | bf16, NoamAnnealing |
| LokaalHub/nemotron-3.5-cy | 웨일스어 | 50h | 99%→22% (−77%) | English 슬롯 사용 (cy 미지원) |
| LokaalHub/nemotron-3.5-da | 덴마크어 | 10h | 36%→16% (−55%) | **단 10시간** |
| LokaalHub/nemotron-3.5-nb-NO | 노르웨이 보크몰 | 87h | 76%→17% (−77%) | nb-NO 전용 슬롯 |
| GaborMadarasz/nemotron-3.5-asr-finetuned-CV24 | 헝가리어 | CV24 | 미공개 | **lr=1e-4 + AdamW** (우리와 동일), fp32 |
| NbAiLab/nb-sami-asr-north-nemotron-lr030-step239000 | North Sami | 390h | — | 위 모델 step 239k 체크포인트 |

### 우리 설정이 외부적으로 확인된 항목

- **lr=1e-4 + AdamW**: Hungarian 모델과 동일 조합 — 안전한 선택 확인
- **d_model=1024 명시**: Frisian이 d_model=512 명시 → OmegaConf interpolation 버그 회피가 업계 관행
- **early stopping (patience=5)**: NbAiLab에서 epoch 38 최적, 이후 WER 재상승 확인 → patience 설정 필수
- **언어 믹스 (ko 80% + en/ja/zh 20%)**: 다른 모델들은 단일 언어만 학습 → 우리가 더 보수적
- **batch_duration=100s on 48GB**: Frisian은 A100 80GB에서 200s 사용 → L40S 48GB 100s 적절

### 주의 사항

- **ko-KR locale 지원 여부**: Welsh(cy)→English, Frisian(fy-NL)→Dutch, North Sami→Finnish 슬롯 재사용 사례 존재. ko-KR이 베이스 모델에서 지원되는지 Phase 1에서 명시적 확인 필요.
- **Noam schedule 실험 가치**: NbAiLab(lr=0.3), LokaalHub(lr=0.1) 모두 Noam schedule 사용 → 풀 학습 성공 후 A/B 실험 고려.
- **데이터 규모 여유**: 단 10h로 55% 개선(Danish) → 우리 7,300h는 충분. 추가 데이터보다 품질 필터링이 더 중요할 수 있음.

---

## References

- [NVIDIA DGX Station 제품 페이지](https://www.nvidia.com/ko-kr/products/workstations/dgx-station/)
- [NGC NeMo 컨테이너](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)
- [Blackwell Architecture (sm_100) — CUDA Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [NeMo GitHub — main branch](https://github.com/NVIDIA/NeMo)
- [RunPod 검증 파이프라인 — CLAUDE.md](./CLAUDE.md) (2026-06-17 검증됨)
