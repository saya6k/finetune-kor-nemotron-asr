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

---

# Round 2: AIHub 데이터셋 확장 및 DGX 전층 동결 학습

> 작성일: 2026-06-21  
> 이전 라운드 대비 변경: 데이터셋 4종 추가, 텍스트 정규화 수정, 오디오 세그멘테이션, DGX 학습 설정 변경

---

## R2-1. 데이터셋 확장

### 배경

Round 1은 Emilia-YODAS 한국어 스트리밍 + LibriSpeech(EN) + AISHELL(ZH) 구성이었음. Round 2는 AIHub 한국어 데이터셋 4종을 추가해 총 877,704발화 / 1,396h로 확장.

JA/DE/FR Emilia 데이터는 확보 실패 (TAR 삭제 후 WAV 경로 소실) — Round 3에서 재다운로드 예정.

### 학습 데이터 구성 (Round 2 최종)

| 데이터셋 | 언어 | 소스 | 발화 수 | 시간 | 평균 길이 |
|---------|------|------|--------|------|---------|
| KsponSpeech | ko-KR | AIHub | 618,720 | 957.7h | 5.6s |
| 회의음성 (Meeting) | ko-KR | AIHub | 81,413 | 150.0h | 6.6s |
| FreeTalk Studio | ko-KR | AIHub | 58,210 | 93.9h | 5.8s |
| FreeTalk Field | ko-KR | AIHub | 47,936 | 74.7h | 5.6s |
| 극한소음 (Extreme Noise) | ko-KR | AIHub | 6,546 | 19.8h | 10.9s |
| LibriSpeech (clean-100) | en-US | OpenSLR | 24,879 | 50.0h | 7.2s |
| AISHELL-1 | zh-CN | AISHELL | 40,000 | 50.3h | 4.5s |
| **합계** | | | **877,704** | **1,396.4h** | **5.7s** |

언어 비율: ko-KR 92.6% (812,825), zh-CN 4.6% (40,000), en-US 2.8% (24,879)

manifest 파일: `/workspace/data/processed/train_manifest_round2.jsonl`

---

## R2-2. 텍스트 정규화

### AIHub 한국어 주석 형식

AIHub 데이터셋은 `(표기형)/(발음형)` 이중 표기 형식을 사용함. 학습에는 **발음형(spoken form)** 을 사용해야 함.

**주의**: 데이터셋마다 순서가 반대임.

| 데이터셋 | 형식 | 발음형 위치 | 예시 |
|---------|------|-----------|------|
| KsponSpeech | `(표기)/(발음)` | 오른쪽 | `(이십 대)/(20대)` → `20대` |
| Extreme Noise | `(표기)/(발음)` | 오른쪽 | KsponSpeech와 동일 |
| 회의음성 (Meeting) | `(발음)/(표기)` | **왼쪽** | `(이십 대)/(20대)` → `이십 대` |

### KsponSpeech / Extreme Noise 정규화 (`convert_ksponspeech.py`, `_normalize()`)

```python
text = re.sub(r"\([^)]+\)/\(([^)]+)\)", r"\1", text)  # (표기)/(발음) → 발음
text = re.sub(r"\s?[bnluo]/\s?", " ", text)            # b/ n/ l/ u/ o/ 비언어 마커
text = re.sub(r"\s?\+/\s?", " ", text)                 # +/ 연장 마커
text = re.sub(r"\S+\+", "", text)                       # word+ 잘린 발화
text = re.sub(r"\S*\*\S*", "", text)                   # word*, *word, * 독립 마커
text = re.sub(r"(\S)/", r"\1", text)                   # 그/ → 그 (발화 겹침)
```

**핵심 버그 이력**: `\S+\*` → `\S*\*\S*` 수정 (2026-06-21)
- `\S+\*`는 `word*` 형태만 처리, `*아니야`(접두) / ` * `(독립) 누락
- 수정 후 KsponSpeech 618,720발화 중 추가 333건 변경됨

### 회의음성 정규화 (`convert_meeting.py`, `_normalize()`)

```python
text = re.sub(r"\(([^)]+)\)/\([^)]+\)", r"\1", text)  # (발음)/(표기) → 발음 (순서 주의!)
text = re.sub(r"/\s*\([^)]+\)", "", text)              # /(bgm), /(noise) 배경음 태그
text = re.sub(r"@\S+", "", text)                       # @이름N 화자 태그
```

**순서 중요**: `(A)/(B)→A` 추출 먼저, `/(bgm)` 제거 나중. 반대로 하면 `(이십 대)/(20대)`에서 `/(20대)`가 먼저 제거되어 `(이십 대)` 잔존.

### 정규화 후처리 도구

기존 manifest 재정규화가 필요한 경우:
```bash
python3 scripts/normalize_manifests.py --type ksponspeech data/processed/ksponspeech_train.jsonl
python3 scripts/normalize_manifests.py --type meeting data/processed/meeting_train.jsonl
```

---

## R2-3. 오디오 세그멘테이션

### 문제: max_duration=20으로 전체 녹음 파일 필터링

NeMo 학습 설정 `max_duration=20`은 20초 초과 발화를 드롭함. AIHub 원본 파일은 녹음 단위가 매우 길어 직접 사용 불가:

| 데이터셋 | 원본 평균 길이 | 문제 |
|---------|-------------|------|
| 극한소음 | ~830s | 전량 필터링 |
| 회의음성 | ~2,372s | 전량 필터링 |

세그멘테이션으로 해결: 타임스탬프 기반으로 개별 발화를 분리해 1-20s 클립으로 저장.

### 회의음성 세그멘테이션 (`convert_meeting.py`)

- 레이블: JSON utterance 배열, 각 utterance에 `start`/`end` float 필드
- 처리: soundfile로 전체 WAV 로드 → 타임스탬프 슬라이싱 → 개별 WAV 저장
- 결과: 673 녹음 파일 → **81,413 발화** (1-20s)
- 출력: `/workspace/data/wav_cache/meeting/segments/*.wav`

```python
# MIN_DUR=1.0, MAX_DUR=20.0
for utt_id, start, end, text in segments:
    clip = audio[int(start*sr):int(end*sr)]
    sf.write(clip_path, clip, sr)
```

**실행 환경**: 호스트에 soundfile 없음 → Docker 컨테이너 내에서 실행해야 함
```bash
docker run --rm -v /root/nemotron-asr-finetune:/workspace nemotron-asr-dgx:latest \
  python3 /workspace/finetune-kor-nemotron-asr/scripts/convert_meeting.py \
  --label-zip /workspace/data/raw/meeting/TL5.zip \
  --source-zip /workspace/data/raw/meeting/TS5.zip \
  --wav-cache /workspace/data/wav_cache/meeting \
  --output /workspace/data/processed/meeting_train.jsonl \
  --max-hours 150
```

### 극한소음 세그멘테이션 (`convert_extreme_noise.py`)

- 레이블: 중첩 ZIP 구조 (`labeling.zip` → 24개 서브 ZIP → SRT + JSON 쌍)
- SRT 타임스탬프 파싱: `HH:MM:SS,mmm --> HH:MM:SS,mmm`
- 처리: SRT segments 파싱 → JSON `mediaUrl`로 WAV 매칭 → 슬라이싱
- 결과: 440 녹음 파일 → **6,546 발화** (0.5-20s)
- 출력: `/workspace/data/wav_cache/extreme_noise/segments/*.wav`

```python
# labeling.zip → sub_zip → (srt_map, json_map) 매칭
with zipfile.ZipFile(label_zip_path) as outer:
    for sub_name in sub_zips:  # 24개 서브 ZIP
        with zipfile.ZipFile(io.BytesIO(outer.read(sub_name))) as inner:
            # SRT 파싱 후 JSON mediaUrl로 WAV basename 매핑
```

---

## R2-4. 평가 데이터셋 (Round 2)

| # | 이름 | 용도 | 소스 |
|---|------|------|------|
| 1 | `val_emilia_holdout_ko` | **Validation / Early Stopping** | Emilia-YODAS KO holdout |
| 2 | `test_fleurs_ko` | Test — 주요 KO 벤치마크 | FLEURS ko_kr |
| 3 | `test_emilia_holdout_ko` | Test — val과 분리된 holdout | Emilia-YODAS KO holdout B |
| 4 | `test_zeroth_ko` | Test | Zeroth Korean |
| 5 | `test_mixed_en` | Test | Emilia-YODAS EN |
| 6 | `test_mixed_ja` | Test | Emilia-YODAS JA |
| 7 | `test_mixed_zh` | Test | Emilia-YODAS ZH |
| 8 | `test_fleurs_fr` | Test — 학습 언어 WER | FLEURS fr_fr |
| 9 | `test_fleurs_de` | Test — 학습 언어 WER | FLEURS de_de |
| 10 | `test_fleurs_ru` | **Catastrophic forgetting 프로브** (비학습) | FLEURS ru_ru |

val set: `val_ksponspeech_500.jsonl` (KsponSpeech 에서 무작위 500발화 — 학습 외 보유)  
eval config: `/workspace/data/processed/eval_datasets_round2.json`

---

## R2-5. DGX 학습 설정 (finetune_with_freeze.py)

### 레이어 동결 전략

Round 2는 `finetune_with_freeze.py` 사용. FastConformer 인코더 하위 8 레이어를 동결해 pretrained 표현 보존.

```
encoder.layers[0:8] frozen (208 tensors) | trainable: 447/655
```

- 동결: 인코더 하위 8층 (conv subsampling + 초기 conformer blocks)
- 학습 가능: 상위 16층 + 디코더(RNNT) + joint network

### 학습 명령 (Round 2 확정)

```bash
python3 /workspace/finetune-kor-nemotron-asr/scripts/finetune_with_freeze.py \
  --config-path=../asr/conf/fastconformer/cache_aware_streaming \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_nemo_model=/workspace/data/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming.nemo \
  ++model.train_ds.manifest_filepath=/workspace/data/processed/train_manifest_round2.jsonl \
  ++model.validation_ds.manifest_filepath=/workspace/data/processed/val_ksponspeech_500.jsonl \
  ++model.tokenizer.dir=/workspace/data/checkpoints \
  ++trainer.devices=1 \
  ++trainer.max_epochs=10 \
  ++trainer.precision=bf16 \
  ++trainer.gradient_clip_val=1.0 \
  ++seed_everything=42 \
  ++model.train_ds.batch_duration=5000 \
  ++model.validation_ds.batch_size=1 \
  ++model.train_ds.max_duration=20 \
  ++model.optim.name=adamw \
  ++model.optim.lr=0.00005 \
  ++model.optim.weight_decay=0.001 \
  ++model.optim.sched.name=CosineAnnealing \
  ++model.optim.sched.warmup_ratio=0.05 \
  ++model.optim.sched.warmup_steps=null \
  ++model.optim.sched.min_lr=1e-6 \
  ~model.optim.sched.d_model \
  ++exp_manager.exp_dir=/workspace/data/checkpoints_round2 \
  ++exp_manager.resume_if_exists=true \
  ++exp_manager.resume_ignore_no_checkpoint=true \
  ++exp_manager.create_early_stopping_callback=true \
  ++exp_manager.early_stopping_callback_params.monitor=val_wer \
  ++exp_manager.early_stopping_callback_params.patience=10 \
  ++exp_manager.checkpoint_callback_params.save_top_k=-1 \
  ++exp_manager.checkpoint_callback_params.monitor=val_wer \
  ++exp_manager.checkpoint_callback_params.save_last=true \
  ++exp_manager.checkpoint_callback_params.every_n_train_steps=250 \
  ++exp_manager.checkpoint_callback_params.every_n_epochs=null \
  ++trainer.val_check_interval=250
```

### Round 1 → Round 2 파라미터 변경

| 파라미터 | Round 1 | Round 2 | 이유 |
|---------|---------|---------|------|
| 스크립트 | `speech_to_text_finetune.py` | `finetune_with_freeze.py` | 레이어 동결 |
| `lr` | 1e-4 | **5e-5** | 더 많은 데이터로 보수적 학습률 |
| `batch_duration` | 100s | **5000s** | GB300 VRAM 284GB 활용 |
| `max_epochs` | 20 | **10** | 조기 수렴 예상 |
| `val_check_interval` | 500 | **250** | 더 촘촘한 체크포인트 |
| `every_n_train_steps` | 500 | **250** | 동일 |
| `d_model` 처리 | `++...d_model=1024` | `~model.optim.sched.d_model` | NeMo 3.x Hydra delete |
| validation set | `val_emilia_holdout_ko` | **`val_ksponspeech_500`** | 학습 외 보유 세트 |

### 예상 step 수

총 학습 데이터: 877,704발화 × 5.7s평균 ≈ 5,003,000s  
Steps/epoch (batch_duration=5000): 5,003,000 / 5000 ≈ **1,001 steps**  
체크포인트 간격: 250 steps (≈ 에폭의 25%)

---

## R2-6. 평가 파이프라인 (eval_watcher.py + eval_one.py)

### eval_watcher.py

체크포인트 디렉토리를 감시하다가 새 `.ckpt`/`.nemo` 파일이 생기면 자동으로 eval_one.py를 실행.

```bash
python3 /workspace/finetune-kor-nemotron-asr/scripts/eval_watcher.py \
  --checkpoint-dir /workspace/data/checkpoints_round2 \
  --datasets-json /workspace/data/processed/eval_datasets_round2.json \
  --output-csv /workspace/results/eval_rolling_round2.csv
```

### eval_one.py

단일 체크포인트 × 복수 데이터셋 평가. CER/WER/SER 산출 후 CSV append.

```bash
python3 /workspace/finetune-kor-nemotron-asr/scripts/eval_one.py \
  --ckpt <path>.nemo \
  --datasets-json /workspace/data/processed/eval_datasets_round2.json \
  --output-csv /workspace/results/eval_rolling_round2.csv \
  --device cuda:0
```

결과 파일: `/workspace/results/eval_rolling_round2.csv`

---

## R2-7. 초기 실험 결과 (참고용)

> 아래는 Round 2 이전 별도 실험 (작은 데이터셋, batch_duration=100)에서 epoch 0-12 학습 결과.  
> Round 2 공식 결과는 `eval_rolling_round2.csv` 에 누적 중.

### Baseline (pretrained, fine-tune 없음)

| 데이터셋 | CER | WER | SER |
|---------|-----|-----|-----|
| val_emilia_holdout_ko | 15.6% | 34.9% | 96.9% |
| test_fleurs_ko | 10.3% | 29.2% | 100.0% |
| test_zeroth_ko | 11.3% | 36.4% | 100.0% |
| test_mixed_en | 7.4% | 17.2% | 90.8% |
| test_mixed_ja | 23.9% | 96.6% | 92.6% |
| test_mixed_zh | 36.6% | 140.0% | 97.6% |
| test_fleurs_fr | 8.6% | 26.0% | 100.0% |
| test_fleurs_de | 10.8% | 46.0% | 100.0% |
| test_fleurs_ru | 7.3% | 31.9% | 100.0% |

모델카드 기준: CER 7.12 on FLEURS KO (1.12s chunk, LangID mode)

### Best checkpoint 결과 (val_wer=0.3182, epoch=12)

| 데이터셋 | CER | WER | Δ WER |
|---------|-----|-----|-------|
| val_emilia_holdout_ko | 14.4% | 32.0% | −2.9% |
| test_fleurs_ko | 10.0% | 28.2% | −1.0% |
| test_zeroth_ko | 12.0% | 36.5% | ~0% |
| test_mixed_en | 5.1% | 12.8% | **−4.4%** |
| test_mixed_ja | 19.8% | 87.8% | −8.8% |
| test_mixed_zh | 36.5% | 102.6% | −37.4% |
| test_fleurs_fr | 7.5% | 26.5% | +0.5% |
| test_fleurs_de | 9.5% | 46.0% | ~0% |
| test_fleurs_ru | 9.1% | 35.0% | **+3.1% (forgetting!)** |

**관찰**:
- 한국어 개선 미미 (FLEURS KO WER: 29.2% → 28.2%)
- 러시아어 forgetting 발생 (+3.1 WER) — JA/DE/FR 학습 데이터 미포함이 원인
- JA/ZH는 학습 데이터 없어도 WER 하락 (모델 내 잠재 능력 회복으로 추정)
- Epoch 12 마지막 체크포인트(`val_wer=0.3163`)는 CER/WER=98% 붕괴(collapse) — patience=10 조기 중단 필요성 확인

---

## R2-8. 미해결 과제

| 항목 | 상태 | 비고 |
|------|------|------|
| JA/DE/FR Emilia 데이터 재다운로드 | 미완료 | TAR 삭제로 WAV 소실, Round 3 재수집 필요 |
| Forgetting 완화 (RU) | 미완료 | RU 데이터 추가 또는 regularization 필요 |
| 회의음성 정규화 오염 (1차 pass) | 저우선순위 | 원본 녹음 파일은 max_duration=20 초과로 학습에 미사용 |
| 극한소음 추가 확보 | 고려 | 현재 19.8h — 더 많은 zip 파일 처리 가능 |
| val set 대표성 | 검토 필요 | val_ksponspeech_500은 read-speech; 회의/노이즈 환경 반영 안됨 |
