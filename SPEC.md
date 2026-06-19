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

| 항목 | RunPod (검증됨) | GB300 DGX Station (탐구) |
|------|----------------|--------------------------|
| CPU 아키텍처 | x86_64 | **aarch64** (NVIDIA Grace ARM) |
| GPU 아키텍처 | Ada Lovelace sm_89 (L40S) | **Blackwell sm_100** |
| VRAM | 48GB GDDR6 | **~192GB HBM3e** (예상, 모델별 상이) |
| CPU-GPU 연결 | PCIe | **NVLink-C2C** (통합 메모리) |
| 컨테이너 | runpod/pytorch (직접 pip 설치) | **NGC NeMo 공식 컨테이너** |
| CUDA toolkit | 12.8 (driver 12.4와 불일치) | 일치 예상 (DGX 관리형) |
| nv_one_logger | 미포함 → stub 필요 | NGC 컨테이너에 포함 추정 |
| GPU 수 | 1 | 1 (초기 검증) |

---

## 3. NGC 컨테이너 선택

### 조건
- aarch64 멀티아치 빌드 지원
- NeMo ≥ 2.8 (Blackwell sm_100 지원 확인 필요)
- CUDA ≥ 12.8 (Blackwell 공식 지원 최소 버전)
- PyTorch ≥ 2.6 (기존 검증 버전)

### 탐색 순서
```bash
# 1. aarch64 지원 여부 확인
docker manifest inspect nvcr.io/nvidia/nemo:latest | jq '.manifests[].platform'

# 2. GPU 인식 + Blackwell 확인
python3 -c "
import torch
p = torch.cuda.get_device_properties(0)
print(f'GPU: {p.name}')
print(f'SM: {p.major}.{p.minor}')   # Blackwell → 10.0
print(f'VRAM: {p.total_memory / 1e9:.1f} GB')
"

# 3. NeMo 버전 확인
python3 -c "import nemo; print(nemo.__version__)"
```

### 후보 컨테이너 (2026-06 기준)
```
nvcr.io/nvidia/nemo:latest           # 최우선 시도
nvcr.io/nvidia/nemo:25.XX.XX         # 특정 릴리즈 (TBD)
nvcr.io/nvidia/pytorch:25.XX-py3     # NeMo 없이 PyTorch만
```

### 컨테이너 실행 명령

```bash
docker run --gpus all -it --rm \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ipc=host \
  -v /raid/workspace:/workspace \
  -v /raid/hf_cache:/root/.cache/huggingface \
  -e WORKSPACE=/workspace \
  -e HF_TOKEN="hf_..." \
  -e DATASETS_AUDIO_BACKEND=soundfile \
  nvcr.io/nvidia/nemo:latest \
  bash
```

**호스트 사전 준비:**
```bash
mkdir -p /raid/workspace /raid/hf_cache
```

**컨테이너 내부 경로 구조:**
```
/workspace/                                          ← WORKSPACE 루트
├── finetune-kor-nemotron-asr/                       ← 이 repo (git clone)
├── data/base_model/nemotron-3.5-asr-streaming-0.6b.nemo
├── NeMo/                                            ← setup_dgx.sh이 필요 시 clone
├── .setup_dgx_done                                  ← sentinel (idempotent)
└── probe_results.txt                                ← probe 출력

/root/.cache/huggingface/                            ← HF 캐시 (재다운 방지)
```

**컨테이너 진입 후 첫 실행 순서:**
```bash
git clone <repo> /workspace/finetune-kor-nemotron-asr
cd /workspace/finetune-kor-nemotron-asr

# 베이스 모델 다운로드 (최초 1회)
mkdir -p /workspace/data/base_model
huggingface-cli download nvidia/nemotron-3.5-asr-streaming-0.6b \
  nemotron-3.5-asr-streaming-0.6b.nemo \
  --local-dir /workspace/data/base_model/

# 환경 프로파일링 (패치 전)
bash scripts/probe_dgx_environment.sh
```

**플래그 설명:**
- `--shm-size=64g` — NeMo DataLoader shared memory 요구
- `--ipc=host` — 멀티프로세스 DataLoader 필요
- `-v /raid/workspace:/workspace` — DGX NVMe RAID → 컨테이너 workspace
- `-v /raid/hf_cache:/root/.cache/huggingface` — HF 캐시 영속화

---

## 4. 패치 재분류 (RunPod 9개 → DGX 필요 여부)

RunPod `setup_environment.sh`의 각 패치를 DGX Station에서 재검증한다.

| # | 패치 | RunPod 필요 이유 | DGX 예상 상태 | 검증 방법 |
|---|------|-----------------|---------------|-----------|
| 1 | Numba PTX 다운그레이드 (8.7→8.4) | CUDA toolkit 12.8 vs driver 12.4 불일치 | **불필요** (toolkit/driver 일치) | 패치 없이 학습 스텝 실행 |
| 2 | nv_one_logger stub | NGC PyPI 미포함 | **불필요** (NGC 컨테이너 포함 추정) | `python3 -c "import nv_one_logger"` |
| 3 | Prompt model 파일 복사 | NeMo 2.7.3 pip 미포함 | **조건부** (NGC NeMo 버전에 따라) | `import EncDecRNNTBPEModelWithPrompt` |
| 4 | NeMo main 클론 + PYTHONPATH | pip vs main 불일치 | **조건부** (NGC 최신이면 불필요) | config YAML 로드 성공 여부 |
| 5 | WarpRNNT GPU 체크 | sm_89 커널 컴파일 | **필요 (변경)** (sm_100 컴파일 확인) | 첫 학습 스텝 |
| 6 | torchcodec 설치 | CUDA 버전 커플링 방지 | **조건부** (NGC 포함 가능) | `import torchcodec` |
| 7 | datasets Audio monkey-patch | soundfile 폴백 | **필요 가능성 높음** | `import datasets; ds.cast_column(Audio())` |
| 8 | Critical import 검증 | 파이프라인 전 검증 | **필요** (목록만 업데이트) | 동일 방식 |
| 9 | .nemo 모델 존재 확인 | 환경 완결성 | **필요** | 동일 방식 |

> **결과 기록**: 각 항목을 실행 후 `NEEDED` / `NOT_NEEDED` / `MODIFIED` 로 표시.

---

## 5. 환경 셋업 전략

### 접근 방식
NGC 컨테이너가 많은 것을 포함하므로, **최소 개입 원칙**: 필요한 패치만 적용.

```bash
# DGX Station 진입 (컨테이너 실행)
docker run --gpus all -it --rm \
  -v /raid/workspace:/workspace \
  nvcr.io/nvidia/nemo:latest \
  bash

# 프로젝트 클론
git clone <repo> /workspace/finetune-kor-nemotron-asr
cd /workspace/finetune-kor-nemotron-asr

# Phase 0: 환경 프로파일링 (패치 전)
bash scripts/probe_dgx_environment.sh   # 신규 작성 필요 (§7 참조)
```

### setup_environment.sh 수정 전략
- 기존 `setup_environment.sh`를 수정하지 않음
- DGX 전용 `scripts/setup_dgx.sh`를 신규 작성
- 패치 재분류 결과를 반영

---

## 6. 학습 설정 (RunPod → DGX 변경점)

### 변경 필요 파라미터

| 파라미터 | RunPod | DGX (초기값) | 근거 |
|----------|--------|-------------|------|
| `trainer.devices` | `1` | `1` | 단일 GPU 검증 우선 |
| `trainer.precision` | `bf16` | `bf16` | 유지 |
| `model.train_ds.batch_duration` | `100` | `500` (시작점) | HBM3e 192GB, 단계적 증가 |
| `model.train_ds.max_duration` | `20` | `20` | 유지 |
| `exp_manager.exp_dir` | `/workspace/data/checkpoints` | `/raid/checkpoints` 또는 DGX 로컬 NVMe |

### batch_duration 탐색 계획
```
100  → L40S 검증값 (기준점)
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
| WarpRNNT sm_100 미지원 | 중간 | NeMo 업스트림 이슈 확인, CUDA 재컴파일 |
| NGC 컨테이너 aarch64 미지원 | 낮음 | pytorch NGC 컨테이너 + NeMo 수동 설치 fallback |
| `EncDecRNNTBPEModelWithPrompt` NGC 미포함 | 중간 | NeMo main clone + PYTHONPATH (RunPod 방식 유지) |
| Numba aarch64 호환 문제 | 낮음 | Numba >= 0.60 (aarch64 공식 지원) |
| 통합 메모리(NVLink-C2C)에서 batch_duration OOM 예측 불확실 | 중간 | 단계적 탐색으로 확정 |
| `ko-KR` prompt locale 미지원 | 낮음 | Phase 1에서 확인 → 미지원 시 nearest 아시아 locale(ja-JP/zh-CN) 슬롯 사용 (타 언어 사례 참조) |

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
