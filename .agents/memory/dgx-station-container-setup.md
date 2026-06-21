---
name: dgx-station-container-setup
description: "GB300 DGX Station 컨테이너 실행 명령 (Dockerfile.dgx 기반, 실측 확정값)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 76c864a0-4c6d-4e2a-9bcc-517a239c9ebe
---

# GB300 DGX Station — 컨테이너 설정 (2026-06-20 확정)

## 결론: `Dockerfile.dgx` 기반 커스텀 이미지

`nvcr.io/nvidia/nemo:latest`는 `lightning` (PyTorch Lightning) 미포함 → 학습 불가.
`nvcr.io/nvidia/pytorch:26.05-py3`를 base로 `Dockerfile.dgx`를 사용.
이미지 태그: `nemotron-asr-dgx:latest` (24.1 GiB), verify_on_dgx.sh 9/9 PASS.

## 호스트 경로 구조 (실측)

```
/root/nemotron-asr-finetune/          ← 작업 디렉토리 (WORKSPACE 마운트 원본)
├── finetune-kor-nemotron-asr/        ← 이 repo (git clone)
└── data/
    └── base_model/
        └── nemotron-3.5-asr-streaming-0.6b.nemo  (2.3 GiB)
```

## 이미지 빌드 (한 번만)

```bash
cd /root/nemotron-asr-finetune/finetune-kor-nemotron-asr
docker build -f Dockerfile.dgx -t nemotron-asr-dgx:latest .
```

## 학습 실행 명령 (확정됨)

```bash
# VLLM 서비스 먼저 중단 (deepseek-v4-flash, docker-compose 관리)
docker compose -f <경로> down   # 서비스 담당자 확인 후 실행

docker run --rm --gpus all --shm-size=64g --ulimit memlock=-1 --ipc=host \
  -v /root/nemotron-asr-finetune:/workspace \
  -e WORKSPACE=/workspace \
  -e NEMO_DIR=/opt/NeMo \
  -e HF_CKPT=/workspace/data/base_model/nemotron-3.5-asr-streaming-0.6b.nemo \
  -e HF_TOKEN=<your-hf-token> \
  -e DATASETS_AUDIO_BACKEND=soundfile \
  -e SKIP_SETUP_INSTALL=1 \
  nemotron-asr-dgx:latest \
  python3 /workspace/finetune-kor-nemotron-asr/scripts/train_pipeline.py \
  > /root/nemotron-asr-finetune/train.log 2>&1
```

## 컨테이너 내부 경로 구조

```
/opt/NeMo/              ← NeMo git main 3.1.0 (이미지 고정, PYTHONPATH=/opt/NeMo)
/workspace/             ← 호스트 /root/nemotron-asr-finetune 마운트
├── finetune-kor-nemotron-asr/    ← 이 repo
├── data/
│   ├── base_model/nemotron-3.5-asr-streaming-0.6b.nemo
│   ├── processed/       ← 변환된 WAV + manifest JSON
│   └── wav_cache/       ← MP3→WAV 캐시
└── checkpoints/         ← 학습 체크포인트
```

## 핵심 환경 변수

| 변수 | 값 | 이유 |
|------|----|----|
| `NEMO_DIR` | `/opt/NeMo` | 이미지에 포함된 NeMo main |
| `SKIP_SETUP_INSTALL=1` | `1` | pip install 생략 (nemo_toolkit 2.7.3 다운그레이드 방지) |
| `DATASETS_AUDIO_BACKEND` | `soundfile` | torchcodec aarch64 미지원 우회 |
| `HOLD_OUT_N` | `10` | 스모크 테스트 시 SMOKE_N < HOLD_OUT_N 방지 |

## VLLM 충돌 주의

DeepSeek-V4-Flash (`deepseek-v4-flash` 컨테이너)가 GPU 97% (~277 GiB) 상시 점유.
학습 전 `docker compose -f <경로> down`으로 중단 필요.
서비스 담당자 확인 후 진행. `docker compose up -d`로 재시작.

**Why:** 2026-06-20 smoke test 시 VLLM OOM으로 Step 6 실패. 서비스 중단 확인됨.
**How to apply:** 학습 실행 전 VLLM 서비스 상태 확인 (`docker ps | grep deepseek`).
