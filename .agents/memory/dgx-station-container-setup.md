---
name: dgx-station-container-setup
description: "GB300 DGX Station NGC 컨테이너 실행 명령, 볼륨 마운트, 경로 구조"
metadata: 
  node_type: memory
  type: project
  originSessionId: cf86f154-0f33-497f-ad67-94165b26cdb3
---

# GB300 DGX Station — 컨테이너 설정

## docker run 명령

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

## 컨테이너 내부 경로 구조

```
/workspace/                          ← WORKSPACE env var 루트 (모든 스크립트 기준점)
├── finetune-kor-nemotron-asr/       ← 이 repo (git clone)
├── data/
│   └── base_model/
│       └── nemotron-3.5-asr-streaming-0.6b.nemo
├── NeMo/                            ← setup_dgx.sh이 필요 시 자동 clone
├── .setup_dgx_done                  ← sentinel (setup idempotent)
└── probe_results.txt                ← probe_dgx_environment.sh 출력

/root/.cache/huggingface/            ← HF 모델/데이터셋 캐시 (재다운 방지)
```

## 호스트 사전 준비 (DGX NVMe RAID 기준)

```bash
mkdir -p /raid/workspace /raid/hf_cache
```

## 컨테이너 진입 후 첫 실행 순서

```bash
git clone <repo> /workspace/finetune-kor-nemotron-asr
cd /workspace/finetune-kor-nemotron-asr
mkdir -p /workspace/data/base_model
huggingface-cli download nvidia/nemotron-3.5-asr-streaming-0.6b \
  nemotron-3.5-asr-streaming-0.6b.nemo \
  --local-dir /workspace/data/base_model/
bash scripts/probe_dgx_environment.sh
```

## 플래그 이유

- `--shm-size=64g` — NeMo DataLoader shared memory 요구
- `--ipc=host` — 멀티프로세스 DataLoader에 필요
- `/raid/workspace` — DGX NVMe RAID (빠른 로컬 스토리지)
- `/raid/hf_cache` — HF 캐시를 호스트에 영속화 (컨테이너 재시작 시 재다운 방지)

**Why:** 2026-06-18 DGX Station 접속 후 포팅 진행 예정. 이 명령이 진입점.
**How to apply:** DGX Station에서 컨테이너 실행할 때 이 명령 사용.
