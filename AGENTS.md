# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-grade Korean ASR fine-tuning pipeline for NVIDIA's `nemotron-3.5-asr-streaming-0.6b` (600M params, OpenMDW-1.1 license) using the NeMo framework. Uses **Emilia-YODAS Korean** (7,300h, CC BY 4.0) as the primary training dataset, processed via HuggingFace `datasets` streaming to avoid downloading the full 4.5TB.

**Key design decisions** (see `SPEC.md` for rationale — Round 2~6 섹션 포함):
- **Round 2**: `lr=5e-5`, `max_epochs=10`, `batch_duration=5000`, 하위 8 인코더 레이어 동결. ko 92.6% + zh 4.6% + en 2.8% (877K 발화). best val_wer=0.3544.
- **Round 3**: en 3.2% (clean-100) + ja 6.7% (60K). layer freeze 유지. best val_wer=0.3608 → catastrophic forgetting: fl_en WER 74% (normalized), empty HYP 다수 발생.
- **Round 4**: en→13% VoxPopuli, zh 제거. 동결 해제 (E0 중단, E1부터 full FT). patience=20, max_epochs=20. E1 val_wer=0.4757에서 중단 후 R5로 전환.
- **Round 5**: ko 80% + en 14% VoxPopuli + ja 6% (1,015K). no freeze, max_epochs=10. best val_wer=0.3590 (E7). fl_en_raw WER: 26%→63% (VoxPopuli replay 실패 — 의회 연설체 도메인 불일치).
- **Round 6**: ko 93.3% (812K) + FLEURS replay 6.7% (39개 언어 × 1,500발화, 58.5K). Full FT, max_epochs=3, 원본 .nemo에서 재시작. 목표: replay로 decoder 앵커링, 최소 epoch으로 과적합 방지.
- `gradient_clip_val=1.0`, `seed_everything=42`
- Evaluation: 10 datasets (1 validation + 9 test), CER+WER+SER; RU는 catastrophic forgetting 프로브
- 자동 eval: eval_watcher.py → eval_one.py (체크포인트 생성 시 즉시 평가)
- Tokenizer reuse with strict verification (coverage ≥ 98%, byte fallback ≤ 2%, UNK ≈ 0%)
- AIHub 주석 정규화: `(표기)/(발음)` → 발음형; Meeting은 순서 반대 `(발음)/(표기)` → 발음형
- 오디오 세그멘테이션: 회의음성(최대 2372s) → 1-20s 클립, 극한소음(최대 830s) → SRT 기반 0.5-20s 클립

## Project Structure

```
finetune-kor-nemotron-asr/
├── SPEC.md                          # Specification + Round 2 문서
├── CLAUDE.md                        # This file (→ AGENTS.md symlink)
├── README.md                        # Usage guide, RunPod deployment
├── Dockerfile.dgx                   # DGX GB300 컨테이너 빌드 파일
├── asr-finetune-kor.ipynb           # Main fine-tuning notebook (17 cells)
├── data/                            # Runtime directory (DGX: /workspace/data)
│   ├── raw/                         # AIHub ZIP / Emilia TAR 원본
│   ├── processed/                   # manifest JSONL 파일들
│   │   ├── train_manifest_round2.jsonl   # Round 2 학습 매니페스트 (877,704발화)
│   │   ├── val_ksponspeech_500.jsonl     # 검증 세트 (KsponSpeech 500발화)
│   │   ├── ksponspeech_train.jsonl       # KsponSpeech (618,720)
│   │   ├── freetalk_train.jsonl          # FreeTalk field (47,936)
│   │   ├── freetalk_studio.jsonl         # FreeTalk studio (58,210)
│   │   ├── meeting_train.jsonl           # 회의음성 세그멘트 (81,413)
│   │   ├── extreme_noise_train.jsonl     # 극한소음 세그멘트 (6,546)
│   │   └── eval_datasets_round2.json    # 10개 eval 데이터셋 경로 정의
│   └── wav_cache/                   # 변환/세그멘트 WAV 캐시
│       ├── ksponspeech/             # PCM→WAV
│       ├── meeting/segments/        # 회의음성 세그멘트 클립
│       └── extreme_noise/segments/  # 극한소음 세그멘트 클립
├── checkpoints/                     # 원본 pretrained .nemo 모델
├── results/                         # Evaluation CSV 출력
│   └── eval_rolling_round2.csv      # Round 2 체크포인트별 eval 결과
├── scripts/                         # 파이프라인 스크립트
│   ├── train_pipeline.py            # 9-step end-to-end pipeline (RunPod용)
│   ├── finetune_with_freeze.py      # 레이어 동결 학습 (Round 2, DGX용)
│   ├── fast_ingest.py               # 병렬 fast ingest (DGX용)
│   │
│   ├── convert_ksponspeech.py       # KsponSpeech PCM→WAV + 정규화
│   ├── convert_meeting.py           # 회의음성 JSON 타임스탬프 세그멘테이션
│   ├── convert_extreme_noise.py     # 극한소음 SRT 타임스탬프 세그멘테이션
│   ├── convert_freetalk.py          # FreeTalk 변환
│   ├── normalize_manifests.py       # 기존 manifest 재정규화 도구
│   │
│   ├── eval_watcher.py              # 체크포인트 감시 → 자동 eval 트리거
│   ├── eval_one.py                  # 단일 체크포인트 × 복수 데이터셋 평가
│   ├── eval_direct.py               # Direct .nemo evaluation (prompt-aware)
│   ├── eval_checkpoints.py          # Hydra 기반 체크포인트 sweep (구버전)
│   │
│   ├── download_emilia.py           # Emilia-YODAS 다운로드
│   ├── download_eval_datasets.py    # FLEURS/Zeroth 등 eval 데이터셋 다운로드
│   ├── build_manifest.py            # Audio → NeMo manifest JSON builder
│   ├── build_round6_manifest.py       # R6: ko+FLEURS 39개국어 replay manifest
│   ├── compute_metrics.py           # CER + WER + SER computation
│   ├── tts_augment.py               # gTTS-based rare term augmentation
│   │
│   ├── setup_environment.sh         # RunPod 의존성 패치 (수정 금지)
│   ├── setup_dgx.sh                 # DGX 환경 설정
│   ├── probe_dgx_environment.sh     # DGX 환경 프로파일링
│   ├── verify_on_pod.sh             # RunPod pre-flight 검증 (수정 금지)
│   ├── verify_on_dgx.sh             # DGX pre-flight 검증
│   │
│   ├── runpod_auto.py               # RunPod pod lifecycle management
│   ├── patch_numba_codegen.py       # PTX version downgrade for driver 550
│   └── benchmark_gpu.py             # GPU speed/VRAM benchmark
└── configs/
    └── override.yaml                # Korean FT hyperparameter reference
```

## Environment Requirements

- GPU with CUDA support (single GPU, 48GB+ VRAM recommended: RTX 6000 Ada / A6000 / A100)
- Python 3.12
- RunPod GPU Pod (primary deployment target)
- **Proven image**: `runpod/pytorch:1.0.6-cu1281-torch260-ubuntu2204` (L40S, 48GB)
  - CUDA 12.8 toolkit + Driver 550.144.03 (CUDA 12.4, PTX 8.4 max)
  - PyTorch 2.6.0+cu126 (CUDA forward-compatible with driver 12.4)
- System packages: `sox`, `libsndfile1`, `ffmpeg`, `libsox-fmt-mp3`, `jq`
- Python packages installed by setup:
  - `nemo_toolkit[asr]==2.7.3` (pip) + NeMo `main` branch (source, for prompt model)
  - `datasets`, `soundfile`, `librosa`, `tqdm`, `huggingface_hub`, `gTTS`, `jiwer`, `torchcodec`
- **Pre-flight**: Run `bash scripts/setup_environment.sh` once per pod

## Notebook Structure (17 Cells)

| # | Section | Purpose |
|---|---------|---------|
| 1 | Header | Project overview, objectives, dataset info |
| 2 | Setup | GPU check, dependency install, NeMo clone |
| 3 | GPU Benchmark | 500-1000 step speed test → cost estimation |
| 4 | Data Prep (MD) | Emilia-YODAS structure, 6 eval datasets |
| 5 | Data Ingest | Streaming download, DNSMOS filter (optional), MP3→WAV with cache |
| 6 | Manifest (MD) | NeMo manifest format |
| 7 | Build Manifest | train/val/holdout + 5 test eval manifests |
| 8 | Lang Mix | ko=80%, en=10%, ja=5%, zh=5% merge |
| 9 | TTS Aug | Optional gTTS augmentation (default OFF) |
| 10 | Tokenizer (MD) | Verification criteria |
| 11 | Tokenizer Verify | Load from .nemo, check byte fallback/UNK/coverage |
| 12 | Training (MD) | Hyperparameter table + early stopping |
| 13 | Fine-Tuning | `speech_to_text_finetune.py` with production params |
| 14 | Eval Prep (MD) | Checkpoint sweep explanation |
| 15 | Checkpoint Sweep | `eval_checkpoints.py` across all checkpoints × 6 datasets |
| 16 | Results | CER/WER/SER summary, baseline comparison |
| 17 | Cost Report | Actual vs estimated cost, instance recommendation |

## Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Data and checkpoint root | `./data` |
| `NEMO_DIR` | NeMo clone path | `./NeMo` |
| `HF_CKPT` | Path to pretrained `.nemo` model (required) | — |
| `BATCH_DURATION` | Batch size in seconds; reduce if OOM | `100` |
| `SMOKE_N` | Training samples limit (unset = full dataset) | — (full) |
| `PYTHONPATH` | Must include NeMo main clone before pip | `/workspace/NeMo` |
| `MAX_EPOCHS` | 상한선 epochs (early stopping이 실제 결정) | `20` |
| `KOR_DATASET` | Dataset override (`"fleurs"`, `"aihub"`, URL, or path) | — (uses Emilia-YODAS) |
| `LANG_MIX_RATIO` | Korean ratio in training mix (0.70-0.80) | `0.80` |
| `TTS_AUGMENT` | Enable TTS augmentation (`"true"`/`"false"`) | `false` |
| `DNSMOS_THRESHOLD` | DNSMOS filter threshold (unset = keep all) | — (keep all) |

## Manifest Format

Each line in a manifest JSON is a JSON object:
```json
{"audio_filepath": "/path/to/file.wav", "duration": 3.5, "text": "한국어 트랜스크립트", "lang": "ko-KR", "target_lang": "ko-KR"}
```
Non-Korean entries in the mixed manifest retain their original `lang` tag (e.g., `"en-US"`, `"ja-JP"`, `"zh-CN"`).

## Training Command

### Round 6 (DGX GB300 — Ready, manifest built)

```bash
# FLEURS replay 39개국어 58.5K(6.7%) + ko 812K(93.3%) = 871K 발화
# max_epochs=3 (NVIDIA fixed step budget), 원본 .nemo에서 재시작
python3 /workspace/scripts/finetune_with_freeze.py \
  --config-path=../asr/conf/fastconformer/cache_aware_streaming \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_nemo_model=/workspace/data/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming.nemo \
  ++model.train_ds.manifest_filepath=/workspace/data/processed/train_manifest_round6.jsonl \
  ++model.validation_ds.manifest_filepath=/workspace/data/processed/val_ksponspeech_500.jsonl \
  ++model.tokenizer.dir=/workspace/data/checkpoints \
  ++trainer.devices=1 \
  ++trainer.max_epochs=3 \
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
  ++exp_manager.exp_dir=/workspace/data/checkpoints_round6 \
  ++exp_manager.resume_if_exists=true \
  ++exp_manager.resume_ignore_no_checkpoint=true \
  ++exp_manager.create_early_stopping_callback=true \
  ++exp_manager.early_stopping_callback_params.monitor=val_wer \
  ++exp_manager.early_stopping_callback_params.patience=5 \
  ++exp_manager.checkpoint_callback_params.save_top_k=-1 \
  ++exp_manager.checkpoint_callback_params.monitor=val_wer \
  ++exp_manager.checkpoint_callback_params.save_last=true \
  ++exp_manager.checkpoint_callback_params.every_n_train_steps=250 \
  ++exp_manager.checkpoint_callback_params.every_n_epochs=null \
  ++trainer.val_check_interval=250
```

### Round 5 (DGX GB300 — Completed 2026-06-22)

```bash
# ko 80% + en 14% VoxPopuli + ja 6% = 1,015K 발화, no freeze, max_epochs=10
# best val_wer=0.3590(E7), fl_en_raw WER 26→63% (VoxPopuli replay 실패)
python3 /workspace/scripts/finetune_with_freeze.py \
  --config-path=../asr/conf/fastconformer/cache_aware_streaming \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_nemo_model=/workspace/data/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming.nemo \
  ++model.train_ds.manifest_filepath=/workspace/data/processed/train_manifest_round5.jsonl \
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
  ++exp_manager.exp_dir=/workspace/data/checkpoints_round5 \
  ++exp_manager.resume_if_exists=true \
  ++exp_manager.resume_ignore_no_checkpoint=true \
  ++exp_manager.create_early_stopping_callback=true \
  ++exp_manager.early_stopping_callback_params.monitor=val_wer \
  ++exp_manager.early_stopping_callback_params.patience=5 \
  ++exp_manager.checkpoint_callback_params.save_top_k=-1 \
  ++exp_manager.checkpoint_callback_params.monitor=val_wer \
  ++exp_manager.checkpoint_callback_params.save_last=true \
  ++exp_manager.checkpoint_callback_params.every_n_train_steps=250 \
  ++exp_manager.checkpoint_callback_params.every_n_epochs=null \
  ++trainer.val_check_interval=250
```

### Round 4 (DGX GB300 — 중단, R5로 전환)

```bash
# 학습 데이터: ko 812K(~81%) / en 132K(~13%, clean-100+360) / ja 60K(~6%) = 1,005K 발화
# E0 layer freeze → E1부터 full FT. patience=20, max_epochs=20.
# E1 val_wer=0.4757에서 중단 후 R5로 전환
# 원본 .nemo에서 재시작 (Round 3 모델이 FL-en WER 70%+로 en 완전 망각)
python3 /workspace/scripts/finetune_with_freeze.py \
  --config-path=../asr/conf/fastconformer/cache_aware_streaming \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_nemo_model=/workspace/data/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming.nemo \
  ++model.train_ds.manifest_filepath=/workspace/data/processed/train_manifest_round4.jsonl \
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
  ++exp_manager.exp_dir=/workspace/data/checkpoints_round4 \
  ++exp_manager.resume_if_exists=true \
  ++exp_manager.resume_ignore_no_checkpoint=true \
  ++exp_manager.create_early_stopping_callback=true \
  ++exp_manager.early_stopping_callback_params.monitor=val_wer \
  ++exp_manager.early_stopping_callback_params.patience=5 \
  ++exp_manager.checkpoint_callback_params.save_top_k=-1 \
  ++exp_manager.checkpoint_callback_params.monitor=val_wer \
  ++exp_manager.checkpoint_callback_params.save_last=true \
  ++exp_manager.checkpoint_callback_params.every_n_train_steps=250 \
  ++exp_manager.checkpoint_callback_params.every_n_epochs=null \
  ++trainer.val_check_interval=250
```

### Round 3 (DGX GB300 — Completed 2026-06-21)

```bash
# 학습 데이터: ko 812K(90.2%) / en 28K(3.2%, clean-100) / ja 60K(6.7%) = 901K 발화
# best val_wer=0.3544 (E9), FL-en WER E6 70%+ (catastrophic forgetting 확인)
python3 /workspace/finetune-kor-nemotron-asr/scripts/finetune_with_freeze.py \
  --config-path=../asr/conf/fastconformer/cache_aware_streaming \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_nemo_model=/workspace/data/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming/checkpoints/FastConformer-Transducer-BPE-Prompt-Streaming.nemo \
  ++model.train_ds.manifest_filepath=/workspace/data/processed/train_manifest_round3.jsonl \
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
  ++exp_manager.exp_dir=/workspace/data/checkpoints_round3 \
  ++exp_manager.resume_if_exists=true \
  ++exp_manager.resume_ignore_no_checkpoint=true \
  ++exp_manager.create_early_stopping_callback=true \
  ++exp_manager.early_stopping_callback_params.monitor=val_wer \
  ++exp_manager.early_stopping_callback_params.patience=5 \
  ++exp_manager.checkpoint_callback_params.save_top_k=-1 \
  ++exp_manager.checkpoint_callback_params.monitor=val_wer \
  ++exp_manager.checkpoint_callback_params.save_last=true \
  ++exp_manager.checkpoint_callback_params.every_n_train_steps=250 \
  ++exp_manager.checkpoint_callback_params.every_n_epochs=null \
  ++trainer.val_check_interval=250
```

### Round 2 (DGX GB300 — Completed 2026-06-21)

```bash
# 컨테이너 내부 /opt/NeMo에 NeMo 3.x가 있음 (PYTHONPATH 자동 설정됨)
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

### Round 1 (RunPod L40S — Proven 2026-06-17)

```bash
export PYTHONPATH=${NEMO_DIR}:${PYTHONPATH}  # CRITICAL: NeMo main before pip
bash scripts/setup_environment.sh

python ${NEMO_DIR}/examples/asr/speech_to_text_finetune.py \
  --config-path="../asr/conf/fastconformer/cache_aware_streaming" \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_nemo_model=${HF_CKPT} \
  ++model.train_ds.manifest_filepath="${TRAIN_MANIFEST}" \
  ++model.validation_ds.manifest_filepath="${VAL_MANIFEST}" \
  ++model.tokenizer.dir=${CHECKPOINT_DIR} \
  ++trainer.devices=1 \
  ++trainer.max_epochs=20 \
  ++trainer.precision=bf16 \
  ++trainer.gradient_clip_val=1.0 \
  ++seed_everything=42 \
  ++model.train_ds.batch_duration=100 \
  ++model.validation_ds.batch_size=1 \
  ++model.train_ds.max_duration=20 \
  ++model.optim.name="adamw" \
  ++model.optim.lr=0.0001 \
  ++model.optim.weight_decay=0.001 \
  ++model.optim.sched.name="CosineAnnealing" \
  ++model.optim.sched.warmup_ratio=0.05 \
  ++model.optim.sched.warmup_steps=null \
  ++model.optim.sched.d_model=1024 \
  ++model.optim.sched.min_lr=1e-6 \
  ++exp_manager.exp_dir=${CHECKPOINT_DIR} \
  ++exp_manager.resume_if_exists=true \
  ++exp_manager.resume_ignore_no_checkpoint=true \
  ++exp_manager.create_early_stopping_callback=true \
  ++exp_manager.early_stopping_callback_params.monitor=val_wer \
  ++exp_manager.early_stopping_callback_params.patience=10 \
  ++exp_manager.checkpoint_callback_params.save_top_k=-1 \
  ++exp_manager.checkpoint_callback_params.save_last=true \
  ++exp_manager.checkpoint_callback_params.every_n_train_steps=500 \
  ++exp_manager.checkpoint_callback_params.monitor=val_wer \
  ++trainer.val_check_interval=500
```

**Critical constraints**:
- `limit_train_batches` — **never set** (NeMo Issue #15782: silently limits data per epoch)
- `max_steps` — **do not combine with max_epochs** (PyTorch Lightning stops at whichever triggers first)
- `batch_duration=100` — RunPod L40S 48GB 최대값; DGX GB300은 5000 사용 중
- `d_model=1024` — NeMo 2.x (RunPod)에서만 필요. NeMo 3.x (DGX)는 `~model.optim.sched.d_model`로 Hydra delete
- `warmup_steps=null` — warmup_ratio 사용 시 반드시 null
- `every_n_epochs=null` — `every_n_train_steps`와 동시 설정 시 NeMo가 epochs 기준 우선 → 반드시 null
- Checkpoint output path: `$EXP_DIR/FastConformer-Transducer-BPE-Prompt-Streaming/checkpoints/`

## Pre-flight Setup (run once per pod)

```bash
bash scripts/setup_environment.sh
```

This applies:
1. Numba PTX downgrade (8.7→8.4) for driver 550 compatibility
2. nv_one_logger stub (NVIDIA internal package)
3. Prompt model file copies (NeMo main → pip)
4. NeMo main branch clone verification
5. WarpRNNT GPU compatibility check
6. torchcodec verification + install
7. datasets Audio monkey-patch (soundfile backend)
8. Critical import verification (EncDecRNNTBPEModelWithPrompt)
9. Base model existence check

Verification script (`bash scripts/verify_on_pod.sh`) runs a 7-step smoke check:
GPU → setup → imports → split logic → training params → eval_direct → tokenizer.

## Evaluation Datasets

| # | Name | Usage | Source |
|---|------|-------|--------|
| 1 | `val_emilia_holdout_ko` | **Validation** (Early Stopping) | Emilia-YODAS KO hold-out |
| 2 | `test_fleurs_ko` | Test — primary KO benchmark | FLEURS ko_kr |
| 3 | `test_emilia_holdout_ko` | Test | Emilia-YODAS KO hold-out B (val과 분리) |
| 4 | `test_zeroth_ko` | Test | Zeroth Korean |
| 5 | `test_mixed_en` | Test | Emilia-YODAS EN |
| 6 | `test_mixed_ja` | Test | Emilia-YODAS JA |
| 7 | `test_mixed_zh` | Test | Emilia-YODAS ZH |
| 8 | `test_fleurs_fr` | Test — 학습 언어 WER | FLEURS fr_fr |
| 9 | `test_fleurs_de` | Test — 학습 언어 WER | FLEURS de_de |
| 10 | `test_fleurs_ru` | Test — **catastrophic forgetting 프로브** (비학습) | FLEURS ru_ru |

**Model card baseline**: CER 7.12 on FLEURS Korean (1.12s chunk, LangID mode).

## References
- [Model Card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [HuggingFace Discussion #11](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/discussions/11)
- [NVIDIA Riva Fine-Tuning Tutorial](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb)
- [Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)
- [NeMo GitHub](https://github.com/NVIDIA/NeMo)
- [NeMo Issue #15782](https://github.com/NVIDIA-NeMo/NeMo/issues/15782)
