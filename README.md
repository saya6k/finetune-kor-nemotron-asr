# Korean Fine-Tuning for Nemotron 3.5 ASR Streaming (0.6B)

Production-grade Korean ASR fine-tuning pipeline for `nvidia/nemotron-3.5-asr-streaming-0.6b`.

## Overview

- **Model**: Nemotron 3.5 ASR Streaming 0.6B (600M params, OpenMDW-1.1)
- **Dataset**: [Emilia-YODAS Korean](https://huggingface.co/datasets/amphion/Emilia-Dataset) (7,300h, CC BY 4.0)
- **Language Mix**: ko=80%, en=10%, ja=5%, zh=5% (prevents catastrophic forgetting)
- **Target**: CER ≤ 7.12 on FLEURS Korean (model card baseline)
- **Environment**: RunPod GPU Pod (L40S / A6000 / RTX 6000 Ada, 48GB+ VRAM)
- **Primary Entry Point**: `scripts/train_pipeline.py` (9-step end-to-end pipeline)

## Prerequisites

### System
- GPU with CUDA (48GB+ VRAM: L40S, RTX 6000 Ada, A6000, or A100)
- Python 3.12
- System packages: `sox`, `libsndfile1`, `ffmpeg`, `libsox-fmt-mp3`, `jq`

### Proven RunPod Image
```
runpod/pytorch:1.0.6-cu1281-torch260-ubuntu2204
```
- CUDA 12.8 toolkit + Driver 550.144.03 (CUDA 12.4, PTX 8.4 max)
- PyTorch 2.6.0+cu126

### HuggingFace Access
- Login: `huggingface-cli login`
- Accept [Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset) terms
- Download base model: `huggingface-cli download nvidia/nemotron-3.5-asr-streaming-0.6b nemotron-3.5-asr-streaming-0.6b.nemo --local-dir ./data/base_model/`

## Quick Start (RunPod)

```bash
# 1. Launch GPU pod (L40S, public IP + SSH)
python3 scripts/runpod_auto.py launch --cloud-type COMMUNITY

# 2. SSH in, clone repo
ssh root@<public-ip> -p <port>
git clone https://github.com/saya6k/finetune-kor-nemotron-asr.git /workspace/finetune-kor-nemotron-asr
cd /workspace/finetune-kor-nemotron-asr

# 3. Pre-flight setup (once per pod)
bash scripts/setup_environment.sh

# 4. Run verification (optional, ~5 min)
bash scripts/verify_on_pod.sh

# 5. Production training (full 7,300h)
python3.12 scripts/train_pipeline.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_CKPT` | `./data/base_model/nemotron-3.5-asr-streaming-0.6b.nemo` | Path to pretrained `.nemo` model |
| `DATA_DIR` | `./data` | Data, checkpoints, results root |
| `NEMO_DIR` | `./NeMo` | NeMo source clone path |
| `BATCH_DURATION` | `100` | Batch size in seconds; reduce if OOM |
| `MAX_EPOCHS` | `3` | Training epochs (1-3 for 7,300h) |
| `SMOKE_N` | `0` (full) | Limit training samples (e.g., `100` for smoke test) |
| `TTS_AUGMENT` | `false` | Enable gTTS rare-term augmentation |
| `LANG_MIX_RATIO` | `0.80` | Korean ratio in language mix |
| `DNSMOS_THRESHOLD` | (unset) | DNSMOS filter threshold (unset = keep all) |
| `PYTHONPATH` | Must include `$NEMO_DIR` before pip NeMo | Required for prompt model imports |

## Pipeline Steps (train_pipeline.py)

| # | Step | Description |
|---|------|-------------|
| 1 | Setup | GPU check, NeMo clone, pip install, .nemo download |
| 2 | Data Ingest | Emilia-YODAS streaming → MP3→WAV with cache |
| 3 | Build Manifests | train/val/holdout split + 5 test eval manifests |
| 4 | Language Mix | ko=80%, en=10%, ja=5%, zh=5% merge |
| 5 | TTS Augment | Optional gTTS rare-term augmentation (default OFF) |
| 6 | Tokenizer Verify | Extract from .nemo, check byte fallback/UNK/coverage |
| 7 | Fine-Tuning | `speech_to_text_finetune.py` with production params |
| 8 | Checkpoint Sweep | `eval_direct.py` across all checkpoints × 6 datasets |
| 9 | Cost Report | Actual vs estimated cost, auto-shutdown |

## Pre-flight Setup (run once per pod)

```bash
bash scripts/setup_environment.sh
```

Applies:
1. Numba PTX downgrade (8.7→8.4) for driver 550 compatibility
2. `nv_one_logger` stub (NVIDIA internal package, not on PyPI)
3. Prompt model files copy (NeMo main → pip install location)
4. NeMo main branch clone
5. torchcodec verification
6. datasets Audio monkey-patch (soundfile backend)
7. Critical import verification

## Training Command (Canonical)

```bash
export PYTHONPATH=${NEMO_DIR}:${PYTHONPATH}  # CRITICAL: NeMo main before pip

python ${NEMO_DIR}/examples/asr/speech_to_text_finetune.py \
  --config-path="../asr/conf/fastconformer/cache_aware_streaming" \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_nemo_model=${HF_CKPT} \
  ++model.train_ds.manifest_filepath="${TRAIN_MANIFEST}" \
  ++model.validation_ds.manifest_filepath="${VAL_MANIFEST}" \
  ++model.tokenizer.dir=${CHECKPOINT_DIR} \
  ++trainer.devices=1 \
  ++trainer.max_epochs=3 \
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
  ++exp_manager.early_stopping_callback_params.patience=5 \
  ++exp_manager.checkpoint_callback_params.save_top_k=3 \
  ++exp_manager.checkpoint_callback_params.monitor=val_wer
```

**Critical constraints**:
- `PYTHONPATH` MUST be set: NeMo main must precede pip-installed NeMo
- `limit_train_batches` — **never set** (NeMo Issue #15782)
- `max_steps` — **do not combine with max_epochs**
- `batch_duration=100` — proven max for L40S 48GB
- `d_model=1024` — explicit; OmegaConf can't resolve `${model.encoder.d_model}`
- `warmup_steps=null` — must be null when using `warmup_ratio`

## Evaluation Datasets

| # | Name | Usage | Source |
|---|------|-------|--------|
| 1 | `val_emilia_holdout_ko` | **Validation** (Early Stopping) | Emilia-YODAS KO hold-out |
| 2 | `test_fleurs_ko` | Test | FLEURS ko_kr |
| 3 | `test_emilia_holdout_ko` | Test | Emilia-YODAS KO hold-out B |
| 4 | `test_zeroth_ko` | Test | Zeroth Korean |
| 5 | `test_mixed_en` | Test | Emilia-YODAS EN |
| 6 | `test_mixed_ja` | Test | Emilia-YODAS JA |

**Model card baseline**: CER 7.12 on FLEURS Korean (1.12s chunk, LangID mode).

## Cost Estimation (RunPod)

| GPU | VRAM | Rate ($/h) | Est. Time (7,300h × 3 epochs) | Est. Cost |
|-----|------|-----------|----------------------|-----------|
| L40S | 48GB | ~0.79 | 40-60h | $32-47 |
| RTX 6000 Ada | 48GB | ~0.79 | 40-60h | $32-47 |
| A6000 | 48GB | ~0.79 | 40-60h | $32-47 |
| A100 SXM | 80GB | ~1.99 | 20-30h | $40-60 |

Run `scripts/benchmark_gpu.py` for a more accurate estimate on your specific instance.

## Project Structure

```
finetune-kor-nemotron-asr/
├── SPEC.md                          # Specification document
├── CLAUDE.md                        # AI agent instructions
├── README.md                        # This file
├── asr-finetune-kor.ipynb           # Reference notebook (17 cells)
├── scripts/                         # Production pipeline
│   ├── train_pipeline.py            # 9-step end-to-end pipeline (primary)
│   ├── setup_environment.sh         # All-in-one dependency fix (run once)
│   ├── verify_on_pod.sh             # 7-step pre-flight verification
│   ├── runpod_auto.py               # Pod lifecycle management
│   ├── build_manifest.py            # Audio → NeMo manifest JSON builder
│   ├── compute_metrics.py           # CER + WER + SER computation
│   ├── eval_checkpoints.py          # Hydra-based checkpoint sweep (legacy)
│   ├── eval_direct.py               # Direct .nemo evaluation (prompt-aware, active)
│   ├── tts_augment.py               # gTTS-based rare term augmentation
│   ├── patch_numba_codegen.py       # PTX version downgrade for driver 550
│   └── benchmark_gpu.py             # GPU speed/VRAM benchmark
├── configs/
│   └── override.yaml                # Korean FT hyperparameter reference
├── data/                            # Runtime directory (RunPod mount)
│   ├── raw/                         # Raw audio from Emilia-YODAS
│   ├── processed/                   # Converted WAV + manifest JSONs
│   └── wav_cache/                   # MP3→WAV conversion cache
├── checkpoints/                     # Training output (.nemo files)
└── results/                         # Evaluation output (CSV)
```

## References

- [Model Card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [Fine-Tuning Discussion #11](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/discussions/11)
- [NVIDIA Riva Tutorial](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb)
- [Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)
- [NeMo GitHub](https://github.com/NVIDIA/NeMo)
- [NeMo Issue #15782](https://github.com/NVIDIA-NeMo/NeMo/issues/15782)
