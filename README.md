# Korean Fine-Tuning for Nemotron 3.5 ASR Streaming (0.6B)

Production-grade Korean ASR fine-tuning pipeline for `nvidia/nemotron-3.5-asr-streaming-0.6b`.

## Overview

- **Model**: Nemotron 3.5 ASR Streaming 0.6B (600M params, OpenMDW-1.1)
- **Dataset**: [Emilia-YODAS Korean](https://huggingface.co/datasets/amphion/Emilia-Dataset) (7,300h, CC BY 4.0)
- **Language Mix**: ko=80%, en=10%, ja=5%, zh=5% (prevents catastrophic forgetting)
- **Target**: CER ≤ 7.12 on FLEURS Korean (model card baseline)
- **Environment**: RunPod GPU Pod (RTX 6000 Ada / A6000 / A100 recommended)

## Prerequisites

### System
- GPU with CUDA (48GB+ VRAM recommended: RTX 6000 Ada, A6000, or A100)
- System packages: `sox`, `libsndfile1`, `ffmpeg`, `libsox-fmt-mp3`, `jq`

### HuggingFace Access
- Login to HuggingFace: `huggingface-cli login`
- Accept [Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset) terms (requires contact info sharing)
- Download the base model: `huggingface-cli download nvidia/nemotron-3.5-asr-streaming-0.6b`

## Quick Start (RunPod)

```bash
# 1. Start a RunPod GPU Pod (RTX 6000 Ada recommended)
# 2. Set required environment variables
export HF_CKPT=/path/to/nemotron-3.5-asr-streaming-0.6b.nemo
export GPU_HOURLY_RATE=0.79  # your instance hourly rate

# 3. Launch the notebook
jupyter notebook asr-finetune-kor.ipynb

# 4. Run cells 1→17 in order
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_CKPT` | **(required)** | Path to pretrained `.nemo` model |
| `DATA_DIR` | `/workspace/data` | Data, checkpoints, results root |
| `NEMO_DIR` | `/workspace/NeMo` | NeMo source clone path |
| `BATCH_DURATION` | `200` | Batch size in seconds; reduce if OOM |
| `MAX_EPOCHS` | `3` | Training epochs (1-3 for 7,300h) |
| `GPU_HOURLY_RATE` | `0.79` | GPU instance cost for benchmark |
| `DNSMOS_THRESHOLD` | (unset) | DNSMOS filter threshold (unset = keep all) |
| `SMOKE_N` | `5000` | Smoke test subset size |
| `TTS_AUGMENT` | `false` | Enable gTTS augmentation |
| `LANG_MIX_KO` | `0.80` | Korean ratio in language mix |
| `LANG_MIX_EN` | `0.10` | English ratio |
| `LANG_MIX_JA` | `0.05` | Japanese ratio |
| `LANG_MIX_ZH` | `0.05` | Chinese ratio |

## Notebook Structure (17 Cells)

| # | Section | Purpose |
|---|---------|---------|
| 1 | Header | Overview, objectives |
| 2 | Setup | GPU check, deps, NeMo clone |
| 3 | GPU Benchmark | Speed test, cost estimate |
| 4 | Data Prep (MD) | Emilia-YODAS + 6 eval datasets |
| 5 | Data Ingest | Streaming, MP3→WAV cache |
| 6 | Manifest (MD) | NeMo manifest format |
| 7 | Build Manifest | train/val/holdout + eval |
| 8 | Language Mix | ko=80%, en=10%, ja=5%, zh=5% |
| 9 | TTS Aug | Optional (default OFF) |
| 10 | Tokenizer (MD) | Verification criteria |
| 11 | Tokenizer Verify | Coverage, byte fallback, UNK |
| 12 | Training (MD) | Hyperparameters |
| 13 | Fine-Tuning | Production training command |
| 14 | Evaluation (MD) | Checkpoint sweep |
| 15 | Checkpoint Sweep | All checkpoints × 6 datasets |
| 16 | Results | CER/WER/SER + baseline comparison |
| 17 | Cost Report | Actual vs estimated cost |

## Running Full Training (Track B)

For full 7,300h training outside the notebook:

```bash
export HF_CKPT=/path/to/model.nemo
export TRAIN_MANIFEST=data/processed/train_manifest_full_ko.json
export VAL_MANIFEST=data/processed/val_emilia_holdout_ko.json
export CHECKPOINT_DIR=data/checkpoints

python ${NEMO_DIR}/examples/asr/speech_to_text_finetune.py \
  --config-path="../asr/conf/fastconformer/cache_aware_streaming" \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_nemo_model=${HF_CKPT} \
  ++model.train_ds.manifest_filepath="${TRAIN_MANIFEST}" \
  ++model.validation_ds.manifest_filepath="${VAL_MANIFEST}" \
  ++trainer.devices=1 \
  ++trainer.max_epochs=3 \
  ++trainer.precision=bf16 \
  ++trainer.gradient_clip_val=1.0 \
  ++seed_everything=42 \
  ++model.train_ds.batch_duration=200 \
  ++model.optim.name="adamw" \
  ++model.optim.lr=0.0001 \
  ++model.optim.weight_decay=0.001 \
  ++model.optim.sched.name="CosineAnnealing" \
  ++model.optim.sched.warmup_ratio=0.05 \
  ++model.optim.sched.min_lr=1e-6 \
  ++exp_manager.exp_dir=${CHECKPOINT_DIR} \
  ++exp_manager.resume_if_exists=true \
  ++exp_manager.checkpoint_every_n_train_steps=5000 \
  ++exp_manager.save_top_k=3 \
  ++exp_manager.early_stopping_enabled=true \
  ++exp_manager.early_stopping_metric="val_cer" \
  ++exp_manager.early_stopping_patience=5
```

## Cost Estimation (RunPod)

| GPU | VRAM | Rate ($/h) | Est. Time (7,300h × 3) | Est. Cost |
|-----|------|-----------|----------------------|-----------|
| RTX 6000 Ada | 48GB | ~0.79 | 20-40h | $16-32 |
| A6000 | 48GB | ~0.79 | 20-40h | $16-32 |
| A100 SXM | 80GB | ~1.99 | 10-20h | $20-40 |

Run Cell 3 (GPU Benchmark) for a more accurate estimate on your specific instance.

## Evaluation

After training, Cell 15-16 will produce:

- `results/checkpoint_eval.csv` — CER/WER/SER for all checkpoints × all datasets
- Best CER per dataset with checkpoint identification
- FLEURS Korean baseline comparison (target: ≤ 7.12)

## References

- [Model Card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [Fine-Tuning Discussion #11](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/discussions/11)
- [NVIDIA Riva Tutorial](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb)
- [Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)
- [NeMo GitHub](https://github.com/NVIDIA/NeMo)
- [NeMo Issue #15782](https://github.com/NVIDIA-NeMo/NeMo/issues/15782)
