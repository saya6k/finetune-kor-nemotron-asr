# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-grade Korean ASR fine-tuning pipeline for NVIDIA's `nemotron-3.5-asr-streaming-0.6b` (600M params, OpenMDW-1.1 license) using the NeMo framework. Uses **Emilia-YODAS Korean** (7,300h, CC BY 4.0) as the primary training dataset, processed via HuggingFace `datasets` streaming to avoid downloading the full 4.5TB.

**Key design decisions** (see `SPEC.md` and `plans/jazzy-waddling-dragonfly.md` for rationale):
- `lr=1e-4` (conservative, preserves pretrained representations)
- `max_epochs=3` (21,900h exposure from 7,300h data is sufficient)
- `gradient_clip_val=1.0`, `seed_everything=42`
- Language mix: ko=80%, en=10%, ja=5%, zh=5% (same Emilia-YODAS dataset)
- Evaluation: 6 datasets (1 validation + 5 test), CER+WER+SER metrics
- Checkpoint sweep eval across all checkpoints × all datasets
- Tokenizer reuse with strict verification (coverage ≥ 98%, byte fallback ≤ 2%, UNK ≈ 0%)

## Project Structure

```
finetune-kor-nemotron-asr/
├── SPEC.md                          # Specification document
├── CLAUDE.md                        # This file
├── README.md                        # Usage guide, RunPod deployment
├── asr-finetune-kor.ipynb           # Main fine-tuning notebook (17 cells)
├── asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb  # Reference (NVIDIA demo)
├── data/                            # Runtime directory (RunPod mount)
│   ├── raw/                         # Raw audio from Emilia-YODAS
│   ├── processed/                   # Converted WAV + manifest JSONs
│   └── wav_cache/                   # MP3→WAV conversion cache
├── checkpoints/                     # Training output (.nemo files)
├── results/                         # Evaluation output (CSV)
├── scripts/                         # Reusable Python modules
│   ├── build_manifest.py            # Audio → NeMo manifest JSON builder
│   ├── compute_metrics.py           # CER + WER + SER computation
│   ├── tts_augment.py               # gTTS-based rare term augmentation
│   ├── eval_checkpoints.py          # Multi-checkpoint × multi-dataset sweep
│   └── benchmark_gpu.py            # GPU speed/VRAM benchmark
└── configs/
    └── override.yaml                # Korean FT hyperparameter reference
```

## Environment Requirements

- GPU with CUDA support (single GPU, 48GB+ VRAM recommended: RTX 6000 Ada / A6000 / A100)
- Python 3.8–3.12
- RunPod GPU Pod (primary deployment target)
- System packages: `sox`, `libsndfile1`, `ffmpeg`, `libsox-fmt-mp3`, `jq`
- Python packages installed by notebook Cell 2:
  - `nemo_toolkit[asr]` from NeMo `main` branch
  - `datasets`, `soundfile`, `librosa`, `tqdm`, `huggingface_hub`, `gTTS`, `jiwer`

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
| `BATCH_DURATION` | Batch size in seconds; reduce if OOM | `200` |
| `MAX_EPOCHS` | Training epochs (1-3 for 7,300h) | `3` |
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

## Training Command (Canonical)

```bash
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

**Critical constraints**:
- `limit_train_batches` — **never set** (NeMo Issue #15782: silently limits data per epoch)
- `max_steps` — **do not combine with max_epochs** (PyTorch Lightning stops at whichever triggers first)
- Checkpoint output path: `$CHECKPOINT_DIR/FastConformer-Transducer-BPE-Prompt-Streaming/...`

## Evaluation Datasets

| # | Name | Usage | Source |
|---|------|-------|--------|
| 1 | `val_emilia_holdout_ko` | **Validation** (Early Stopping) | Emilia-YODAS KO hold-out |
| 2 | `test_fleurs_ko` | Test | FLEURS ko_kr |
| 3 | `test_emilia_holdout_ko` | Test | Emilia-YODAS KO hold-out B (val과 분리) |
| 4 | `test_zeroth_ko` | Test | Zeroth Korean |
| 5 | `test_mixed_en` | Test | Emilia-YODAS EN |
| 6 | `test_mixed_ja` | Test | Emilia-YODAS JA |

**Model card baseline**: CER 7.12 on FLEURS Korean (1.12s chunk, LangID mode).

## References
- [Model Card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [HuggingFace Discussion #11](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/discussions/11)
- [NVIDIA Riva Fine-Tuning Tutorial](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb)
- [Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)
- [NeMo GitHub](https://github.com/NVIDIA/NeMo)
- [NeMo Issue #15782](https://github.com/NVIDIA-NeMo/NeMo/issues/15782)
