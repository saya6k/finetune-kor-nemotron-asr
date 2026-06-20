# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-grade Korean ASR fine-tuning pipeline for NVIDIA's `nemotron-3.5-asr-streaming-0.6b` (600M params, OpenMDW-1.1 license) using the NeMo framework. Uses **Emilia-YODAS Korean** (7,300h, CC BY 4.0) as the primary training dataset, processed via HuggingFace `datasets` streaming to avoid downloading the full 4.5TB.

**Key design decisions** (see `SPEC.md` and `plans/jazzy-waddling-dragonfly.md` for rationale):
- `lr=1e-4` (conservative, preserves pretrained representations)
- `max_epochs=20` (상한선; 실제 중단은 early stopping이 결정 — patience=10 val checks ≈ 2 epochs)
- `gradient_clip_val=1.0`, `seed_everything=42`
- Language mix: ko=80%, en=10%, ja=4%, zh=4%, fr=1%, de=1% (Emilia-YODAS)
- Evaluation: 10 datasets (1 validation + 9 test), CER+WER+SER metrics; RU probes forgetting
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
│   ├── train_pipeline.py            # 9-step end-to-end pipeline
│   ├── setup_environment.sh          # All-in-one dependency fix (run once)
│   ├── runpod_auto.py               # RunPod pod lifecycle management
│   ├── build_manifest.py            # Audio → NeMo manifest JSON builder
│   ├── compute_metrics.py           # CER + WER + SER computation
│   ├── tts_augment.py               # gTTS-based rare term augmentation
│   ├── eval_checkpoints.py          # Multi-checkpoint × multi-dataset sweep
│   ├── eval_direct.py               # Direct .nemo evaluation (prompt-aware)
│   ├── patch_numba_codegen.py       # PTX version downgrade for driver 550
│   ├── verify_on_pod.sh             # 7-step pre-flight verification script
│   └── benchmark_gpu.py            # GPU speed/VRAM benchmark
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

## Training Command (Canonical — Proven 2026-06-17)

```bash
export PYTHONPATH=${NEMO_DIR}:${PYTHONPATH}  # CRITICAL: NeMo main before pip
bash scripts/setup_environment.sh              # Apply all patches

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
- `PYTHONPATH` MUST be set: NeMo main must precede pip-installed NeMo
- `limit_train_batches` — **never set** (NeMo Issue #15782: silently limits data per epoch)
- `max_steps` — **do not combine with max_epochs** (PyTorch Lightning stops at whichever triggers first)
- `batch_duration=100` — proven max for L40S 48GB (200 causes OOM risk, 400 crashes)
- `d_model=1024` — **NeMo 2.x only** (RunPod). NeMo 2.x config references `${model.encoder.d_model}`; must override. **NeMo 3.x** (DGX): `WarmupAnnealHoldPolicy` rejects `d_model` → use `~model.optim.sched.d_model` (Hydra delete) to remove from .nemo embedded config. `train_pipeline.py` handles this automatically via `_nemo_major()` version check.
- `warmup_steps=null` — must be null when using warmup_ratio
- Checkpoint output path: `$CHECKPOINT_DIR/FastConformer-Transducer-BPE-Prompt-Streaming/...`

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
