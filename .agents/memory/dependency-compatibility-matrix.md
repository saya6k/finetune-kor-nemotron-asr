---
name: dependency-compatibility-matrix
description: "PyTorch, NeMo, CUDA version compatibility findings for RunPod ASR fine-tuning"
metadata: 
  node_type: memory
  type: project
  originSessionId: a56bbf84-5976-448a-8f18-2c8a12171796
---

# Dependency Compatibility Matrix for Nemotron ASR Fine-Tuning

## Proven Working Configuration (2026-06-17)

### Pod Image
- `runpod/pytorch:1.0.6-cu1281-torch260-ubuntu2204` (RunPod new naming scheme)
- GPU: L40S (48GB VRAM), Driver 550.144.03 (CUDA 12.4, PTX 8.4 max)
- CUDA Toolkit in container: 12.8 (CUDA forward-compatible with driver 12.4)
- Python: 3.12
- PyTorch: 2.6.0+cu126

### Software Stack
| Component | Version | Notes |
|-----------|---------|-------|
| NeMo (pip) | 2.7.3 | Base install, provides most ASR code |
| NeMo (source) | main branch | Prompt model, streaming config, example scripts |
| PyTorch | 2.6.0+cu126 | CUDA forward-compat: cu126 works on driver 12.4 |
| CUDA toolkit | 12.8 | Container toolkit > driver (forward compat) |
| Numba | latest | JIT-compiles WarpRNNT CUDA kernels |
| datasets | latest | Audio feature needs torchcodec or monkey-patch |
| torchcodec | latest | Now pip-installable for CUDA 12.8 |

### Key Patches Applied (setup_environment.sh)
1. **Numba PTX downgrade**: 8.7 → 8.4 (patch_numba_codegen.py)
2. **nv_one_logger stub**: NVIDIA internal package, not on PyPI
3. **Prompt model files**: Copy from NeMo main to pip install
4. **PYTHONPATH**: NeMo main MUST precede pip (for prompt model config)
5. **datasets Audio monkey-patch**: Use soundfile, avoid torchcodec coupling

### Training Config (Proven)
- Config: `fastconformer_transducer_bpe_streaming_prompt.yaml` (NeMo main only)
- batch_duration: 100 (L40S 48GB max — 400 causes OOM at step 82)
- Optimizer: AdamW, lr=1e-4, CosineAnnealing, warmup_ratio=0.05
- Scheduler: d_model=1024 explicit, warmup_steps=null (use ratio)
- exp_manager: main branch schema (create_early_stopping_callback, checkpoint_callback_params)
- Validation: batch_size=1, max_duration=20

### NeMo pip vs main Incompatibilities
- `RNNTBPEDecoding.set_strip_lang_tags`: main only (not in 2.7.3)
- `GreedyBatchedRNNTLabelLoopingComputer`: main has `preserve_step_confidence` kwarg
- `transducer_decoding/label_looping_base.py`: circular import if mixed
- **Strategy**: Use PYTHONPATH for main, DON'T mix files between main and pip
- Exception: prompt model files (rnnt_bpe_models_prompt.py, audio_to_text_lhotse_prompt_index.py) must be copied to pip

### Evaluation Pipeline
- `transcribe()` with prompt models: Cut language = None bug
- **Workaround**: Monkey-patch `LhotseSpeechToTextBpeDatasetWithPromptIndex._get_prompt_index_for_cut`
- Call with `target_lang=ko-KR` (NOT `prompt=ko-KR`)
- Handle Hypothesis objects in return (extract .text property)

### Models
- Base: `nvidia/nemotron-3.5-asr-streaming-0.6b` (638M params)
- License: OpenMDW-1.1
- Architecture: FastConformer-Transducer-BPE-Prompt (EncDecRNNTBPEModelWithPrompt)
