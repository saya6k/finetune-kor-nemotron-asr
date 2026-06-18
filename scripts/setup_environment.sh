#!/bin/bash
# ============================================================================
# Nemotron ASR Fine-Tuning — Comprehensive Environment Setup
# ============================================================================
# Run ONCE per pod before training. Idempotent (safe to re-run).
# Applies ALL known fixes for NeMo 2.7.3 (pip) + main branch (source).
#
# Target: runpod/pytorch:1.0.6-cu1281-torch260-ubuntu2204 (L40S, 48GB)
# Driver: 550.144.03 (CUDA 12.4, PTX 8.4 max)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=$(ls /usr/local/bin/python3.* 2>/dev/null | head -1)
PYTHON_BIN="${PYTHON_BIN:-python3}"
SENTINEL="/workspace/.setup_done"

if [ -f "$SENTINEL" ]; then
    echo "=== Setup already completed ($(cat $SENTINEL)) ==="
    echo "    To re-run: rm $SENTINEL && bash $0"
    exit 0
fi

echo "=== Nemotron ASR Environment Setup ==="
echo "  Python: $($PYTHON_BIN --version)"
echo "  PyTorch: $($PYTHON_BIN -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $($PYTHON_BIN -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# ── 1. Numba PTX version patch ─────────────────────────────────────────────
# Driver 550.144.03 supports max PTX 8.4. CUDA 12.8 toolkit generates PTX 8.7.
# WarpRNNT loss uses @cuda.jit which compiles to PTX at runtime → must downgrade.
echo "[1/9] Patching numba PTX version (8.7 → 8.4)..."
$PYTHON_BIN "$SCRIPT_DIR/patch_numba_codegen.py" 2>&1 || {
    echo "  WARNING: numba patch failed — first training step may crash"
    echo "  If training fails with 'PTX .version 8.7' error, check:"
    echo "  /usr/local/lib/python*/dist-packages/numba/cuda/codegen.py"
}

# ── 2. nv_one_logger stub ──────────────────────────────────────────────────
# NVIDIA internal package, not on PyPI. NeMo main imports it unconditionally.
echo "[2/9] Creating nv_one_logger stub..."
NEMO_LIGHTNING_DIR="$($PYTHON_BIN -c 'import nemo.lightning; import os; print(os.path.dirname(nemo.lightning.__file__))' 2>/dev/null || echo '')"
if [ -z "$NEMO_LIGHTNING_DIR" ]; then
    NEMO_LIGHTNING_DIR="/usr/local/lib/python3.12/dist-packages/nemo/lightning"
fi
mkdir -p "$NEMO_LIGHTNING_DIR"

# Only replace if the original contains the real import (not our stub)
if [ -f "$NEMO_LIGHTNING_DIR/one_logger_callback.py" ]; then
    if grep -q "from nv_one_logger" "$NEMO_LIGHTNING_DIR/one_logger_callback.py" 2>/dev/null; then
        cp "$NEMO_LIGHTNING_DIR/one_logger_callback.py" "$NEMO_LIGHTNING_DIR/one_logger_callback.py.bak"
        echo "  Backed up original one_logger_callback.py"
    fi
fi

cat > "$NEMO_LIGHTNING_DIR/one_logger_callback.py" << 'PYEOF'
"""Stub: nv_one_logger is NVIDIA internal, not available on PyPI."""
from lightning.pytorch.callbacks import Callback
class OneLoggerNeMoCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__()
PYEOF
echo "  Done"

# ── 3. Prompt model files ──────────────────────────────────────────────────
# EncDecRNNTBPEModelWithPrompt was added after NeMo 2.7.3 release.
# Copy from main branch clone to pip install location.
echo "[3/9] Copying prompt model files (NeMo main → pip)..."
NEMO_PKG="$($PYTHON_BIN -c 'import nemo; import os; print(os.path.dirname(nemo.__file__))' 2>/dev/null)"
NEMO_MAIN="${NEMO_DIR:-/workspace/NeMo}"

# Core prompt model
cp "$NEMO_MAIN/nemo/collections/asr/models/rnnt_bpe_models_prompt.py" \
   "$NEMO_PKG/collections/asr/models/" 2>/dev/null && echo "  rnnt_bpe_models_prompt.py OK" || {
    echo "  ERROR: Failed to copy rnnt_bpe_models_prompt.py"
    echo "  Check NEMO_DIR=$NEMO_MAIN"
}

# Prompt-aware Lhotse dataset
cp "$NEMO_MAIN/nemo/collections/asr/data/audio_to_text_lhotse_prompt_index.py" \
   "$NEMO_PKG/collections/asr/data/" 2>/dev/null && echo "  audio_to_text_lhotse_prompt_index.py OK" || echo "  WARNING: prompt index file copy failed"

# Mixins (PromptStreamingMixin)
MIXINS_DIR="$NEMO_PKG/collections/asr/parts/mixins"
mkdir -p "$MIXINS_DIR"
cp "$NEMO_MAIN/nemo/collections/asr/parts/mixins/__init__.py" "$MIXINS_DIR/" 2>/dev/null || true
cp "$NEMO_MAIN/nemo/collections/asr/parts/mixins/mixins.py" "$MIXINS_DIR/" 2>/dev/null && echo "  mixins.py OK" || echo "  WARNING: mixins.py copy failed"

# ── 4. NeMo main clone ─────────────────────────────────────────────────────
echo "[4/9] Verifying NeMo main clone..."
if [ ! -d "$NEMO_MAIN" ]; then
    echo "  Cloning NeMo main branch..."
    git clone --depth 1 https://github.com/NVIDIA/NeMo.git "$NEMO_MAIN" 2>&1 | tail -1
fi
echo "  NeMo main: $NEMO_MAIN ($(cd $NEMO_MAIN && git log --oneline -1 2>/dev/null || echo 'no git'))"

# ── 5. WarpRNNT verification ───────────────────────────────────────────────
echo "[5/9] Verifying WarpRNNT (numba JIT)..."
$PYTHON_BIN -c "
import torch
dummy = torch.zeros(1, 10, 1024, device='cuda')
print('  GPU tensor OK, numba JIT will compile on first training step')
" 2>&1 || echo "  WARNING: GPU check failed"

# ── 6. Torchcodec ──────────────────────────────────────────────────────────
echo "[6/9] Checking torchcodec (datasets Audio feature)..."
$PYTHON_BIN -c "import torchcodec" 2>/dev/null && echo "  torchcodec OK" || {
    echo "  Installing torchcodec..."
    pip install torchcodec 2>&1 | tail -2
}

# ── 7. Datasets Audio monkey-patch ─────────────────────────────────────────
# HuggingFace datasets >=2.14 requires torchcodec for Audio feature.
# We patch to use soundfile backend to avoid CUDA version coupling.
echo "[7/9] Patching datasets Audio backend..."
$PYTHON_BIN -c "
import os
os.environ['DATASETS_AUDIO_BACKEND'] = 'soundfile'
print('  DATASETS_AUDIO_BACKEND=soundfile')
"

# ── 8. Critical import verification ────────────────────────────────────────
echo "[8/9] Verifying critical imports..."
$PYTHON_BIN -c "
import sys
sys.path.insert(0, '$NEMO_MAIN')
from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import LhotseSpeechToTextBpeDatasetWithPromptIndex
print('  All critical imports OK (PYTHONPATH=$NEMO_MAIN)')
" 2>&1 | grep -v "NeMo W\|megatron\|nemo_logging" || {
    echo "  ERROR: Import verification failed"
    echo "  Check that NeMo main is at: $NEMO_MAIN"
    exit 1
}

# ── 9. Verify .nemo model exists ───────────────────────────────────────────
echo "[9/9] Checking base model..."
HF_CKPT="${HF_CKPT:-/workspace/data/base_model/nemotron-3.5-asr-streaming-0.6b.nemo}"
if [ -f "$HF_CKPT" ]; then
    echo "  Base model: $HF_CKPT ($(du -h $HF_CKPT | cut -f1))"
else
    echo "  WARNING: Base model not found at $HF_CKPT"
    echo "  Download with: huggingface-cli download nvidia/nemotron-3.5-asr-streaming-0.6b \\"
    echo "      nemotron-3.5-asr-streaming-0.6b.nemo --local-dir /workspace/data/base_model/"
fi

# ── Done ───────────────────────────────────────────────────────────────────
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$SENTINEL"
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Environment:"
echo "  PYTHONPATH: $NEMO_MAIN (MUST be set before training)"
echo "  BASE_MODEL: $HF_CKPT"
echo ""
echo "Known non-critical warnings:"
echo "  - CUDA graphs disabled (driver 12.4 < 12.6 needed)"
echo "  - Numba grid size warnings (cosmetic)"
echo "  - Lhotse dataloader config key warnings"
echo "  - Megatron num_microbatches_calculator (harmless)"
echo ""
echo "To train:"
echo "  export PYTHONPATH=$NEMO_MAIN:\$PYTHONPATH"
echo "  python3 train_pipeline.py"
