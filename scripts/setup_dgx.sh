#!/bin/bash
# ============================================================================
# Nemotron ASR — DGX Station Setup
# ============================================================================
# Run ONCE per container session before training. Idempotent (safe to re-run).
#
# Applies only the patches confirmed necessary on GB300 DGX Station.
# Each patch section checks if it's needed before applying.
#
# Run probe_dgx_environment.sh FIRST to understand what will be skipped.
#
# Target: NGC NeMo container on GB300 DGX Station (aarch64 / sm_100 Blackwell)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=$(which python3 2>/dev/null || echo "python3")
WORKSPACE="${WORKSPACE:-/workspace}"
NEMO_DIR="${NEMO_DIR:-$WORKSPACE/NeMo}"
SENTINEL="$WORKSPACE/.setup_dgx_done"

if [ -f "$SENTINEL" ]; then
    echo "=== DGX Setup already completed ($(cat $SENTINEL)) ==="
    echo "    To re-run: rm $SENTINEL && bash $0"
    exit 0
fi

echo "=== Nemotron ASR — DGX Station Setup ==="
echo "  Python:     $($PYTHON_BIN --version)"
echo "  Arch:       $($PYTHON_BIN -c 'import platform; print(platform.machine())')"
echo "  CUDA avail: $($PYTHON_BIN -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# Detect GPU SM version for conditional patches
SM_MAJOR=$($PYTHON_BIN -c "
import torch, sys
if torch.cuda.is_available():
    print(torch.cuda.get_device_capability(0)[0])
else:
    print(0)
" 2>/dev/null || echo "0")
echo "  SM_MAJOR: $SM_MAJOR"
echo ""

# ── 1. Numba PTX patch ─────────────────────────────────────────────────────
# SKIP on Blackwell (sm_100+): toolkit and driver match on DGX Station.
# Blackwell uses PTX 9.x natively — downgrading to 8.4 would break things.
echo "[1/9] Numba PTX patch..."
if [ "${SM_MAJOR:-0}" -ge 10 ] 2>/dev/null; then
    echo "  SKIP: Blackwell sm_100 — PTX downgrade not needed"
else
    echo "  Applying PTX 8.4 downgrade (non-Blackwell GPU detected)..."
    $PYTHON_BIN "$SCRIPT_DIR/patch_numba_codegen.py" 2>&1 || {
        echo "  WARNING: numba patch failed — first training step may crash with PTX error"
    }
fi

# ── 2. nv_one_logger stub ──────────────────────────────────────────────────
# NGC NeMo container typically includes nv_one_logger — skip if already importable.
echo "[2/9] nv_one_logger..."
if $PYTHON_BIN -c "import nv_one_logger" 2>/dev/null; then
    echo "  SKIP: nv_one_logger already available"
else
    echo "  Creating nv_one_logger stub..."
    NEMO_LIGHTNING_DIR="$($PYTHON_BIN -c 'import nemo.lightning, os; print(os.path.dirname(nemo.lightning.__file__))' 2>/dev/null || echo "$WORKSPACE/nemo_stub/nemo/lightning")"
    mkdir -p "$NEMO_LIGHTNING_DIR"
    cat > "$NEMO_LIGHTNING_DIR/one_logger_callback.py" << 'PYEOF'
"""Stub: nv_one_logger is NVIDIA internal, not available on PyPI."""
from lightning.pytorch.callbacks import Callback
class OneLoggerNeMoCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__()
PYEOF
    echo "  Stub created: $NEMO_LIGHTNING_DIR/one_logger_callback.py"
fi

# ── 3 & 4. Prompt model files + NeMo main clone ───────────────────────────
# If EncDecRNNTBPEModelWithPrompt is already in the pip install (NGC NeMo latest),
# skip the clone. Otherwise, clone NeMo main and copy prompt files.
echo "[3/9] Prompt model (EncDecRNNTBPEModelWithPrompt)..."
PROMPT_IMPORT=$($PYTHON_BIN -c "
from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
print('OK')
" 2>/dev/null || echo "FAIL")

if [ "$PROMPT_IMPORT" = "OK" ]; then
    echo "  SKIP: EncDecRNNTBPEModelWithPrompt already importable from pip install"
else
    echo "  Not in pip install — cloning NeMo main branch..."
    echo "[4/9] NeMo main clone..."
    if [ ! -d "$NEMO_DIR" ]; then
        git clone --depth 1 https://github.com/NVIDIA/NeMo.git "$NEMO_DIR" 2>&1 | tail -1
    else
        echo "  NeMo main already at $NEMO_DIR ($(cd $NEMO_DIR && git log --oneline -1 2>/dev/null || echo 'no git'))"
    fi

    export PYTHONPATH="${NEMO_DIR}:${PYTHONPATH:-}"
    NEMO_PKG="$($PYTHON_BIN -c 'import nemo, os; print(os.path.dirname(nemo.__file__))' 2>/dev/null)"

    # Copy prompt model files
    for src_rel in \
        "nemo/collections/asr/models/rnnt_bpe_models_prompt.py" \
        "nemo/collections/asr/data/audio_to_text_lhotse_prompt_index.py"; do
        src="$NEMO_DIR/$src_rel"
        dst_dir="$NEMO_PKG/$(dirname "${src_rel#nemo/}")"
        mkdir -p "$dst_dir"
        if [ -f "$src" ]; then
            cp "$src" "$dst_dir/" && echo "  Copied: $(basename $src)" || echo "  WARNING: copy failed for $src"
        else
            echo "  WARNING: $src not found in NeMo main"
        fi
    done

    # Copy mixins
    MIXINS_SRC="$NEMO_DIR/nemo/collections/asr/parts/mixins"
    MIXINS_DST="$NEMO_PKG/collections/asr/parts/mixins"
    if [ -d "$MIXINS_SRC" ]; then
        mkdir -p "$MIXINS_DST"
        cp "$MIXINS_SRC/__init__.py" "$MIXINS_DST/" 2>/dev/null || true
        cp "$MIXINS_SRC/mixins.py" "$MIXINS_DST/" 2>/dev/null && echo "  mixins.py OK" || echo "  WARNING: mixins.py copy failed"
    fi
fi

# ── 4. NeMo main PYTHONPATH (always set if clone exists) ──────────────────
echo "[4/9] PYTHONPATH..."
if [ -d "$NEMO_DIR" ]; then
    export PYTHONPATH="${NEMO_DIR}:${PYTHONPATH:-}"
    echo "  PYTHONPATH set: $NEMO_DIR"
else
    echo "  SKIP: NeMo main not cloned (prompt model available in pip)"
fi

# ── 5. WarpRNNT GPU check ─────────────────────────────────────────────────
echo "[5/9] WarpRNNT GPU check (sm_100)..."
$PYTHON_BIN -c "
import torch
if torch.cuda.is_available():
    dummy = torch.zeros(1, 10, 1024, device='cuda')
    sm = torch.cuda.get_device_capability(0)
    print(f'  GPU tensor OK — SM {sm[0]}.{sm[1]}')
    if sm[0] >= 10:
        print('  Blackwell sm_100 — WarpRNNT will compile for sm_100 on first training step')
    else:
        print(f'  GPU SM {sm[0]}.{sm[1]} — WarpRNNT numba JIT will compile on first step')
else:
    print('  WARNING: No GPU — skipping GPU check')
" 2>&1 || echo "  WARNING: GPU tensor check failed"

# ── 6. torchcodec ────────────────────────────────────────────────────────
echo "[6/9] torchcodec..."
if $PYTHON_BIN -c "import torchcodec" 2>/dev/null; then
    echo "  torchcodec OK (already installed)"
else
    echo "  Installing torchcodec..."
    pip install torchcodec 2>&1 | tail -2
fi

# ── 7. datasets Audio backend ────────────────────────────────────────────
echo "[7/9] datasets Audio backend..."
$PYTHON_BIN -c "
import os
os.environ['DATASETS_AUDIO_BACKEND'] = 'soundfile'
print('  DATASETS_AUDIO_BACKEND=soundfile (set in environment)')
"
# Ensure the env var persists for this session
export DATASETS_AUDIO_BACKEND=soundfile

# ── 8. Critical import verification ─────────────────────────────────────
echo "[8/9] Critical imports..."
$PYTHON_BIN -c "
import sys
nemo_dir = '${NEMO_DIR}'
import os
if os.path.isdir(nemo_dir):
    sys.path.insert(0, nemo_dir)
from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import LhotseSpeechToTextBpeDatasetWithPromptIndex
print('  All critical imports OK')
" 2>&1 | grep -v "NeMo W\|megatron\|nemo_logging" || {
    echo "  ERROR: Import verification failed"
    echo "  Check NeMo version and PYTHONPATH"
    exit 1
}

# ── 9. Base model check ──────────────────────────────────────────────────
echo "[9/9] Base model..."
HF_CKPT="${HF_CKPT:-$WORKSPACE/data/base_model/nemotron-3.5-asr-streaming-0.6b.nemo}"
if [ -f "$HF_CKPT" ]; then
    echo "  Base model: $HF_CKPT ($(du -h $HF_CKPT | cut -f1))"
else
    echo "  WARNING: Base model not found at $HF_CKPT"
    echo "  Download: huggingface-cli download nvidia/nemotron-3.5-asr-streaming-0.6b \\"
    echo "      nemotron-3.5-asr-streaming-0.6b.nemo --local-dir $(dirname $HF_CKPT)"
fi

# ── Done ─────────────────────────────────────────────────────────────────
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$SENTINEL"
echo ""
echo "=== DGX Setup Complete ==="
echo ""
echo "Environment:"
if [ -d "$NEMO_DIR" ]; then
    echo "  PYTHONPATH: $NEMO_DIR (set — NeMo main for prompt model)"
else
    echo "  PYTHONPATH: not needed (prompt model in NGC NeMo pip install)"
fi
echo "  BASE_MODEL: $HF_CKPT"
echo "  DATASETS_AUDIO_BACKEND: soundfile"
echo ""
echo "To train:"
[ -d "$NEMO_DIR" ] && echo "  export PYTHONPATH=$NEMO_DIR:\$PYTHONPATH"
echo "  export DATASETS_AUDIO_BACKEND=soundfile"
echo "  python3 scripts/train_pipeline.py"
