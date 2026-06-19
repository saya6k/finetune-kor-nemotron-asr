#!/bin/bash
# ============================================================================
# GB300 DGX Station — Environment Probe
# ============================================================================
# Run ONCE on a fresh DGX Station (before any patches) to determine which
# RunPod patches are still needed on the new hardware.
#
# Usage:
#   bash scripts/probe_dgx_environment.sh
#
# Output:
#   - Colored table to stdout
#   - ${WORKSPACE:-/workspace}/probe_results.txt (plain text copy)
#
# Each check outputs one of:
#   NOT_NEEDED  (green)  — patch can be skipped
#   NEEDED      (yellow) — patch should be applied
#   UNKNOWN     (red)    — could not determine (e.g. no GPU)
# ============================================================================
set -uo pipefail
# Note: no -e — we continue even if individual checks fail

WORKSPACE="${WORKSPACE:-/workspace}"
RESULTS_FILE="$WORKSPACE/probe_results.txt"
PYTHON_BIN=$(which python3 2>/dev/null || echo "python3")

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
BOLD='\033[1m'

needed()     { echo -e "  ${YELLOW}NEEDED${NC}     $1"; echo "NEEDED     $1" >> "$RESULTS_FILE"; }
not_needed() { echo -e "  ${GREEN}NOT_NEEDED${NC} $1"; echo "NOT_NEEDED $1" >> "$RESULTS_FILE"; }
unknown()    { echo -e "  ${RED}UNKNOWN${NC}    $1"; echo "UNKNOWN    $1" >> "$RESULTS_FILE"; }
info()       { echo -e "             $1"; echo "INFO       $1" >> "$RESULTS_FILE"; }

# ── Init results file ───────────────────────────────────────────────────────
mkdir -p "$WORKSPACE"
echo "# GB300 DGX Station Environment Probe" > "$RESULTS_FILE"
echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo ""
echo -e "${BOLD}=============================================${NC}"
echo -e "${BOLD} GB300 DGX Station Environment Probe${NC}"
echo -e "${BOLD} $(date -u +%Y-%m-%dT%H:%M:%SZ)${NC}"
echo -e "${BOLD}=============================================${NC}"
echo ""

# ── [1] GPU: Name, SM version, VRAM ─────────────────────────────────────────
echo -e "${BOLD}── [1/8] GPU${NC}"
GPU_INFO=$($PYTHON_BIN -c "
import torch, sys
if not torch.cuda.is_available():
    print('NO_GPU')
    sys.exit(0)
p = torch.cuda.get_device_properties(0)
sm = f'{p.major}.{p.minor}'
vram_gb = p.total_memory / 1e9
print(f'{p.name}|SM={sm}|VRAM={vram_gb:.1f}GB')
" 2>/dev/null || echo "TORCH_ERROR")

if [ "$GPU_INFO" = "NO_GPU" ] || [ "$GPU_INFO" = "TORCH_ERROR" ]; then
    unknown "GPU not detected or torch error — cannot determine SM version"
else
    GPU_NAME=$(echo "$GPU_INFO" | cut -d'|' -f1)
    SM_VER=$(echo "$GPU_INFO" | cut -d'|' -f2 | cut -d'=' -f2)
    VRAM=$(echo "$GPU_INFO" | cut -d'|' -f3 | cut -d'=' -f2)
    SM_MAJOR=$(echo "$SM_VER" | cut -d'.' -f1)
    info "GPU: $GPU_NAME | SM: $SM_VER | VRAM: $VRAM"
    echo "GPU_NAME=$GPU_NAME" >> "$RESULTS_FILE"
    echo "SM_MAJOR=$SM_MAJOR" >> "$RESULTS_FILE"
    if [ "$SM_MAJOR" -ge 10 ] 2>/dev/null; then
        not_needed "[Patch 1] Blackwell sm_100+ — Numba PTX downgrade NOT needed"
    else
        needed "[Patch 1] Non-Blackwell GPU — Numba PTX downgrade may be needed"
    fi
fi

# ── [2] CUDA: Toolkit vs Driver version match ────────────────────────────────
echo ""
echo -e "${BOLD}── [2/8] CUDA Toolkit vs Driver${NC}"
TOOLKIT_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/' || echo "UNKNOWN")
DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "UNKNOWN")
TORCH_CUDA=$($PYTHON_BIN -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "UNKNOWN")

info "nvcc toolkit:  $TOOLKIT_VER"
info "driver CUDA:   $DRIVER_CUDA"
info "torch CUDA:    $TORCH_CUDA"
echo "CUDA_TOOLKIT=$TOOLKIT_VER" >> "$RESULTS_FILE"
echo "CUDA_DRIVER=$DRIVER_CUDA" >> "$RESULTS_FILE"

# If toolkit major == driver major, no PTX version mismatch
TK_MAJOR=$(echo "$TOOLKIT_VER" | cut -d'.' -f1)
DRV_MAJOR=$(echo "$DRIVER_CUDA" | cut -d'.' -f1)
if [ "$TK_MAJOR" = "$DRV_MAJOR" ] 2>/dev/null; then
    not_needed "CUDA toolkit/driver major versions match — no PTX version mismatch"
elif [ "$TOOLKIT_VER" = "UNKNOWN" ] || [ "$DRIVER_CUDA" = "UNKNOWN" ]; then
    unknown "Could not determine CUDA toolkit or driver version"
else
    needed "CUDA toolkit ($TOOLKIT_VER) > driver ($DRIVER_CUDA) — PTX mismatch possible"
fi

# ── [3] Python: Version + Architecture ───────────────────────────────────────
echo ""
echo -e "${BOLD}── [3/8] Python${NC}"
PY_VER=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
PY_ARCH=$($PYTHON_BIN -c "import platform; print(platform.machine())" 2>/dev/null || echo "UNKNOWN")
info "Python: $PY_VER | Arch: $PY_ARCH"
echo "PYTHON_VERSION=$PY_VER" >> "$RESULTS_FILE"
echo "PYTHON_ARCH=$PY_ARCH" >> "$RESULTS_FILE"
if [ "$PY_ARCH" = "aarch64" ]; then
    info "aarch64 (Grace CPU) confirmed"
elif [ "$PY_ARCH" = "x86_64" ]; then
    info "x86_64 (standard)"
else
    unknown "Architecture: $PY_ARCH"
fi

# ── [4] NeMo: Version + EncDecRNNTBPEModelWithPrompt ─────────────────────────
echo ""
echo -e "${BOLD}── [4/8] NeMo + Prompt Model${NC}"
NEMO_VER=$($PYTHON_BIN -c "import nemo; print(nemo.__version__)" 2>/dev/null || echo "NOT_INSTALLED")
info "NeMo pip version: $NEMO_VER"
echo "NEMO_VERSION=$NEMO_VER" >> "$RESULTS_FILE"

PROMPT_IMPORT=$($PYTHON_BIN -c "
from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
print('OK')
" 2>/dev/null || echo "FAIL")
echo "PROMPT_MODEL_IMPORT=$PROMPT_IMPORT" >> "$RESULTS_FILE"
if [ "$PROMPT_IMPORT" = "OK" ]; then
    not_needed "[Patch 3,4] EncDecRNNTBPEModelWithPrompt available in pip install — NeMo main clone may not be needed"
else
    needed "[Patch 3,4] EncDecRNNTBPEModelWithPrompt NOT in pip — NeMo main clone + PYTHONPATH required"
fi

# ── [5] nv_one_logger ────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}── [5/8] nv_one_logger${NC}"
NV_ONE_LOGGER=$($PYTHON_BIN -c "import nv_one_logger; print('OK')" 2>/dev/null || echo "FAIL")
echo "NV_ONE_LOGGER=$NV_ONE_LOGGER" >> "$RESULTS_FILE"
if [ "$NV_ONE_LOGGER" = "OK" ]; then
    not_needed "[Patch 2] nv_one_logger available — stub NOT needed"
else
    needed "[Patch 2] nv_one_logger not found — stub required"
fi

# ── [6] Numba: PTX version + sm compatibility ────────────────────────────────
echo ""
echo -e "${BOLD}── [6/8] Numba PTX${NC}"
NUMBA_INFO=$($PYTHON_BIN -c "
import numba
print(f'numba={numba.__version__}')
try:
    from numba.cuda.cudadrv import nvvm
    ptx = nvvm.NVVM().get_ir_version()
    print(f'PTX={ptx}')
except Exception as e:
    print(f'PTX=UNKNOWN ({e})')
" 2>/dev/null || echo "NOT_INSTALLED")
info "$NUMBA_INFO"
echo "NUMBA_INFO=$NUMBA_INFO" >> "$RESULTS_FILE"

# Check if codegen.py already patched
NUMBA_PURELIB=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null || echo "")
CODEGEN_PATH="$NUMBA_PURELIB/numba/cuda/codegen.py"
if [ -f "$CODEGEN_PATH" ]; then
    if grep -q "_patch_re" "$CODEGEN_PATH" 2>/dev/null; then
        not_needed "[Patch 1] Numba codegen.py already patched"
    else
        info "Numba codegen.py at: $CODEGEN_PATH (not yet patched)"
    fi
else
    unknown "Numba codegen.py not found at expected path: $CODEGEN_PATH"
fi

# ── [7] datasets Audio backend ───────────────────────────────────────────────
echo ""
echo -e "${BOLD}── [7/8] datasets Audio Backend${NC}"
AUDIO_BACKEND=$($PYTHON_BIN -c "
import os
# Check default without env override
try:
    import datasets.features.audio as _a
    import inspect
    src = inspect.getsource(_a)
    if 'torchcodec' in src and 'soundfile' in src:
        print('BOTH_SUPPORTED')
    elif 'torchcodec' in src:
        print('TORCHCODEC_ONLY')
    elif 'soundfile' in src:
        print('SOUNDFILE_ONLY')
    else:
        print('UNKNOWN')
except ImportError:
    print('DATASETS_NOT_INSTALLED')
" 2>/dev/null || echo "ERROR")
info "datasets Audio: $AUDIO_BACKEND"
echo "DATASETS_AUDIO=$AUDIO_BACKEND" >> "$RESULTS_FILE"

TORCHCODEC_OK=$($PYTHON_BIN -c "import torchcodec; print('OK')" 2>/dev/null || echo "FAIL")
info "torchcodec import: $TORCHCODEC_OK"
echo "TORCHCODEC=$TORCHCODEC_OK" >> "$RESULTS_FILE"
if [ "$TORCHCODEC_OK" = "OK" ]; then
    not_needed "[Patch 6] torchcodec available — datasets Audio should work"
else
    needed "[Patch 6,7] torchcodec missing — soundfile monkey-patch needed"
fi

# ── [8] PYTHONPATH: NeMo main needed for prompt config YAML? ─────────────────
echo ""
echo -e "${BOLD}── [8/8] PYTHONPATH / Prompt Config YAML${NC}"
# Check if streaming prompt config exists in pip install
PROMPT_YAML=$($PYTHON_BIN -c "
import nemo, os
nemo_dir = os.path.dirname(nemo.__file__)
yaml_path = os.path.join(nemo_dir, 'examples', 'asr', 'conf', 'fastconformer',
    'cache_aware_streaming', 'fastconformer_transducer_bpe_streaming_prompt.yaml')
alt_path = os.path.join(os.path.dirname(nemo_dir), 'examples', 'asr', 'conf',
    'fastconformer', 'cache_aware_streaming',
    'fastconformer_transducer_bpe_streaming_prompt.yaml')
for p in [yaml_path, alt_path]:
    if os.path.exists(p):
        print(f'FOUND:{p}')
        break
else:
    print('NOT_FOUND')
" 2>/dev/null || echo "ERROR")
info "Prompt config YAML: $PROMPT_YAML"
echo "PROMPT_YAML=$PROMPT_YAML" >> "$RESULTS_FILE"

if echo "$PROMPT_YAML" | grep -q "^FOUND:"; then
    not_needed "[Patch 4] Prompt config YAML found in pip install — NeMo main clone may not be needed for YAML"
else
    needed "[Patch 4] Prompt config YAML not in pip install — NeMo main clone required"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}=============================================${NC}"
echo -e "${BOLD} Summary${NC}"
echo -e "${BOLD}=============================================${NC}"
NEEDED_COUNT=$(grep -c "^NEEDED" "$RESULTS_FILE" 2>/dev/null || echo 0)
NOT_NEEDED_COUNT=$(grep -c "^NOT_NEEDED" "$RESULTS_FILE" 2>/dev/null || echo 0)
UNKNOWN_COUNT=$(grep -c "^UNKNOWN" "$RESULTS_FILE" 2>/dev/null || echo 0)
echo -e "  ${GREEN}NOT_NEEDED${NC}: $NOT_NEEDED_COUNT checks"
echo -e "  ${YELLOW}NEEDED${NC}:     $NEEDED_COUNT checks"
echo -e "  ${RED}UNKNOWN${NC}:    $UNKNOWN_COUNT checks"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Next step: Update SPEC.md §4 with these findings, then run:"
echo "  bash scripts/setup_dgx.sh"
