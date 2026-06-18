#!/bin/bash
set -euo pipefail

# Source environment
export $(grep -v '^#' /workspace/.env | xargs)

# Pipeline defaults
export WORKSPACE=/workspace
export DATA_DIR=/workspace/data
export NEMO_DIR=/workspace/NeMo
export SMOKE_N=${SMOKE_N:-5000}
export HOLD_OUT_N=${HOLD_OUT_N:-1000}
export MAX_EPOCHS=${MAX_EPOCHS:-3}
export BATCH_DURATION=${BATCH_DURATION:-200}
export HF_REPO_ID=${HF_REPO_ID:-saya6k/nemotron-kor-checkpoints}
export AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-false}

# Use the Python with PyTorch (3.12)
export PATH="/usr/local/bin:$PATH"

echo "=== Pipeline Environment ==="
echo "Python: $(python3.12 --version)"
echo "PyTorch: $(python3.12 -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python3.12 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)MiB"
echo "Workspace: $WORKSPACE"
echo "============================"

cd /workspace
exec python3.12 scripts/train_pipeline.py 2>&1
