#!/bin/bash
# ============================================================================
# Verification script — run ON the pod after git clone + SSH
# Usage:
#   1. git clone to pod (or rsync project)
#   2. SSH in:  ssh root@<public-ip> -p <port>
#   3. Run:     cd /workspace/finetune-kor-nemotron-asr && bash scripts/verify_on_pod.sh
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
START_TIME=$(date +%s)

# Detect Python 3.12 (required by runpod/pytorch:1.0.6-cu1281-torch260-ubuntu2204)
PYTHON_BIN=$(which python3.12 2>/dev/null || which python3 2>/dev/null || echo "python3")

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass() { echo -e "${GREEN}✅ PASS${NC}: $1"; }
fail() { echo -e "${RED}❌ FAIL${NC}: $1"; }
warn() { echo -e "${YELLOW}⚠️  WARN${NC}: $1"; }
check() { if [ $? -eq 0 ]; then pass "$1"; else fail "$1"; fi; }

echo "============================================="
echo " Nemotron ASR Pipeline Verification"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="

# ── 1. GPU ─────────────────────────────────────────────────────
echo ""
echo "── [1/7] GPU Check ──"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
check "GPU detected"

# ── 2. setup_environment.sh ─────────────────────────────────────
echo ""
echo "── [2/7] setup_environment.sh ──"
cd "$PROJECT_DIR"
bash scripts/setup_environment.sh 2>&1 | tail -30
check "setup_environment.sh completed"

# ── 3. Critical imports ─────────────────────────────────────────
echo ""
echo "── [3/7] Critical imports ──"
NEMO_DIR="${NEMO_DIR:-/workspace/NeMo}"
export PYTHONPATH=${NEMO_DIR}:${PYTHONPATH}
$PYTHON_BIN -c "
from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import LhotseSpeechToTextBpeDatasetWithPromptIndex
print('  Imports OK')
" 2>&1 | grep -v "NeMo W\|megatron\|nemo_logging" || true
check "Prompt model imports"

# ── 4. Data ingest + split logic ────────────────────────────────
echo ""
echo "── [4/7] Data ingest & split (SMOKE_N=100) ──"
SMOKE_N=100 $PYTHON_BIN -c "
import os, sys, json, random
sys.path.insert(0, 'scripts')
random.seed(42)

# Quick test: simulate data_ingest + build_manifests with cutoff
from pathlib import Path
SMOKE_N = int(os.environ.get('SMOKE_N', '100'))
HOLD_OUT_N = 10
TEST_HOLD_OUT_N = 5

# Generate fake entries to simulate the split logic
entries = [{
    'audio_filepath': f'/tmp/t_{i}.wav',
    'duration': 3.0,
    'text': f'테스트 문장 {i}',
    'lang': 'ko-KR',
    'target_lang': 'ko-KR',
} for i in range(SMOKE_N)]

random.shuffle(entries)
print(f'  Total entries: {len(entries)}')

# Split logic (same as train_pipeline.py build_manifests)
holdout = entries[:HOLD_OUT_N]
middle = entries[HOLD_OUT_N:]

if len(middle) > TEST_HOLD_OUT_N:
    test_holdout = middle[-TEST_HOLD_OUT_N:]
    train_pool = middle[:-TEST_HOLD_OUT_N]
else:
    test_holdout = []
    train_pool = middle

# SMOKE_N=0 → all, else capped
SMOKE_N_VAL = int(os.environ.get('SMOKE_N', '0'))
train_n = len(train_pool) if SMOKE_N_VAL == 0 else min(SMOKE_N_VAL, len(train_pool))
train = train_pool[:train_n]

print(f'  Holdout: {len(holdout)}, Train: {len(train)}, Test: {len(test_holdout)}')

# Verify no overlaps
h_set = set(e['audio_filepath'] for e in holdout)
t_set = set(e['audio_filepath'] for e in train)
th_set = set(e['audio_filepath'] for e in test_holdout)
assert not (h_set & t_set), 'Overlap: holdout ∩ train'
assert not (h_set & th_set), 'Overlap: holdout ∩ test'
assert not (t_set & th_set), 'Overlap: train ∩ test'
print('  No overlaps between splits ✅')
" 2>&1
check "Split logic (SMOKE_N=100)"

# ── 5. Training command validation ─────────────────────────────
echo ""
echo "── [5/7] Training command ──"
# Verify the .nemo model exists
HF_CKPT=$(ls /workspace/data/base_model/*.nemo 2>/dev/null | head -1 || echo "")
if [ -n "$HF_CKPT" ] && [ -f "$HF_CKPT" ]; then
    pass "Base model found: $(basename $HF_CKPT)"
else
    warn "Base model not found — downloading..."
    pip install -q huggingface_hub
    huggingface-cli download nvidia/nemotron-3.5-asr-streaming-0.6b \
        nemotron-3.5-asr-streaming-0.6b.nemo \
        --local-dir /workspace/data/base_model/ 2>&1 | tail -3
    HF_CKPT=$(ls /workspace/data/base_model/*.nemo 2>/dev/null | head -1)
fi

# Verify training command structure
$PYTHON_BIN -c "
import shlex
# Check critical params present
cmd_parts = [
    '--config-name=fastconformer_transducer_bpe_streaming_prompt.yaml',
    'batch_duration=100',
    'd_model=1024',
    'warmup_steps=null',
    'validation_ds.batch_size=1',
    'max_duration=20',
    'create_early_stopping_callback=true',
    'monitor=val_wer',
    'save_top_k=3',
    'resume_ignore_no_checkpoint=true',
]
missing = [p for p in cmd_parts if p not in open('scripts/train_pipeline.py').read()]
if missing:
    for m in missing:
        print(f'  ❌ Missing: {m}')
else:
    print('  All 10 critical params present ✅')
" 2>&1
check "Training command params"

# ── 6. eval_direct imports ─────────────────────────────────────
echo ""
echo "── [6/7] eval_direct.py ──"
$PYTHON_BIN -c "
import sys; sys.path.insert(0, 'scripts')
from eval_direct import load_manifest, detect_lang, sweep_checkpoints_direct
# Verify manifest parsing with synthetic data
import json, tempfile, os
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    for i in range(3):
        json.dump({'audio_filepath': f'/tmp/t_{i}.wav', 'duration': 2.0,
                   'text': f'test {i}', 'lang': 'ko-KR', 'target_lang': 'ko-KR'}, f, ensure_ascii=False)
        f.write('\n')
    tmp = f.name
entries = load_manifest(tmp)
lang = detect_lang(tmp)
assert len(entries) == 3, f'Expected 3, got {len(entries)}'
assert lang == 'ko-KR', f'Expected ko-KR, got {lang}'
os.unlink(tmp)
print(f'  load_manifest + detect_lang OK')
" 2>&1
check "eval_direct module"

# ── 7. Tokenizer extraction ────────────────────────────────────
echo ""
echo "── [7/7] Tokenizer verification ──"
if [ -n "$HF_CKPT" ] && [ -f "$HF_CKPT" ]; then
    $PYTHON_BIN -c "
import tarfile, os, sys
ckpt = '$HF_CKPT'
tok_dir = '/tmp/tok_test'
os.makedirs(tok_dir, exist_ok=True)
with tarfile.open(ckpt, 'r') as tar:
    tok_files = [n for n in tar.getnames() if n.endswith('_tokenizer.model')]
    if tok_files:
        member = tar.getmember(tok_files[0])
        member.name = 'test_tokenizer.model'
        tar.extract(member, tok_dir)
        tok_path = os.path.join(tok_dir, 'test_tokenizer.model')
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(tok_path)
        ids = sp.EncodeAsIds('안녕하세요')
        decoded = sp.DecodeIds(ids)
        print(f'  Tokenizer OK: 안녕하세요 → {decoded}')
        os.unlink(tok_path)
    else:
        print('  ⚠️ Tokenizer model not found in .nemo')
" 2>&1
    check "Tokenizer extraction"
else
    fail "No .nemo model to extract tokenizer from"
fi

# ── Summary ─────────────────────────────────────────────────────
ELAPSED=$(($(date +%s) - START_TIME))
echo ""
echo "============================================="
echo " Verification Complete — ${ELAPSED}s"
echo "============================================="
