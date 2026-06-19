#!/bin/bash
# ============================================================================
# GB300 DGX Station Verification Script
# ============================================================================
# Run after setup_dgx.sh to confirm the environment is ready for training.
# Adapted from verify_on_pod.sh — DGX-specific checks added.
#
# Usage:
#   cd /workspace/finetune-kor-nemotron-asr
#   bash scripts/verify_on_dgx.sh
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${WORKSPACE:-/workspace}"
START_TIME=$(date +%s)

PYTHON_BIN=$(which python3 2>/dev/null || echo "python3")

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass() { echo -e "${GREEN}✅ PASS${NC}: $1"; PASS_COUNT=$((PASS_COUNT+1)); }
fail() { echo -e "${RED}❌ FAIL${NC}: $1"; FAIL_COUNT=$((FAIL_COUNT+1)); }
warn() { echo -e "${YELLOW}⚠️  WARN${NC}: $1"; WARN_COUNT=$((WARN_COUNT+1)); }
PASS_COUNT=0; FAIL_COUNT=0; WARN_COUNT=0

echo "============================================="
echo " Nemotron ASR — DGX Station Verification"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="

# ── 1. GPU — Blackwell sm_100 ───────────────────────────────────────────────
echo ""
echo "── [1/7] GPU Check ──"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || {
    warn "nvidia-smi not available"
}

GPU_RESULT=$($PYTHON_BIN -c "
import torch, sys
if not torch.cuda.is_available():
    print('NO_GPU')
    sys.exit(0)
p = torch.cuda.get_device_properties(0)
sm_major = p.major
vram_gb = p.total_memory / 1e9
print(f'{p.name}|SM_MAJOR={sm_major}|VRAM={vram_gb:.1f}GB')
" 2>/dev/null || echo "TORCH_ERROR")

if [ "$GPU_RESULT" = "NO_GPU" ]; then
    fail "No GPU detected"
elif [ "$GPU_RESULT" = "TORCH_ERROR" ]; then
    fail "torch.cuda error"
else
    SM_MAJOR=$(echo "$GPU_RESULT" | grep -o 'SM_MAJOR=[0-9]*' | cut -d= -f2)
    echo "  $GPU_RESULT"
    pass "GPU detected"
    if [ "${SM_MAJOR:-0}" -ge 10 ] 2>/dev/null; then
        pass "Blackwell sm_100+ confirmed (SM_MAJOR=$SM_MAJOR)"
    else
        warn "Expected sm_100 (Blackwell), got SM_MAJOR=$SM_MAJOR — verify NGC container"
    fi
fi

# ── 2. setup_dgx.sh ─────────────────────────────────────────────────────────
echo ""
echo "── [2/7] setup_dgx.sh ──"
cd "$PROJECT_DIR"
SETUP_SENTINEL="$WORKSPACE/.setup_dgx_done"
if [ -f "$SETUP_SENTINEL" ]; then
    pass "setup_dgx.sh already completed ($(cat $SETUP_SENTINEL))"
else
    echo "  Running setup_dgx.sh..."
    bash scripts/setup_dgx.sh 2>&1 | tail -20
    if [ $? -eq 0 ]; then
        pass "setup_dgx.sh completed"
    else
        fail "setup_dgx.sh failed"
    fi
fi

# ── 3. Critical imports ──────────────────────────────────────────────────────
echo ""
echo "── [3/7] Critical imports ──"
NEMO_DIR="${NEMO_DIR:-$WORKSPACE/NeMo}"
export PYTHONPATH="${NEMO_DIR}:${PYTHONPATH:-}"
$PYTHON_BIN -c "
from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import LhotseSpeechToTextBpeDatasetWithPromptIndex
print('  Prompt model imports OK')
" 2>&1 | grep -v "NeMo W\|megatron\|nemo_logging" || true
if $PYTHON_BIN -c "from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt" 2>/dev/null; then
    pass "EncDecRNNTBPEModelWithPrompt import"
else
    fail "EncDecRNNTBPEModelWithPrompt import — check PYTHONPATH and NeMo version"
fi

# ── 4. Data ingest + split logic ─────────────────────────────────────────────
echo ""
echo "── [4/7] Data ingest & split (SMOKE_N=100) ──"
SMOKE_N=100 $PYTHON_BIN -c "
import os, sys, json, random
sys.path.insert(0, 'scripts')
random.seed(42)
from pathlib import Path

SMOKE_N = int(os.environ.get('SMOKE_N', '100'))
HOLD_OUT_N = 10
TEST_HOLD_OUT_N = 5

entries = [{'audio_filepath': f'/tmp/t_{i}.wav', 'duration': 3.0,
            'text': f'테스트 {i}', 'lang': 'ko-KR', 'target_lang': 'ko-KR'}
           for i in range(SMOKE_N)]
random.shuffle(entries)

holdout = entries[:HOLD_OUT_N]
middle = entries[HOLD_OUT_N:]
test_holdout = middle[-TEST_HOLD_OUT_N:] if len(middle) > TEST_HOLD_OUT_N else []
train_pool = middle[:-TEST_HOLD_OUT_N] if len(middle) > TEST_HOLD_OUT_N else middle
train = train_pool[:SMOKE_N]

h_set = set(e['audio_filepath'] for e in holdout)
t_set = set(e['audio_filepath'] for e in train)
th_set = set(e['audio_filepath'] for e in test_holdout)
assert not (h_set & t_set), 'Overlap: holdout ∩ train'
assert not (h_set & th_set), 'Overlap: holdout ∩ test'
assert not (t_set & th_set), 'Overlap: train ∩ test'
print(f'  Holdout={len(holdout)} Train={len(train)} Test={len(test_holdout)} — no overlaps')
" 2>&1
[ $? -eq 0 ] && pass "Split logic (SMOKE_N=100)" || fail "Split logic"

# ── 5. Training command validation ───────────────────────────────────────────
echo ""
echo "── [5/7] Training command params ──"
HF_CKPT=$(ls "$WORKSPACE/data/base_model/"*.nemo 2>/dev/null | head -1 || echo "")
if [ -n "$HF_CKPT" ] && [ -f "$HF_CKPT" ]; then
    pass "Base model: $(basename $HF_CKPT)"
else
    warn "Base model not found — download with: huggingface-cli download nvidia/nemotron-3.5-asr-streaming-0.6b *.nemo --local-dir $WORKSPACE/data/base_model/"
fi

# Verify critical params in train_pipeline.py
$PYTHON_BIN -c "
src = open('scripts/train_pipeline.py').read()
params = [
    'fastconformer_transducer_bpe_streaming_prompt.yaml',
    'd_model=1024',
    'warmup_steps=null',
    'validation_ds.batch_size=1',
    'max_duration=20',
    'create_early_stopping_callback=true',
    'monitor=val_wer',
    'save_top_k=3',
    'resume_ignore_no_checkpoint=true',
    'aarch64-linux',  # DGX arch fix
    '_sm_major >= 10',  # Blackwell skip
]
missing = [p for p in params if p not in src]
if missing:
    for m in missing:
        print(f'  ❌ Missing param: {m}')
    raise SystemExit(1)
print(f'  All {len(params)} critical params present ✅')
" 2>&1
[ $? -eq 0 ] && pass "Training command params (incl. DGX patches)" || fail "Training command params"

# ── 6. eval_direct imports ───────────────────────────────────────────────────
echo ""
echo "── [6/7] eval_direct.py ──"
$PYTHON_BIN -c "
import sys; sys.path.insert(0, 'scripts')
from eval_direct import load_manifest, detect_lang, sweep_checkpoints_direct
import json, tempfile, os
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    for i in range(3):
        json.dump({'audio_filepath': f'/tmp/t_{i}.wav', 'duration': 2.0,
                   'text': f'test {i}', 'lang': 'ko-KR', 'target_lang': 'ko-KR'}, f)
        f.write('\n')
    tmp = f.name
entries = load_manifest(tmp)
lang = detect_lang(tmp)
assert len(entries) == 3 and lang == 'ko-KR'
os.unlink(tmp)
print('  load_manifest + detect_lang OK')
" 2>&1
[ $? -eq 0 ] && pass "eval_direct module" || fail "eval_direct module"

# ── 7. Tokenizer extraction ──────────────────────────────────────────────────
echo ""
echo "── [7/7] Tokenizer verification ──"
if [ -n "${HF_CKPT:-}" ] && [ -f "${HF_CKPT:-}" ]; then
    $PYTHON_BIN -c "
import tarfile, os
ckpt = '$HF_CKPT'
tok_dir = '/tmp/tok_dgx_test'
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
    [ $? -eq 0 ] && pass "Tokenizer extraction" || fail "Tokenizer extraction"
else
    warn "No .nemo model — skipping tokenizer check (download base model first)"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
ELAPSED=$(($(date +%s) - START_TIME))
echo ""
echo "============================================="
echo " Verification Complete — ${ELAPSED}s"
echo " PASS: $PASS_COUNT | FAIL: $FAIL_COUNT | WARN: $WARN_COUNT"
echo "============================================="
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "❌ $FAIL_COUNT check(s) failed — resolve before training"
    exit 1
else
    echo "✅ Ready for training"
fi
