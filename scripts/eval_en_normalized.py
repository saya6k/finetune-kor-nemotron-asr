#!/usr/bin/env python3
"""One-off English WER probe: raw vs normalized, for a single checkpoint.

Separates formatting artifacts (casing/punctuation) from genuine word-level errors.
Reuses model loading from eval_one.py. Reports three WERs against the given manifest:
  - raw          : no normalization (what eval_one/watcher reports)
  - lowercase    : lowercase both sides only
  - normalized   : lowercase + strip punctuation both sides (standard ASR WER)

Usage:
    python3 scripts/eval_en_normalized.py \
        --ckpt /path/to/ckpt_or_nemo \
        --manifest /workspace/data/processed/test_fleurs_en.json \
        --base-nemo-dir /workspace/data/checkpoints_round5 \
        --label R3-best
"""
import argparse
import re
import string
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).parent))
from eval_one import (  # noqa: E402
    load_manifest, detect_lang, patch_prompt_dataloader,
    load_model_from_ckpt, load_model_from_nemo,
)

_PUNCT = str.maketrans("", "", string.punctuation)


def norm_lower(s):
    return s.lower().strip()


def norm_full(s):
    s = s.lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", s).strip()


def _extract_text(item):
    if isinstance(item, str):
        return item
    if hasattr(item, "text"):
        return item.text
    return str(item)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-nemo-dir", default="/workspace/data/checkpoints")
    ap.add_argument("--label", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    is_ckpt = Path(args.ckpt).suffix == ".ckpt"
    if is_ckpt:
        model = load_model_from_ckpt(args.ckpt, args.base_nemo_dir, args.device)
    else:
        model = load_model_from_nemo(args.ckpt, args.device)

    target_lang = detect_lang(args.manifest)
    patch_prompt_dataloader(target_lang)
    data = load_manifest(args.manifest)
    refs = [e["text"] for e in data]
    audio_paths = [e["audio_filepath"] for e in data]

    hyps = []
    bs = 32
    for i in range(0, len(audio_paths), bs):
        batch = audio_paths[i:i + bs]
        res = model.transcribe(batch, batch_size=len(batch), target_lang=target_lang, verbose=False)
        hyps.extend(_extract_text(r) for r in res)

    raw = jiwer.wer(refs, hyps) * 100
    low = jiwer.wer([norm_lower(r) for r in refs], [norm_lower(h) for h in hyps]) * 100
    full = jiwer.wer([norm_full(r) for r in refs], [norm_full(h) for h in hyps]) * 100

    print("\n========== EN WER PROBE ==========", flush=True)
    print(f"label     : {args.label}", flush=True)
    print(f"ckpt      : {Path(args.ckpt).name}", flush=True)
    print(f"manifest  : {Path(args.manifest).name}  (n={len(refs)})", flush=True)
    print(f"raw WER       : {raw:6.2f}", flush=True)
    print(f"lowercase WER : {low:6.2f}", flush=True)
    print(f"normalized WER: {full:6.2f}", flush=True)
    print("--- samples ---", flush=True)
    for r, h in list(zip(refs, hyps))[:3]:
        print(f"  REF: {r[:90]}", flush=True)
        print(f"  HYP: {h[:90]}", flush=True)
    print("==================================", flush=True)


if __name__ == "__main__":
    main()
