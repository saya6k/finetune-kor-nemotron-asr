#!/usr/bin/env python3
"""Rebuild FLEURS en_us test manifests directly from the dataset (clean, deduplicated).

The pre-existing test_fleurs_en.json has 647 entries but only 350 unique FLEURS ids
(297 duplicates) and uses `transcription` (lowercase, no punctuation). Lowercase refs
mismatch models trained on cased+punctuated text (e.g. VoxPopuli raw_text) and inflate WER.

This rebuilds both:
  test_fleurs_en.json      — `transcription` (lowercase), clean 350 entries
  test_fleurs_en_raw.json  — `raw_transcription` (cased + punctuated), clean 350 entries

Reuses already-cached WAVs (fleurs_en_us_<id>.wav). Duration from FLEURS num_samples.

Usage:
    python3 scripts/build_fleurs_en_raw.py --data-dir /workspace/data
"""
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset, Audio as HFAudio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--rebuild-lowercase", action="store_true",
                        help="also rebuild clean test_fleurs_en.json (lowercase)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    processed = data_dir / "processed"
    wav_cache = data_dir / "wav_cache"

    print("Loading FLEURS en_us test ...", flush=True)
    ds = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
    ds = ds.cast_column("audio", HFAudio(decode=False))

    raw_entries, low_entries = [], []
    missing_wav = 0
    seen = set()
    for s in ds:
        sid = str(s["id"])
        if sid in seen:
            continue
        seen.add(sid)
        wav_path = wav_cache / f"fleurs_en_us_{sid}.wav"
        if not wav_path.exists():
            missing_wav += 1
            continue
        duration = round(s["num_samples"] / 16000, 3)
        base = {
            "audio_filepath": str(wav_path),
            "duration": duration,
            "lang": "en-US",
            "target_lang": "en-US",
        }
        raw_entries.append({**base, "text": s["raw_transcription"].strip()})
        low_entries.append({**base, "text": s["transcription"].strip()})

    out_raw = processed / "test_fleurs_en_raw.json"
    with open(out_raw, "w", encoding="utf-8") as f:
        for e in raw_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Done: {len(raw_entries)} entries → {out_raw} (missing_wav={missing_wav})", flush=True)

    if args.rebuild_lowercase:
        out_low = processed / "test_fleurs_en.json"
        with open(out_low, "w", encoding="utf-8") as f:
            for e in low_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Done: {len(low_entries)} entries → {out_low}", flush=True)


if __name__ == "__main__":
    main()
