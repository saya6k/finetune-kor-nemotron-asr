#!/usr/bin/env python3
"""Download VoxPopuli English and convert to NeMo manifest.

Audio: ogg/vorbis via HF datasets → saved as 16kHz mono WAV.
Text: normalized_text field (ASR-normalized, lowercase, no punctuation).
      Pass --use-raw-text to use raw parliamentary transcripts (proper case + punctuation).

Usage:
    python3 scripts/convert_voxpopuli_en.py \
        --wav-cache /workspace/data/wav_cache/voxpopuli_en \
        --output /workspace/data/processed/voxpopuli_en.jsonl
"""
import argparse
import json
import re
import time
from pathlib import Path

import io
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio as HFAudio

SAMPLE_RATE = 16000


def save_wav(array: np.ndarray, sr: int, out_path: Path) -> None:
    if sr != SAMPLE_RATE:
        import librosa
        array = librosa.resample(array.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    sf.write(str(out_path), array, SAMPLE_RATE, subtype="PCM_16")


def clean_raw_text(text: str) -> str:
    # Light normalization for raw parliamentary text: remove XML tags, normalize whitespace
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-cache", default="/workspace/data/wav_cache/voxpopuli_en")
    parser.add_argument("--output", default="/workspace/data/processed/voxpopuli_en.jsonl")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--gold-only", action="store_true", default=True,
                        help="Only include gold (human) transcripts")
    parser.add_argument("--use-raw-text", action="store_true", default=False,
                        help="Use raw parliamentary text (proper case+punctuation) instead of normalized_text")
    args = parser.parse_args()

    wav_cache = Path(args.wav_cache)
    wav_cache.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text_field = "raw_text" if args.use_raw_text else "normalized_text"
    print(f"Loading VoxPopuli English (text_field={text_field}, gold_only={args.gold_only})...", flush=True)

    ds = load_dataset("facebook/voxpopuli", "en", split="train", streaming=True)
    ds = ds.cast_column("audio", HFAudio(decode=False))  # raw bytes → manual soundfile decode

    entries = []
    skipped_gold = 0
    skipped_text = 0
    skipped_dur = 0
    t0 = time.time()

    for i, sample in enumerate(ds):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            print(f"  [{i}] kept={len(entries)} skipped_gold={skipped_gold} "
                  f"skipped_text={skipped_text} skipped_dur={skipped_dur} "
                  f"rate={rate:.0f}/s", flush=True)

        if args.gold_only and not sample.get("is_gold_transcript", False):
            skipped_gold += 1
            continue

        text = (sample.get(text_field) or "").strip()
        if args.use_raw_text:
            text = clean_raw_text(text)
        if not text:
            skipped_text += 1
            continue

        audio = sample["audio"]
        audio_bytes = audio.get("bytes")
        audio_path = audio.get("path")
        try:
            src = io.BytesIO(audio_bytes) if audio_bytes else audio_path
            with sf.SoundFile(src) as f:
                array = f.read(dtype="float32")
                sr = f.samplerate
        except Exception:
            skipped_dur += 1
            continue
        duration = len(array) / sr

        if duration < args.min_duration or duration > args.max_duration:
            skipped_dur += 1
            continue

        audio_id = sample.get("audio_id") or f"vox_{i:07d}"
        wav_path = wav_cache / f"{audio_id}.wav"
        if not wav_path.exists():
            save_wav(array, sr, wav_path)

        entries.append({
            "audio_filepath": str(wav_path),
            "duration": round(duration, 3),
            "text": text,
            "lang": "en-US",
            "target_lang": "en-US",
        })

        if args.max_samples > 0 and len(entries) >= args.max_samples:
            break

    total_h = sum(e["duration"] for e in entries) / 3600
    print(f"\nDone: {len(entries)} entries, {total_h:.1f}h", flush=True)
    print(f"Skipped: gold={skipped_gold}, empty_text={skipped_text}, duration={skipped_dur}", flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Written → {out_path}", flush=True)


if __name__ == "__main__":
    main()
