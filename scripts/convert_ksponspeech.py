#!/usr/bin/env python3
"""KsponSpeech → NeMo manifest converter.

Handles the AI Hub KsponSpeech format:
  - PCM audio files inside numbered zip parts
  - .trn files with format: "path/to/file.pcm :: transcript"

Usage:
  python3 scripts/convert_ksponspeech.py \
    --trn-file /workspace/data/processed/ksponspeech/train.trn \
    --zip-dir /workspace/data/raw/ksponspeech/한국어_음성/한국어_음성_분야/ \
    --wav-cache /workspace/data/wav_cache/ksponspeech \
    --output /workspace/data/processed/ksponspeech_train.jsonl \
    --max-samples 50000   # ~400h equivalent
"""
import argparse, io, json, os, re, sys, time, zipfile
from pathlib import Path

import librosa
import soundfile as sf


def parse_trn(trn_path: str):
    """Parse KsponSpeech .trn file into list of (pcm_path, transcript)."""
    entries = []
    with open(trn_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or " :: " not in line:
                continue
            pcm_path, text = line.split(" :: ", 1)
            # Extract spoken (pronunciation) form: (표기형)/(발음형) → 발음형
            text = re.sub(r"\([^)]+\)/\(([^)]+)\)", r"\1", text)
            # Remove non-speech disfluency markers: b/ n/ l/ u/ o/ +/
            text = re.sub(r"\s?[bnluo]/\s?", " ", text)
            text = re.sub(r"\s?\+/\s?", " ", text)
            # Remove truncated words (word+) and inaudible (word*, *word, standalone *)
            text = re.sub(r"\S+\+", "", text)
            text = re.sub(r"\S*\*\S*", "", text)
            # Remove overlap slash while keeping the word: 그/ → 그
            text = re.sub(r"(\S)/", r"\1", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            entries.append((pcm_path.strip(), text))
    return entries


def pcm_to_wav(pcm_bytes: bytes, wav_path: Path, target_sr: int = 16000):
    """Convert raw PCM (16kHz/16bit/mono) bytes to WAV file."""
    import numpy as np
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(str(wav_path), arr, target_sr)
    return len(arr) / target_sr


def process_zip(zip_path: Path, trn_entries: list, wav_cache: Path,
                max_samples: int = 0, skip_existing: bool = True):
    """Extract PCM files from a zip and convert to WAV.

    Returns list of manifest entries written.
    """
    # trn_entries is a dict: {pcm_path: text}
    text_map = trn_entries
    manifest = []
    with zipfile.ZipFile(zip_path) as zf:
        pcm_files = [n for n in zf.namelist() if n.endswith(".pcm") and n in text_map]

        for pcm_name in pcm_files:
            wav_name = pcm_name.replace("/", "_").replace(".pcm", ".wav")
            wav_path = wav_cache / wav_name
            try:
                if skip_existing and wav_path.exists():
                    duration = librosa.get_duration(path=str(wav_path))
                else:
                    pcm_bytes = zf.read(pcm_name)
                    duration = pcm_to_wav(pcm_bytes, wav_path)
            except Exception:
                continue  # Skip corrupted/unreadable PCM files

            text = text_map.get(pcm_name, "")
            manifest.append({
                "audio_filepath": str(wav_path),
                "duration": round(duration, 3),
                "text": text,
                "lang": "ko-KR",
                "target_lang": "ko-KR",
            })

            if max_samples > 0 and len(manifest) >= max_samples:
                break

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trn-file", required=True)
    parser.add_argument("--zip-dir", required=True)
    parser.add_argument("--wav-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Max utterances (50000 ≈ 400h @ ~5s avg)")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="Max zip files to process (0=all)")
    args = parser.parse_args()

    wav_cache = Path(args.wav_cache)
    wav_cache.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse transcription
    print(f"Parsing {args.trn_file} ...", flush=True)
    entries = parse_trn(args.trn_file)
    # Build text lookup by pcm path
    text_map = dict(entries)
    print(f"  {len(entries)} valid utterances", flush=True)

    # Find zip files
    zip_dir = Path(args.zip_dir)
    zips = sorted(zip_dir.glob("KsponSpeech_*.zip"))
    if args.max_shards > 0:
        zips = zips[:args.max_shards]
    print(f"Found {len(zips)} zip files", flush=True)

    # Process zips
    total = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        for i, zp in enumerate(zips, 1):
            if args.max_samples > 0 and total >= args.max_samples:
                break

            remaining = args.max_samples - total if args.max_samples > 0 else 0
            manifest = process_zip(zp, text_map, wav_cache,
                                   max_samples=remaining)
            for entry in manifest:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total += len(manifest)

            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            print(f"  [{i}/{len(zips)}] +{len(manifest)} | "
                  f"total={total} | {rate:.0f} samples/s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {total} samples in {elapsed/60:.1f}min → {out_path}", flush=True)


if __name__ == "__main__":
    main()
