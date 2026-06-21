#!/usr/bin/env python3
"""Convert AIHub 자유대화(일반남녀) source audio + labels → NeMo manifest.

Label format (JSON):
  발화정보.stt    = transcript text
  발화정보.fileNm = audio filename (.wav)
  발화정보.recrdTime = duration in seconds

Usage:
  python3 convert_freetalk.py \
    --label-zip /path/to/Training/[라벨]2.음성수집도구.zip \
    --source-zip /path/to/Training/[원천]2.음성수집도구_5.zip \
    --wav-cache /workspace/data/wav_cache/free_talk \
    --output /workspace/data/processed/freetalk_train.jsonl \
    --max-hours 150
"""
import argparse, json, os, random, sys, time, zipfile
from pathlib import Path


def parse_labels(label_zip_path: str, max_utterances: int = 0) -> dict:
    """Extract and parse label JSONs from label zip.
    Returns dict mapping wav_filename -> (text, duration_seconds).
    """
    mapping = {}
    with zipfile.ZipFile(label_zip_path) as zf:
        json_files = [n for n in zf.namelist() if n.endswith(".json")]
        for name in json_files:
            try:
                data = json.loads(zf.read(name).decode("utf-8"))
                info = data.get("발화정보", {})
                text = info.get("stt", "").strip()
                file_nm = info.get("fileNm", "")
                # Normalize: some labels have .wavp instead of .wav
                if file_nm.endswith("p") and not file_nm.endswith(".zip"):
                    file_nm = file_nm[:-1]  # .wavp -> .wav
                duration = float(info.get("recrdTime", 0))
                # Clean noise markers
                text = text.replace("(NO:)", "").replace("(no:)", "").strip()
                if text and file_nm:
                    mapping[file_nm] = (text, duration)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        if max_utterances > 0:
            keys = list(mapping.keys())
            random.seed(42)
            random.shuffle(keys)
            mapping = {k: mapping[k] for k in keys[:max_utterances]}
    return mapping


def process_source(source_zip_path: str, label_map: dict,
                   wav_cache: Path, max_hours: float = 0):
    """Extract WAVs from source zip that match labels, build manifest entries."""
    entries = []
    total_sec = 0.0
    max_sec = max_hours * 3600 if max_hours > 0 else float("inf")
    wav_cache.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source_zip_path) as zf:
        wav_files = [n for n in zf.namelist() if n.endswith(".wav") and
                     n.split("/")[-1] in label_map]
        for wav_name in wav_files:
            basename = wav_name.split("/")[-1]
            if basename not in label_map:
                continue
            text, duration = label_map[basename]

            # Extract WAV
            wav_path = wav_cache / basename
            if not wav_path.exists():
                # Read from zip and write
                with zf.open(wav_name) as src:
                    with open(wav_path, "wb") as dst:
                        dst.write(src.read())

            entries.append({
                "audio_filepath": str(wav_path),
                "duration": round(duration, 3),
                "text": text,
                "lang": "ko-KR",
                "target_lang": "ko-KR",
            })
            total_sec += duration
            if total_sec >= max_sec:
                break

    return entries, total_sec / 3600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-zips", nargs="+", required=True)
    parser.add_argument("--source-zips", nargs="+", required=True)
    parser.add_argument("--wav-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-hours", type=float, default=300)
    args = parser.parse_args()

    wav_cache = Path(args.wav_cache)
    wav_cache.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build combined label map
    full_label_map = {}
    for lz in args.label_zips:
        print(f"Parsing labels: {Path(lz).name}", flush=True)
        label_map = parse_labels(lz)
        full_label_map.update(label_map)
        print(f"  {len(label_map)} valid labels", flush=True)
    print(f"Total labels: {len(full_label_map)}", flush=True)

    # Process source zips
    hours_per_source = args.max_hours / len(args.source_zips)
    all_entries = []
    t0 = time.time()
    for sz in args.source_zips:
        print(f"\nProcessing source: {Path(sz).name}", flush=True)
        entries, hours = process_source(sz, full_label_map, wav_cache,
                                        max_hours=hours_per_source)
        all_entries.extend(entries)
        print(f"  {len(entries)} entries, {hours:.1f}h", flush=True)

    total_h = sum(e["duration"] for e in all_entries) / 3600
    print(f"\nTotal: {len(all_entries)} entries, {total_h:.1f}h", flush=True)

    # Shuffle and write manifest
    random.seed(42)
    random.shuffle(all_entries)
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f}min → {out_path}", flush=True)


if __name__ == "__main__":
    main()
