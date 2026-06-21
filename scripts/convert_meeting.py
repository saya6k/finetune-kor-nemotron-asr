#!/usr/bin/env python3
"""Convert AIHub 회의음성 label JSONs + source WAV → NeMo manifest.

Segments individual utterances using start/end timestamps from label JSON.
Each utterance becomes its own short WAV file (2-20s range).

Usage:
  python3 convert_meeting.py \
    --label-zip /path/to/TL5.zip \
    --source-zip /path/to/TS5.zip \
    --wav-cache /workspace/data/wav_cache/meeting \
    --output /workspace/data/processed/meeting_train.jsonl \
    --max-hours 150
"""
import argparse, json, os, random, re, time, zipfile
from pathlib import Path


MIN_DUR = 1.0   # drop utterances shorter than 1s
MAX_DUR = 20.0  # NeMo max_duration limit


def _normalize(text: str) -> str:
    text = re.sub(r"\(([^)]+)\)/\([^)]+\)", r"\1", text)  # (발음)/(표기) → 발음
    text = re.sub(r"/\s*\([^)]+\)", "", text)              # /(bgm), /(noise)
    text = re.sub(r"@\S+", "", text)                       # @이름N speaker tags
    return re.sub(r"\s+", " ", text).strip()


def parse_labels(label_zip_path: str) -> dict:
    """Parse meeting label JSONs. Returns {title: [(utt_id, start, end, text)]}."""
    mapping = {}
    with zipfile.ZipFile(label_zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            try:
                data = json.loads(zf.read(name).decode("utf-8"))
                title = data.get("metadata", {}).get("title", "")
                utterances = data.get("utterance", [])
                if not title or not utterances:
                    continue
                segments = []
                for u in utterances:
                    text = _normalize(u.get("form", "").strip())
                    try:
                        start = float(u.get("start", 0))
                        end = float(u.get("end", 0))
                    except (TypeError, ValueError):
                        continue
                    dur = end - start
                    if text and MIN_DUR <= dur <= MAX_DUR:
                        segments.append((u.get("id", ""), start, end, text))
                if segments:
                    mapping[title] = segments
            except (json.JSONDecodeError, KeyError):
                continue
    return mapping


def process_source(source_zip_path: str, label_map: dict, wav_cache: Path,
                   max_hours: float):
    """Extract WAVs, segment by utterance timestamps, build manifest."""
    import soundfile as sf

    entries = []
    total_sec = 0.0
    max_sec = max_hours * 3600
    seg_cache = wav_cache / "segments"
    seg_cache.mkdir(parents=True, exist_ok=True)

    print("Processing source...", flush=True)
    with zipfile.ZipFile(source_zip_path) as zf:
        wav_names = [n for n in zf.namelist() if n.endswith(".wav")]
        matched = [(n, n.rsplit("/", 1)[-1].replace(".wav", ""))
                   for n in wav_names
                   if n.rsplit("/", 1)[-1].replace(".wav", "") in label_map]
        print(f"  {len(matched)} matched of {len(wav_names)} WAVs", flush=True)

        for wav_name, title in matched:
            segments = label_map[title]
            full_wav = wav_cache / f"{title}.wav"
            if not full_wav.exists():
                with zf.open(wav_name) as src:
                    with open(full_wav, "wb") as dst:
                        dst.write(src.read())

            try:
                audio, sr = sf.read(str(full_wav), dtype="float32")
            except Exception as e:
                print(f"  Skip {title}: {e}", flush=True)
                continue

            for utt_id, start, end, text in segments:
                s0 = int(start * sr)
                s1 = int(end * sr)
                clip = audio[s0:s1]
                dur = len(clip) / sr

                safe_id = str(utt_id).replace("/", "_").replace(".", "_")
                clip_path = seg_cache / f"{title}_{safe_id}.wav"
                if not clip_path.exists():
                    sf.write(str(clip_path), clip, sr)

                entries.append({
                    "audio_filepath": str(clip_path),
                    "duration": round(dur, 3),
                    "text": text,
                    "lang": "ko-KR",
                    "target_lang": "ko-KR",
                })
                total_sec += dur
                if total_sec >= max_sec:
                    break

            if total_sec >= max_sec:
                break

    return entries, total_sec / 3600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-zip", required=True)
    parser.add_argument("--source-zip", required=True)
    parser.add_argument("--wav-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-hours", type=float, default=150)
    args = parser.parse_args()

    wav_cache = Path(args.wav_cache)
    wav_cache.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Parsing labels: {Path(args.label_zip).name}", flush=True)
    label_map = parse_labels(args.label_zip)
    total_recordings = len(label_map)
    total_segments = sum(len(v) for v in label_map.values())
    print(f"  {total_recordings} recordings, {total_segments} utterances", flush=True)

    entries, hours = process_source(args.source_zip, label_map, wav_cache,
                                    args.max_hours)
    print(f"Extracted: {len(entries)} utterances, {hours:.1f}h", flush=True)

    random.seed(42)
    random.shuffle(entries)
    with open(args.output, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Done in {(time.time()-t0)/60:.1f}min -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
