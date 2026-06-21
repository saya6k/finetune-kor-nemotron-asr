#!/usr/bin/env python3
"""Convert AIHub 극한소음 source audio + JSON/SRT labels → NeMo manifest.

Segments individual utterances using SRT timestamps (not whole recordings).
Labels are in a nested zip: labeling.zip → TL_*.zip → {json, srt} pairs.
Source WAVs are in [원천]*.zip files.

Usage:
  python3 convert_extreme_noise.py \
    --label-zip  /workspace/data/raw/extreme_noise/labeling.zip \
    --source-zips /workspace/data/raw/extreme_noise/Training/[원천]*.zip \
    --wav-cache  /workspace/data/wav_cache/extreme_noise \
    --output     /workspace/data/processed/extreme_noise_train.jsonl \
    --max-hours  100
"""
import argparse, io, json, os, re, time, zipfile
from pathlib import Path


MIN_DUR = 0.5
MAX_DUR = 20.0


def _srt_to_sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_srt(content: str) -> list:
    """Parse SRT content into [(start_sec, end_sec, text)] segments."""
    segments = []
    for block in re.split(r'\n\s*\n', content.strip()):
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        time_idx = None
        for i, line in enumerate(lines):
            if '-->' in line:
                time_idx = i
                break
        if time_idx is None:
            continue
        times = re.findall(r'(\d+):(\d+):(\d+)[,.](\d+)', lines[time_idx])
        if len(times) < 2:
            continue
        start = _srt_to_sec(*times[0])
        end = _srt_to_sec(*times[1])
        raw_text = ' '.join(lines[time_idx + 1:])
        text = _normalize(raw_text)
        dur = end - start
        if text and MIN_DUR <= dur <= MAX_DUR:
            segments.append((start, end, text))
    return segments


def _normalize(text: str) -> str:
    text = re.sub(r'\([^)]+\)/\(([^)]+)\)', r'\1', text)  # (표기)/(발음) → 발음
    text = re.sub(r'\S+\+', '', text)
    text = re.sub(r'\S*\*\S*', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def parse_labels(label_zip_path: str) -> dict:
    """Parse nested labeling.zip → {wav_basename: [(start, end, text)]}."""
    mapping = {}
    with zipfile.ZipFile(label_zip_path) as outer:
        sub_zips = [n for n in outer.namelist() if n.endswith('.zip')]
        print(f"  {len(sub_zips)} sub-zips in labeling.zip", flush=True)
        for sub_name in sub_zips:
            sub_data = io.BytesIO(outer.read(sub_name))
            with zipfile.ZipFile(sub_data) as inner:
                srt_map = {}
                for name in inner.namelist():
                    if name.endswith('.srt'):
                        base = name.rsplit('/', 1)[-1].replace('.srt', '')
                        content = inner.read(name).decode('utf-8')
                        segs = _parse_srt(content)
                        if segs:
                            srt_map[base] = segs
                for name in inner.namelist():
                    if not name.endswith('.json'):
                        continue
                    try:
                        data = json.loads(inner.read(name).decode('utf-8'))
                        media_url = data.get('mediaUrl', '')
                        wav_base = media_url.rsplit('/', 1)[-1].replace('.wav', '')
                        segs = srt_map.get(wav_base)
                        if segs:
                            wav_filename = media_url.rsplit('/', 1)[-1]
                            mapping[wav_filename] = segs
                    except (json.JSONDecodeError, KeyError):
                        continue
    print(f"  {len(mapping)} labeled WAVs with SRT segments", flush=True)
    return mapping


def process_sources(source_zips: list, label_map: dict, wav_cache: Path,
                    max_hours: float):
    import soundfile as sf

    entries = []
    total_sec = 0.0
    max_sec = max_hours * 3600
    seg_cache = wav_cache / 'segments'
    seg_cache.mkdir(parents=True, exist_ok=True)

    for sz_path in source_zips:
        print(f"Processing: {Path(sz_path).name}", flush=True)
        with zipfile.ZipFile(sz_path) as zf:
            wav_names = [n for n in zf.namelist() if n.endswith('.wav')]
            matched = [(n, n.rsplit('/', 1)[-1])
                       for n in wav_names
                       if n.rsplit('/', 1)[-1] in label_map]
            print(f"  {len(matched)} matched of {len(wav_names)} WAVs", flush=True)

            for wav_name, basename in matched:
                segs = label_map[basename]
                full_wav = wav_cache / basename
                if not full_wav.exists():
                    with zf.open(wav_name) as src:
                        with open(full_wav, 'wb') as dst:
                            dst.write(src.read())
                try:
                    audio, sr = sf.read(str(full_wav), dtype='float32')
                except Exception as e:
                    print(f"  Skip {basename}: {e}", flush=True)
                    continue

                stem = Path(basename).stem
                for idx, (start, end, text) in enumerate(segs):
                    s0 = int(start * sr)
                    s1 = int(end * sr)
                    clip = audio[s0:s1]
                    dur = len(clip) / sr
                    if dur < MIN_DUR or dur > MAX_DUR:
                        continue

                    clip_path = seg_cache / f'{stem}_{idx:04d}.wav'
                    if not clip_path.exists():
                        sf.write(str(clip_path), clip, sr)

                    entries.append({
                        'audio_filepath': str(clip_path),
                        'duration': round(dur, 3),
                        'text': text,
                        'lang': 'ko-KR',
                        'target_lang': 'ko-KR',
                    })
                    total_sec += dur
                    if total_sec >= max_sec:
                        break

                if total_sec >= max_sec:
                    break

        if total_sec >= max_sec:
            break

    return entries, total_sec / 3600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-zip', required=True)
    parser.add_argument('--source-zips', nargs='+', required=True)
    parser.add_argument('--wav-cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-hours', type=float, default=100)
    args = parser.parse_args()

    wav_cache = Path(args.wav_cache)
    wav_cache.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Parsing labels: {Path(args.label_zip).name}", flush=True)
    label_map = parse_labels(args.label_zip)

    import random
    entries, hours = process_sources(args.source_zips, label_map, wav_cache,
                                     args.max_hours)
    print(f"Extracted: {len(entries)} utterances, {hours:.1f}h", flush=True)

    random.seed(42)
    random.shuffle(entries)
    with open(args.output, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    print(f"Done in {(time.time()-t0)/60:.1f}min -> {args.output}", flush=True)


if __name__ == '__main__':
    main()
