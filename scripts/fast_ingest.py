#!/usr/bin/env python3
"""Fast parallel MP3→WAV ingest from local Emilia-YODAS TAR files.

Replaces the sequential streaming data_ingest() in train_pipeline.py.
Uses multiprocessing to process multiple TARs in parallel — ~16x speedup.

Usage:
    python3 scripts/fast_ingest.py \
        --emilia-dir /workspace/emilia_local/Emilia-YODAS/KO \
        --wav-cache /workspace/data/wav_cache \
        --output /workspace/data/processed/_temp_ingest_ko.jsonl \
        --lang ko-KR \
        --workers 32

Output: JSONL file with one manifest entry per line.
"""
import argparse
import io
import json
import os
import tarfile
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"


def _process_tar(args):
    tar_path, wav_cache_str, lang_tag, partial_out, max_entries_per_tar = args
    import soundfile as sf
    import librosa
    import io

    wav_cache = Path(wav_cache_str)
    lang_prefix = lang_tag.split('-')[0].lower()
    entries = []
    skipped = 0

    try:
        with tarfile.open(tar_path, 'r:*') as tar:
            members = tar.getmembers()
            # Index by base key (strip extension)
            mp3s = {}
            jsons = {}
            for m in members:
                name = m.name
                if name.endswith('.mp3'):
                    key = name[:-4]
                    mp3s[key] = m
                elif name.endswith('.json'):
                    key = name[:-5]
                    jsons[key] = m

            for key, mp3_m in mp3s.items():
                if max_entries_per_tar > 0 and len(entries) >= max_entries_per_tar:
                    break
                json_m = jsons.get(key)
                if json_m is None:
                    skipped += 1
                    continue

                try:
                    with tar.extractfile(json_m) as jf:
                        meta = json.load(jf)
                except Exception:
                    skipped += 1
                    continue

                text = meta.get('text', '').strip()
                if not text:
                    skipped += 1
                    continue

                sample_id = key.replace('/', '_')
                wav_name = f"{lang_prefix}_{sample_id}.wav"
                wav_path = wav_cache / wav_name

                if wav_path.exists():
                    try:
                        duration = librosa.get_duration(path=str(wav_path))
                    except Exception:
                        duration = 0.0
                else:
                    try:
                        with tar.extractfile(mp3_m) as mf:
                            mp3_bytes = mf.read()
                        arr, sr = sf.read(io.BytesIO(mp3_bytes), dtype='float32')
                        if sr != 16000:
                            arr = librosa.resample(arr, orig_sr=float(sr), target_sr=16000.0)
                        sf.write(str(wav_path), arr, 16000)
                        duration = len(arr) / 16000.0
                    except Exception:
                        skipped += 1
                        continue

                if duration <= 0.1:
                    skipped += 1
                    continue

                entries.append({
                    "audio_filepath": str(wav_path),
                    "duration": round(duration, 3),
                    "text": text,
                    "lang": lang_tag,
                    "target_lang": lang_tag,
                })
    except Exception as e:
        print(f"  ERROR {tar_path}: {e}", flush=True)
        return 0

    with open(partial_out, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    return len(entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emilia-dir', required=True,
                        help='Path to Emilia-YODAS/<LANG>/ directory containing TAR files')
    parser.add_argument('--wav-cache', required=True,
                        help='WAV cache directory')
    parser.add_argument('--output', required=True,
                        help='Output JSONL manifest path')
    parser.add_argument('--lang', default='ko-KR',
                        help='Language tag (e.g. ko-KR, en-US)')
    parser.add_argument('--workers', type=int, default=min(32, cpu_count()),
                        help='Parallel workers (default: min(32, cpu_count))')
    parser.add_argument('--max-shards', type=int, default=0,
                        help='Max TARs to process (0 = all)')
    parser.add_argument('--max-entries', type=int, default=0,
                        help='Stop after collecting this many entries (0 = unlimited)')
    args = parser.parse_args()

    emilia_dir = Path(args.emilia_dir)
    wav_cache = Path(args.wav_cache)
    wav_cache.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    partials_dir = out_path.parent / f'_partials_{out_path.stem}'
    partials_dir.mkdir(exist_ok=True)

    tars = sorted(emilia_dir.rglob('*.tar'))
    if args.max_shards > 0:
        tars = tars[:args.max_shards]

    print(f"Found {len(tars)} TARs in {emilia_dir}", flush=True)
    print(f"Workers: {args.workers}, lang: {args.lang}, max_entries: {args.max_entries}", flush=True)
    print(f"WAV cache: {wav_cache}", flush=True)
    print(f"Output: {out_path}", flush=True)

    # Build task list (skip already-done TARs if partial output exists)
    tasks = []
    already_done = 0
    for tar_path in tars:
        partial = partials_dir / f'{tar_path.stem}.jsonl'
        if partial.exists() and partial.stat().st_size > 0:
            already_done += 1
            continue
        # max_entries per TAR: if total max_entries set, rough per-TAR limit
        max_per_tar = 0
        if args.max_entries > 0:
            max_per_tar = max(1, args.max_entries // max(1, len(tars)))
        tasks.append((str(tar_path), str(wav_cache), args.lang, str(partial), max_per_tar))

    print(f"To process: {len(tasks)} TARs ({already_done} already done)", flush=True)

    t0 = time.time()
    total = 0

    if tasks:
        with Pool(processes=args.workers) as pool:
            for i, n in enumerate(pool.imap_unordered(_process_tar, tasks), 1):
                if args.max_entries > 0 and total >= args.max_entries:
                    pool.terminate()
                    break
                total += n
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - i) * (elapsed / i)
                print(f"  [{i}/{len(tasks)}] +{n} entries | "
                      f"total={total} | {rate:.0f}/s | ETA {eta/60:.1f}min",
                      flush=True)

    # Merge all partials into final output
    print(f"\nMerging {len(list(partials_dir.glob('*.jsonl')))} partial files...", flush=True)
    merged = 0
    with open(out_path, 'w', encoding='utf-8') as out:
        for partial in sorted(partials_dir.glob('*.jsonl')):
            with open(partial) as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        merged += 1

    elapsed = time.time() - t0
    print(f"\nDone: {merged} entries in {elapsed/3600:.2f}h → {out_path}", flush=True)


if __name__ == '__main__':
    main()
