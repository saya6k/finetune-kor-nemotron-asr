#!/usr/bin/env python3
"""Download ReazonSpeech Japanese ASR dataset and build NeMo manifest.

Downloads directly from https://corpus.reazon-research.org/ (no HF token needed).
Audio: 16kHz WAV saved to --wav-dir
Output manifest: --output (JSONL, NeMo format)

Usage:
    python3 scripts/download_reazonspeech.py \
        --data-dir /workspace/data \
        --subset small \
        --max-samples 60000
"""
import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://corpus.reazon-research.org/"

SUBSETS = {
    # name: (tsv_path, audio_pattern, nfiles)
    "tiny":   ("reazonspeech-v2/tsv/tiny.tsv",   "reazonspeech-v2/data/{:03x}.tar", 1),
    "small":  ("reazonspeech-v2/tsv/small.tsv",  "reazonspeech-v2/data/{:03x}.tar", 12),
    "medium": ("reazonspeech-v2/tsv/medium.tsv", "reazonspeech-v2/data/{:03x}.tar", 116),
    "large":  ("reazonspeech-v2/tsv/large.tsv",  "reazonspeech-v2/data/{:03x}.tar", 579),
}


def download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def load_meta(tsv_url: str) -> dict:
    """Returns {filename: transcription}."""
    data = download_bytes(tsv_url).decode('utf-8')
    meta = {}
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t', 1)
        if len(parts) == 2:
            meta[parts[0]] = parts[1]
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/workspace/data')
    parser.add_argument('--subset', default='small', choices=list(SUBSETS.keys()))
    parser.add_argument('--max-samples', type=int, default=60000)
    parser.add_argument('--min-duration', type=float, default=1.0)
    parser.add_argument('--max-duration', type=float, default=20.0)
    args = parser.parse_args()

    import soundfile as sf
    import librosa

    data_dir = Path(args.data_dir)
    wav_dir = data_dir / 'wav_cache' / 'reazonspeech'
    wav_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / 'processed' / 'reazonspeech_ja.jsonl'

    if out_path.exists():
        existing = sum(1 for _ in open(out_path))
        print(f'Already exists: {existing} entries at {out_path}')
        if existing >= args.max_samples:
            print('Target already met, exiting.')
            return

    # Collect already-saved filenames to support resume
    saved_names = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                entry = json.loads(line)
                saved_names.add(Path(entry['audio_filepath']).stem)

    tsv_rel, audio_pattern, nfiles = SUBSETS[args.subset]
    tsv_url = BASE_URL + tsv_rel

    print(f'Downloading TSV metadata from {tsv_url} ...', flush=True)
    meta = load_meta(tsv_url)
    print(f'  {len(meta)} entries in TSV', flush=True)

    written = len(saved_names)
    skipped_dur = 0
    skipped_empty = 0

    with open(out_path, 'a', encoding='utf-8') as f_out:
        for idx in range(nfiles):
            if written >= args.max_samples:
                break

            tar_url = BASE_URL + audio_pattern.format(idx)
            print(f'[{idx+1}/{nfiles}] {tar_url} (written={written}) ...', flush=True)

            try:
                tar_bytes = download_bytes(tar_url)
            except Exception as e:
                print(f'  SKIP (download error): {e}', flush=True)
                continue

            with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
                for member in tar.getmembers():
                    if written >= args.max_samples:
                        break
                    if not member.isfile():
                        continue

                    name = member.name.lstrip('./')
                    if name not in meta:
                        continue

                    stem = Path(name).stem
                    if stem in saved_names:
                        continue

                    text = meta[name].strip()
                    if not text:
                        skipped_empty += 1
                        continue

                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue
                    audio_bytes = fobj.read()

                    try:
                        arr, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
                    except Exception:
                        continue

                    duration = len(arr) / sr
                    if duration < args.min_duration or duration > args.max_duration:
                        skipped_dur += 1
                        continue

                    wav_path = wav_dir / f'{stem}.wav'
                    if not wav_path.exists():
                        if sr != 16000:
                            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                        sf.write(str(wav_path), arr, 16000)

                    entry = {
                        'audio_filepath': str(wav_path),
                        'duration': round(duration, 3),
                        'text': text,
                        'lang': 'ja-JP',
                        'target_lang': 'ja-JP',
                    }
                    f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f_out.flush()
                    saved_names.add(stem)
                    written += 1

            print(f'  → {written} total (skipped: {skipped_dur} dur, {skipped_empty} empty)',
                  flush=True)

    print(f'Done: {written} entries → {out_path}', flush=True)


if __name__ == '__main__':
    main()
