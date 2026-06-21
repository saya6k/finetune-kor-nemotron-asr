#!/usr/bin/env python3
"""Download English ASR datasets for Round 4 catastrophic-forgetting mitigation.

Datasets:
  librispeech_360  LibriSpeech clean-360  ~104K utterances  CC BY 4.0
  librispeech_500  LibriSpeech other-500  ~148K utterances  CC BY 4.0
  common_voice     Common Voice 17 en      validated split   CC0
  voxpopuli        VoxPopuli en            train split       CC0

Outputs:
  data/processed/librispeech_clean360.jsonl
  data/processed/librispeech_other500.jsonl
  data/processed/common_voice_en.jsonl
  data/processed/voxpopuli_en.jsonl

Usage:
    # LibriSpeech clean-360 only (Round 4 default)
    python3 scripts/download_en_asr.py --data-dir /workspace/data

    # All datasets
    python3 scripts/download_en_asr.py --data-dir /workspace/data \\
        --datasets librispeech_360 librispeech_500 common_voice voxpopuli

    # Common Voice with sample cap
    python3 scripts/download_en_asr.py --data-dir /workspace/data \\
        --datasets common_voice --max-samples 80000
"""
import argparse
import json
import os
from pathlib import Path

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"


def _patch_datasets_audio():
    import io
    try:
        import soundfile as sf
        import datasets.features.audio as _am

        def _decode(self, value, token_per_repo_id=None):
            if isinstance(value, dict) and value.get('bytes'):
                arr, sr = sf.read(io.BytesIO(value['bytes']), dtype='float32')
                return {'array': arr, 'sampling_rate': sr, 'path': value.get('path')}
            if isinstance(value, dict) and value.get('array') is not None:
                return value
            return _orig_decode(self, value, token_per_repo_id=token_per_repo_id)

        _orig_decode = _am.Audio.decode_example
        _am.Audio.decode_example = _decode
        print('Audio patch applied (soundfile backend)', flush=True)
    except Exception as e:
        print(f'Audio patch skipped: {e}', flush=True)


def _load_existing_stems(jsonl_path: Path) -> set:
    stems = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    stems.add(Path(json.loads(line)['audio_filepath']).stem)
    return stems


def _write_wav(arr, sr, wav_path: Path):
    import soundfile as sf
    import librosa
    if not wav_path.exists():
        if sr != 16000:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        sf.write(str(wav_path), arr, 16000)


def download_librispeech(data_dir: Path, subset: str, max_samples: int = 0):
    """
    subset: '360' (clean-360, ~104K) or '500' (other-500, ~148K)
    """
    from datasets import load_dataset
    _patch_datasets_audio()

    if subset == '360':
        hf_config, hf_split = 'clean', 'train.360'
        out_name = 'librispeech_clean360'
    else:
        hf_config, hf_split = 'other', 'train.500'
        out_name = 'librispeech_other500'

    out_path = data_dir / 'processed' / f'{out_name}.jsonl'
    wav_dir = data_dir / 'wav_cache' / f'librispeech_{subset}'
    wav_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_stems(out_path)
    if existing:
        print(f'LibriSpeech {hf_config}-{subset}: {len(existing)} already cached', flush=True)

    print(f'Loading openslr/librispeech_asr ({hf_config}, {hf_split})...', flush=True)
    ds = load_dataset('openslr/librispeech_asr', hf_config, split=hf_split)
    print(f'  {len(ds):,} total utterances', flush=True)

    written = 0
    with open(out_path, 'a', encoding='utf-8') as f_out:
        for sample in ds:
            if max_samples and written >= max_samples:
                break

            stem = sample.get('id') or Path(sample.get('file', f'ls_{written}')).stem
            if stem in existing:
                continue

            audio = sample['audio']
            arr, sr = audio['array'], audio['sampling_rate']
            duration = len(arr) / sr
            if not (1.0 <= duration <= 20.0):
                continue

            text = sample['text'].lower().strip()
            if not text:
                continue

            wav_path = wav_dir / f'{stem}.wav'
            _write_wav(arr, sr, wav_path)

            f_out.write(json.dumps({
                'audio_filepath': str(wav_path),
                'duration': round(duration, 3),
                'text': text,
                'lang': 'en-US',
                'target_lang': 'en-US',
            }, ensure_ascii=False) + '\n')
            existing.add(stem)
            written += 1
            if written % 5000 == 0:
                print(f'  {written:,} written...', flush=True)

    print(f'LibriSpeech {hf_config}-{subset}: {written:,} entries → {out_path}', flush=True)
    return written


def download_common_voice(data_dir: Path, max_samples: int = 0):
    from datasets import load_dataset
    _patch_datasets_audio()

    out_path = data_dir / 'processed' / 'common_voice_en.jsonl'
    wav_dir = data_dir / 'wav_cache' / 'common_voice_en'
    wav_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_stems(out_path)
    if existing:
        print(f'Common Voice en: {len(existing):,} already cached', flush=True)

    hf_token = os.environ.get('HF_TOKEN') or None
    print('Loading mozilla-foundation/common_voice_17_0 (en, validated)...', flush=True)
    ds = load_dataset('mozilla-foundation/common_voice_17_0', 'en',
                      split='validated', token=hf_token)
    print(f'  {len(ds):,} validated utterances', flush=True)

    written = 0
    with open(out_path, 'a', encoding='utf-8') as f_out:
        for sample in ds:
            if max_samples and written >= max_samples:
                break

            # Use audio path stem as stable ID for resume support
            stem = Path(sample['audio']['path']).stem
            if stem in existing:
                continue

            audio = sample['audio']
            arr, sr = audio['array'], audio['sampling_rate']
            duration = len(arr) / sr
            if not (1.0 <= duration <= 20.0):
                continue

            text = sample['sentence'].strip()
            if not text:
                continue

            wav_path = wav_dir / f'{stem}.wav'
            _write_wav(arr, sr, wav_path)

            f_out.write(json.dumps({
                'audio_filepath': str(wav_path),
                'duration': round(duration, 3),
                'text': text,
                'lang': 'en-US',
                'target_lang': 'en-US',
            }, ensure_ascii=False) + '\n')
            existing.add(stem)
            written += 1
            if written % 10000 == 0:
                print(f'  {written:,} written...', flush=True)

    print(f'Common Voice en: {written:,} entries → {out_path}', flush=True)
    return written


def download_voxpopuli(data_dir: Path, max_samples: int = 0):
    from datasets import load_dataset
    _patch_datasets_audio()

    out_path = data_dir / 'processed' / 'voxpopuli_en.jsonl'
    wav_dir = data_dir / 'wav_cache' / 'voxpopuli_en'
    wav_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_stems(out_path)
    if existing:
        print(f'VoxPopuli en: {len(existing):,} already cached', flush=True)

    print('Loading facebook/voxpopuli (en, train)...', flush=True)
    ds = load_dataset('facebook/voxpopuli', 'en', split='train')
    print(f'  {len(ds):,} utterances', flush=True)

    written = 0
    with open(out_path, 'a', encoding='utf-8') as f_out:
        for i, sample in enumerate(ds):
            if max_samples and written >= max_samples:
                break

            audio_id = sample.get('audio_id', f'vp_{i:07d}')
            stem = f'vp_en_{audio_id}'
            if stem in existing:
                continue

            audio = sample['audio']
            arr, sr = audio['array'], audio['sampling_rate']
            duration = len(arr) / sr
            if not (1.0 <= duration <= 20.0):
                continue

            text = (sample.get('normalized_text') or sample.get('raw_text', '')).strip()
            if not text:
                continue

            wav_path = wav_dir / f'{stem}.wav'
            _write_wav(arr, sr, wav_path)

            f_out.write(json.dumps({
                'audio_filepath': str(wav_path),
                'duration': round(duration, 3),
                'text': text,
                'lang': 'en-US',
                'target_lang': 'en-US',
            }, ensure_ascii=False) + '\n')
            existing.add(stem)
            written += 1
            if written % 5000 == 0:
                print(f'  {written:,} written...', flush=True)

    print(f'VoxPopuli en: {written:,} entries → {out_path}', flush=True)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/workspace/data')
    parser.add_argument('--datasets', nargs='+',
                        default=['librispeech_360'],
                        choices=['librispeech_360', 'librispeech_500',
                                 'common_voice', 'voxpopuli'])
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Per-dataset sample cap (0=all); applies to all specified datasets')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    (data_dir / 'processed').mkdir(parents=True, exist_ok=True)

    total = 0
    for ds in args.datasets:
        print(f'\n=== {ds} ===', flush=True)
        if ds == 'librispeech_360':
            total += download_librispeech(data_dir, '360', args.max_samples)
        elif ds == 'librispeech_500':
            total += download_librispeech(data_dir, '500', args.max_samples)
        elif ds == 'common_voice':
            total += download_common_voice(data_dir, args.max_samples)
        elif ds == 'voxpopuli':
            total += download_voxpopuli(data_dir, args.max_samples)

    print(f'\nAll done: {total:,} total new entries.', flush=True)


if __name__ == '__main__':
    main()
