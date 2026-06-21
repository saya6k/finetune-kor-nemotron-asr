#!/usr/bin/env python3
"""Pre-download and cache evaluation datasets for ASR fine-tuning.

Downloads:
  - FLEURS ko-KR test set  → data/processed/test_fleurs_ko.json
  - Zeroth Korean test set → data/raw/zeroth/data/test/{wav,text}
  - (Emilia holdout comes from KO TAR download, no action needed here)

Usage:
    python3 scripts/download_eval_datasets.py --data-dir /workspace/data
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

# Must be set before any datasets import to avoid torchcodec on aarch64.
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"


def _patch_datasets_audio():
    try:
        import soundfile as sf
        import numpy as np
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

HF_TOKEN = os.environ.get('HF_TOKEN', '')


FLEURS_LOCALES = {
    # locale_key: (hf_locale, lang_tag, manifest_name)
    'ko': ('ko_kr', 'ko-KR', 'test_fleurs_ko'),
    'fr': ('fr_fr', 'fr-FR', 'test_fleurs_fr'),
    'de': ('de_de', 'de-DE', 'test_fleurs_de'),
    'ru': ('ru_ru', 'ru-RU', 'test_fleurs_ru'),  # catastrophic-forgetting probe
    'ja': ('ja_jp', 'ja-JP', 'test_fleurs_ja'),
    'zh': ('cmn_hans_cn', 'zh-CN', 'test_fleurs_zh'),
    'en': ('en_us', 'en-US', 'test_fleurs_en'),
}


def download_fleurs(data_dir: Path, wav_cache: Path, langs=('ko', 'fr', 'de', 'ru')):
    from datasets import load_dataset
    import soundfile as sf
    import librosa
    _patch_datasets_audio()

    processed = data_dir / 'processed'
    processed.mkdir(parents=True, exist_ok=True)

    for lang_key in langs:
        hf_locale, lang_tag, manifest_name = FLEURS_LOCALES[lang_key]
        out_path = processed / f'{manifest_name}.json'

        if out_path.exists():
            with open(out_path) as f:
                n = sum(1 for l in f if l.strip())
            print(f'FLEURS {lang_key}: already cached ({n} entries)', flush=True)
            continue

        print(f'Downloading FLEURS {hf_locale} test...', flush=True)
        ds = load_dataset('google/fleurs', hf_locale, split='test',
                          token=HF_TOKEN or None)

        entries = []
        for sample in ds:
            wav_path = wav_cache / f"fleurs_{hf_locale}_{sample['id']}.wav"
            audio = sample['audio']
            arr = audio['array']
            sr = audio['sampling_rate']
            if not wav_path.exists():
                resampled = librosa.resample(arr, orig_sr=sr, target_sr=16000) if sr != 16000 else arr
                sf.write(str(wav_path), resampled, 16000)
            text = sample['transcription'].strip()
            if lang_tag == 'zh-CN':
                # FLEURS zh stores space-separated characters; remove spaces for correct CER
                text = text.replace(' ', '')
            entries.append({
                'audio_filepath': str(wav_path),
                'duration': round(len(arr) / sr, 3),
                'text': text,
                'lang': lang_tag,
                'target_lang': lang_tag,
            })

        with open(out_path, 'w', encoding='utf-8') as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
        print(f'FLEURS {lang_key}: {len(entries)} entries → {out_path}', flush=True)


def download_zeroth(data_dir: Path, wav_cache: Path):
    from datasets import load_dataset
    import soundfile as sf
    import librosa
    _patch_datasets_audio()

    zeroth_test = data_dir / 'raw' / 'zeroth' / 'data' / 'test'
    wav_dir = zeroth_test / 'wav'
    text_path = zeroth_test / 'text'

    if text_path.exists() and wav_dir.exists():
        n = sum(1 for _ in wav_dir.glob('*.flac'))
        print(f'Zeroth Korean: already cached ({n} files) at {zeroth_test}', flush=True)
        return

    wav_dir.mkdir(parents=True, exist_ok=True)

    print('Downloading Zeroth Korean test (kresnik/zeroth_korean)...', flush=True)
    ds = load_dataset('kresnik/zeroth_korean', split='test',
                      token=HF_TOKEN or None)

    kaldi_lines = []
    for i, sample in enumerate(ds):
        utt_id = sample.get('id', f'zeroth_{i:06d}')
        flac_path = wav_dir / f'{utt_id}.flac'
        if not flac_path.exists():
            audio = sample['audio']
            arr = audio['array']
            sr = audio['sampling_rate']
            if sr != 16000:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            sf.write(str(flac_path), arr, 16000, format='FLAC')
        text = sample.get('text', sample.get('transcription', '')).strip()
        kaldi_lines.append(f'{utt_id} {text}')

    with open(text_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(kaldi_lines) + '\n')
    print(f'Zeroth Korean: {len(kaldi_lines)} entries → {zeroth_test}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/workspace/data')
    parser.add_argument('--datasets', nargs='+', default=['fleurs', 'zeroth'],
                        choices=['fleurs', 'zeroth'])
    parser.add_argument('--fleurs-langs', nargs='+', default=['ko', 'fr', 'de', 'ru'],
                        choices=list(FLEURS_LOCALES.keys()))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    wav_cache = data_dir / 'wav_cache'
    wav_cache.mkdir(parents=True, exist_ok=True)

    if not HF_TOKEN:
        print('Warning: HF_TOKEN not set (public datasets should still work)', flush=True)

    for name in args.datasets:
        if name == 'fleurs':
            download_fleurs(data_dir, wav_cache, langs=args.fleurs_langs)
        elif name == 'zeroth':
            download_zeroth(data_dir, wav_cache)

    print('Done.', flush=True)


if __name__ == '__main__':
    main()
