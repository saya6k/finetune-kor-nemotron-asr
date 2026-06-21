#!/usr/bin/env python3
"""Build train_manifest_round4.jsonl for Round 4 fine-tuning.

Data mix:
  ko  812,825  (train_manifest_round2.jsonl — ko-KR only)
  en  ~132K    (librispeech_clean100 + clean360 combined, ~13% of total)
  ja   60,000  (reazonspeech_ja.jsonl, normalized)
  zh       0   (removed)

Run download_en_asr.py first to prepare clean360:
    python3 scripts/download_en_asr.py --data-dir /workspace/data \\
        --datasets librispeech_360

Usage:
    python3 scripts/build_round4_manifest.py --data-dir /workspace/data
"""
import argparse
import json
import random
import re
import unicodedata
from pathlib import Path


def normalize_ja(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[（(][^）)]{0,20}[）)]', '', text)
    text = re.sub(r'[。、！？「」『』【】《》〈〉〔〕［］…]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_jsonl(path: Path) -> list:
    entries = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/workspace/data')
    parser.add_argument('--ja-samples', type=int, default=60000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)
    processed = data_dir / 'processed'
    out_path = processed / 'train_manifest_round4.jsonl'

    # 1. Korean: ko-KR only from round2
    ko_path = processed / 'train_manifest_round2.jsonl'
    print(f'Loading ko-KR from {ko_path} ...', flush=True)
    ko_entries = [e for e in load_jsonl(ko_path) if e.get('lang') == 'ko-KR']
    print(f'  ko: {len(ko_entries):,}', flush=True)

    # 2. English: LibriSpeech clean-100 + clean-360 (~132K, 13% of total)
    en_sources = [
        ('librispeech_clean100.jsonl', 'clean-100'),
        ('librispeech_clean360.jsonl', 'clean-360'),
    ]
    en_entries = []
    for fname, label in en_sources:
        p = processed / fname
        if not p.exists():
            print(f'  [SKIP] {fname} not found (run download_en_asr.py first)', flush=True)
            continue
        batch = load_jsonl(p)
        print(f'  en {label}: {len(batch):,}', flush=True)
        en_entries.extend(batch)
    print(f'  en total: {len(en_entries):,}', flush=True)

    # 3. Japanese: normalize and sample
    ja_path = processed / 'reazonspeech_ja.jsonl'
    print(f'Loading ja from {ja_path} ...', flush=True)
    ja_raw = load_jsonl(ja_path)
    ja_normalized = []
    skipped_empty = 0
    for e in ja_raw:
        text = normalize_ja(e['text'])
        if not text:
            skipped_empty += 1
            continue
        e['text'] = text
        ja_normalized.append(e)
    print(f'  ja: {len(ja_normalized):,} after normalize (skipped {skipped_empty} empty)', flush=True)
    ja_entries = random.sample(ja_normalized, min(args.ja_samples, len(ja_normalized)))
    print(f'  ja: {len(ja_entries):,} sampled', flush=True)

    # 4. Merge and shuffle
    all_entries = ko_entries + en_entries + ja_entries
    random.shuffle(all_entries)

    total = len(all_entries)
    print(f'\nMix summary:', flush=True)
    print(f'  ko: {len(ko_entries):,} ({len(ko_entries)/total*100:.1f}%)', flush=True)
    print(f'  en: {len(en_entries):,} ({len(en_entries)/total*100:.1f}%)', flush=True)
    print(f'  ja: {len(ja_entries):,} ({len(ja_entries)/total*100:.1f}%)', flush=True)
    print(f'  total: {total:,}', flush=True)

    # 5. Write
    print(f'\nWriting {out_path} ...', flush=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f'Done → {out_path}', flush=True)


if __name__ == '__main__':
    main()
