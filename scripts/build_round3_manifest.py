#!/usr/bin/env python3
"""Build train_manifest_round3.jsonl for Round 3 fine-tuning.

Data mix:
  ko  812,825  (train_manifest_round2.jsonl — unchanged)
  en   80,000  (sampled from _temp_ingest_en.jsonl)
  ja  ≤60,000  (reazonspeech_ja.jsonl, normalized)
  zh      0   (removed — catastrophic forgetting in Round 2)

Japanese text normalization applied inline:
  - NFKC (全角→半角: １２３→123, Ａ→A)
  - Remove punctuation: 。、！？「」『』【】《》〈〉〔〕［］…
  - Remove parenthetical stage directions: (笑) （拍手）
  - Collapse whitespace

Also fixes FLEURS ja eval manifest (strips spaces between phrases).

Usage:
    python3 scripts/build_round3_manifest.py --data-dir /workspace/data
"""
import argparse
import json
import random
import re
import unicodedata
from pathlib import Path


def normalize_ja(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[（(][^）)]{0,20}[）)]', '', text)  # stage directions
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


def fix_fleurs_ja(data_dir: Path):
    """Strip inter-phrase spaces from FLEURS ja eval manifest (fixes CER inflation)."""
    ja_path = data_dir / 'processed' / 'test_fleurs_ja.json'
    if not ja_path.exists():
        print('  test_fleurs_ja.json not found, skipping fix', flush=True)
        return
    entries = load_jsonl(ja_path)
    fixed = 0
    new_entries = []
    for e in entries:
        original = e['text']
        e['text'] = re.sub(r' ', '', original)
        if e['text'] != original:
            fixed += 1
        new_entries.append(e)
    with open(ja_path, 'w', encoding='utf-8') as f:
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f'  FLEURS ja: stripped spaces from {fixed}/{len(entries)} entries', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/workspace/data')
    parser.add_argument('--en-samples', type=int, default=0,
                        help='Max en samples (0 = use all available)')
    parser.add_argument('--ja-samples', type=int, default=60000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)
    processed = data_dir / 'processed'
    out_path = processed / 'train_manifest_round3.jsonl'

    # 1. Fix FLEURS ja eval text (space stripping)
    print('Fixing FLEURS ja eval manifest...', flush=True)
    fix_fleurs_ja(data_dir)

    # 2. Korean: filter ko-KR only from round2 (excludes zh/en that were mixed in)
    ko_path = processed / 'train_manifest_round2.jsonl'
    print(f'Loading ko-KR from {ko_path} ...', flush=True)
    ko_entries = [e for e in load_jsonl(ko_path) if e.get('lang') == 'ko-KR']
    print(f'  ko: {len(ko_entries):,} entries', flush=True)

    # 3. English: librispeech_clean100 (files verified on disk; _temp_ingest_en WAVs missing)
    en_path = processed / 'librispeech_clean100.jsonl'
    print(f'Loading en from {en_path} ...', flush=True)
    en_all = load_jsonl(en_path)
    n_en = args.en_samples if args.en_samples > 0 else len(en_all)
    en_entries = random.sample(en_all, min(n_en, len(en_all)))
    print(f'  en: {len(en_entries):,} (of {len(en_all):,} available)', flush=True)
    del en_all

    # 4. Japanese: normalize and sample
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

    # 5. Merge and shuffle
    all_entries = ko_entries + en_entries + ja_entries
    random.shuffle(all_entries)

    total = len(all_entries)
    ko_pct = len(ko_entries) / total * 100
    en_pct = len(en_entries) / total * 100
    ja_pct = len(ja_entries) / total * 100

    print(f'\nMix summary:', flush=True)
    print(f'  ko: {len(ko_entries):,} ({ko_pct:.1f}%)', flush=True)
    print(f'  en: {len(en_entries):,} ({en_pct:.1f}%)', flush=True)
    print(f'  ja: {len(ja_entries):,} ({ja_pct:.1f}%)', flush=True)
    print(f'  total: {total:,}', flush=True)

    # 6. Write
    print(f'\nWriting {out_path} ...', flush=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f'Done → {out_path}', flush=True)


if __name__ == '__main__':
    main()
