#!/usr/bin/env python3
"""Build train_manifest_round5.jsonl for Round 5 fine-tuning.

Data mix:
  ko  812,825  (train_manifest_round2.jsonl — ko-KR only)
  en  all      (voxpopuli_en.jsonl — VoxPopuli English, normalized_text)
  ja   60,000  (reazonspeech_ja.jsonl, already normalized)

Usage:
    python3 scripts/build_round5_manifest.py --data-dir /workspace/data
"""
import argparse
import json
import random
from pathlib import Path


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
    out_path = processed / 'train_manifest_round5.jsonl'

    # 1. Korean
    ko_path = processed / 'train_manifest_round2.jsonl'
    print(f'Loading ko-KR from {ko_path} ...', flush=True)
    ko_entries = [e for e in load_jsonl(ko_path) if e.get('lang') == 'ko-KR']
    print(f'  ko: {len(ko_entries):,}', flush=True)

    # 2. English — VoxPopuli (normalized_text, already clean)
    en_path = processed / 'voxpopuli_en.jsonl'
    print(f'Loading en from {en_path} ...', flush=True)
    en_entries = load_jsonl(en_path)
    print(f'  en: {len(en_entries):,}', flush=True)

    # 3. Japanese
    ja_path = processed / 'reazonspeech_ja.jsonl'
    print(f'Loading ja from {ja_path} ...', flush=True)
    ja_all = [e for e in load_jsonl(ja_path) if e.get('text', '').strip()]
    ja_entries = random.sample(ja_all, min(args.ja_samples, len(ja_all)))
    print(f'  ja: {len(ja_entries):,} sampled from {len(ja_all):,}', flush=True)

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
