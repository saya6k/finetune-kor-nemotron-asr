#!/usr/bin/env python3
"""Normalize English ASR manifest text in-place.

Normalization applied:
  - Remove punctuation: . , ! ? ; : ( ) [ ] { } - ... etc.
  - Keep apostrophes between letters (contractions: don't, it's)
  - Normalize whitespace

Usage:
    python3 scripts/normalize_en_manifest.py \
        /workspace/data/processed/librispeech_clean360.jsonl
    python3 scripts/normalize_en_manifest.py \
        /workspace/data/processed/librispeech_clean360.jsonl \
        /workspace/data/processed/librispeech_clean100.jsonl
"""
import json
import re
import sys
from pathlib import Path


def normalize_en(text: str) -> str:
    # keep only lowercase letters, digits, spaces, apostrophes
    text = re.sub(r"[^a-z0-9 ']", ' ', text)
    # remove apostrophes not surrounded by letters (standalone quotes)
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_file(path: Path):
    entries = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    changed = 0
    for e in entries:
        original = e['text']
        e['text'] = normalize_en(original)
        if e['text'] != original:
            changed += 1

    with open(path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    print(f'{path.name}: {len(entries)} entries, {changed} normalized', flush=True)


if __name__ == '__main__':
    paths = sys.argv[1:]
    if not paths:
        print('Usage: normalize_en_manifest.py <file.jsonl> [file2.jsonl ...]')
        sys.exit(1)
    for p in paths:
        normalize_file(Path(p))
    print('Done.', flush=True)
