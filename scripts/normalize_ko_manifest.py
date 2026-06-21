#!/usr/bin/env python3
"""Normalize Korean text in NeMo manifest JSONL files in-place.

Normalization applied to ko-KR entries:
  - NFC unicode normalization
  - Remove punctuation: . , ! ? ; : " ' ( ) 「」【】《》〈〉…
  - Keep: Korean syllables, digits, Latin letters (brand names), spaces
  - Normalize whitespace

Non-ko-KR entries are untouched.

Usage:
    python3 scripts/normalize_ko_manifest.py \
        /workspace/data/processed/train_manifest_round4.jsonl
"""
import json
import re
import sys
import unicodedata
from pathlib import Path


def normalize_ko(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    # remove punctuation; keep Korean (\w covers CJK in Python 3), digits, spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # \w includes underscore — remove it
    text = re.sub(r'_', ' ', text)
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
    skipped = 0
    for e in entries:
        if e.get('lang') != 'ko-KR':
            continue
        original = e['text']
        normalized = normalize_ko(original)
        if not normalized:
            skipped += 1
            continue
        e['text'] = normalized
        if normalized != original:
            changed += 1

    entries = [e for e in entries if e.get('lang') != 'ko-KR' or e['text']]

    with open(path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    ko_total = sum(1 for e in entries if e.get('lang') == 'ko-KR')
    print(f'{path.name}: ko={ko_total}, changed={changed}, dropped={skipped}', flush=True)


if __name__ == '__main__':
    paths = sys.argv[1:]
    if not paths:
        print('Usage: normalize_ko_manifest.py <file.jsonl> [...]')
        sys.exit(1)
    for p in paths:
        normalize_file(Path(p))
    print('Done.', flush=True)
