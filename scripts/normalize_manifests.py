#!/usr/bin/env python3
"""Normalize text fields in existing NeMo manifest JSONL files in-place.

Fixes (A)/(B) alternate forms, truncated words (+), inaudible (*), overlap
slashes, non-speech markers, and speaker tags without re-running full conversion.

Usage:
  python3 scripts/normalize_manifests.py --type ksponspeech <path1> [<path2> ...]
  python3 scripts/normalize_manifests.py --type extreme_noise <path>
  python3 scripts/normalize_manifests.py --type meeting <path>
"""
import re, json, shutil, argparse
from pathlib import Path


def normalize_ksponspeech(text: str) -> str:
    text = re.sub(r"\([^)]+\)/\(([^)]+)\)", r"\1", text)  # (표기)/(발음) → 발음
    text = re.sub(r"\s?[bnluo]/\s?", " ", text)            # b/ n/ l/ u/ o/
    text = re.sub(r"\s?\+/\s?", " ", text)                 # +/
    text = re.sub(r"\S+\+", "", text)                       # word+ truncated
    text = re.sub(r"\S*\*\S*", "", text)                   # word*, *word, standalone *
    text = re.sub(r"(\S)/", r"\1", text)                   # overlap slash
    return re.sub(r"\s+", " ", text).strip()


def normalize_extreme_noise(text: str) -> str:
    text = re.sub(r"\([^)]+\)/\(([^)]+)\)", r"\1", text)  # (표기)/(발음) → 발음
    text = re.sub(r"\S+\+", "", text)
    text = re.sub(r"\S+\*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_meeting(text: str) -> str:
    # Extract spoken form FIRST (before /(bgm) removal, which would eat the right side)
    text = re.sub(r"\(([^)]+)\)/\([^)]+\)", r"\1", text)  # (발음)/(표기) → 발음
    text = re.sub(r"/\s*\([^)]+\)", "", text)              # remaining /(bgm), /(noise)
    text = re.sub(r"@\S+", "", text)                       # @이름N tags
    return re.sub(r"\s+", " ", text).strip()


NORMALIZERS = {
    "ksponspeech": normalize_ksponspeech,
    "extreme_noise": normalize_extreme_noise,
    "meeting": normalize_meeting,
}


def process_file(path: str, norm_fn) -> tuple:
    in_path = Path(path)
    tmp_path = in_path.with_suffix(".tmp")
    changed = dropped = total = 0
    with open(in_path, encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            orig = d["text"]
            d["text"] = norm_fn(orig)
            if d["text"] != orig:
                changed += 1
            if d["text"]:
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                total += 1
            else:
                dropped += 1
    shutil.move(str(tmp_path), str(in_path))
    return total, changed, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=list(NORMALIZERS), required=True)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    fn = NORMALIZERS[args.type]
    for path in args.paths:
        total, changed, dropped = process_file(path, fn)
        print(f"{path}: {total} kept, {changed} modified, {dropped} dropped", flush=True)


if __name__ == "__main__":
    main()
