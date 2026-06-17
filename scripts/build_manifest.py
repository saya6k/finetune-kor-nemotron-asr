"""
Build NeMo-compatible manifest JSON from audio files and transcripts.

Supports multiple transcript formats and Korean text normalization.
Usable both as an importable module and as a CLI tool.

Usage:
    python build_manifest.py --audio-dir data/wavs/train \
        --output data/processed/train_manifest.json \
        --transcript-file data/transcripts.csv --transcript-format csv \
        --lang ko-KR
"""

import json
import csv
import glob
import os
import sys
import unicodedata
from pathlib import Path
from typing import Optional, Callable, List, Dict

try:
    import librosa
except ImportError:
    librosa = None


def normalize_korean_text(text: str) -> str:
    """
    Normalize Korean text for consistency in manifest and CER computation.

    - NFKC normalization (compose/decompose Unicode consistently)
    - Collapse whitespace
    - Strip leading/trailing whitespace
    - Preserve case (Korean doesn't have case, but English mixed text does)
    """
    text = unicodedata.normalize('NFKC', text)
    text = ' '.join(text.split())
    text = text.strip()
    return text


def build_manifest(
    audio_dir: str,
    output_path: str,
    transcript_file: Optional[str] = None,
    transcript_format: str = "csv",
    lang: str = "ko-KR",
    target_lang: str = "ko-KR",
    audio_ext: str = ".wav",
    normalize_func: Optional[Callable[[str], str]] = None,
    min_duration: float = 0.1,
    max_duration: float = 30.0,
    skip_missing_transcripts: bool = True,
) -> int:
    """
    Build a NeMo manifest JSON file from audio files and transcripts.

    Args:
        audio_dir: Directory containing audio files (searched recursively).
        output_path: Path to write the manifest JSON lines file.
        transcript_file: Path to transcript file. If None, looks for
            companion .txt files per audio file (same stem).
        transcript_format: One of "csv", "flat", "kaldi", or "auto".
            - "csv": CSV with columns: filename_stem,transcript
            - "flat": One transcript per line, matched alphabetically to sorted audio list.
            - "kaldi": utterance_id transcript per line, utterance_id matches filename stem.
        lang: Language tag for manifest entries (e.g., "ko-KR").
        target_lang: Target language tag.
        audio_ext: Audio file extension to search for (e.g., ".wav", ".mp3").
        normalize_func: Optional text normalization function.
        min_duration: Minimum audio duration in seconds (shorter clips skipped).
        max_duration: Maximum audio duration in seconds (longer clips skipped).
        skip_missing_transcripts: If True, skip audio files without transcripts.
            If False, raise ValueError.

    Returns:
        Number of entries written to the manifest.

    Raises:
        ValueError: If transcript file is missing or format is unknown.
        FileNotFoundError: If audio_dir does not exist.
    """
    if normalize_func is None:
        normalize_func = normalize_korean_text

    audio_dir_path = Path(audio_dir)
    if not audio_dir_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    # 1. Discover audio files
    audio_files = sorted(glob.glob(os.path.join(audio_dir, f'**/*{audio_ext}'), recursive=True))
    if not audio_files:
        raise ValueError(f"No {audio_ext} files found in {audio_dir}")

    audio_stems = {Path(f).stem: f for f in audio_files}
    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    # 2. Load transcripts
    transcripts: Dict[str, str] = {}
    if transcript_file and os.path.isfile(transcript_file):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            if transcript_format == "csv":
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        stem = row[0].strip()
                        text = ','.join(row[1:]).strip()
                        if stem and text:
                            transcripts[stem] = normalize_func(text)
            elif transcript_format == "flat":
                lines = [line.strip() for line in f if line.strip()]
                if len(lines) != len(audio_files):
                    print(f"Warning: {len(lines)} transcripts != {len(audio_files)} audio files")
                sorted_stems = sorted(audio_stems.keys())
                for stem, text in zip(sorted_stems, lines):
                    transcripts[stem] = normalize_func(text)
            elif transcript_format == "kaldi":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        uid, text = parts
                        transcripts[uid.strip()] = normalize_func(text)
            else:
                raise ValueError(f"Unknown transcript_format: {transcript_format}")
    elif transcript_file:
        raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
    # else: no transcript file — will try companion .txt files or raise

    # 3. Build manifest entries
    entries: List[Dict] = []
    skipped_empty_text = 0
    skipped_duration = 0
    skipped_missing = 0
    skipped_duration_error = 0

    for audio_path in audio_files:
        stem = Path(audio_path).stem

        # Get transcript
        if stem in transcripts:
            text = transcripts[stem]
        elif transcript_file is None:
            # Try companion .txt file
            txt_path = os.path.splitext(audio_path)[0] + '.txt'
            if os.path.isfile(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = normalize_func(f.read())
            elif skip_missing_transcripts:
                skipped_missing += 1
                continue
            else:
                raise ValueError(f"No transcript found for {stem}")
        elif skip_missing_transcripts:
            skipped_missing += 1
            continue
        else:
            raise ValueError(f"No transcript found for {stem}")

        if not text:
            skipped_empty_text += 1
            continue

        # Compute duration
        try:
            if librosa is not None:
                duration = float(librosa.get_duration(path=audio_path))
            else:
                import soundfile as sf
                info = sf.info(audio_path)
                duration = info.duration
        except Exception as e:
            print(f"Warning: Could not compute duration for {audio_path}: {e}")
            skipped_duration_error += 1
            continue

        if duration < min_duration or duration > max_duration:
            skipped_duration += 1
            continue

        entries.append({
            "audio_filepath": audio_path,
            "duration": round(duration, 3),
            "text": text,
            "lang": lang,
            "target_lang": target_lang,
        })

    # 4. Write manifest
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # 5. Summary
    print(f"Manifest written: {output_path}")
    print(f"  Entries: {len(entries)}")
    print(f"  Skipped (missing transcript): {skipped_missing}")
    print(f"  Skipped (empty text): {skipped_empty_text}")
    print(f"  Skipped (duration out of range): {skipped_duration}")
    print(f"  Skipped (duration error): {skipped_duration_error}")
    return len(entries)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build NeMo manifest JSON from audio files and transcripts"
    )
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio files")
    parser.add_argument("--output", required=True, help="Output manifest JSON path")
    parser.add_argument("--transcript-file", default=None, help="Transcript file path")
    parser.add_argument("--transcript-format", default="csv",
                        choices=["csv", "flat", "kaldi"],
                        help="Transcript file format")
    parser.add_argument("--lang", default="ko-KR", help="Language tag")
    parser.add_argument("--target-lang", default="ko-KR", help="Target language tag")
    parser.add_argument("--audio-ext", default=".wav", help="Audio file extension")
    parser.add_argument("--min-duration", type=float, default=0.1)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--no-nfkc", action="store_true", help="Skip NFKC normalization")
    parser.add_argument("--no-skip", action="store_true", help="Fail on missing transcripts")

    args = parser.parse_args()

    normalize = None if args.no_nfkc else normalize_korean_text
    count = build_manifest(
        audio_dir=args.audio_dir,
        output_path=args.output,
        transcript_file=args.transcript_file,
        transcript_format=args.transcript_format,
        lang=args.lang,
        target_lang=args.target_lang,
        audio_ext=args.audio_ext,
        normalize_func=normalize,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        skip_missing_transcripts=not args.no_skip,
    )
    sys.exit(0 if count > 0 else 1)
