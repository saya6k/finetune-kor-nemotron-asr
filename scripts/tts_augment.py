"""
Generate synthetic speech samples for rare domain-specific terms using gTTS.

Default: OFF. With 7,300h of real Emilia-YODAS data, gTTS audio (overly clean,
monotone intonation) has negligible impact. Use only when you have genuinely
rare domain terms not covered by the main dataset.

Usage:
    python tts_augment.py --terms "반도체,인공지능,클라우드" \
        --output-dir data/tts_augmented --lang ko
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional


def generate_tts_samples(
    term_list: List[str],
    output_dir: str,
    lang: str = "ko",
    engine: str = "gtts",
    sr: int = 16000,
    prefix: str = "tts_aug",
    max_terms: Optional[int] = None,
) -> List[Dict]:
    """
    Generate synthetic speech for a list of terms.

    Args:
        term_list: List of text terms to synthesize.
        output_dir: Directory to save generated WAV files and manifest.
        lang: Language code for TTS engine (default: "ko").
        engine: TTS engine: "gtts" (free, no GPU) or "coqui" (higher quality).
        sr: Target sample rate (Hz).
        prefix: Filename prefix for generated audio.
        max_terms: Maximum number of terms to generate (None = all).

    Returns:
        List of manifest entry dicts with keys: audio_filepath, duration,
        text, lang, target_lang.
    """
    if not term_list:
        print("Empty term list. Nothing to generate.")
        return []

    terms = term_list[:max_terms] if max_terms else term_list
    os.makedirs(output_dir, exist_ok=True)

    entries = []
    skipped = 0

    for i, term in enumerate(terms):
        term = term.strip()
        if not term:
            continue

        wav_path = os.path.join(output_dir, f"{prefix}_{i:04d}.wav")

        try:
            if engine == "gtts":
                duration = _gtts_generate(term, wav_path, lang, sr)
            elif engine == "coqui":
                duration = _coqui_generate(term, wav_path, lang, sr)
            else:
                raise ValueError(f"Unknown TTS engine: {engine}")
        except Exception as e:
            print(f"Warning: Failed to generate TTS for '{term}': {e}")
            skipped += 1
            continue

        entries.append({
            "audio_filepath": wav_path,
            "duration": round(duration, 3),
            "text": term,
            "lang": "ko-KR",
            "target_lang": "ko-KR",
        })

    # Write manifest
    manifest_path = os.path.join(output_dir, "tts_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"TTS augmentation: generated {len(entries)} samples "
          f"(skipped {skipped}), manifest: {manifest_path}")
    return entries


def _gtts_generate(term: str, output_path: str, lang: str, sr: int) -> float:
    """
    Generate a single TTS sample using gTTS.

    Returns duration in seconds.
    """
    from gtts import gTTS
    import soundfile as sf

    # For very short terms, use a carrier sentence for natural prosody
    if len(term) <= 10 and lang == "ko":
        text = f"다음 단어는 {term}입니다."
    else:
        text = term

    tmp_mp3 = output_path.replace('.wav', '.mp3')
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(tmp_mp3)

        # Convert MP3 to WAV using librosa or soundfile
        try:
            import librosa
            audio, native_sr = librosa.load(tmp_mp3, sr=sr, mono=True)
        except Exception:
            # Fallback: use soundfile with pydub
            from pydub import AudioSegment
            audio_seg = AudioSegment.from_mp3(tmp_mp3)
            audio_seg = audio_seg.set_frame_rate(sr).set_channels(1)
            import numpy as np
            audio = np.array(audio_seg.get_array_of_samples(), dtype=np.float32) / 32768.0

        sf.write(output_path, audio, sr)
        duration = len(audio) / sr
    finally:
        if os.path.exists(tmp_mp3):
            os.remove(tmp_mp3)

    return duration


def _coqui_generate(term: str, output_path: str, lang: str, sr: int) -> float:
    """
    Placeholder for Coqui TTS generation (higher quality, requires GPU).

    Install: pip install TTS
    """
    raise NotImplementedError(
        "Coqui TTS support is not yet implemented. "
        "Install with: pip install TTS, then implement model loading in _coqui_generate()."
    )


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate TTS augmentation samples for rare terms"
    )
    parser.add_argument("--terms", required=True,
                        help="Comma-separated list of terms to synthesize")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save generated WAV files")
    parser.add_argument("--lang", default="ko", help="TTS language code")
    parser.add_argument("--engine", default="gtts", choices=["gtts", "coqui"],
                        help="TTS engine")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--max-terms", type=int, default=None,
                        help="Maximum number of terms to generate")

    args = parser.parse_args()

    terms = [t.strip() for t in args.terms.split(',') if t.strip()]
    entries = generate_tts_samples(
        term_list=terms,
        output_dir=args.output_dir,
        lang=args.lang,
        engine=args.engine,
        sr=args.sr,
        max_terms=args.max_terms,
    )

    print(f"Done. Generated {len(entries)} samples.")
    sys.exit(0 if entries else 1)
