#!/usr/bin/env python3
"""Build train_manifest_round6.jsonl for Round 6 fine-tuning.

KO ~90% main data + FLEURS replay ~10% (all nemotron-3.5-asr-streaming supported languages).

Replay anchors the decoder output across all 40+ languages the base model originally
supported. Every clip carries a proper target_lang tag so the prompt mechanism stays
active per language. Text uses raw_transcription (cased+punctuated) to match the base
model's output style, per NVIDIA guidance.

Usage:
    python3 scripts/build_round6_manifest.py --data-dir /workspace/data

Refs:
    https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr
    https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

from datasets import load_dataset, Audio as HFAudio

# ── Nemotron supported language-locales → FLEURS config name ──────────────
# From model card (43 locales). Map to FLEURS where available.
# BCP-47 codes are what the base model was trained with; target_lang must match.
NEMOTRON_TO_FLEURS = {
    "ar": "ar_eg",       # Arabic
    "bg": "bg_bg",       # Bulgarian
    "ca": "ca_es",       # Catalan
    "hr": "hr_hr",       # Croatian
    "cs": "cs_cz",       # Czech
    "da": "da_dk",       # Danish
    "nl": "nl_nl",       # Dutch
    "en": "en_us",       # English (US)
    "et": "et_ee",       # Estonian
    "fi": "fi_fi",       # Finnish
    "fr": "fr_fr",       # French
    "gl": "gl_es",       # Galician
    "de": "de_de",       # German
    "el": "el_gr",       # Greek
    "he": "he_il",       # Hebrew
    "hi": "hi_in",       # Hindi
    "hu": "hu_hu",       # Hungarian
    "id": "id_id",       # Indonesian
    "it": "it_it",       # Italian
    "ja": "ja_jp",       # Japanese
    "kk": "kk_kz",       # Kazakh
    "lv": "lv_lv",       # Latvian
    "lt": "lt_lt",       # Lithuanian
    "ms": "ms_my",       # Malay
    "zh": "cmn_hant_tw", # Mandarin (FLEURS has Traditional; text is Han script)
    "no": "nb_no",       # Norwegian (Bokmål)
    "fa": "fa_ir",       # Persian
    "pl": "pl_pl",       # Polish
    "pt": "pt_br",       # Portuguese (Brazil)
    "ro": "ro_ro",       # Romanian
    "ru": "ru_ru",       # Russian
    "sk": "sk_sk",       # Slovak
    "sl": "sl_si",       # Slovenian
    "es": "es_es",       # Spanish
    "sv": "sv_se",       # Swedish
    "th": "th_th",       # Thai
    "tr": "tr_tr",       # Turkish
    "uk": "uk_ua",       # Ukrainian
    "ur": "ur_pk",       # Urdu
    "vi": "vi_vn",       # Vietnamese
    "cy": "cy_gb",       # Welsh
}

# BCP-47 target_lang tag per nemotron code (what the model expects).
# For most, <iso>-<region> with the dominant region.
_BCP47 = {
    "ar": "ar-EG", "bg": "bg-BG", "ca": "ca-ES", "hr": "hr-HR",
    "cs": "cs-CZ", "da": "da-DK", "nl": "nl-NL", "en": "en-US",
    "et": "et-EE", "fi": "fi-FI", "fr": "fr-FR", "gl": "gl-ES",
    "de": "de-DE", "el": "el-GR", "he": "he-IL", "hi": "hi-IN",
    "hu": "hu-HU", "id": "id-ID", "it": "it-IT", "ja": "ja-JP",
    "kk": "kk-KZ", "lv": "lv-LV", "lt": "lt-LT", "ms": "ms-MY",
    "zh": "zh-CN", "no": "no-NO", "fa": "fa-IR", "pl": "pl-PL",
    "pt": "pt-BR", "ro": "ro-RO", "ru": "ru-RU", "sk": "sk-SK",
    "sl": "sl-SI", "es": "es-ES", "sv": "sv-SE", "th": "th-TH",
    "tr": "tr-TR", "uk": "uk-UA", "ur": "ur-PK", "vi": "vi-VN",
    "cy": "cy-GB",
}

# Languages not in FLEURS: eu (Basque) — skipped with notice
# ko-KR excluded intentionally (main training language, not replay)


def load_jsonl(path: Path) -> list:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--replay-ratio", type=float, default=0.095,
                        help="Replay fraction of total mix (default: 9.5%%)")
    parser.add_argument("--max-per-lang", type=int, default=3000,
                        help="Cap entries per replay language")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ko-manifest", default="train_manifest_round2.jsonl",
                        help="KO source manifest filename in processed/")
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)
    processed = data_dir / "processed"
    wav_cache = data_dir / "wav_cache" / "fleurs_replay"
    wav_cache.mkdir(parents=True, exist_ok=True)
    out_path = processed / "train_manifest_round6.jsonl"

    # ── 1. Korean main data ────────────────────────────────────────────
    ko_path = processed / args.ko_manifest
    if not ko_path.exists():
        print(f"ERROR: KO manifest not found: {ko_path}", flush=True)
        sys.exit(1)
    ko_entries = [e for e in load_jsonl(ko_path) if e.get("lang") == "ko-KR"]
    print(f"KO: {len(ko_entries):,} entries", flush=True)

    # ── 2. FLEURS replay — all nemotron-supported languages ────────────
    replay_entries = []
    skipped = {}
    lang_counts = {}

    for nem_code, fleurs_code in sorted(NEMOTRON_TO_FLEURS.items()):
        target_lang = _BCP47.get(nem_code, f"{nem_code}-{nem_code.upper()}")
        print(f"\n{nem_code:>4s} → {fleurs_code:<14s} target_lang={target_lang} ...",
              end=" ", flush=True)

        try:
            ds = load_dataset("google/fleurs", fleurs_code, split="train", streaming=True)
            ds = ds.cast_column("audio", HFAudio(decode=False))
            # take() caps stream iteration — prevents scanning entire split
            ds = ds.take(args.max_per_lang * 2)

            lang_entries = []
            for s in ds:
                try:
                    sid = str(s["id"])
                    wav_path = wav_cache / f"fleurs_{fleurs_code}_{sid}.wav"

                    if not wav_path.exists():
                        audio_bytes = s["audio"]["bytes"]
                        with open(wav_path, "wb") as f:
                            f.write(audio_bytes)

                    n_samples = s.get("num_samples", 0)
                    duration = round(n_samples / 16000, 3) if n_samples else 0.0

                    text = s["raw_transcription"].strip()
                    if not text:
                        continue

                    lang_entries.append({
                        "audio_filepath": str(wav_path),
                        "duration": duration,
                        "text": text,
                        "lang": target_lang,
                        "target_lang": target_lang,
                    })

                    if len(lang_entries) >= args.max_per_lang:
                        break
                except Exception:
                    continue

            if lang_entries:
                replay_entries.extend(lang_entries)
                total_dur = sum(e["duration"] for e in lang_entries)
                lang_counts[nem_code] = len(lang_entries)
                print(f"{len(lang_entries):>4d} entries  {total_dur/3600:.1f}h", flush=True)
            else:
                skipped[nem_code] = "no entries"
                print("SKIP (no entries)", flush=True)

        except Exception as exc:
            skipped[nem_code] = str(exc)
            print(f"ERROR: {exc}", flush=True)

    # ── 3. Cap replay to target ratio ──────────────────────────────────
    n_ko = len(ko_entries)
    target_replay = int(n_ko * args.replay_ratio / (1 - args.replay_ratio))
    if len(replay_entries) > target_replay:
        print(f"\nCapping replay: {len(replay_entries):,} → {target_replay:,}",
              flush=True)
        replay_entries = random.sample(replay_entries, target_replay)

    # ── 4. Merge + shuffle ─────────────────────────────────────────────
    all_entries = ko_entries + replay_entries
    random.shuffle(all_entries)
    total = len(all_entries)

    # ── 5. Report ──────────────────────────────────────────────────────
    replay_total = len(replay_entries)
    ko_pct = n_ko / total * 100
    replay_pct = replay_total / total * 100
    total_h = sum(e["duration"] for e in all_entries) / 3600

    print(f"\n{'='*60}")
    print(f"R6 Manifest Summary")
    print(f"{'='*60}")
    print(f"  KO      : {n_ko:>8,}  ({ko_pct:.1f}%)")
    print(f"  Replay  : {replay_total:>8,}  ({replay_pct:.1f}%)")
    print(f"  ─────────────────────────────")
    print(f"  Total   : {total:>8,}  ({total_h:,.0f}h)")
    print(f"{'='*60}")
    print(f"\nReplay language breakdown:")
    for code, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = count / replay_total * 100 if replay_total else 0
        print(f"  {_BCP47.get(code, code):>8s}  {count:>5d}  ({pct:.1f}%)")
    if skipped:
        print(f"\nSkipped: {len(skipped)} languages")
        for code, reason in sorted(skipped.items()):
            print(f"  {code}: {reason}")

    # ── 6. Write ───────────────────────────────────────────────────────
    print(f"\nWriting → {out_path} ...", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Done. {total:,} lines written.", flush=True)


if __name__ == "__main__":
    main()
