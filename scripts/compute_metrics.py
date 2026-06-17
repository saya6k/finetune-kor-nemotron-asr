"""
Compute ASR evaluation metrics: CER, WER, SER.

For Korean, CER (Character Error Rate) is the primary metric (per model card).
WER and SER are secondary — they catch number/English-mixing/proper-noun
errors that CER alone may not reveal.

Usage:
    python compute_metrics.py --ref test_manifest.json --hyp hypothesis_output.json \
        --baseline 7.12 --output results.csv
"""

import json
import sys
import unicodedata
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ── Text normalization ─────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """NFKC normalization + whitespace collapse."""
    text = unicodedata.normalize('NFKC', text)
    text = ' '.join(text.split())
    return text.strip()


# ── Levenshtein distance with operation breakdown ──────────────────────────

def _levenshtein_operations(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """
    Compute Levenshtein distance and return (distance, insertions, deletions, substitutions).
    Uses character-level alignment with full backtracking.
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = (distance, insertions, deletions, substitutions)
    dp: List[List[Tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        prev = dp[i - 1][0]
        dp[i][0] = (prev[0] + 1, prev[1], prev[2] + 1, prev[3])

    for j in range(1, m + 1):
        prev = dp[0][j - 1]
        dp[0][j] = (prev[0] + 1, prev[1] + 1, prev[2], prev[3])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # substitution
                sub = dp[i - 1][j - 1]
                sub_cost = (sub[0] + 1, sub[1], sub[2], sub[3] + 1)
                # deletion
                dele = dp[i - 1][j]
                dele_cost = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])
                # insertion
                ins = dp[i][j - 1]
                ins_cost = (ins[0] + 1, ins[1] + 1, ins[2], ins[3])

                dp[i][j] = min([sub_cost, dele_cost, ins_cost], key=lambda x: x[0])

    return dp[n][m]


def _word_tokenize(text: str) -> List[str]:
    """Simple whitespace-based word tokenization."""
    return text.split()


def _char_tokenize(text: str) -> List[str]:
    """Character-level tokenization. Korean composed syllables are single characters."""
    return list(text)


# ── Metric computation ─────────────────────────────────────────────────────

def compute_cer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute Character Error Rate for a single pair.

    Returns dict with keys: cer, distance, insertions, deletions,
    substitutions, ref_length.
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    ref_chars = _char_tokenize(reference)
    hyp_chars = _char_tokenize(hypothesis)

    ref_len = len(ref_chars)
    if ref_len == 0:
        return {"cer": 0.0, "distance": 0, "insertions": 0,
                "deletions": 0, "substitutions": 0, "ref_length": 0}

    dist, ins, dele, sub = _levenshtein_operations(ref_chars, hyp_chars)
    cer = (dist / ref_len) * 100.0

    return {
        "cer": round(cer, 2),
        "distance": dist,
        "insertions": ins,
        "deletions": dele,
        "substitutions": sub,
        "ref_length": ref_len,
    }


def compute_wer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute Word Error Rate for a single pair.
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    ref_words = _word_tokenize(reference)
    hyp_words = _word_tokenize(hypothesis)

    ref_len = len(ref_words)
    if ref_len == 0:
        return {"wer": 0.0, "distance": 0, "insertions": 0,
                "deletions": 0, "substitutions": 0, "ref_length": 0}

    dist, ins, dele, sub = _levenshtein_operations(ref_words, hyp_words)
    wer = (dist / ref_len) * 100.0

    return {
        "wer": round(wer, 2),
        "distance": dist,
        "insertions": ins,
        "deletions": dele,
        "substitutions": sub,
        "ref_length": ref_len,
    }


def compute_metrics_from_files(
    reference_file: str,
    hypothesis_file: str,
    ref_key: str = "text",
    hyp_key: str = "pred_text",
    normalize: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Compute CER + WER + SER across two manifest files.

    Files can be NeMo manifest JSON lines (ref_key/hyp_key per entry)
    or plain text files (one utterance per line).

    Returns:
        {
            "cer": float, "wer": float, "ser": float,
            "total_chars": int, "total_words": int,
            "total_insertions": int, "total_deletions": int, "total_substitutions": int,
            "num_samples": int, "num_errors": int,
            "baseline": float or None,
        }
    """
    # Load references
    ref_texts = _load_texts(reference_file, ref_key, normalize)
    hyp_texts = _load_texts(hypothesis_file, hyp_key, normalize)

    # Align by min length
    n = min(len(ref_texts), len(hyp_texts))
    if len(ref_texts) != len(hyp_texts):
        print(f"Warning: ref has {len(ref_texts)} entries, hyp has {len(hyp_texts)}. "
              f"Using {n} aligned pairs.")

    # Aggregate metrics
    total_chars = 0
    total_words = 0
    total_cer_dist = 0
    total_wer_dist = 0
    total_ins = 0
    total_dele = 0
    total_sub = 0
    num_errors = 0  # sentences with any error (for SER)

    for i in range(n):
        ref = ref_texts[i] if i < len(ref_texts) else ""
        hyp = hyp_texts[i] if i < len(hyp_texts) else ""

        cer_r = compute_cer(ref, hyp, normalize=False)  # already normalized
        wer_r = compute_wer(ref, hyp, normalize=False)

        total_chars += cer_r["ref_length"]
        total_words += wer_r["ref_length"]
        total_cer_dist += cer_r["distance"]
        total_wer_dist += wer_r["distance"]
        total_ins += cer_r["insertions"]
        total_dele += cer_r["deletions"]
        total_sub += cer_r["substitutions"]

        if cer_r["distance"] > 0:
            num_errors += 1

        # Print first 5 samples for spot-checking
        if verbose and i < 5:
            print(f"\n--- Sample {i + 1} ---")
            print(f"  Ref: {ref[:80]}{'...' if len(ref) > 80 else ''}")
            print(f"  Hyp: {hyp[:80]}{'...' if len(hyp) > 80 else ''}")
            print(f"  CER: {cer_r['cer']:.1f}%, WER: {wer_r['wer']:.1f}%")

    cer = (total_cer_dist / total_chars * 100.0) if total_chars > 0 else 0.0
    wer = (total_wer_dist / total_words * 100.0) if total_words > 0 else 0.0
    ser = (num_errors / n * 100.0) if n > 0 else 0.0

    return {
        "cer": round(cer, 2),
        "wer": round(wer, 2),
        "ser": round(ser, 2),
        "total_chars": total_chars,
        "total_words": total_words,
        "total_insertions": total_ins,
        "total_deletions": total_dele,
        "total_substitutions": total_sub,
        "num_samples": n,
        "num_errors": num_errors,
    }


def compare_with_baseline(results: Dict, baseline: float = 7.12) -> str:
    """
    Generate a formatted comparison table against the model card baseline.

    Args:
        results: Dict from compute_metrics_from_files().
        baseline: Model card CER value.

    Returns:
        Formatted multi-line string.
    """
    cer = results["cer"]
    better = cer <= baseline
    diff = cer - baseline

    lines = [
        "=" * 60,
        "  Evaluation Results vs Model Card Baseline",
        "=" * 60,
        f"  CER:       {cer:.2f}%",
        f"  WER:       {results['wer']:.2f}%",
        f"  SER:       {results['ser']:.2f}%",
        f"  Baseline:  {baseline:.2f}% (FLEURS ko-KR, 1.12s chunk, LangID)",
        f"  Difference: {'+' if diff > 0 else ''}{diff:.2f}pp",
        f"  Samples:   {results['num_samples']}",
        f"  Errors:    {results['num_errors']} sentences",
        "-" * 60,
    ]

    if better:
        lines.append(f"  RESULT: PASS — CER ({cer:.2f}%) ≤ baseline ({baseline:.2f}%)")
    else:
        lines.append(f"  RESULT: BELOW BASELINE — CER ({cer:.2f}%) > baseline "
                      f"({baseline:.2f}%). Consider more training or data tuning.")

    lines.append("=" * 60)
    return '\n'.join(lines)


def _load_texts(filepath: str, key: str, normalize: bool) -> List[str]:
    """Load texts from a manifest JSON or plain text file."""
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)

        # Detect format: JSON manifest or plain text
        if first_line.startswith('{'):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    text = entry.get(key, '')
                    if normalize:
                        text = normalize_text(text)
                    texts.append(text)
                except json.JSONDecodeError:
                    continue
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if normalize:
                    line = normalize_text(line)
                texts.append(line)

    return texts


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute CER/WER/SER from NeMo inference output"
    )
    parser.add_argument("--ref", required=True, help="Reference manifest or text file")
    parser.add_argument("--hyp", required=True, help="Hypothesis manifest or text file")
    parser.add_argument("--ref-key", default="text", help="Key for reference text in JSON")
    parser.add_argument("--hyp-key", default="pred_text", help="Key for hypothesis text in JSON")
    parser.add_argument("--baseline", type=float, default=7.12,
                        help="Model card baseline CER (default: 7.12)")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    parser.add_argument("--no-normalize", action="store_true", help="Skip text normalization")

    args = parser.parse_args()

    results = compute_metrics_from_files(
        reference_file=args.ref,
        hypothesis_file=args.hyp,
        ref_key=args.ref_key,
        hyp_key=args.hyp_key,
        normalize=not args.no_normalize,
        verbose=True,
    )

    # Attach baseline for compare function
    results["baseline"] = args.baseline
    results["better_than_baseline"] = results["cer"] <= args.baseline

    print(compare_with_baseline(results, baseline=args.baseline))

    if args.output:
        import csv as csv_mod
        with open(args.output, 'w', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)
        print(f"\nResults saved to: {args.output}")

    sys.exit(0 if results.get("better_than_baseline", True) else 1)
