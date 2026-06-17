"""
Sweep evaluation across all checkpoints × multiple eval datasets.

Finds all .nemo checkpoint files in a directory tree, runs NeMo streaming
inference on each against every specified eval manifest, and aggregates
CER/WER/SER results into a CSV.

Supports --watch mode for live checkpoint monitoring during training.

Usage:
    python eval_checkpoints.py \\
        --checkpoint-dir checkpoints/ \\
        --datasets '{"fleurs_ko":"data/eval_fleurs.json","holdout":"data/eval_holdout.json"}' \\
        --output results/checkpoint_eval.csv \\
        --nemo-dir /workspace/NeMo \\
        --target-lang ko-KR
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import csv as csv_mod


def find_checkpoints(checkpoint_dir: str) -> List[str]:
    """
    Find all .nemo checkpoint files, sorted by modification time (oldest first).

    Returns list of absolute paths.
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    nemo_files = sorted(ckpt_path.rglob('*.nemo'), key=lambda p: p.stat().st_mtime)
    if not nemo_files:
        raise FileNotFoundError(f"No .nemo files found under {checkpoint_dir}")

    return [str(p) for p in nemo_files]


def run_inference(
    checkpoint_path: str,
    manifest_path: str,
    nemo_dir: str,
    target_lang: str = "ko-KR",
    output_dir: Optional[str] = None,
    batch_size: int = 8,
    cuda: int = 0,
) -> str:
    """
    Run NeMo streaming inference on a single checkpoint + manifest pair.

    Returns path to the hypothesis output file.
    """
    if output_dir is None:
        output_dir = os.path.dirname(manifest_path)
    os.makedirs(output_dir, exist_ok=True)

    infer_script = os.path.join(
        nemo_dir,
        'examples', 'asr', 'asr_cache_aware_streaming',
        'speech_to_text_cache_aware_streaming_infer.py'
    )

    cmd = [
        sys.executable, infer_script,
        f"model_path={checkpoint_path}",
        f"dataset_manifest={manifest_path}",
        f"target_lang={target_lang}",
        'att_context_size=[56,3]',
        'decoder_type=rnnt',
        'pad_and_drop_preencoded=true',
        f'batch_size={batch_size}',
        f'cuda={cuda}',
        'strip_lang_tags=false',
        f'output_dir={output_dir}',
    ]

    print(f"  Running inference: {Path(checkpoint_path).name} -> "
          f"{Path(manifest_path).stem}")
    print(f"  Command: {' '.join(cmd[:5])} ...")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour timeout per eval run
    )

    if result.returncode != 0:
        stderr_tail = result.stderr.strip().split('\n')[-10:]
        print(f"  WARNING: Inference failed (exit code {result.returncode})")
        print(f"  STDERR: {'; '.join(stderr_tail)}")
        raise RuntimeError(f"Inference failed for {checkpoint_path}: {result.stderr[-500:]}")

    # NeMo writes output alongside the input manifest
    # Look for the hypothesis file
    manifest_stem = Path(manifest_path).stem
    hypothesis_patterns = [
        os.path.join(output_dir, f"{manifest_stem}_hypothesis.json"),
        os.path.join(output_dir, f"{manifest_stem}_predictions.json"),
        os.path.join(output_dir, f"{manifest_stem}_output.json"),
    ]

    for pattern in hypothesis_patterns:
        if os.path.exists(pattern):
            return pattern

    # Fallback: NeMo may append pred_text to the manifest itself
    print(f"  Hypothesis file not found in standard locations, checking manifest directory...")
    for f in sorted(Path(output_dir).glob('*.json')):
        if 'hypothesis' in f.name or 'pred' in f.name or 'output' in f.name:
            return str(f)

    raise FileNotFoundError(f"Could not find hypothesis output for {manifest_path}")


def sweep_checkpoints(
    checkpoint_dir: str,
    datasets: Dict[str, str],
    nemo_dir: str,
    output_csv: str,
    target_lang: str = "ko-KR",
    batch_size: int = 8,
    cuda: int = 0,
) -> str:
    """
    Run inference for all checkpoints against all datasets, compute metrics,
    and write results to CSV.

    Args:
        checkpoint_dir: Directory containing .nemo checkpoint files.
        datasets: Dict mapping dataset name -> manifest path.
        nemo_dir: Path to NeMo source directory.
        output_csv: Path to write results CSV.
        target_lang: Target language for inference (e.g., "ko-KR").
        batch_size: Batch size for inference.
        cuda: CUDA device index.

    Returns:
        Path to the output CSV file.
    """
    # Lazy import to avoid circular dependency
    sys.path.insert(0, str(Path(__file__).parent))
    from compute_metrics import compute_metrics_from_files

    checkpoints = find_checkpoints(checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoint(s) to evaluate")
    print(f"Datasets: {list(datasets.keys())}")
    print()

    results = []
    total_runs = len(checkpoints) * len(datasets)
    run_num = 0

    for ckpt_path in checkpoints:
        ckpt_name = Path(ckpt_path).name
        print(f"[Checkpoint: {ckpt_name}]")

        for ds_name, manifest_path in datasets.items():
            run_num += 1
            print(f"  [{run_num}/{total_runs}] {ds_name}")

            if not os.path.exists(manifest_path):
                print(f"  SKIP: Manifest not found: {manifest_path}")
                continue

            try:
                hyp_path = run_inference(
                    checkpoint_path=ckpt_path,
                    manifest_path=manifest_path,
                    nemo_dir=nemo_dir,
                    target_lang=target_lang,
                    batch_size=batch_size,
                    cuda=cuda,
                )

                metrics = compute_metrics_from_files(
                    reference_file=manifest_path,
                    hypothesis_file=hyp_path,
                    verbose=False,
                )

                results.append({
                    "checkpoint": ckpt_name,
                    "checkpoint_path": ckpt_path,
                    "dataset": ds_name,
                    "cer": metrics["cer"],
                    "wer": metrics["wer"],
                    "ser": metrics["ser"],
                    "total_chars": metrics["total_chars"],
                    "total_words": metrics["total_words"],
                    "num_samples": metrics["num_samples"],
                    "num_errors": metrics["num_errors"],
                })

                print(f"    CER={metrics['cer']:.2f}%, WER={metrics['wer']:.2f}%, "
                      f"SER={metrics['ser']:.2f}%")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "checkpoint": ckpt_name,
                    "checkpoint_path": ckpt_path,
                    "dataset": ds_name,
                    "cer": None,
                    "wer": None,
                    "ser": None,
                    "total_chars": 0,
                    "total_words": 0,
                    "num_samples": 0,
                    "num_errors": 0,
                })

    # Write CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    fieldnames = ["checkpoint", "checkpoint_path", "dataset",
                  "cer", "wer", "ser", "total_chars", "total_words",
                  "num_samples", "num_errors"]

    with open(output_csv, 'w', newline='') as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print(f"\nResults written to: {output_csv}")
    _print_summary(results)

    return output_csv


def _print_summary(results: List[Dict]) -> None:
    """Print a summary table of checkpoint evaluation results."""
    # Group by dataset
    datasets = sorted(set(r["dataset"] for r in results))

    print("\n" + "=" * 80)
    print("  Checkpoint Evaluation Summary")
    print("=" * 80)

    for ds in datasets:
        ds_results = [r for r in results if r["dataset"] == ds and r["cer"] is not None]
        if not ds_results:
            print(f"  {ds}: No valid results")
            continue

        best = min(ds_results, key=lambda r: r["cer"])
        worst = max(ds_results, key=lambda r: r["cer"])

        print(f"\n  {ds}:")
        print(f"    Best  CER: {best['cer']:.2f}% @ {best['checkpoint']}")
        print(f"    Worst CER: {worst['cer']:.2f}% @ {worst['checkpoint']}")

        # CER trend
        if len(ds_results) >= 2:
            first = ds_results[0]["cer"]
            last = ds_results[-1]["cer"]
            trend = "↓ improving" if last < first else "↑ degrading" if last > first else "→ stable"
            print(f"    Trend: {first:.2f}% → {last:.2f}% ({trend})")

    # Overall best for FLEURS (if present)
    fleurs_results = [r for r in results if "fleurs" in r["dataset"].lower() and r["cer"] is not None]
    if fleurs_results:
        best_fleurs = min(fleurs_results, key=lambda r: r["cer"])
        baseline = 7.12
        diff = best_fleurs["cer"] - baseline
        print(f"\n  FLEURS Korean Best: {best_fleurs['cer']:.2f}% "
              f"(baseline: {baseline:.2f}%, {'+' if diff > 0 else ''}{diff:.2f}pp)")
        if best_fleurs["cer"] <= baseline:
            print("  RESULT: CER at or below model card baseline.")
        else:
            print("  RESULT: CER above baseline. Consider more training or hyperparameter tuning.")

    print("=" * 80)


def watch_mode(
    checkpoint_dir: str,
    datasets: Dict[str, str],
    nemo_dir: str,
    output_csv: str,
    target_lang: str = "ko-KR",
    poll_interval: int = 60,
) -> None:
    """
    Watch checkpoint directory for new .nemo files and evaluate them as they appear.

    Runs until interrupted (Ctrl+C).
    """
    print(f"Watch mode: polling {checkpoint_dir} every {poll_interval}s")
    print(f"Press Ctrl+C to stop.\n")

    seen = set()
    try:
        while True:
            try:
                current = set(find_checkpoints(checkpoint_dir))
                new = current - seen
                if new:
                    print(f"\n[{time.strftime('%H:%M:%S')}] New checkpoint(s): "
                          f"{[Path(p).name for p in new]}")
                    # Only sweep new checkpoints
                    sweep_checkpoints(
                        checkpoint_dir=checkpoint_dir,
                        datasets=datasets,
                        nemo_dir=nemo_dir,
                        output_csv=output_csv,
                        target_lang=target_lang,
                    )
                    seen = current
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No new checkpoints. "
                          f"Watching ({len(seen)} seen)...")
            except Exception as e:
                print(f"Watch iteration error: {e}")

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nWatch mode stopped.")
        _print_summary([dict(zip(
            ["checkpoint", "checkpoint_path", "dataset", "cer", "wer", "ser",
             "total_chars", "total_words", "num_samples", "num_errors"],
            []
        ))] if not seen else [])  # noop for clean exit


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sweep evaluation across checkpoints × datasets"
    )
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Directory containing .nemo checkpoint files")
    parser.add_argument("--datasets", required=True,
                        help='JSON string: {"name":"path/to/manifest.json",...}')
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--nemo-dir", required=True, help="NeMo source directory")
    parser.add_argument("--target-lang", default="ko-KR", help="Target language tag")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index")
    parser.add_argument("--watch", action="store_true",
                        help="Watch mode: poll for new checkpoints")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Poll interval in seconds (watch mode)")

    args = parser.parse_args()

    datasets = json.loads(args.datasets)
    if not datasets:
        parser.error("--datasets must be a non-empty JSON object")

    if args.watch:
        watch_mode(
            checkpoint_dir=args.checkpoint_dir,
            datasets=datasets,
            nemo_dir=args.nemo_dir,
            output_csv=args.output,
            target_lang=args.target_lang,
            poll_interval=args.poll_interval,
        )
    else:
        sweep_checkpoints(
            checkpoint_dir=args.checkpoint_dir,
            datasets=datasets,
            nemo_dir=args.nemo_dir,
            output_csv=args.output,
            target_lang=args.target_lang,
            batch_size=args.batch_size,
            cuda=args.cuda,
        )
