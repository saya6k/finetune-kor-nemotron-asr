#!/usr/bin/env python3
"""Checkpoint watcher: evaluates new .nemo checkpoints as they appear during training.

Polls checkpoint_dir every POLL_INTERVAL seconds. When a new .nemo file is found,
loads it once and evaluates all configured datasets, then appends results to CSV.

Usage:
    python3 scripts/eval_watcher.py \
        --checkpoint-dir /workspace/data/checkpoints \
        --datasets-json /workspace/data/processed/eval_datasets.json \
        --output-csv /workspace/results/eval_rolling.csv

datasets_json format:
    {"test_fleurs_ko": "/path/to/manifest.json", "test_fleurs_ru": "...", ...}
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

POLL_INTERVAL = 60  # seconds between directory scans
STABLE_WAIT = 30    # wait for checkpoint file to stop growing before evaluating


def is_stable(path: Path, wait: int = STABLE_WAIT) -> bool:
    """Return True if file size hasn't changed in `wait` seconds."""
    try:
        size1 = path.stat().st_size
        time.sleep(wait)
        size2 = path.stat().st_size
        return size1 == size2 and size1 > 0
    except FileNotFoundError:
        return False


def append_results(output_csv: str, rows: list):
    fieldnames = ["checkpoint", "dataset", "cer", "wer", "ser", "num_samples", "eval_time"]
    exists = os.path.exists(output_csv)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)


def load_evaluated(output_csv: str) -> set:
    """Return set of (checkpoint_name, dataset) pairs already in CSV."""
    done = set()
    if not os.path.exists(output_csv):
        return done
    with open(output_csv) as f:
        for row in csv.DictReader(f):
            done.add((row["checkpoint"], row["dataset"]))
    return done


def find_checkpoints(checkpoint_dir: str) -> list:
    return sorted(Path(checkpoint_dir).rglob("*.nemo"), key=lambda p: p.stat().st_mtime)


def run_watcher(checkpoint_dir: str, datasets: dict, output_csv: str, device: str):
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_direct import eval_checkpoint

    evaluated = load_evaluated(output_csv)
    print(f"Watcher started. Polling {checkpoint_dir} every {POLL_INTERVAL}s", flush=True)
    print(f"Output: {output_csv}", flush=True)
    print(f"Datasets: {list(datasets.keys())}", flush=True)
    print(f"Already evaluated: {len(evaluated)} (checkpoint, dataset) pairs", flush=True)

    while True:
        checkpoints = find_checkpoints(checkpoint_dir)

        for ckpt in checkpoints:
            ckpt_name = ckpt.name
            pending_datasets = {
                ds: path for ds, path in datasets.items()
                if (ckpt_name, ds) not in evaluated
            }
            if not pending_datasets:
                continue

            print(f"\nNew checkpoint: {ckpt_name} — waiting for stability...", flush=True)
            if not is_stable(ckpt):
                print(f"  {ckpt_name} still being written, skipping for now.", flush=True)
                continue

            print(f"  Evaluating {len(pending_datasets)} datasets...", flush=True)
            t0 = time.time()
            try:
                results = eval_checkpoint(str(ckpt), pending_datasets, device=device)
                for r in results:
                    r["eval_time"] = round(time.time() - t0, 1)
                    evaluated.add((ckpt_name, r["dataset"]))
                append_results(output_csv, results)
                print(f"  Done: {len(results)} results appended to {output_csv}", flush=True)
            except Exception as e:
                import traceback
                print(f"  ERROR evaluating {ckpt_name}: {e}", flush=True)
                traceback.print_exc()

        print(f"[{time.strftime('%H:%M:%S')}] Checked {len(checkpoints)} checkpoints, "
              f"sleeping {POLL_INTERVAL}s ...", flush=True)
        time.sleep(POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--datasets-json", required=True,
                        help="JSON file mapping dataset_name -> manifest_path")
    parser.add_argument("--output-csv", default="/workspace/results/eval_rolling.csv")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with open(args.datasets_json) as f:
        datasets = json.load(f)

    run_watcher(args.checkpoint_dir, datasets, args.output_csv, args.device)


if __name__ == "__main__":
    main()
