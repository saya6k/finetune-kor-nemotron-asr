#!/usr/bin/env python3
"""Checkpoint watcher: evaluates new .nemo and .ckpt checkpoints as they appear.

Polls checkpoint_dir every POLL_INTERVAL seconds. When a new .nemo or .ckpt file is
found, loads it once and evaluates all configured datasets, then appends results to CSV.
-last.ckpt files are skipped (duplicate of the epoch's primary .ckpt).

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
    """Find all evaluatable checkpoints (.nemo + .ckpt), excluding -last.ckpt duplicates."""
    nemo = Path(checkpoint_dir).rglob("*.nemo")
    ckpt = [p for p in Path(checkpoint_dir).rglob("*.ckpt")
            if "-last.ckpt" not in p.name]
    return sorted(list(nemo) + list(ckpt), key=lambda p: p.stat().st_mtime)


def run_watcher(checkpoint_dir: str, datasets: dict, output_csv: str, device: str,
                datasets_json_path: str = ""):
    import subprocess
    import tempfile

    evaluated = load_evaluated(output_csv)
    eval_one_script = str(Path(__file__).parent / "eval_one.py")

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

            print(f"  Evaluating {len(pending_datasets)} datasets via subprocess...", flush=True)
            t0 = time.time()
            tmp_json = tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False, dir='/tmp'
            )
            json.dump(pending_datasets, tmp_json)
            tmp_json.close()
            try:
                result = subprocess.run(
                    [sys.executable, eval_one_script,
                     "--ckpt", str(ckpt),
                     "--datasets-json", tmp_json.name,
                     "--output-csv", output_csv,
                     "--base-nemo-dir", checkpoint_dir,
                     "--device", device],
                    capture_output=True, text=True, timeout=3600,
                )
                # Print subprocess stdout for visibility
                if result.stdout:
                    for line in result.stdout.strip().splitlines():
                        print(f"  {line}", flush=True)
                if result.stderr:
                    # stderr may contain NeMo warnings — print selectively
                    for line in result.stderr.strip().splitlines():
                        if "ERROR" in line or "Traceback" in line or "Error" in line:
                            print(f"  [stderr] {line}", flush=True)

                if result.returncode == 0:
                    for ds_name in pending_datasets:
                        evaluated.add((ckpt_name, ds_name))
                    elapsed = time.time() - t0
                    print(f"  Done in {elapsed:.0f}s: {len(pending_datasets)} results → {output_csv}", flush=True)
                else:
                    print(f"  Subprocess exited with code {result.returncode}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT evaluating {ckpt_name} (1h limit)", flush=True)
            except Exception as e:
                import traceback
                print(f"  ERROR evaluating {ckpt_name}: {e}", flush=True)
                traceback.print_exc()
            finally:
                Path(tmp_json.name).unlink(missing_ok=True)

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

    run_watcher(args.checkpoint_dir, datasets, args.output_csv, args.device,
                datasets_json_path=args.datasets_json)


if __name__ == "__main__":
    main()
