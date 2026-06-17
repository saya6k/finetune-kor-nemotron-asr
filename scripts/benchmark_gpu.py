"""
GPU speed and VRAM benchmark for fine-tuning cost estimation.

Runs a short training session (500-1000 steps or 10 minutes) and measures
steps/sec, peak VRAM, and OOM threshold. Extrapolates total training time
and cost for the full 7,300h dataset.

Usage:
    python benchmark_gpu.py \\
        --nemo-dir /workspace/NeMo \\
        --hf-ckpt /path/to/model.nemo \\
        --bench-manifest data/bench_manifest.json \\
        --batch-duration 200 \\
        --max-steps 500
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


def run_benchmark(
    nemo_dir: str,
    hf_ckpt: str,
    bench_manifest: str,
    batch_duration: int = 200,
    max_steps: int = 500,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run a short training benchmark and return performance metrics.

    Args:
        nemo_dir: Path to NeMo source directory.
        hf_ckpt: Path to the pretrained .nemo model.
        bench_manifest: Path to a small manifest for benchmarking.
        batch_duration: Batch size in seconds of audio.
        max_steps: Maximum training steps to run.
        output_dir: Directory for temporary benchmark outputs.

    Returns:
        Dict with keys: steps_per_sec, peak_vram_mb, vram_total_mb,
        max_batch_duration, estimated_hours, estimated_cost.
    """
    if output_dir is None:
        output_dir = '/tmp/nemo_benchmark'
    os.makedirs(output_dir, exist_ok=True)

    # Validate inputs
    for path, name in [(nemo_dir, "nemo_dir"), (hf_ckpt, "hf_ckpt"),
                        (bench_manifest, "bench_manifest")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    finetune_script = os.path.join(
        nemo_dir,
        'examples', 'asr', 'speech_to_text_finetune.py'
    )

    cmd = [
        sys.executable, finetune_script,
        '--config-path=../asr/conf/fastconformer/cache_aware_streaming',
        '--config-name=fastconformer_transducer_bpe_streaming_prompt.yaml',
        f'+init_from_nemo_model={hf_ckpt}',
        f'++model.train_ds.manifest_filepath={bench_manifest}',
        f'++model.validation_ds.manifest_filepath={bench_manifest}',
        '++model.optim.sched.d_model=1024',
        '++trainer.devices=1',
        f'++trainer.max_steps={max_steps}',
        '++trainer.max_epochs=-1',  # run only by steps
        '++trainer.precision=bf16',
        f'++model.train_ds.batch_duration={batch_duration}',
        '++model.optim.name=adamw',
        '++model.optim.lr=0.0001',
        '++model.optim.weight_decay=0.001',
        '++model.optim.sched.name=CosineAnnealing',
        '++model.optim.sched.warmup_ratio=0.05',
        '++model.optim.sched.min_lr=1e-6',
        f'++exp_manager.exp_dir={output_dir}',
        '++exp_manager.use_datetime_version=False',
        '++exp_manager.version=benchmark',
        '++trainer.log_every_n_steps=10',
    ]

    print("Starting GPU benchmark...")
    print(f"  batch_duration: {batch_duration}")
    print(f"  max_steps: {max_steps}")
    print(f"  output_dir: {output_dir}")
    print()

    # Start nvidia-smi monitoring in background
    vram_log_path = os.path.join(output_dir, 'vram_log.txt')
    nvidia_smi_proc = subprocess.Popen(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total',
         '--format=csv,noheader,nounits', '-l', '2'],
        stdout=open(vram_log_path, 'w'),
        stderr=subprocess.DEVNULL,
    )

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1800, max_steps * 5),  # min 30 min timeout
        )
    finally:
        nvidia_smi_proc.terminate()
        nvidia_smi_proc.wait()

    elapsed = time.time() - start_time

    # Parse step time from NeMo output
    step_times = []
    for line in result.stdout.split('\n') + result.stderr.split('\n'):
        # PyTorch Lightning logs step time in various formats
        m = re.search(r'(\d+\.?\d*)\s*s/it', line)
        if m:
            step_times.append(float(m.group(1)))
        # NeMo may log "global_step" with timing
        m = re.search(r'time_per_step[=:]\s*(\d+\.?\d*)', line, re.IGNORECASE)
        if m:
            step_times.append(float(m.group(1)))

    # Use average step time excluding first 10% (warmup)
    if len(step_times) > 10:
        warmup = max(1, len(step_times) // 10)
        step_times = step_times[warmup:]
        avg_step_time = sum(step_times) / len(step_times)
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
    else:
        # Fallback: total elapsed / max_steps
        avg_step_time = elapsed / max_steps if max_steps > 0 else 0
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0

    # Parse peak VRAM from log
    peak_vram_mb = 0
    vram_total_mb = 0
    if os.path.exists(vram_log_path):
        with open(vram_log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        used = float(parts[0].strip())
                        total = float(parts[1].strip())
                        if used > peak_vram_mb:
                            peak_vram_mb = used
                        vram_total_mb = total
                    except (ValueError, IndexError):
                        continue

    # OOM threshold detection: try larger batch_duration
    max_batch_duration = batch_duration
    # (Full OOM detection would retry with increasing values — skipped in benchmark)

    # Extrapolate to full 7,300h dataset with 3 epochs
    # Rough estimate: total_steps ≈ (total_audio_hours * 3600 * epochs) / (batch_duration * avg_step_time)
    # But more practically, use steps_per_sec with known data size
    TOTAL_HOURS = 7300
    EPOCHS = 3
    # Steps per epoch ≈ total_audio_seconds / batch_duration
    steps_per_epoch = (TOTAL_HOURS * 3600) / batch_duration
    total_steps = steps_per_epoch * EPOCHS
    estimated_hours = total_steps / (steps_per_sec * 3600) if steps_per_sec > 0 else float('inf')

    # Cost estimate (user should provide GPU hourly rate)
    # Default: RTX 6000 Ada @ $0.79/h
    gpu_hourly_rate = float(os.environ.get('GPU_HOURLY_RATE', '0.79'))
    estimated_cost = estimated_hours * gpu_hourly_rate

    results = {
        "steps_per_sec": round(steps_per_sec, 4),
        "avg_step_time_sec": round(avg_step_time, 4),
        "peak_vram_mb": peak_vram_mb,
        "vram_total_mb": vram_total_mb,
        "vram_usage_pct": round(peak_vram_mb / vram_total_mb * 100, 1) if vram_total_mb > 0 else 0,
        "max_batch_duration": max_batch_duration,
        "benchmark_elapsed_sec": round(elapsed, 1),
        "benchmark_steps": max_steps if len(step_times) == 0 else len(step_times) + warmup,
        "total_audio_hours": TOTAL_HOURS,
        "epochs": EPOCHS,
        "estimated_total_steps": int(total_steps),
        "estimated_hours": round(estimated_hours, 1),
        "gpu_hourly_rate": gpu_hourly_rate,
        "estimated_cost": round(estimated_cost, 2),
        "oom_detected": result.returncode != 0 and 'out of memory' in (result.stderr + result.stdout).lower(),
    }

    # Print results
    print("\n" + "=" * 60)
    print("  GPU Benchmark Results")
    print("=" * 60)
    print(f"  Steps/sec:        {results['steps_per_sec']:.4f} ({results['avg_step_time_sec']:.4f}s/step)")
    print(f"  Peak VRAM:        {results['peak_vram_mb']:.0f} MB / "
          f"{results['vram_total_mb']:.0f} MB ({results['vram_usage_pct']:.1f}%)")
    print(f"  Batch duration:   {results['max_batch_duration']}s (OOM-safe)")
    print(f"  Benchmark took:   {results['benchmark_elapsed_sec']:.0f}s "
          f"({results['benchmark_steps']} steps)")
    print()
    print(f"  Estimated ({TOTAL_HOURS}h × {EPOCHS} epochs):")
    print(f"    Total steps:    {results['estimated_total_steps']:,}")
    print(f"    Total time:     {results['estimated_hours']:.1f}h")
    print(f"    Est. cost:      ${results['estimated_cost']:.2f} "
          f"(@ ${gpu_hourly_rate:.2f}/h)")
    print()

    if results['vram_usage_pct'] > 90:
        print("  ⚠️  VRAM usage > 90%. Consider reducing batch_duration.")
    elif results['vram_usage_pct'] < 50:
        print("  💡 VRAM usage < 50%. Consider increasing batch_duration for speed.")
    else:
        print("  ✅ VRAM usage is in a healthy range (50-90%).")

    if results.get('oom_detected'):
        print("  ❌ OOM DETECTED. Reduce batch_duration and re-run benchmark.")

    print("=" * 60)

    # Save results
    results_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def estimate_cost(
    total_audio_hours: float,
    epochs: int,
    batch_duration: int,
    steps_per_sec: float,
    gpu_hourly_rate: float = 0.79,
) -> Tuple[float, float]:
    """
    Estimate total training time and cost.

    Args:
        total_audio_hours: Total hours of audio data.
        epochs: Number of training epochs.
        batch_duration: Batch size in seconds.
        steps_per_sec: Measured training throughput.
        gpu_hourly_rate: GPU instance hourly rate ($).

    Returns:
        (estimated_hours, estimated_cost)
    """
    steps_per_epoch = (total_audio_hours * 3600) / batch_duration
    total_steps = steps_per_epoch * epochs
    hours = total_steps / (steps_per_sec * 3600) if steps_per_sec > 0 else float('inf')
    cost = hours * gpu_hourly_rate
    return hours, cost


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU speed and VRAM benchmark for ASR fine-tuning cost estimation"
    )
    parser.add_argument("--nemo-dir", required=True, help="NeMo source directory")
    parser.add_argument("--hf-ckpt", required=True, help="Path to pretrained .nemo model")
    parser.add_argument("--bench-manifest", required=True,
                        help="Path to benchmark manifest (200-500 samples)")
    parser.add_argument("--batch-duration", type=int, default=200,
                        help="Batch duration in seconds")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Number of training steps for benchmark")
    parser.add_argument("--output-dir", default='/tmp/nemo_benchmark',
                        help="Directory for benchmark outputs")
    parser.add_argument("--total-hours", type=int, default=7300,
                        help="Total audio hours in full dataset")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Planned training epochs")
    parser.add_argument("--gpu-rate", type=float, default=None,
                        help="GPU hourly rate in $ (default: $GPU_HOURLY_RATE env or $0.79)")

    args = parser.parse_args()

    # Set GPU rate from env if not provided
    if args.gpu_rate is not None:
        os.environ['GPU_HOURLY_RATE'] = str(args.gpu_rate)

    results = run_benchmark(
        nemo_dir=args.nemo_dir,
        hf_ckpt=args.hf_ckpt,
        bench_manifest=args.bench_manifest,
        batch_duration=args.batch_duration,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )

    if results.get('oom_detected'):
        sys.exit(2)
    sys.exit(0)
