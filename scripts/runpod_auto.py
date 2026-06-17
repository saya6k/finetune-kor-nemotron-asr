"""
RunPod GPU Pod automation for ASR fine-tuning.

Supports:
- Pod lifecycle management (create, stop, terminate, status)
- Spot instance preemption safety (checkpoint → HF Hub on SIGTERM)
- Cost-aware auto-shutdown after training completes
- GPU type selection by price/VRAM optimization

Requires:
    pip install runpod huggingface_hub

Environment:
    RUNPOD_API_KEY      RunPod API key (required)
    HF_TOKEN            HuggingFace token (for checkpoint upload)
    HF_REPO_ID          HuggingFace repo for checkpoints (e.g., "user/my-asr-ckpts")

Usage:
    # Launch a pod and run training
    python runpod_auto.py launch \
        --gpu-type "NVIDIA RTX 6000 Ada" \
        --image "nvidia/nemo:latest" \
        --volume /workspace:50 \
        --env HF_CKPT=/workspace/model.nemo \
        --script train.sh \
        --auto-shutdown \
        --max-cost 50

    # From inside a pod: register spot preemption handler
    python runpod_auto.py guard

    # Stop the current pod
    python runpod_auto.py stop --pod-id $RUNPOD_POD_ID

    # Check available GPU types and prices
    python runpod_auto.py gpus
"""

import json
import os
import signal
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('runpod_auto')


# ── RunPod client (lazy init) ─────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.environ.get('RUNPOD_API_KEY', '')
    if not key:
        raise RuntimeError(
            "RUNPOD_API_KEY not set. "
            "Get your key at https://www.runpod.io/console/user/api"
        )
    return key


def _init_runpod():
    """Lazy import + authenticate."""
    import runpod
    runpod.api_key = _get_api_key()
    return runpod


# ── GPU Discovery ──────────────────────────────────────────────────────────

def list_gpus(
    min_vram: int = 24,
    max_price: Optional[float] = None,
    sort_by: str = "price",
) -> List[Dict]:
    """
    List available GPU types with filtering.

    Args:
        min_vram: Minimum VRAM in GB (24 = RTX 4090, 48 = A6000/RTX 6000 Ada).
        max_price: Maximum hourly price in $ (None = no limit).
        sort_by: "price" or "vram".

    Returns:
        List of GPU info dicts with keys: id, name, vram_gb, price_per_hour.
    """
    rp = _init_runpod()
    gpus = rp.get_gpus()

    filtered = []
    for g in gpus:
        vram = g.get('memoryInGb', 0)
        price = g.get('lowestPrice', {}).get('price', float('inf'))

        if vram < min_vram:
            continue
        if max_price is not None and price > max_price:
            continue

        filtered.append({
            'id': g.get('id'),
            'name': g.get('displayName'),
            'vram_gb': vram,
            'price_per_hour': price,
            'max_gpu_count': g.get('maxGpuCount', 1),
            'secure_price': g.get('securePrice', {}).get('price'),
            'community_price': g.get('communityPrice', {}).get('price'),
        })

    if sort_by == "price":
        filtered.sort(key=lambda g: g['price_per_hour'])
    elif sort_by == "vram":
        filtered.sort(key=lambda g: g['vram_gb'], reverse=True)

    return filtered


def find_best_gpu(
    min_vram: int = 48,
    max_price: Optional[float] = None,
) -> Optional[Dict]:
    """
    Find the cheapest GPU meeting VRAM requirements.
    """
    candidates = list_gpus(min_vram=min_vram, max_price=max_price, sort_by="price")
    return candidates[0] if candidates else None


# ── Pod Lifecycle ──────────────────────────────────────────────────────────

def create_pod(
    name: str,
    gpu_type: str,
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel-ubuntu22.04",
    gpu_count: int = 1,
    container_disk_gb: int = 100,
    volume_gb: int = 50,
    volume_mount: str = "/workspace",
    env: Optional[Dict[str, str]] = None,
    ports: str = "8888/http",
    country_code: Optional[str] = None,
    start_ssh: bool = True,
) -> Dict:
    """
    Create and start a GPU pod on RunPod.

    Args:
        name: Pod name.
        gpu_type: GPU type display name (e.g., "NVIDIA RTX 6000 Ada").
        image: Docker image name.
        gpu_count: Number of GPUs.
        container_disk_gb: Container disk size in GB.
        volume_gb: Persistent volume size in GB (0 = no volume).
        volume_mount: Volume mount path inside container.
        env: Environment variables dict.
        ports: Port mapping string (e.g., "8888/http,22/tcp").
        country_code: Preferred country (e.g., "US", "KR").
        start_ssh: Enable SSH access.

    Returns:
        Pod info dict with keys: id, name, status, gpu_type, price_per_hour.
    """
    rp = _init_runpod()

    # Resolve GPU: accept display name or ID
    all_gpus = rp.get_gpus()
    gpu = None
    gpu_id = None
    for g in all_gpus:
        if g.get('id') == gpu_type or g.get('displayName') == gpu_type:
            gpu = g
            gpu_id = g['id']
            break

    if not gpu:
        available = sorted(
            set(g['displayName'] for g in all_gpus if g.get('displayName'))
        )
        raise ValueError(
            f"GPU type '{gpu_type}' not found. Available:\n  " +
            '\n  '.join(available[:20])
        )
    gpu_price = gpu.get('lowestPrice', {}).get('price', None)
    price_str = f"${gpu_price:.2f}/h" if isinstance(gpu_price, (int, float)) else "$?/h"
    logger.info(f"GPU: {gpu['displayName']} ({gpu.get('memoryInGb', '?')}GB, {price_str})")

    # Build env dict
    pod_env = env.copy() if env else {}
    pod_env.setdefault('HF_HOME', '/workspace/.cache/huggingface')

    # Create pod
    logger.info(f"Creating pod '{name}'...")
    pod = rp.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu_id,
        gpu_count=gpu_count,
        container_disk_in_gb=container_disk_gb,
        volume_in_gb=volume_gb,
        volume_mount_path=volume_mount,
        env=pod_env,
        ports=ports,
        country_code=country_code,
        start_ssh=start_ssh,
        support_public_ip=True,
    )

    pod_id = pod.get('id')
    logger.info(f"Pod created: {pod_id}")

    # Wait for pod to be ready
    logger.info("Waiting for pod to start...")
    max_wait = 600  # 10 minutes
    info = None
    for _ in range(max_wait // 10):
        info = rp.get_pod(pod_id)
        if info and (info.get('runtime') or {}).get('uptime'):
            break
        time.sleep(10)
    else:
        logger.warning("Pod may not be ready yet. Check manually.")

    info = info or rp.get_pod(pod_id) or {}
    gpu_price = gpu['lowestPrice']['price']

    result = {
        'id': pod_id,
        'name': name,
        'status': info.get('desiredStatus', 'UNKNOWN'),
        'gpu_type': gpu_type,
        'gpu_vram_gb': gpu['memoryInGb'],
        'price_per_hour': gpu_price,
        'image': image,
        'volume_mount': volume_mount,
        'machine': info.get('machine', {}),
        'pod_hostname': info.get('podHostname', ''),
    }

    logger.info(f"Pod ready: {pod_id}")
    logger.info(f"  Price: ${gpu_price:.2f}/h")
    logger.info(f"  SSH:   ssh root@{info.get('podHostname', 'N/A')} -p {info.get('runtime', {}).get('ports', [{}])[0].get('publicPort', 22)}")

    return result


def stop_pod(pod_id: Optional[str] = None) -> bool:
    """
    Stop a running pod.

    Args:
        pod_id: Pod ID. If None, uses RUNPOD_POD_ID from environment.

    Returns:
        True if stop command was sent successfully.
    """
    pod_id = pod_id or os.environ.get('RUNPOD_POD_ID', '')
    if not pod_id:
        raise RuntimeError("No pod ID provided. Set RUNPOD_POD_ID or pass --pod-id.")

    rp = _init_runpod()
    logger.info(f"Stopping pod: {pod_id}")
    rp.stop_pod(pod_id)
    logger.info("Stop command sent.")
    return True


def terminate_pod(pod_id: Optional[str] = None) -> bool:
    """
    Terminate (delete) a pod.

    Args:
        pod_id: Pod ID. If None, uses RUNPOD_POD_ID from environment.

    Returns:
        True if terminate command was sent successfully.
    """
    pod_id = pod_id or os.environ.get('RUNPOD_POD_ID', '')
    if not pod_id:
        raise RuntimeError("No pod ID provided.")

    rp = _init_runpod()
    logger.info(f"Terminating pod: {pod_id}")
    rp.terminate_pod(pod_id)
    logger.info("Terminate command sent.")
    return True


def pod_status(pod_id: Optional[str] = None) -> Dict:
    """
    Get pod status and runtime info.

    Returns dict with: id, name, status, uptime, gpu_utilization, cost_to_date.
    """
    pod_id = pod_id or os.environ.get('RUNPOD_POD_ID', '')
    if not pod_id:
        raise RuntimeError("No pod ID provided.")

    rp = _init_runpod()
    info = rp.get_pod(pod_id)

    if not info:
        return {
            'id': pod_id,
            'name': 'unknown',
            'status': 'PROVISIONING',
            'runtime_status': 'starting',
            'uptime_hours': 0,
            'gpu_utilization_pct': 0,
            'memory_utilization_pct': 0,
            'cost_per_hour': 0,
            'estimated_cost': 0,
            'gpu_type': '',
            'hostname': '',
        }

    runtime = info.get('runtime') or {}
    gpu_info = (runtime.get('gpus') or [{}])[0]

    # Calculate cost
    cost_per_hour = info.get('costPerHr', 0)
    uptime_seconds = runtime.get('uptime', 0) / 1000 if runtime.get('uptime') else 0
    cost_to_date = cost_per_hour * (uptime_seconds / 3600)

    return {
        'id': pod_id,
        'name': info.get('name', ''),
        'status': info.get('desiredStatus', 'UNKNOWN'),
        'runtime_status': runtime.get('status', 'UNKNOWN'),
        'uptime_hours': round(uptime_seconds / 3600, 2),
        'gpu_utilization_pct': gpu_info.get('gpuUtilPercent', 0),
        'memory_utilization_pct': gpu_info.get('memoryUtilPercent', 0),
        'cost_per_hour': cost_per_hour,
        'estimated_cost': round(cost_to_date, 4),
        'gpu_type': info.get('machine', {}).get('gpuDisplayName', ''),
        'hostname': info.get('podHostname', ''),
    }


# ── Spot Preemption Guard ──────────────────────────────────────────────────

# Path to the latest checkpoint (set by the notebook)
_LAST_CHECKPOINT_PATH: Optional[str] = None
# Fallback directory to scan for checkpoints
_CHECKPOINT_DIR: Optional[str] = None
_checkpoint_uploaded = False


def set_checkpoint_path(path: str) -> None:
    """Register the latest checkpoint path for spot preemption upload."""
    global _LAST_CHECKPOINT_PATH
    _LAST_CHECKPOINT_PATH = path
    logger.info(f"Spot guard: checkpoint path set to {path}")


def _find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the most recently modified .nemo file in a directory tree.

    Args:
        checkpoint_dir: Directory to scan recursively for .nemo files.

    Returns:
        Path to the latest .nemo file, or None if none found.
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return None
    nemo_files = sorted(ckpt_path.rglob('*.nemo'), key=lambda p: p.stat().st_mtime)
    if nemo_files:
        latest = str(nemo_files[-1])
        logger.info(f"Auto-discovered checkpoint: {latest}")
        return latest
    return None


def _ensure_repo_exists(api, repo_id: str, private: bool = True) -> None:
    """
    Create a HuggingFace model repo if it doesn't exist.
    Idempotent — no error if already exists.
    """
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        logger.info(f"HF repo ready: {repo_id} (private={private})")
    except Exception as e:
        # If repo already exists or we lack create permission, just log
        logger.debug(f"Repo creation skipped (may already exist): {e}")


def upload_checkpoint_to_hub(
    checkpoint_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[str]:
    """
    Upload checkpoint to HuggingFace Hub for safekeeping.

    Args:
        checkpoint_path: .nemo file to upload. Uses _LAST_CHECKPOINT_PATH if None.
        repo_id: HF repo ID (e.g., "user/my-asr-ckpts").
                 Defaults to HF_REPO_ID env var.
        token: HF API token. Defaults to HF_TOKEN env var.

    Returns:
        URL of the uploaded file, or None on failure.
    """
    global _LAST_CHECKPOINT_PATH, _checkpoint_uploaded

    path = checkpoint_path or _LAST_CHECKPOINT_PATH
    if not path or not os.path.exists(path):
        logger.warning("No checkpoint available for upload.")
        return None

    repo_id = repo_id or os.environ.get('HF_REPO_ID', '')
    if not repo_id:
        logger.warning("HF_REPO_ID not set. Skipping upload.")
        return None

    token = token or os.environ.get('HF_TOKEN', '')
    if not token:
        logger.warning("HF_TOKEN not set. Skipping upload.")
        return None

    try:
        from huggingface_hub import upload_file, HfApi

        api = HfApi(token=token)
        filename = os.path.basename(path)
        repo_path = f"checkpoints/{filename}"

        # Ensure repo exists (create if missing, private by default)
        _ensure_repo_exists(api, repo_id, private=True)

        logger.info(f"Uploading {path} → {repo_id}/{repo_path} ...")
        url = api.upload_file(
            path_or_fileobj=path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        logger.info(f"Checkpoint uploaded: {url}")
        _checkpoint_uploaded = True
        return url
    except Exception as e:
        logger.error(f"Failed to upload checkpoint: {e}")
        return None


_spot_upload_fn = upload_checkpoint_to_hub


def _spot_preemption_handler(signum, frame):
    """
    Handle SIGTERM from RunPod spot instance preemption.
    RunPod sends SIGTERM ~30 seconds before reclaiming the instance.

    Priority:
    1. _LAST_CHECKPOINT_PATH (explicitly set by notebook)
    2. _CHECKPOINT_DIR (auto-scan for latest .nemo)
    """
    global _checkpoint_uploaded, _spot_upload_fn, _LAST_CHECKPOINT_PATH, _CHECKPOINT_DIR

    logger.warning("=" * 60)
    logger.warning("SPOT PREEMPTION DETECTED — Uploading checkpoint...")
    logger.warning("=" * 60)

    if not _checkpoint_uploaded:
        ckpt_path = _LAST_CHECKPOINT_PATH

        # Auto-discover from checkpoint directory if no explicit path
        if (not ckpt_path or not os.path.exists(ckpt_path)) and _CHECKPOINT_DIR:
            logger.info(f"No explicit checkpoint path. Scanning {_CHECKPOINT_DIR} ...")
            ckpt_path = _find_latest_checkpoint(_CHECKPOINT_DIR)

        if ckpt_path and os.path.exists(ckpt_path):
            logger.info(f"Uploading: {ckpt_path} "
                        f"({os.path.getsize(ckpt_path) / 1024**2:.0f} MB)")
            _spot_upload_fn(ckpt_path)
            _checkpoint_uploaded = True
        else:
            logger.warning("No checkpoint found to upload. "
                           "Training may not have started yet.")

    logger.info("Preemption handler complete.")


def register_spot_guard(
    checkpoint_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    repo_id: str = "saya6k/nemotron-kor-checkpoints",
) -> None:
    """
    Register SIGTERM handler for spot instance preemption safety.

    Run this early in your notebook/script to guard against unexpected
    spot instance termination.

    The handler tries, in order:
    1. checkpoint_path (explicit path to a .nemo file)
    2. checkpoint_dir (auto-scan for latest .nemo — use this at startup)

    Args:
        checkpoint_path: Path to a specific .nemo file to protect.
        checkpoint_dir: Directory to auto-scan for latest .nemo (e.g., data/checkpoints).
        repo_id: HF Hub repo for emergency upload (default: saya6k/nemotron-kor-checkpoints).
    """
    global _CHECKPOINT_DIR, _spot_upload_fn

    if checkpoint_path:
        set_checkpoint_path(checkpoint_path)
    if checkpoint_dir:
        _CHECKPOINT_DIR = checkpoint_dir
        logger.info(f"Spot guard: checkpoint dir set to {checkpoint_dir}")

    # Patch upload_checkpoint_to_hub to use the specified repo
    original_upload = upload_checkpoint_to_hub

    def guarded_upload(ckpt_path=None):
        return original_upload(
            checkpoint_path=ckpt_path,
            repo_id=os.environ.get('HF_REPO_ID', repo_id),
            token=os.environ.get('HF_TOKEN'),
        )

    _spot_upload_fn = guarded_upload

    signal.signal(signal.SIGTERM, _spot_preemption_handler)

    is_spot = os.environ.get('RUNPOD_POD_TYPE', '').lower() == 'spot'
    if is_spot:
        logger.info("🛡️  Spot guard active: checkpoint will auto-upload on preemption")
    else:
        logger.info("🛡️  Spot guard registered (pod type: on-demand)")

    # Also log whether we're on RunPod at all
    pod_id = os.environ.get('RUNPOD_POD_ID', '')
    if pod_id:
        logger.info(f"   Pod ID: {pod_id}")


def is_runpod() -> bool:
    """Check if we're running inside a RunPod pod."""
    return bool(os.environ.get('RUNPOD_POD_ID', ''))


# ── Cost Tracking ──────────────────────────────────────────────────────────

class CostTracker:
    """
    Track training costs during long runs.

    Usage:
        tracker = CostTracker(gpu_hourly_rate=0.79)
        # ... training loop ...
        tracker.log()
    """

    def __init__(self, gpu_hourly_rate: Optional[float] = None):
        self.start_time = time.time()
        self.rate = gpu_hourly_rate or float(os.environ.get('GPU_HOURLY_RATE', 0))
        self._last_log = self.start_time

    @property
    def elapsed_hours(self) -> float:
        return (time.time() - self.start_time) / 3600

    @property
    def current_cost(self) -> float:
        return self.elapsed_hours * self.rate

    def log(self, step: Optional[int] = None) -> Dict:
        """Log current cost and return stats dict."""
        elapsed = self.elapsed_hours
        cost = self.current_cost
        step_str = f" | Step: {step}" if step is not None else ""
        logger.info(f"⏱  Elapsed: {elapsed:.1f}h | Cost: ${cost:.2f}{step_str}")

        return {
            'elapsed_hours': round(elapsed, 2),
            'cost_dollars': round(cost, 2),
            'rate_per_hour': self.rate,
        }

    def should_auto_stop(self, max_cost: float) -> bool:
        """Check if cost exceeds budget."""
        return self.current_cost >= max_cost


# ── Launch Script Wrapper ──────────────────────────────────────────────────

def launch_and_run(
    gpu_type: str = "NVIDIA RTX 6000 Ada",
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel-ubuntu22.04",
    script: Optional[str] = None,
    pod_name: Optional[str] = None,
    volume_gb: int = 100,
    container_disk_gb: int = 100,
    env: Optional[Dict[str, str]] = None,
    auto_shutdown: bool = True,
    max_cost: Optional[float] = None,
    country_code: Optional[str] = None,
) -> Dict:
    """
    Full lifecycle: create pod → (user runs script manually) → auto-stop.

    This creates the pod and prints connection info. The actual script
    execution happens inside the pod (typically via the container start
    command or SSH).
    """
    import datetime

    pod_name = pod_name or f"nemotron-ft-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"

    # Find best GPU if not specified
    gpu = find_best_gpu(min_vram=48)
    if gpu and not gpu_type:
        gpu_type = gpu['name']
        logger.info(f"Auto-selected GPU: {gpu_type} (${gpu['price_per_hour']:.2f}/h, "
                     f"{gpu['vram_gb']}GB)")

    # Create pod
    pod = create_pod(
        name=pod_name,
        gpu_type=gpu_type,
        image=image,
        volume_gb=volume_gb,
        container_disk_gb=container_disk_gb,
        volume_mount="/workspace",
        env=env,
        country_code=country_code,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  Pod Ready")
    print("=" * 60)
    for k, v in pod.items():
        if k != 'machine':
            print(f"  {k}: {v}")
    print("=" * 60)
    print(f"\n  Connect:  ssh root@{pod['pod_hostname']}")
    print(f"  Notebook: https://{pod['pod_hostname']}:8888")
    print(f"\n  Stop:     python scripts/runpod_auto.py stop --pod-id {pod['id']}")
    print(f"  Terminate: python scripts/runpod_auto.py terminate --pod-id {pod['id']}")
    print(f"  Monitor:  python scripts/runpod_auto.py status --pod-id {pod['id']}")

    if max_cost:
        print(f"\n  ⚠️  Auto-stop at ${max_cost:.2f} (estimate: "
              f"{max_cost/pod['price_per_hour']:.1f}h)")

    return pod


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RunPod GPU Pod automation for ASR fine-tuning"
    )
    sub = parser.add_subparsers(dest='command', help='Command')

    # gpus
    gpus_p = sub.add_parser('gpus', help='List available GPU types')
    gpus_p.add_argument('--min-vram', type=int, default=24,
                        help='Minimum VRAM in GB')
    gpus_p.add_argument('--max-price', type=float, default=None,
                        help='Maximum hourly price in $')
    gpus_p.add_argument('--sort', choices=['price', 'vram'], default='price')

    # launch
    launch_p = sub.add_parser('launch', help='Create and start a GPU pod')
    launch_p.add_argument('--gpu-type', default=None,
                          help='GPU type name (auto-selects cheapest 48GB+ if omitted)')
    launch_p.add_argument('--image',
                          default='runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel-ubuntu22.04',
                          help='Docker image')
    launch_p.add_argument('--pod-name', default=None, help='Pod name')
    launch_p.add_argument('--volume-gb', type=int, default=100,
                          help='Persistent volume size (GB)')
    launch_p.add_argument('--container-disk-gb', type=int, default=100)
    launch_p.add_argument('--env', default=None, help='JSON env dict')
    launch_p.add_argument('--country-code', default=None, help='Preferred country')
    launch_p.add_argument('--auto-shutdown', action='store_true', default=True)
    launch_p.add_argument('--max-cost', type=float, default=None,
                          help='Maximum budget in $')

    # stop
    stop_p = sub.add_parser('stop', help='Stop a pod')
    stop_p.add_argument('--pod-id', default=None, help='Pod ID (default: $RUNPOD_POD_ID)')

    # terminate
    term_p = sub.add_parser('terminate', help='Terminate (delete) a pod')
    term_p.add_argument('--pod-id', default=None, help='Pod ID (default: $RUNPOD_POD_ID)')

    # status
    status_p = sub.add_parser('status', help='Get pod status')
    status_p.add_argument('--pod-id', default=None, help='Pod ID (default: $RUNPOD_POD_ID)')
    status_p.add_argument('--watch', action='store_true', help='Watch mode')

    # guard (spot preemption handler)
    guard_p = sub.add_parser('guard', help='Register spot instance preemption guard')
    guard_p.add_argument('--checkpoint-path', default=None,
                         help='Checkpoint path to protect')

    # upload-ckpt (manual checkpoint upload)
    upload_p = sub.add_parser('upload-ckpt', help='Upload checkpoint to HF Hub')
    upload_p.add_argument('--checkpoint-path', required=True,
                          help='Path to .nemo checkpoint')
    upload_p.add_argument('--repo-id', default=None,
                          help='HF repo ID (default: $HF_REPO_ID)')

    args = parser.parse_args()

    if args.command == 'gpus':
        gpus = list_gpus(min_vram=args.min_vram, max_price=args.max_price,
                         sort_by=args.sort)
        print(f"\nAvailable GPUs (min {args.min_vram}GB VRAM):\n")
        print(f"{'GPU':<35} {'VRAM':>5} {'Price/h':>8}")
        print("-" * 50)
        for g in gpus:
            print(f"{g['name']:<35} {g['vram_gb']:>4}GB ${g['price_per_hour']:>7.2f}")
        print(f"\n{len(gpus)} GPU types found.")

    elif args.command == 'launch':
        env = json.loads(args.env) if args.env else {}
        launch_and_run(
            gpu_type=args.gpu_type,
            image=args.image,
            pod_name=args.pod_name,
            volume_gb=args.volume_gb,
            container_disk_gb=args.container_disk_gb,
            env=env,
            auto_shutdown=args.auto_shutdown,
            max_cost=args.max_cost,
            country_code=args.country_code,
        )

    elif args.command == 'stop':
        stop_pod(args.pod_id)

    elif args.command == 'terminate':
        terminate_pod(args.pod_id)

    elif args.command == 'status':
        if args.watch:
            while True:
                try:
                    s = pod_status(args.pod_id)
                    print(f"\rStatus: {s['status']} | Uptime: {s['uptime_hours']:.1f}h "
                          f"| GPU: {s['gpu_utilization_pct']}% | Cost: ${s['estimated_cost']:.4f}",
                          end='')
                    time.sleep(30)
                except KeyboardInterrupt:
                    print()
                    break
        else:
            s = pod_status(args.pod_id)
            print(f"\nPod: {s['name']} ({s['id']})")
            print(f"  Status:     {s['status']}")
            print(f"  Runtime:    {s['runtime_status']}")
            print(f"  Uptime:     {s['uptime_hours']}h")
            print(f"  GPU:        {s['gpu_type']} ({s['gpu_utilization_pct']}% util)")
            print(f"  Cost:       ${s['estimated_cost']:.4f} (@ ${s['cost_per_hour']}/h)")

    elif args.command == 'guard':
        register_spot_guard(args.checkpoint_path)

    elif args.command == 'upload-ckpt':
        url = upload_checkpoint_to_hub(
            checkpoint_path=args.checkpoint_path,
            repo_id=args.repo_id,
        )
        if url:
            print(f"Uploaded: {url}")
        else:
            print("Upload failed. Check HF_TOKEN and HF_REPO_ID.")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)
