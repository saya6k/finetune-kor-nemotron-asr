"""Direct evaluation using .nemo model restore + transcribe (prompt-aware).

Workaround for NeMo bug: transcribe() with raw audio paths on prompt-based models
doesn't propagate language to Cut objects. We monkey-patch the prompt dataloader
to use a fallback language when cut.supervisions[0].language is None/'None'.
"""
import gc, json, os, sys, csv, time
from pathlib import Path
import torch
import jiwer


def load_manifest(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def detect_lang(manifest_path):
    """Detect the target language from the first manifest entry."""
    with open(manifest_path) as f:
        first = json.loads(f.readline())
    return first.get("target_lang", first.get("lang", "ko-KR"))


def patch_prompt_dataloader(fallback_lang: str):
    """Monkey-patch the prompt index dataloader to handle missing language on Cuts.

    NeMo's transcribe() creates Cut objects from raw audio paths, which don't have
    language metadata set. This causes ValueError("Unknown prompt key: 'None'").
    We patch _get_prompt_index_for_cut to use the fallback language when None.
    """
    from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import (
        LhotseSpeechToTextBpeDatasetWithPromptIndex,
    )

    original_fn = LhotseSpeechToTextBpeDatasetWithPromptIndex._get_prompt_index_for_cut

    def patched_get_prompt_index(self, cut):
        lang = cut.supervisions[0].language if cut.supervisions else None
        if lang is None or lang == "None":
            # Use fallback — occurs when Cut is created from raw audio path
            return self._get_prompt_index(fallback_lang)
        return original_fn(self, cut)

    LhotseSpeechToTextBpeDatasetWithPromptIndex._get_prompt_index_for_cut = (
        patched_get_prompt_index
    )
    return True


def _patch_nemo_torch_load():
    """Monkey-patch NeMo's save_restore_connector to use weights_only=False.

    PyTorch 2.6+ defaults to weights_only=True, which breaks loading of .nemo
    files whose weights include custom reduce functions (e.g. encoder bias).
    """
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

    original_fn = SaveRestoreConnector._load_state_dict_from_disk
    _torch = torch  # capture module-level torch in closure (avoid import-in-func)

    @staticmethod
    def patched_load_state_dict(model_weights, map_location=None):
        return _torch.load(model_weights, map_location=map_location, weights_only=False)

    SaveRestoreConnector._load_state_dict_from_disk = patched_load_state_dict
    return True


def _load_model_from_ckpt(ckpt_path: str, base_nemo_dir: str, device: str = "cuda:0"):
    """Load a .ckpt (PyTorch Lightning) checkpoint by restoring the original
    pretrained .nemo architecture and then loading the fine-tuned state_dict."""
    gc.collect()
    torch.cuda.empty_cache()
    _patch_nemo_torch_load()
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt

    # Find the ORIGINAL pretrained base model (not the training-updated one).
    # Priority: 1) /workspace/data/base_model/  2) checkpoint parent dir
    original_candidates = [
        "/workspace/data/base_model/nemotron-3.5-asr-streaming-0.6b.nemo",
        "/workspace/data/base_model/nemotron-3.5-asr-streaming-0.6b-v1.nemo",
    ]
    base_nemo = None
    for cand in original_candidates:
        if os.path.exists(cand):
            base_nemo = cand
            break
    if not base_nemo:
        # Fallback: search near the checkpoint or in base_nemo_dir
        base_dir = Path(ckpt_path).parent
        nemo_files = sorted(base_dir.rglob("*.nemo"))
        if not nemo_files:
            nemo_files = sorted(Path(base_nemo_dir).rglob("*.nemo"))
        if nemo_files:
            base_nemo = str(nemo_files[0])
    if not base_nemo:
        raise FileNotFoundError(f"No base .nemo found near {ckpt_path}")

    print(f"  Loading base model from {Path(base_nemo).name} ...", flush=True)
    t0 = time.time()
    model = EncDecRNNTBPEModelWithPrompt.restore_from(base_nemo, map_location=device)
    model = model.to(device)
    print(f"    Base loaded in {time.time()-t0:.1f}s", flush=True)

    print(f"  Loading state_dict from {Path(ckpt_path).name} ...", flush=True)
    t1 = time.time()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    print(f"    State dict loaded in {time.time()-t1:.1f}s", flush=True)
    return model


def eval_checkpoint(ckpt_path: str, datasets: dict, device: str = "cuda:0",
                    base_nemo_dir: str = None) -> list:
    """Load checkpoint once, evaluate all datasets, return list of result dicts.

    Handles both .nemo (NeMo archive) and .ckpt (PyTorch Lightning) checkpoints.
    """
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt

    gc.collect()
    torch.cuda.empty_cache()

    ckpt_path = Path(ckpt_path)
    is_ckpt = ckpt_path.suffix == ".ckpt"

    print(f"\n{'='*60}")
    print(f"Loading {ckpt_path.name} ...")
    t0 = time.time()

    if is_ckpt:
        model = _load_model_from_ckpt(str(ckpt_path),
                                      base_nemo_dir or str(ckpt_path.parent),
                                      device)
    else:
        _patch_nemo_torch_load()
        gc.collect()
        torch.cuda.empty_cache()
        model = EncDecRNNTBPEModelWithPrompt.restore_from(str(ckpt_path), map_location=device)
        model = model.to(device)
        model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    results = []
    for ds_name, manifest_path in datasets.items():
        if not manifest_path or not os.path.exists(manifest_path):
            print(f"  SKIP {ds_name}: manifest not found")
            continue
        if os.path.getsize(manifest_path) == 0:
            print(f"  SKIP {ds_name}: empty manifest")
            continue
        try:
            r = _eval_with_model(model, ckpt_path, manifest_path, ds_name, device)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  ERROR {ds_name}: {e}")
            traceback.print_exc()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def _eval_with_model(model, checkpoint_path, manifest_path, ds_name, device="cuda:0"):
    """Evaluate one dataset against an already-loaded model."""
    ckpt_name = Path(checkpoint_path).name
    target_lang = detect_lang(manifest_path)
    print(f"  [{ds_name}] lang={target_lang}", flush=True)

    data = load_manifest(manifest_path)
    refs = [e["text"] for e in data]
    audio_paths = [e["audio_filepath"] for e in data]

    patch_prompt_dataloader(target_lang)

    hyps = []
    batch_size = 8
    t1 = time.time()
    for batch_start in range(0, len(audio_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(audio_paths))
        batch = audio_paths[batch_start:batch_end]
        try:
            result = model.transcribe(batch, batch_size=len(batch), target_lang=target_lang, verbose=False)
            hyps.extend(_extract_text(r) for r in result)
        except Exception as e:
            print(f"    batch [{batch_start}:{batch_end}] fallback: {e}")
            for ap in batch:
                try:
                    h = model.transcribe([ap], batch_size=1, target_lang=target_lang, verbose=False)[0]
                except Exception:
                    h = ""
                hyps.append(_extract_text(h))

        progress = min(batch_end, len(audio_paths))
        if progress % 100 == 0 or progress == len(audio_paths):
            partial_cer = jiwer.cer(refs[:progress], hyps[:progress]) * 100
            print(f"    [{progress}/{len(audio_paths)}] CER={partial_cer:.2f}%", flush=True)

    elapsed = time.time() - t1
    refs_n = [r.lower() for r in refs]
    hyps_n = [h.lower() for h in hyps]

    # Filter empty hypothesis pairs (empty output = model silence, handled separately)
    pairs = [(r, h) for r, h in zip(refs_n, hyps_n) if h.strip()]
    empty_count = len(refs_n) - len(pairs)
    empty_rate = empty_count / len(refs_n) * 100 if refs_n else 0
    if empty_count:
        print(f"    Empty outputs: {empty_count}/{len(refs_n)} ({empty_rate:.1f}%)", flush=True)
    refs_f = [r for r, _ in pairs]
    hyps_f = [h for _, h in pairs]

    cer = jiwer.cer(refs_f, hyps_f) * 100 if refs_f else 0.0
    wer = jiwer.wer(refs_f, hyps_f) * 100 if refs_f else 0.0
    ser = sum(1 for r, h in zip(refs_n, hyps_n) if r != h) / len(refs) * 100
    print(f"  [{ds_name}] CER={cer:.2f}% WER={wer:.2f}% empty={empty_rate:.1f}% ({elapsed:.0f}s)", flush=True)

    return {"checkpoint": ckpt_name, "dataset": ds_name, "cer": cer, "wer": wer,
            "ser": ser, "num_samples": len(data), "empty_rate": empty_rate}


def _extract_text(item):
    if isinstance(item, str):
        return item
    if hasattr(item, "text"):
        return item.text
    return str(item)


def run_eval(checkpoint_path, manifest_path, ds_name, device="cuda:0"):
    # Restore model once per checkpoint
    print(f"Loading model from {checkpoint_path}...")
    t0 = time.time()
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
    model = EncDecRNNTBPEModelWithPrompt.restore_from(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    ckpt_name = Path(checkpoint_path).name
    target_lang = detect_lang(manifest_path)
    print(f"  Dataset: {ds_name} ({Path(manifest_path).stat().st_size//1024}KB), lang={target_lang}")

    # Read all references and audio paths
    data = load_manifest(manifest_path)
    refs = [e["text"] for e in data]
    audio_paths = [e["audio_filepath"] for e in data]

    # Monkey-patch dataloader to handle missing Cut language
    patch_prompt_dataloader(target_lang)

    # Transcribe using model's built-in batching with target_lang parameter
    print(f"  Transcribing {len(audio_paths)} samples...")
    t1 = time.time()

    def _extract_text(item):
        """Handle both string and Hypothesis return types from transcribe()."""
        if isinstance(item, str):
            return item
        if hasattr(item, "text"):
            return item.text
        return str(item)

    hyps = []
    batch_size = 8
    for batch_start in range(0, len(audio_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(audio_paths))
        batch = audio_paths[batch_start:batch_end]
        try:
            result = model.transcribe(batch, batch_size=len(batch), target_lang=target_lang, verbose=False)
            hyps.extend(_extract_text(r) for r in result)
        except Exception as e:
            print(f"  Batch [{batch_start}:{batch_end}] failed: {e}, falling back to one-by-one")
            for ap in batch:
                try:
                    h = model.transcribe([ap], batch_size=1, target_lang=target_lang, verbose=False)[0]
                except Exception:
                    h = ""
                hyps.append(_extract_text(h))

        progress = min(batch_end, len(audio_paths))
        if progress % 100 == 0 or progress == len(audio_paths):
            hyps_text = [_extract_text(h) for h in hyps]
            partial_cer = jiwer.cer(refs[:progress], hyps_text[:progress]) * 100
            print(f"  [{progress}/{len(audio_paths)}] partial CER={partial_cer:.2f}%")

    elapsed = time.time() - t1
    print(f"  Done in {elapsed:.1f}s ({len(audio_paths)/elapsed:.1f} samples/s)")

    refs_n = [r.lower() for r in refs]
    hyps_n = [h.lower() for h in hyps]

    pairs = [(r, h) for r, h in zip(refs_n, hyps_n) if h.strip()]
    empty_count = len(refs_n) - len(pairs)
    empty_rate = empty_count / len(refs_n) * 100 if refs_n else 0
    if empty_count:
        print(f"  Empty outputs: {empty_count}/{len(refs_n)} ({empty_rate:.1f}%)")
    refs_f = [r for r, _ in pairs]
    hyps_f = [h for _, h in pairs]

    cer = jiwer.cer(refs_f, hyps_f) * 100 if refs_f else 0.0
    wer = jiwer.wer(refs_f, hyps_f) * 100 if refs_f else 0.0
    ser = sum(1 for r, h in zip(refs_n, hyps_n) if r != h) / len(refs) * 100

    print(f"  CER={cer:.2f}% WER={wer:.2f}% empty={empty_rate:.1f}% SER={ser:.2f}%")
    del model
    torch.cuda.empty_cache()
    return {"checkpoint": ckpt_name, "dataset": ds_name, "cer": cer, "wer": wer, "ser": ser, "num_samples": len(data)}


def sweep_checkpoints_direct(
    checkpoint_dir: str,
    datasets: dict,
    output_csv: str,
    device: str = "cuda:0",
) -> str:
    """Sweep all .nemo checkpoints against all datasets using direct evaluation.

    Uses monkey-patched prompt dataloader (bypasses NeMo Hydra inference) to
    handle the Cut language=None bug. Each checkpoint is loaded once and used
    across all datasets before unloading.

    Args:
        checkpoint_dir: Directory containing .nemo checkpoint files.
        datasets: Dict mapping dataset name → manifest path.
        output_csv: Path to write results CSV.
        device: CUDA device for inference.

    Returns:
        Path to the output CSV, or empty string if no checkpoints found.
    """
    ckpt_files = sorted(Path(checkpoint_dir).rglob("*.nemo"))
    if not ckpt_files:
        print("No .nemo checkpoints found.")
        return ""

    print(f"Checkpoints: {[c.name for c in ckpt_files]}")
    print(f"Datasets: {list(datasets.keys())}")

    results = []
    for ckpt in ckpt_files:
        for ds_name, manifest_path in datasets.items():
            if not os.path.exists(manifest_path):
                print(f"SKIP {ds_name}: manifest not found")
                continue
            if os.path.getsize(manifest_path) == 0:
                print(f"SKIP {ds_name}: manifest is empty")
                continue
            try:
                r = run_eval(str(ckpt), manifest_path, ds_name, device=device)
                results.append(r)
            except Exception as e:
                print(f"ERROR: {ckpt.name} x {ds_name}: {e}")
                import traceback
                traceback.print_exc()

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint", "dataset", "cer", "wer", "ser", "num_samples"])
        w.writeheader()
        w.writerows(results)
    print(f"\nResults -> {output_csv}")

    # Summary
    print(f"\n{'='*80}")
    print(f"  Evaluation Summary")
    print(f"{'='*80}")
    print(f"  {'Dataset':<30s} {'CER':>8s} {'WER':>8s} {'SER':>8s}  Samples")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['dataset']:<30s} {r['cer']:>7.2f}% {r['wer']:>7.2f}% {r['ser']:>7.2f}% {r['num_samples']:>8d}")

    fleurs = [r for r in results if "fleurs" in r["dataset"].lower()]
    if fleurs:
        best = min(fleurs, key=lambda r: r["cer"])
        baseline = 7.12
        print(f"\n  FLEURS Best CER: {best['cer']:.2f}% (baseline: {baseline}%)")

    return output_csv


def main():
    ckpt_dir = sys.argv[1]
    datasets_json = sys.argv[2]
    output_csv = sys.argv[3]

    datasets = json.loads(datasets_json)
    sweep_checkpoints_direct(ckpt_dir, datasets, output_csv)


if __name__ == "__main__":
    main()
