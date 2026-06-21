#!/usr/bin/env python3
"""Evaluate a single checkpoint against all configured datasets in a clean process.

Designed to be called as a subprocess from eval_watcher.py. Each invocation runs in
a fresh Python interpreter, isolating GPU/Numba/pickle state across checkpoints.

Usage:
    python3 scripts/eval_one.py \
        --ckpt /path/to/checkpoint.ckpt \
        --datasets-json /path/to/eval_datasets.json \
        --output-csv /path/to/eval_rolling.csv \
        --base-nemo-dir /workspace/data/checkpoints

Output: Appends one CSV row per dataset to --output-csv.
Exit code 0 on success, 1 on error.
"""
import argparse, csv, gc, json, os, sys, time
from pathlib import Path
import torch
import jiwer


def load_manifest(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def detect_lang(manifest_path):
    with open(manifest_path) as f:
        first = json.loads(f.readline())
    return first.get("target_lang", first.get("lang", "ko-KR"))


def patch_prompt_dataloader(fallback_lang: str):
    from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import (
        LhotseSpeechToTextBpeDatasetWithPromptIndex,
    )
    original_fn = LhotseSpeechToTextBpeDatasetWithPromptIndex._get_prompt_index_for_cut

    def patched_get_prompt_index(self, cut):
        lang = cut.supervisions[0].language if cut.supervisions else None
        if lang is None or lang == "None":
            return self._get_prompt_index(fallback_lang)
        return original_fn(self, cut)

    LhotseSpeechToTextBpeDatasetWithPromptIndex._get_prompt_index_for_cut = patched_get_prompt_index
    return True


def _patch_nemo_torch_load():
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
    _torch = torch

    @staticmethod
    def patched_load_state_dict(model_weights, map_location=None):
        return _torch.load(model_weights, map_location=map_location, weights_only=False)

    SaveRestoreConnector._load_state_dict_from_disk = patched_load_state_dict
    return True


def load_model_from_ckpt(ckpt_path: str, base_nemo_dir: str, device: str = "cuda:0"):
    gc.collect()
    torch.cuda.empty_cache()
    _patch_nemo_torch_load()
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt

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


def load_model_from_nemo(nemo_path: str, device: str = "cuda:0"):
    gc.collect()
    torch.cuda.empty_cache()
    _patch_nemo_torch_load()
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
    model = EncDecRNNTBPEModelWithPrompt.restore_from(nemo_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model


def eval_one_dataset(model, manifest_path, ds_name, device="cuda:0"):
    target_lang = detect_lang(manifest_path)
    print(f"  [{ds_name}] lang={target_lang}", flush=True)

    data = load_manifest(manifest_path)
    refs = [e["text"] for e in data]
    audio_paths = [e["audio_filepath"] for e in data]

    patch_prompt_dataloader(target_lang)

    def _extract_text(item):
        if isinstance(item, str):
            return item
        if hasattr(item, "text"):
            return item.text
        return str(item)

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
    cer = jiwer.cer(refs, hyps) * 100
    wer = jiwer.wer(refs, hyps) * 100
    ser = sum(1 for r, h in zip(refs, hyps) if r != h) / len(refs) * 100
    print(f"  [{ds_name}] CER={cer:.2f}% WER={wer:.2f}% ({elapsed:.0f}s)", flush=True)

    return {"dataset": ds_name, "cer": cer, "wer": wer, "ser": ser, "num_samples": len(data)}


def append_results(output_csv: str, rows: list):
    fieldnames = ["checkpoint", "dataset", "cer", "wer", "ser", "num_samples", "eval_time"]
    exists = os.path.exists(output_csv)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint path (.nemo or .ckpt)")
    parser.add_argument("--datasets-json", required=True, help="JSON file mapping dataset_name -> manifest_path")
    parser.add_argument("--output-csv", required=True, help="CSV to append results to")
    parser.add_argument("--base-nemo-dir", default="/workspace/data/checkpoints")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with open(args.datasets_json) as f:
        datasets = json.load(f)

    ckpt_path = Path(args.ckpt)
    ckpt_name = ckpt_path.name
    is_ckpt = ckpt_path.suffix == ".ckpt"

    print(f"\n{'='*60}")
    print(f"Loading {ckpt_name} ...")
    t0 = time.time()

    if is_ckpt:
        model = load_model_from_ckpt(str(ckpt_path), args.base_nemo_dir, args.device)
    else:
        model = load_model_from_nemo(str(ckpt_path), args.device)
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
            r = eval_one_dataset(model, manifest_path, ds_name, args.device)
            r["checkpoint"] = ckpt_name
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  ERROR {ds_name}: {e}")
            traceback.print_exc()

    eval_time = round(time.time() - t0, 1)
    for r in results:
        r["eval_time"] = eval_time

    append_results(args.output_csv, results)
    print(f"\n  Done: {len(results)} results → {args.output_csv}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
