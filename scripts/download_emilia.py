#!/usr/bin/env python3
"""Download Emilia-YODAS language shards from HuggingFace to local disk.

Usage:
    python3 scripts/download_emilia.py --langs KO [EN JA ZH] --local-dir /workspace/emilia_local
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HF_TOKEN = os.environ.get('HF_TOKEN', '')
REPO_ID = 'amphion/Emilia-Dataset'


def list_lang_files(api, lang: str, token: str):
    items = api.list_repo_tree(
        repo_id=REPO_ID,
        repo_type='dataset',
        path_in_repo=f'Emilia-YODAS/{lang}',
        token=token,
        recursive=False,
    )
    return [item.path for item in items
            if hasattr(item, 'path') and item.path.endswith('.tar')]


def download_one(path: str, local_dir: str, token: str):
    from huggingface_hub import hf_hub_download
    dest = Path(local_dir) / path
    if dest.exists():
        return f'skip {path}'
    dest.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type='dataset',
        filename=path,
        local_dir=local_dir,
        token=token,
    )
    return f'done {path}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--langs', nargs='+', default=['KO'])
    parser.add_argument('--local-dir', default='/workspace/emilia_local')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-shards', type=int, default=0,
                        help='Max TAR shards per language (0 = all)')
    args = parser.parse_args()

    token = HF_TOKEN or os.environ.get('HUGGING_FACE_HUB_TOKEN', '')
    if not token:
        sys.exit('ERROR: HF_TOKEN not set')

    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

    from huggingface_hub import HfApi
    api = HfApi()

    all_files = []
    for lang in args.langs:
        print(f'Listing {lang} files...', flush=True)
        files = list_lang_files(api, lang, token)
        if args.max_shards and len(files) > args.max_shards:
            files = files[:args.max_shards]
            print(f'  {lang}: {len(files)} TAR files (capped at {args.max_shards})', flush=True)
        else:
            print(f'  {lang}: {len(files)} TAR files', flush=True)
        all_files.extend(files)

    print(f'Total: {len(all_files)} files → {args.local_dir}', flush=True)

    done = skip = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_one, f, args.local_dir, token): f for f in all_files}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                result = fut.result()
                if result.startswith('skip'):
                    skip += 1
                else:
                    done += 1
                print(f'[{i}/{len(all_files)}] {result}', flush=True)
            except Exception as e:
                fail += 1
                print(f'[{i}/{len(all_files)}] ERROR {futs[fut]}: {e}', flush=True)

    print(f'\nFinished: {done} downloaded, {skip} skipped, {fail} failed', flush=True)


if __name__ == '__main__':
    main()
