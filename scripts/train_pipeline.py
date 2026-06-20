#!/usr/bin/env python3
"""
Production Korean ASR Fine-Tuning Pipeline for Nemotron 3.5 ASR Streaming.

Single executable script — replaces the 17-cell notebook with direct execution.
Designed for RunPod GPU Pods with public IP + SSH access.

Usage:
    export HF_TOKEN="hf_..."
    export RUNPOD_API_KEY="rpa_..."
    python3 train_pipeline.py

Env vars (all optional with defaults):
    SMOKE_N=0           Samples to process (0 = 전체 데이터, unlimited)
    HOLD_OUT_N=1000     Validation holdout samples
    MAX_EPOCHS=3        Training epochs
    BATCH_DURATION=100  Batch size in seconds (100 proven on L40S 48GB)
    HF_REPO_ID=saya6k/nemotron-kor-checkpoints
"""

import os, sys, subprocess, logging, json, shlex, random, time, gc
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Allow imports from sibling scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('train_pipeline')

# ── Configuration ────────────────────────────────────────────────────
WORKSPACE = Path(os.environ.get('WORKSPACE', '/workspace'))
DATA_DIR = Path(os.environ.get('DATA_DIR', str(WORKSPACE / 'data')))
NEMO_DIR = Path(os.environ.get('NEMO_DIR', str(WORKSPACE / 'NeMo')))
BATCH_DURATION = int(os.environ.get('BATCH_DURATION', '100'))  # 100 proven on L40S 48GB
MAX_EPOCHS = int(os.environ.get('MAX_EPOCHS', '3'))
SMOKE_N = int(os.environ.get('SMOKE_N', '0'))  # 0 = 전체 데이터 (unlimited)
HOLD_OUT_N = int(os.environ.get('HOLD_OUT_N', '1000'))
TEST_HOLD_OUT_N = 500
HF_REPO_ID = os.environ.get('HF_REPO_ID', 'saya6k/nemotron-kor-checkpoints')

# Audio backend (avoid torchcodec CUDA mismatch)
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

# Monkey-patch datasets Audio to use soundfile instead of torchcodec.
# Newer datasets (≥3.x) require torchcodec which conflicts with pod CUDA 12.1.
# We patch both decode_example (bytes→array) and encode_example (array→dict).
def _audio_bytes_to_array(value):
    """Decode raw audio bytes to (array, sampling_rate) using soundfile."""
    import soundfile as sf
    import io
    import numpy as np
    array, sr = sf.read(io.BytesIO(value), dtype='float32')
    return array, sr

def _patch_datasets_audio():
    """Replace datasets Audio decode/encode with soundfile-based versions."""
    try:
        import datasets.features.audio as _audio_mod
        _orig_decode = _audio_mod.Audio.decode_example
        _orig_encode = _audio_mod.Audio.encode_example

        def _patched_decode(self, value, token_per_repo_id=None):
            if isinstance(value, bytes):
                array, sr = _audio_bytes_to_array(value)
                return {"array": array, "sampling_rate": sr}
            if isinstance(value, dict) and value.get('bytes'):
                array, sr = _audio_bytes_to_array(value['bytes'])
                result = {"array": array, "sampling_rate": sr}
                if value.get('path'):
                    result['path'] = value['path']
                return result
            if isinstance(value, dict) and value.get('array') is not None:
                return value
            return _orig_decode(self, value, token_per_repo_id=token_per_repo_id)

        def _patched_encode(self, value):
            import numpy as np
            if isinstance(value, bytes):
                array, sr = _audio_bytes_to_array(value)
                return {"array": array, "sampling_rate": sr}
            if isinstance(value, dict) and value.get('bytes'):
                array, sr = _audio_bytes_to_array(value['bytes'])
                result = {"array": array, "sampling_rate": sr}
                if value.get('path'):
                    result['path'] = value['path']
                return result
            if isinstance(value, dict) and value.get('array') is not None:
                return value
            if isinstance(value, np.ndarray):
                return {"array": value, "sampling_rate": self.sampling_rate or 16000}
            return _orig_encode(self, value)

        _audio_mod.Audio.decode_example = _patched_decode
        _audio_mod.Audio.encode_example = _patched_encode
        logger.info("Audio patch: soundfile 기반으로 datasets Audio 우회 (decode+encode)")
    except Exception as e:
        logger.warning(f"Audio patch 실패 (fallback to torchcodec): {e}")

random.seed(42)

# ── Path helpers ─────────────────────────────────────────────────────
PROCESSED_DIR = DATA_DIR / 'processed'
WAV_CACHE_DIR = DATA_DIR / 'wav_cache'
CHECKPOINT_DIR = DATA_DIR / 'checkpoints'
RESULTS_DIR = DATA_DIR / 'results'

for d in [DATA_DIR / 'raw', PROCESSED_DIR, WAV_CACHE_DIR,
          CHECKPOINT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: SETUP                                                     ║
# ╚═════════════════════════════════════════════════════════════════════╝

def setup() -> Dict:
    """GPU check, dependency installs, NeMo clone, .nemo model resolution."""
    import torch

    # GPU check
    logger.info("GPU 확인 중...")
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        logger.info(f"  GPU: {parts[0]} | VRAM: {parts[1]}MB total, {parts[2]}MB free")

    logger.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  Device: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"  VRAM: {total_mem / 1024**3:.1f} GB")

    # Dependency installs — skip when SKIP_SETUP_INSTALL=1 (e.g. nemotron-asr-dgx image
    # already has all deps pre-installed at correct versions).
    if os.environ.get('SKIP_SETUP_INSTALL', '0') != '1':
        logger.info("Dependencies 설치 중...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
            'wget', 'text-unidecode', 'matplotlib>=3.3.2', 'Cython'], check=False)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
            'huggingface-hub', 'librosa', 'datasets', 'soundfile', 'tqdm',
            'jiwer', 'gTTS', 'runpod', 'sentencepiece', 'torchcodec'], check=False)
        subprocess.run(['apt-get', 'install', '-y', '-qq',
            'sox', 'libsndfile1', 'ffmpeg', 'libsox-fmt-mp3', 'jq'],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # cuDNN 9 (required by NeMo, installed via pip into Python package dir)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
            'nvidia-cudnn-cu12>=9', 'nvidia-nvjitlink-cu12'], check=False)
        py_ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
        cudnn_lib = f'/usr/local/lib/{py_ver}/dist-packages/nvidia/cudnn/lib'
        if os.path.isdir(cudnn_lib):
            os.environ['LD_LIBRARY_PATH'] = cudnn_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

        # NeMo 2.7.3 pip + main branch source
        logger.info("NeMo 2.7.3 (pip) + main branch (source) 설치 중...")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q',
             'nemo_toolkit[asr]==2.7.3'],
            check=False
        )
    else:
        logger.info("SKIP_SETUP_INSTALL=1 — pre-built image, skipping pip installs")

    if not NEMO_DIR.exists():
        logger.info(f"NeMo main 클론 중... {NEMO_DIR}")
        subprocess.run(['git', 'clone', '--depth', '1',
            'https://github.com/NVIDIA/NeMo.git', str(NEMO_DIR)], check=True)
    os.environ["PYTHONPATH"] = f"{NEMO_DIR}:{os.environ.get('PYTHONPATH', '')}"

    # ── Critical patches for driver 550 compatibility ──────────────────
    # Numba PTX downgrade: CUDA 12.8 toolkit → PTX 8.7, driver supports max 8.4
    # Skipped on Blackwell (sm_100+): toolkit/driver match, no PTX version clash
    _sm_major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    if _sm_major >= 10:
        logger.info("Blackwell sm_100 감지 — Numba PTX 패치 건너뜀 (PTX 9.x 네이티브 지원)")
    else:
        logger.info("Numba PTX 패치 적용 중...")
        patch_script = Path(__file__).resolve().parent / 'patch_numba_codegen.py'
        if patch_script.exists():
            subprocess.run([sys.executable, str(patch_script)], check=False)
        else:
            logger.warning(f"patch_numba_codegen.py not found at {patch_script}")

    # nv_one_logger stub (NVIDIA internal package, not on PyPI)
    logger.info("nv_one_logger stub 생성 중...")
    nemo_lightning_dir = Path(f'/usr/local/lib/{py_ver}/dist-packages/nemo/lightning')
    nemo_lightning_dir.mkdir(parents=True, exist_ok=True)
    stub_path = nemo_lightning_dir / 'one_logger_callback.py'
    stub_path.write_text(
        '"""Stub: nv_one_logger is NVIDIA internal, not available on PyPI."""\n'
        'from lightning.pytorch.callbacks import Callback\n'
        'class OneLoggerNeMoCallback(Callback):\n'
        '    def __init__(self, *args, **kwargs):\n'
        '        super().__init__()\n'
    )
    logger.info(f"  Stub created: {stub_path}")

    # Copy prompt model files from main branch to pip install location
    logger.info("Prompt model 파일 복사 중...")
    pkg_nemo = f'/usr/local/lib/{py_ver}/dist-packages/nemo'
    for src_rel, dst_rel in [
        ('nemo/collections/asr/models/rnnt_bpe_models_prompt.py',
         'collections/asr/models/rnnt_bpe_models_prompt.py'),
        ('nemo/collections/asr/data/audio_to_text_lhotse_prompt_index.py',
         'collections/asr/data/audio_to_text_lhotse_prompt_index.py'),
    ]:
        src = NEMO_DIR / src_rel
        dst = Path(pkg_nemo) / dst_rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text())
            logger.info(f"  Copied: {dst_rel}")
        else:
            logger.warning(f"  MISSING: {src}")

    # Copy mixins for PromptStreamingMixin
    mixins_src = NEMO_DIR / 'nemo/collections/asr/parts/mixins'
    mixins_dst = Path(pkg_nemo) / 'collections/asr/parts/mixins'
    mixins_dst.mkdir(parents=True, exist_ok=True)
    for f in mixins_src.glob('*.py'):
        dst = mixins_dst / f.name
        dst.write_text(f.read_text())
    logger.info("  Copied mixins")

    # .nemo model resolution
    hf_ckpt = os.environ.get('HF_CKPT', '')
    if not hf_ckpt or not os.path.isfile(hf_ckpt):
        logger.info(".nemo 파일 검색 중...")
        candidates = sorted(WORKSPACE.rglob('*.nemo'))
        if candidates:
            hf_ckpt = str(candidates[0])
            logger.info(f"  발견: {hf_ckpt}")
        else:
            logger.info("  HuggingFace Hub에서 다운로드...")
            from huggingface_hub import snapshot_download
            model_repo = os.environ.get('HF_MODEL_REPO',
                'nvidia/nemotron-3.5-asr-streaming-0.6b')
            downloaded = snapshot_download(
                repo_id=model_repo,
                local_dir=str(DATA_DIR / 'base_model'),
                local_dir_use_symlinks=False,
            )
            nemo_files = sorted(Path(downloaded).rglob('*.nemo'))
            if nemo_files:
                hf_ckpt = str(nemo_files[0])
                logger.info(f"  다운로드 완료: {hf_ckpt}")
            else:
                raise FileNotFoundError(f".nemo file not found in {model_repo}")

    assert os.path.isfile(hf_ckpt), f".nemo not found: {hf_ckpt}"
    file_size_gb = os.path.getsize(hf_ckpt) / 1024**3
    logger.info(f"HF_CKPT: {hf_ckpt} ({file_size_gb:.2f} GB)")

    # RunPod spot guard
    from runpod_auto import is_runpod, register_spot_guard
    ON_RUNPOD = is_runpod()
    if ON_RUNPOD:
        pod_id = os.environ.get('RUNPOD_POD_ID', 'unknown')
        logger.info(f"RunPod Pod: {pod_id}")
        register_spot_guard(checkpoint_dir=str(CHECKPOINT_DIR))

    return {'hf_ckpt': hf_ckpt, 'on_runpod': ON_RUNPOD}


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  STEP 2: DATA INGEST (stream-to-file)                              ║
# ╚═════════════════════════════════════════════════════════════════════╝

def data_ingest() -> Tuple[str, int]:
    """
    Stream Emilia-YODAS Korean samples and write directly to a temp manifest file.
    Returns (temp_manifest_path, total_entries_written).
    """
    import soundfile as sf
    import librosa
    from datasets import load_dataset
    _patch_datasets_audio()  # apply after datasets is available

    logger.info("Emilia-YODAS Korean streaming 시작 (stream-to-file)...")
    logger.info(f"  SMOKE_N: {SMOKE_N}")

    ds = load_dataset(
        "amphion/Emilia-Dataset",
        data_files={"train": "Emilia-YODAS/KO/**/*.tar"},
        split="train",
        streaming=True
    )

    temp_path = PROCESSED_DIR / '_temp_ingest_ko.jsonl'
    entries_written = 0
    skipped_text = 0
    wav_cache_hits = 0

    with open(temp_path, 'w', encoding='utf-8') as f:
        for sample in ds:
            # Emilia-YODAS structure: json.text + mp3 (24kHz MP3 audio)
            meta = sample.get('json', {})
            text = meta.get('text', '').strip()
            if not text:
                skipped_text += 1
                continue

            # WAV conversion (with cache)
            audio = sample['mp3']
            sample_id = meta.get('_id', sample.get('__key__', str(abs(hash(text)))))
            wav_filename = f"ko_{sample_id}.wav".replace('/', '_')
            wav_path = WAV_CACHE_DIR / wav_filename

            if wav_path.exists():
                wav_cache_hits += 1
                duration = librosa.get_duration(path=str(wav_path))
            else:
                audio_array = audio['array']
                sr = audio['sampling_rate']
                if sr != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                sf.write(str(wav_path), audio_array, 16000)
                duration = len(audio_array) / 16000

            entry = {
                "audio_filepath": str(wav_path),
                "duration": round(duration, 3),
                "text": text,
                "lang": "ko-KR",
                "target_lang": "ko-KR",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            entries_written += 1

            if entries_written % 1000 == 0:
                logger.info(f"  {entries_written} entries written...")

            if SMOKE_N > 0 and entries_written >= SMOKE_N:
                logger.info(f"SMOKE_N={SMOKE_N} 도달, 처리 중단")
                break

    # Force garbage collection (tar buffers)
    del ds
    gc.collect()

    logger.info(f"Data ingest 완료: {entries_written} entries")
    logger.info(f"  WAV 캐시 히트: {wav_cache_hits}")
    logger.info(f"  빈 텍스트 스킵: {skipped_text}")
    logger.info(f"  Temp file: {temp_path}")

    return str(temp_path), entries_written


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  STEP 3: BUILD MANIFESTS (shuffle + split)                         ║
# ╚═════════════════════════════════════════════════════════════════════╝

def build_manifests(temp_path: str) -> Dict[str, str]:
    """
    Shuffle temp manifest and split into train/val/test.
    Returns dict mapping manifest name → path.
    """
    logger.info("Manifest 빌드 중...")

    # Read all entries from temp file
    with open(temp_path, 'r', encoding='utf-8') as f:
        all_entries = [json.loads(line) for line in f if line.strip()]

    random.shuffle(all_entries)
    logger.info(f"  총 entries: {len(all_entries)}")

    # Split: val holdout from front, test holdout from back, train from middle
    holdout_entries = all_entries[:HOLD_OUT_N]
    middle_entries = all_entries[HOLD_OUT_N:]

    # Reserve TEST_HOLD_OUT_N from the end for test holdout
    if len(middle_entries) > TEST_HOLD_OUT_N:
        test_holdout_entries = middle_entries[-TEST_HOLD_OUT_N:]
        train_pool = middle_entries[:-TEST_HOLD_OUT_N]
    else:
        # Not enough data for a separate test holdout — use all remaining for train
        test_holdout_entries = []
        train_pool = middle_entries

    # Train: SMOKE_N=0이면 전체, 아니면 SMOKE_N까지만
    train_n = len(train_pool) if SMOKE_N == 0 else min(SMOKE_N, len(train_pool))
    train_entries = train_pool[:train_n]

    manifests = {}

    def write_manifest(name, entries, path):
        with open(path, 'w', encoding='utf-8') as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
        manifests[name] = path
        logger.info(f"  {name}: {len(entries)} entries → {path}")

    write_manifest('train_ko', train_entries,
                   str(PROCESSED_DIR / 'train_manifest_ko.json'))
    write_manifest('val_emilia_holdout_ko', holdout_entries,
                   str(PROCESSED_DIR / 'val_emilia_holdout_ko.json'))
    write_manifest('test_emilia_holdout_ko', test_holdout_entries,
                   str(PROCESSED_DIR / 'test_emilia_holdout_ko.json'))

    # FLEURS Korean eval
    logger.info("FLEURS Korean eval manifest 생성 중...")
    try:
        from datasets import load_dataset
        import librosa, soundfile as sf
        fleurs_ds = load_dataset("google/fleurs", "ko_kr", split="test")
        fleurs_entries = []
        for sample in fleurs_ds:
            audio = sample['audio']
            wav_path = str(WAV_CACHE_DIR / f"fleurs_{sample['id']}.wav")
            audio_array = audio['array']
            sr = audio['sampling_rate']
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            sf.write(wav_path, audio_array, 16000)
            fleurs_entries.append({
                "audio_filepath": wav_path,
                "duration": round(len(audio_array) / 16000, 3),
                "text": sample['transcription'].strip(),
                "lang": "ko-KR",
                "target_lang": "ko-KR"
            })
        write_manifest('test_fleurs_ko', fleurs_entries,
                       str(PROCESSED_DIR / 'test_fleurs_ko.json'))
    except Exception as e:
        logger.warning(f"FLEURS manifest 실패: {e}")
        manifests['test_fleurs_ko'] = None

    # Zeroth Korean eval (requires manual download to data/raw/zeroth/)
    zeroth_dir = DATA_DIR / 'raw' / 'zeroth'
    if zeroth_dir.exists():
        logger.info("Zeroth Korean eval manifest 생성 중...")
        try:
            from build_manifest import build_manifest as bm
            zeroth_test = zeroth_dir / 'data' / 'test'
            zeroth_output = str(PROCESSED_DIR / 'test_zeroth_ko.json')
            n = bm(
                audio_dir=str(zeroth_test / 'wav') if (zeroth_test / 'wav').exists() else str(zeroth_test),
                output_path=zeroth_output,
                transcript_file=str(zeroth_test / 'text'),
                transcript_format='kaldi',
                lang='ko-KR',
                target_lang='ko-KR',
                audio_ext='.flac',
                skip_missing_transcripts=True,
            )
            if n > 0:
                manifests['test_zeroth_ko'] = zeroth_output
                logger.info(f"  Zeroth Korean: {n} entries")
        except Exception as e:
            logger.warning(f"Zeroth manifest 실패: {e}")

    # Cleanup temp file
    os.remove(temp_path)
    logger.info("Manifest 빌드 완료.")

    return manifests


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  STEP 4: LANGUAGE MIX                                              ║
# ╚═════════════════════════════════════════════════════════════════════╝

def language_mix(manifests: Dict[str, str]) -> str:
    """
    Add EN/JA/ZH samples from Emilia-YODAS to create a mixed training manifest.
    Returns path to mixed manifest.
    """
    import soundfile as sf
    import librosa
    from datasets import load_dataset
    _patch_datasets_audio()  # apply after datasets is available

    lang_mix = {
        "ko": float(os.environ.get('LANG_MIX_KO', '0.80')),
        "en": float(os.environ.get('LANG_MIX_EN', '0.10')),
        "ja": float(os.environ.get('LANG_MIX_JA', '0.05')),
        "zh": float(os.environ.get('LANG_MIX_ZH', '0.05')),
    }
    assert abs(sum(lang_mix.values()) - 1.0) < 0.01, f"Sum={sum(lang_mix.values())}"

    # Load Korean entries
    ko_path = manifests.get('train_ko', str(PROCESSED_DIR / 'train_manifest_ko.json'))
    with open(ko_path, 'r', encoding='utf-8') as f:
        ko_entries = [json.loads(line) for line in f if line.strip()]
    n_ko = len(ko_entries)
    if n_ko == 0:
        raise RuntimeError(
            "Korean train manifest is empty. "
            "Check: (1) SMOKE_N is set correctly, (2) data ingest produced entries, "
            "(3) build_manifests split didn't allocate all entries to holdout."
        )

    total_entries = int(n_ko / lang_mix["ko"])
    other_total = total_entries - n_ko

    logger.info(f"Language Mix: ko={n_ko} ({lang_mix['ko']:.0%}), "
                f"other={other_total} ({1-lang_mix['ko']:.0%})")

    other_langs = [
        ("en", "Emilia-YODAS/EN/**/*.tar", "en-US"),
        ("ja", "Emilia-YODAS/JA/**/*.tar", "ja-JP"),
        ("zh", "Emilia-YODAS/ZH/**/*.tar", "zh-CN"),
    ]

    mixed_entries = list(ko_entries)
    eval_other = {}

    for lang_code, data_pattern, lang_tag in other_langs:
        target_n = int(other_total * lang_mix[lang_code] / (1 - lang_mix["ko"]))
        logger.info(f"  {lang_code} 로드 중 (목표: {target_n})...")

        try:
            other_ds = load_dataset(
                "amphion/Emilia-Dataset",
                data_files={"train": data_pattern},
                split="train",
                streaming=True
            )

            lang_entries = []
            for sample in other_ds:
                if len(lang_entries) >= target_n + 500:
                    break
                meta = sample.get('json', {})
                text = meta.get('text', '').strip()
                if not text:
                    continue

                audio = sample['mp3']
                sample_id = meta.get('_id', sample.get('__key__', str(abs(hash(text)))))
                wav_filename = f"{lang_code}_{sample_id}.wav".replace('/', '_')
                wav_path = WAV_CACHE_DIR / wav_filename

                if not wav_path.exists():
                    audio_array = audio['array']
                    sr = audio['sampling_rate']
                    if sr != 16000:
                        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                    sf.write(str(wav_path), audio_array, 16000)

                duration = librosa.get_duration(path=str(wav_path))
                lang_entries.append({
                    "audio_filepath": str(wav_path),
                    "duration": round(duration, 3),
                    "text": text,
                    "lang": lang_tag,
                    "target_lang": lang_tag,
                })

            mixed_entries.extend(lang_entries[:target_n])
            if len(lang_entries) > target_n:
                eval_entries = lang_entries[target_n:target_n + 500]
                eval_path = str(PROCESSED_DIR / f'test_mixed_{lang_code}.json')
                with open(eval_path, 'w', encoding='utf-8') as fe:
                    for e in eval_entries:
                        fe.write(json.dumps(e, ensure_ascii=False) + '\n')
                eval_other[f"mixed_{lang_code}"] = eval_path

            logger.info(f"    {len(lang_entries[:target_n])} entries + "
                        f"{min(500, max(0, len(lang_entries)-target_n))} eval")

            del other_ds
            gc.collect()

        except Exception as e:
            logger.warning(f"  {lang_code} 로드 실패: {e}")

    # Shuffle and write mixed manifest
    random.shuffle(mixed_entries)
    mixed_path = str(PROCESSED_DIR / 'train_manifest_mixed.json')
    with open(mixed_path, 'w', encoding='utf-8') as f:
        for e in mixed_entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    # Ratio validation
    ko_count = sum(1 for e in mixed_entries if e.get('lang') in ('ko-KR',) or e.get('target_lang') in ('ko-KR',))
    actual_ko = ko_count / len(mixed_entries)
    assert 0.70 <= actual_ko <= 0.85, f"KO ratio {actual_ko:.1%} out of [70%,85%]"

    logger.info(f"혼합 완료: {len(mixed_entries)} entries, ko={actual_ko:.1%}")

    # Store eval paths for later use
    manifests.update(eval_other)

    return mixed_path


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  STEP 5: TOKENIZER VERIFICATION                                    ║
# ╚═════════════════════════════════════════════════════════════════════╝

def verify_tokenizer(hf_ckpt: str) -> bool:
    """Check tokenizer coverage and jamo decomposition on Korean text.

    Extracts SentencePiece tokenizer from the .nemo tar archive directly,
    avoiding PyTorch/CUDA version conflicts from full model loading.
    """
    import tarfile
    import unicodedata

    logger.info("Tokenizer 검증 중...")

    # Extract tokenizer.model from .nemo tar (avoid full model load)
    tokenizer_model_path = CHECKPOINT_DIR / '_extracted_tokenizer.model'
    with tarfile.open(hf_ckpt, 'r') as tar:
        tok_files = [n for n in tar.getnames() if n.endswith('_tokenizer.model')]
        if not tok_files:
            logger.error("Tokenizer model not found in .nemo archive")
            return False
        tok_member = tar.getmember(tok_files[0])
        tok_member.name = '_extracted_tokenizer.model'
        tar.extract(tok_member, str(CHECKPOINT_DIR))

    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(tokenizer_model_path))

    test_texts = [
        "안녕하세요, 음성 인식 모델입니다.",
        "한국어 데이터로 파인튜닝을 수행합니다.",
        "네모 트로나이트 모델을 사용합니다.",
        "인공지능과 반도체는 대한민국의 핵심 산업입니다.",
        "오늘 날씨가 매우 좋네요.",
    ]

    total_orig, total_decoded = 0, 0
    for text in test_texts:
        ids = tokenizer.EncodeAsIds(text)
        decoded = tokenizer.DecodeIds(ids)
        total_orig += len(text.replace(' ', ''))
        total_decoded += len(decoded.replace(' ', ''))
        logger.info(f"  {text[:40]}... → {decoded[:40]}...")

    coverage = total_decoded / max(total_orig, 1)
    logger.info(f"Coverage: {coverage:.1%}")

    # Jamo decomposition check
    sample = "안녕하세요"
    decoded = tokenizer.DecodeIds(tokenizer.EncodeAsIds(sample))
    nfd = unicodedata.normalize('NFD', decoded)
    has_jamo = any(0x1100 <= ord(c) <= 0x11FF or 0x3130 <= ord(c) <= 0x318F for c in nfd)

    if has_jamo:
        logger.warning("⚠️  음절 분해 감지. Tokenizer adaptation 검토 필요.")
    else:
        logger.info("음절 분해: 없음 (정상)")

    if coverage >= 0.98:
        logger.info(f"✅ Tokenizer 양호: {coverage:.1%} ≥ 98%")
        ok = True
    elif coverage >= 0.80:
        logger.warning(f"⚠️  Coverage 부족: {coverage:.1%} < 98%")
        ok = False
    else:
        logger.error(f"❌ Coverage 심각: {coverage:.1%} < 80%")
        ok = False

    # Cleanup extracted tokenizer
    os.remove(tokenizer_model_path)
    return ok


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  STEP 6: TRAINING                                                  ║
# ╚═════════════════════════════════════════════════════════════════════╝

def train(hf_ckpt: str, mixed_manifest: str, val_manifest: str) -> int:
    """Run NeMo fine-tuning. Returns process exit code."""
    logger.info("=" * 60)
    logger.info("  Fine-Tuning 시작")
    logger.info(f"  Mixed manifest: {mixed_manifest}")
    logger.info(f"  Val manifest:   {val_manifest}")
    logger.info(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    logger.info(f"  Epochs: {MAX_EPOCHS}, Batch duration: {BATCH_DURATION}")
    logger.info("=" * 60)

    finetune_script = str(NEMO_DIR / 'examples' / 'asr' / 'speech_to_text_finetune.py')

    cmd = [
        sys.executable, finetune_script,
        '--config-path=../asr/conf/fastconformer/cache_aware_streaming',
        '--config-name=fastconformer_transducer_bpe_streaming_prompt.yaml',
        f'+init_from_nemo_model={shlex.quote(hf_ckpt)}',
        f'++model.train_ds.manifest_filepath={shlex.quote(mixed_manifest)}',
        f'++model.validation_ds.manifest_filepath={shlex.quote(val_manifest)}',
        f'++model.tokenizer.dir={shlex.quote(str(CHECKPOINT_DIR))}',
        '++trainer.devices=1',
        f'++trainer.max_epochs={MAX_EPOCHS}',
        '++trainer.precision=bf16',
        '++trainer.gradient_clip_val=1.0',
        '++seed_everything=42',
        f'++model.train_ds.batch_duration={BATCH_DURATION}',
        '++model.validation_ds.batch_size=1',        # prevent val OOM
        '++model.train_ds.max_duration=20',           # prevent OOM on long samples
        '++model.optim.name=adamw',
        '++model.optim.lr=0.0001',
        '++model.optim.weight_decay=0.001',
        '++model.optim.sched.name=CosineAnnealing',
        '++model.optim.sched.warmup_ratio=0.05',
        '++model.optim.sched.warmup_steps=null',      # use ratio, not steps
        '++model.optim.sched.d_model=1024',           # explicit: OmegaConf can't resolve
        '++model.optim.sched.min_lr=1e-6',
        f'++exp_manager.exp_dir={shlex.quote(str(CHECKPOINT_DIR))}',
        '++exp_manager.resume_if_exists=true',
        '++exp_manager.resume_ignore_no_checkpoint=true',
        '++exp_manager.create_early_stopping_callback=true',
        '++exp_manager.early_stopping_callback_params.monitor=val_wer',
        '++exp_manager.early_stopping_callback_params.patience=5',
        '++exp_manager.checkpoint_callback_params.save_top_k=3',
        '++exp_manager.checkpoint_callback_params.monitor=val_wer',
    ]

    logger.info(f"Command: {' '.join(cmd[:3])} ... (truncated)")
    # Pass pip-installed CUDA libraries (cuDNN 9, nvJitLink) before system paths
    env = os.environ.copy()
    py_ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
    import platform as _platform
    _arch = "aarch64-linux" if _platform.machine() == "aarch64" else "x86_64-linux"
    pip_cuda_libs = [
        f'/usr/local/lib/{py_ver}/dist-packages/nvidia/cudnn/lib',
        f'/usr/local/lib/{py_ver}/dist-packages/nvidia/nvjitlink/lib',
        '/usr/local/cuda/lib64',
        f'/usr/local/cuda-{os.environ.get("CUDA_VERSION", "12.4")}/targets/{_arch}/lib',
    ]
    ld_paths = [p for p in pip_cuda_libs if os.path.isdir(p)]
    env['LD_LIBRARY_PATH'] = ':'.join(ld_paths + [env.get('LD_LIBRARY_PATH', '')])
    result = subprocess.run(cmd, cwd=str(NEMO_DIR / 'examples' / 'asr'), env=env)
    logger.info(f"Training 완료 (exit code: {result.returncode})")
    return result.returncode


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  STEP 7: HF HUB BACKUP                                             ║
# ╚═════════════════════════════════════════════════════════════════════╝

def backup_checkpoint() -> Optional[str]:
    """Upload latest checkpoint to HF Hub."""
    from runpod_auto import upload_checkpoint_to_hub

    nemo_files = sorted(CHECKPOINT_DIR.rglob('*.nemo'),
                        key=lambda p: p.stat().st_mtime)
    if not nemo_files:
        logger.warning("백업할 체크포인트 없음")
        return None

    latest = str(nemo_files[-1])
    size_mb = os.path.getsize(latest) / 1024**2
    logger.info(f"체크포인트 백업: {latest} ({size_mb:.0f} MB) → {HF_REPO_ID}")

    url = upload_checkpoint_to_hub(
        checkpoint_path=latest,
        repo_id=HF_REPO_ID,
        token=os.environ.get('HF_TOKEN'),
    )
    if url:
        logger.info(f"✅ 백업 완료: {url}")
    else:
        logger.warning("⚠️  백업 실패 — HF_TOKEN 확인")
    return url


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  STEP 8: EVALUATION                                                ║
# ╚═════════════════════════════════════════════════════════════════════╝

def eval_sweep(manifests: Dict[str, str]) -> str:
    """Run checkpoint sweep evaluation across all eval datasets using direct eval.

    Uses eval_direct.py which monkey-patches the prompt dataloader to handle
    Cut language=None bug, bypassing NeMo Hydra inference entirely.
    """
    from eval_direct import sweep_checkpoints_direct

    # Build eval dataset dict (exclude training manifests)
    eval_datasets = {}
    for key, path in manifests.items():
        if path and os.path.exists(path) and 'train' not in key.lower():
            eval_datasets[key] = path

    # Filter: keep only our 6 eval datasets
    desired = {'val_emilia_holdout_ko', 'test_emilia_holdout_ko', 'test_fleurs_ko',
               'test_zeroth_ko', 'mixed_en', 'mixed_ja', 'mixed_zh'}
    eval_datasets = {k: v for k, v in eval_datasets.items()
                     if any(d in k for d in desired) or k in desired}
    # Rename mixed_* to test_mixed_*
    renamed = {}
    for k, v in eval_datasets.items():
        if k.startswith('mixed_'):
            renamed[f'test_{k}'] = v
        else:
            renamed[k] = v
    eval_datasets = renamed

    if not eval_datasets:
        logger.warning("Eval datasets 없음. Sweep 건너뜁니다.")
        return ""

    logger.info(f"Checkpoint sweep 시작: {len(eval_datasets)} datasets")
    for k, v in eval_datasets.items():
        logger.info(f"  {k}: {v}")

    output_csv = str(RESULTS_DIR / 'checkpoint_eval.csv')
    return sweep_checkpoints_direct(
        checkpoint_dir=str(CHECKPOINT_DIR),
        datasets=eval_datasets,
        output_csv=output_csv,
    )


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  STEP 9: COST REPORT                                               ║
# ╚═════════════════════════════════════════════════════════════════════╝

def cost_report(on_runpod: bool):
    """Print cost summary and optionally auto-shutdown."""
    from runpod_auto import pod_status, stop_pod, is_runpod

    print("\n" + "=" * 60)
    print("  Cost Report")
    print("=" * 60)

    if is_runpod():
        pod_id = os.environ.get('RUNPOD_POD_ID', '')
        try:
            status = pod_status(pod_id)
            print(f"  Pod:         {status['name']} ({status['id']})")
            print(f"  Status:      {status['status']}")
            print(f"  Uptime:      {status['uptime_hours']:.1f}h")
            print(f"  GPU:         {status['gpu_type']} ({status['gpu_utilization_pct']}%)")
            print(f"  Cost:        ${status['estimated_cost']:.4f} (@ ${status['cost_per_hour']}/h)")
        except Exception as e:
            logger.warning(f"RunPod API 조회 실패: {e}")

    # Auto-shutdown
    auto_shutdown = os.environ.get('AUTO_SHUTDOWN', 'false').lower() == 'true'
    if auto_shutdown and is_runpod():
        pod_id = os.environ.get('RUNPOD_POD_ID', '')
        logger.warning("AUTO_SHUTDOWN 활성화 — 30초 후 Pod 종료")
        for i in range(30, 0, -5):
            print(f"  종료까지 {i}초...")
            time.sleep(5)
        try:
            stop_pod(pod_id)
            print("  ✅ Pod 종료 완료")
        except Exception as e:
            logger.error(f"Pod 종료 실패: {e}")
    elif is_runpod():
        print(f"\n  자동 종료: 비활성화 (export AUTO_SHUTDOWN=true)")
        print(f"  수동 종료: python scripts/runpod_auto.py stop")
    print("=" * 60)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                              ║
# ╚═════════════════════════════════════════════════════════════════════╝

def main():
    start_time = time.time()

    print("=" * 60)
    print("  Korean ASR Fine-Tuning Pipeline")
    print(f"  Model: nemotron-3.5-asr-streaming-0.6b")
    print(f"  Dataset: Emilia-YODAS Korean (7,300h, CC BY 4.0)")
    print(f"  Target: CER ≤ 7.12 on FLEURS Korean")
    print("=" * 60)

    try:
        # Step 1: Setup
        logger.info("── Step 1/9: Setup ──")
        cfg = setup()

        # Step 2: Data Ingest
        logger.info("── Step 2/9: Data Ingest ──")
        temp_path, n_entries = data_ingest()
        if n_entries == 0:
            raise RuntimeError("No data ingested. Check dataset access.")

        # Step 3: Build Manifests
        logger.info("── Step 3/9: Build Manifests ──")
        manifests = build_manifests(temp_path)

        # Step 4: Language Mix
        logger.info("── Step 4/9: Language Mix ──")
        mixed_path = language_mix(manifests)

        # Step 5: Tokenizer Verification
        logger.info("── Step 5/9: Tokenizer Verification ──")
        tokenizer_ok = verify_tokenizer(cfg['hf_ckpt'])
        if not tokenizer_ok:
            logger.warning("Tokenizer coverage 낮음. Training 계속 진행합니다.")

        # Step 6: Training
        logger.info("── Step 6/9: Fine-Tuning ──")
        ret = train(cfg['hf_ckpt'], mixed_path,
                    manifests.get('val_emilia_holdout_ko',
                                  str(PROCESSED_DIR / 'val_emilia_holdout_ko.json')))

        # Step 7: Backup
        logger.info("── Step 7/9: HF Hub Backup ──")
        backup_checkpoint()

        # Step 8: Evaluation
        logger.info("── Step 8/9: Checkpoint Sweep Evaluation ──")
        eval_csv = eval_sweep(manifests)
        if eval_csv:
            logger.info(f"Results: {eval_csv}")

        # Step 9: Cost Report
        logger.info("── Step 9/9: Cost Report ──")
        cost_report(cfg['on_runpod'])

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed_h = (time.time() - start_time) / 3600
    logger.info(f"Pipeline 완료. 총 소요 시간: {elapsed_h:.1f}h")


if __name__ == "__main__":
    main()
