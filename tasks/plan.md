# 구현 계획 (v3 · Emilia-YODAS Korean 반영): Nemotron 3.5 ASR 한국어 파인튜닝

> **목표**: 실사용 가능한 한국어 ASR 모델 산출.
> **v2 → v3 변경**: 데이터셋을 KsponSpeech → **Emilia-YODAS Korean**(7,300h, CC BY 4.0, HuggingFace 직접 접근)으로 교체. EUC-KR·헤더리스 PCM·이중표기 함정 제거, 대신 WebDataset tar 파싱·자동전사 품질 필터링 추가.

## 검증된 핵심 사실

### 모델 (변경 없음)
- **`nvidia/nemotron-3.5-asr-streaming-0.6b`** (600M, OpenMDW-1.1), **NeMo 26.06** 필요. ([모델 카드](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b))
- 한국어 ko-KR 기지원, CER 7.12%. **토크나이저 재사용**. `target_lang="ko-KR"` 필수.
- 파인튜닝 레시피: `speech_to_text_finetune.py` + `fastconformer_transducer_bpe_streaming_prompt.yaml`. ([파인튜닝 블로그](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr))
- **`limit_train_batches` 제거 필수** (전체 데이터 학습 시). ([Issue #15782](https://github.com/NVIDIA-NeMo/NeMo/issues/15782))

### Emilia-YODAS Korean 데이터셋
- **규모**: 한국어 7,300h (Emilia-YODAS) + 200h (Emilia Korean) = ~7,500h. ([데이터셋](https://huggingface.co/datasets/amphion/Emilia-Dataset))
- **라이선스**: Emilia-YODAS는 **CC BY 4.0** (상업 이용 가능). Emilia(비YODAS)는 CC-BY-NC 4.0 (비상업).
- **오디오 포맷**: MP3, WebDataset tar 파일로 패킹. 각 tar 내: `000000.mp3` + `000000.json`.
- **메타데이터 JSON 필드**: `id`, `wav`, `text`, `duration`, `speaker`, `language`, `dnsmos`.
- **품질 필터**: DNSMOS > 2.4가 이미 적용됨 (원본 YODAS 422k시간 → 114k시간 필터링).
- **자동 전사**: Whisper 계열로 추정. 수동 전사 대비 오류 가능성 있음 — **추가 필터링 권장**.
- **다운로드**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("amphion/Emilia-Dataset",
                    data_files={"train": "Emilia-YODAS/KO/**/*.tar"},
                    split="train", streaming=True)
  ```

## 아키텍처/접근 결정

1. **2-트랙 구조** (유지):
   - **트랙 A (노트북)**: 한글화 + 데이터 준비/정규화/매니페스트 생성 + 스모크 학습 + 평가.
   - **트랙 B (스크립트/가이드)**: `train_emilia_ko.sh` + `RUN.md`. 7,500h 전체 학습은 GPU 인프라 필요.
2. **토크나이저 재사용** (확정).
3. **전사 정규화 (v3 단순화)**: 특수기호 복잡 처리 불필요. 단, 자동 전사 오류 필터로 `dnsmos > 3.0` 추가 적용 권장.
4. **MP3 → WAV 변환**: `soundfile`+`librosa` 또는 `ffmpeg`. 44100Hz → 16kHz 리샘플링 필요.
5. **처리 전략**: 7,500h 전체를 한 번에 WAV로 변환하면 수백 GB. **streaming 처리**(HF datasets streaming=True로 읽으면서 매니페스트 생성 + 선택적 WAV 저장) 또는 **배치 단위 처리**로 관리.
6. **단계적 확장** (확정): 1차 = Emilia-YODAS KO 일부 tar(~200h 상당) 스모크 → CER 확인 → 전체 확장.

## 의존성 그래프

```
[Phase 0] 환경·체크포인트 확보 (NeMo 26.06, .nemo 다운로드)
     │
[Task 1] 노트북 전체 한글화 ──────── [Task 9] 리소스/다음단계 섹션
     │
[Task 2] 의존성 셀 정정 (NeMo 26.06 핀, datasets/soundfile/tqdm)
     │
[Task 3] 데이터 경로/HF 모델 다운로드 셀
     │
[Task 4] Emilia-YODAS KO 전처리 (streaming 다운로드, MP3→WAV, dnsmos 필터, 매니페스트)
     │
     ├── [Task 5] 오디오 샘플 셀
     ├── [Task 6] 스모크 학습 셀 (일부 tar, limit 경고)
     │         └── [Task 8] 평가 셀 (한국어 CER)
     └── [Task 7] 트랙 B: 본학습 스크립트 + RUN.md
```

---

## Phase 0: 환경 및 체크포인트 확보

### Task 0: NeMo 26.06 환경 + 베이스 .nemo 다운로드
**설명:** NeMo 26.06 컨테이너 준비, `huggingface-hub`로 `nvidia/nemotron-3.5-asr-streaming-0.6b` `.nemo` 다운로드. OpenMDW-1.1 라이선스 동의.
**수락 기준:**
- [ ] `import nemo.collections.asr` 성공
- [ ] `.nemo` 파일 존재, `HF_CKPT` 경로 확정
**규모:** S (노트북 외부)

---

## Phase 1: 노트북 한글화

### Task 1: 마크다운 + 코드 주석 한글 번역
**설명:** cell-0~6, 8, 10~11, 13~17, 19~20, 22~24 마크다운과 cell-3, 18 주석을 한국어로. 모델명 `nemotron-3.5-asr-streaming-0.6b`, 데이터셋 Emilia-YODAS Korean으로 반영.
**수락 기준:**
- [ ] 모든 마크다운 셀 한국어, 기술 용어 원어 병기
- [ ] 모델명 정정됨
- [ ] 데이터셋 설명이 Emilia-YODAS로 업데이트됨
**검증:** `python -m json.tool *.ipynb > /dev/null`
**규모:** M

### Task 2: 의존성 셀 정정
**설명:** cell-3에서 NeMo를 **26.06으로 핀**, `datasets`, `soundfile`, `tqdm`, `librosa` 추가. 한국어 주석으로 용도 설명.
**변경:**
```python
# NeMo 26.06 고정
!pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@r26.06"
# 데이터 처리
!pip install datasets soundfile librosa tqdm huggingface_hub
```
**수락 기준:**
- [ ] NeMo 26.06 버전 핀 적용
- [ ] `datasets`, `soundfile`, `librosa`, `tqdm`, `huggingface_hub` 추가
**규모:** XS

### Checkpoint 1
- [ ] JSON 유효성 / 마크다운 한국어 / 모델명·데이터셋명 정정

---

## Phase 2: Emilia-YODAS 데이터 파이프라인

### Task 3: 데이터 경로 + HF 모델 다운로드 셀 교체
**설명:** cell-7, 9를 (a) `DATA_DIR` 설정 + (b) HF `.nemo` 다운로드로 교체. Emilia-YODAS는 Task 4에서 streaming으로 처리하므로 별도 수동 다운로드 불필요.
```python
import os
from huggingface_hub import snapshot_download

DATA_DIR = '/data/emilia_ko'
os.makedirs(DATA_DIR, exist_ok=True)
os.environ['DATA_DIR'] = DATA_DIR

# 베이스 모델 체크포인트 다운로드 (라이선스 동의 필요)
HF_CKPT = snapshot_download(
    repo_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
    local_dir=f"{DATA_DIR}/nemotron_ckpt"
)
# .nemo 파일 경로 확정
import glob
nemo_files = glob.glob(f"{HF_CKPT}/*.nemo")
assert nemo_files, "베이스 .nemo 체크포인트를 찾을 수 없습니다."
HF_CKPT_PATH = nemo_files[0]
print(f"베이스 체크포인트: {HF_CKPT_PATH}")
```
**수락 기준:**
- [ ] `DATA_DIR` 설정, 디렉토리 생성
- [ ] `.nemo` 파일 존재 assert 통과
- [ ] `HF_CKPT_PATH` 변수 확정
**규모:** S

### Task 4: Emilia-YODAS Korean 전처리 — streaming + MP3→WAV + dnsmos 필터 + 매니페스트
**설명:** cell-12 전면 교체. Emilia-YODAS KO를 HF datasets streaming으로 읽으면서 MP3→WAV 변환, dnsmos 필터링, NeMo 매니페스트 생성.

**파이프라인:**
1. HF datasets streaming으로 Emilia-YODAS KO 파티션 로드
2. `dnsmos > 3.0` 추가 필터 적용 (기본 2.4 + 추가 품질 확보)
3. MP3 bytes → `librosa.load(sr=16000)` → WAV 저장 (16kHz mono)
4. 전사 텍스트는 그대로 사용 (UTF-8, 별도 정규화 불필요)
5. 매니페스트: `{"audio_filepath", "duration", "text", "target_lang": "ko-KR"}`
6. 스모크용 서브셋(N=5000 샘플)과 전체 매니페스트 분리 생성

```python
from datasets import load_dataset
import soundfile as sf
import librosa
import json, os
from tqdm import tqdm

WAV_DIR = f"{DATA_DIR}/wavs"
os.makedirs(WAV_DIR, exist_ok=True)

# 스모크용: 처음 N개 샘플만
SMOKE_N = 5000
DNSMOS_THRESHOLD = 3.0

ds = load_dataset(
    "amphion/Emilia-Dataset",
    data_files={"train": "Emilia-YODAS/KO/**/*.tar"},
    split="train",
    streaming=True
)

smoke_manifest, full_manifest = [], []

for i, sample in enumerate(tqdm(ds)):
    if sample.get('dnsmos', 0) < DNSMOS_THRESHOLD:
        continue
    
    audio_array = sample['audio']['array']  # 이미 디코딩된 경우
    sr = sample['audio']['sampling_rate']
    
    # 16kHz 리샘플링
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    
    wav_path = f"{WAV_DIR}/{sample['id']}.wav"
    sf.write(wav_path, audio_array, 16000)
    duration = len(audio_array) / 16000
    
    entry = {
        "audio_filepath": wav_path,
        "duration": round(duration, 3),
        "text": sample['text'].strip(),
        "target_lang": "ko-KR"
    }
    
    if len(smoke_manifest) < SMOKE_N:
        smoke_manifest.append(entry)
    full_manifest.append(entry)

# 매니페스트 저장
with open(f"{DATA_DIR}/smoke_manifest.json", 'w', encoding='utf-8') as f:
    for e in smoke_manifest:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')

with open(f"{DATA_DIR}/train_manifest.json", 'w', encoding='utf-8') as f:
    for e in full_manifest:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')

print(f"스모크: {len(smoke_manifest)}개 / 전체: {len(full_manifest)}개")
```

> **주의**: 7,500h 전체 처리는 수 시간 소요, 수백 GB 디스크 필요. 스모크 학습엔 `smoke_manifest.json` 사용.

**수락 기준:**
- [ ] AN4·KsponSpeech 코드 전부 제거
- [ ] streaming 로드 + dnsmos 필터 적용
- [ ] MP3 → 16kHz WAV 변환 (librosa 리샘플링)
- [ ] `smoke_manifest.json` (N=5000) + `train_manifest.json` 생성
- [ ] 모든 항목 `target_lang: "ko-KR"`, `text` 비어있지 않음
- [ ] eval용 매니페스트 (`eval_manifest.json`) 별도 생성 (Emilia KO 또는 스모크 hold-out)

**검증:**
- [ ] `head -1 $DATA_DIR/train_manifest.json | python -m json.tool`
- [ ] `grep -c ko-KR train_manifest.json` > 0
- [ ] 빈 텍스트 없음: `python -c "import json; [print(r) for r in open('train_manifest.json') if not json.loads(r)['text'].strip()]"`
- [ ] WAV 파일 1개 재생 가능

**의존성:** Task 3
**규모:** L

### Task 5: 오디오 샘플 셀
**설명:** cell-14를 `smoke_manifest.json` 첫 항목 동적 참조 + `text`/`dnsmos`/`duration` 출력으로 교체.
**수락 기준:** 동적 경로, 메타데이터 출력, `ipd.Audio()` 렌더링
**의존성:** Task 4
**규모:** XS

### Checkpoint 2
- [ ] cell-3~14 순차 실행 오류 없음
- [ ] 매니페스트 생성·`ko-KR`·텍스트 정상
- [ ] AN4 잔존 없음: `grep -i "an4\|en-US" *.ipynb` → 없음

---

## Phase 3: 학습 및 평가

### Task 6: 노트북 스모크 학습 셀
**설명:** cell-18을 `smoke_manifest.json`(5,000샘플) 기반 스모크 학습으로 정정.
- 경로: `$DATA_DIR/smoke_manifest.json` (train), `$DATA_DIR/eval_manifest.json` (val)
- `init_from_nemo_model={HF_CKPT_PATH}`
- `limit_train_batches=100` (스모크용 유지)
- **본학습에서 `limit_train_batches` 제거 경고 주석** (Issue #15782)
- `trainer.max_epochs=3`, `trainer.precision=bf16`

**수락 기준:**
- [ ] manifest 경로 `$DATA_DIR/smoke_manifest.json`
- [ ] `limit_train_batches` 함정 경고 주석
- [ ] `HF_CKPT_PATH` assert
- [ ] 스모크·본학습 파라미터 구분 주석
**검증:** 명령 구문 오류 없음, 경로 일관성
**의존성:** Task 4
**규모:** S

### Task 7: 트랙 B — 본학습 스크립트 + RUN.md
**설명:** `train_emilia_ko.sh` + `RUN.md`. 단계적 확장 전략:
- **1단계**: Emilia-YODAS KO 일부 (~200h, `smoke_manifest` 확대판) → CER 검증
- **2단계**: 전체 7,500h → CER 재검증

핵심 설정:
- `limit_train_batches` 미사용
- `trainer.devices` 멀티GPU
- `exp_manager.resume_if_exists=true`
- `warmup_steps=2000`, `weight_decay=0.001`, `optimizer=adamw`, `precision=bf16`
- LR: 보수적 시작값 명시 + 손실 곡선 기준 조정 안내

**수락 기준:**
- [ ] 2단계 데이터 확장 절차 문서화 (매니페스트 병합 방법 포함)
- [ ] `limit_train_batches` 미사용, 멀티GPU·재개·로깅 포함
- [ ] 로컬·클라우드(vast.ai/runpod) 양쪽 안내
- [ ] 예상 처리 시간·디스크 요구사항 명시
**의존성:** Task 4
**규모:** M

### Task 8: 평가 셀 (한국어 CER)
**설명:** cell-21을 eval_manifest 경로로 교체. 한국어는 **CER** 측정임을 주석으로 명시 (베이스라인 7.12%).
**수락 기준:**
- [ ] eval_manifest 경로 `$DATA_DIR/eval_manifest.json`
- [ ] CER 보고 안내 주석 (베이스라인 7.12% 비교 기준)
- [ ] `nemo_file_path` 학습 출력과 일치
**의존성:** Task 6
**규모:** XS

### Task 9: 리소스/다음 단계 섹션
**설명:** cell-22~24 한국어화 + Emilia-YODAS·Nemotron 모델 카드·파인튜닝 블로그·Riva 배포 링크.
**의존성:** Task 1 (독립)
**규모:** XS

### Checkpoint 3 (최종)
- [ ] `python -m json.tool *.ipynb > /dev/null`
- [ ] `grep -i "an4\|en-US\|ksponspeech\|an268" *.ipynb` → 없음
- [ ] 매니페스트 `target_lang` 전부 `ko-KR`
- [ ] 트랙 A cell-3~14 + 스모크 학습 무오류
- [ ] 트랙 B RUN.md로 본학습 재현 가능

---

## 리스크 및 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| 자동 전사 오류로 학습 품질 저하 | 중간 | DNSMOS ≥ 3.0 추가 필터, 텍스트 길이 필터(10자 미만·500자 초과 제외) |
| 7,500h WAV 저장 시 수백 GB 디스크 필요 | 높음 | 스모크는 5,000샘플만, 본학습은 배치 처리 또는 MP3 직접 사용 검토 |
| HF datasets streaming 속도 느림 | 중간 | 노트북에서는 서브셋, 본학습은 사전 다운로드·캐시 활용 |
| `limit_train_batches` 로 epoch 데이터 제한 (#15782) | 높음 | 본학습 스크립트에서 제거, 노트북 경고 주석 |
| Emilia(비YODAS) CC-BY-NC 라이선스 혼용 | 중간 | 상업 용도면 YODAS 파티션만 사용, 라이선스 명시 |
| NeMo 26.06 ≠ main, config 부재 | 중간 | 26.06 태그 핀, config 경로 사전 확인 |
| MP3 → 16kHz 리샘플링 품질 | 낮음 | librosa 기본값(kaiser_best), 샘플 검청으로 확인 |

## 결정 사항 (확정)

1. ✅ **데이터셋**: Emilia-YODAS Korean (7,300h, CC BY 4.0, HF streaming)
2. ✅ **단계적 확장**: 스모크(5,000샘플) → 1단계(~200h) → 전체(7,500h)
3. ✅ **전처리**: MP3→16kHz WAV (librosa), dnsmos ≥ 3.0, UTF-8 그대로
4. ✅ **모델**: `nemotron-3.5-asr-streaming-0.6b`, NeMo 26.06, 토크나이저 재사용
5. ✅ **환경**: 트랙 A (노트북 스모크), 트랙 B (로컬·클라우드 본학습)

## 남은 미결

- **eval 데이터셋**: Emilia KO 중 hold-out으로 쓸지, 별도 한국어 평가셋(예: FLEURS Korean) 사용할지.
- **WAV 저장 vs MP3 직접**: NeMo가 MP3를 직접 읽을 수 있으면 변환 생략 가능 — NeMo 26.06 데이터로더 지원 여부 확인 필요.
