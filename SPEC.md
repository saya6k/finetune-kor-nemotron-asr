# SPEC: Korean Fine-Tuning for Nemotron-3.5-ASR-Streaming-0.6B

## 1. Objective

**nemotron-3.5-asr-streaming-0.6b** 모델을 한국어 음성인식(ASR)에 최적화하는 fine-tuning 파이프라인을 구축한다. NVIDIA 공식 튜토리얼(`asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb`)과 HuggingFace Discussion #11의 fine-tuning 팁을 기반으로, 한국어 ASR에서 **모델 카드 기준 CER 7.12(FLEURS, 1.12s chunk, LangID)과 경쟁력 있는 CER을 달성**하는 것을 목표로 한다.

### Target Users
- 한국어 음성인식을 자체 fine-tuning하려는 ML 엔지니어
- RunPod GPU 환경에서 실행 가능한 재현 가능한 파이프라인이 필요

### Success Criteria
| Metric | Baseline (Model Card) | Target |
|--------|----------------------|--------|
| CER on FLEURS ko-KR (1.12s, LangID) | 7.12 | ≤ baseline (7.12 이하) |
| CER on 자체 평가셋 (1.12s, LangID) | N/A | ≤ baseline + 2pp |

## 2. Commands / Notebook Workflow

단일 Jupyter notebook(`asr-finetune-kor.ipynb`)으로 구성하며, 아래 셀 순서로 진행한다.

### Cell Map

| # | Section | Type | Purpose |
|---|---------|------|---------|
| 1 | Header | Markdown | 프로젝트명, NVIDIA 로고, 개요 |
| 2 | Environment Setup | Code | RunPod용 패키지 설치 (NeMo main 브랜치, 시스템 패키지) |
| 3 | Data Preparation | Markdown | 데이터 디렉토리 구조 설명 |
| 4 | Data Download/Ingest | Code | 한국어 코퍼스 다운로드 및 DATA_DIR 구성 |
| 5 | Pre-processing | Markdown | 매니페스트 생성 설명 |
| 6 | Build Manifest | Code | 한국어 오디오 → NeMo manifest JSON 생성 (`lang: ko-KR`, `target_lang: ko-KR`) |
| 7 | Language Mix | Code | **다른 언어 20~30%** 포함된 학습 매니페스트 병합 (catastrophic forgetting 방지) |
| 8 | TTS Augmentation (Optional) | Code | 희귀 도메인 용어에 대한 TTS 생성 및 매니페스트 추가 |
| 9 | Tokenizer | Markdown | 토크나이저 전략 설명 |
| 10 | Tokenizer Reuse | Code | Pretrained 모델에서 토크나이저 로드 (< 50 hrs 데이터 기준) |
| 11 | Training Config | Markdown | 하이퍼파라미터 설명 |
| 12 | Fine-Tuning | Code | `speech_to_text_finetune.py` 실행 (→ 아래 Training Command 참조) |
| 13 | Evaluation | Markdown | 평가 방법 설명 |
| 14 | Evaluation Run | Code | `speech_to_text_cache_aware_streaming_infer.py` 실행 및 CER 계산 |
| 15 | Results Summary | Code | CER 결과를 모델 카드 기준값과 비교 출력 |

### Core Training Command

```bash
python ${NEMO_DIR}/examples/asr/speech_to_text_finetune.py \
  --config-path="../asr/conf/fastconformer/cache_aware_streaming" \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_nemo_model=${HF_CKPT} \
  ++model.train_ds.manifest_filepath="${DATA_DIR}/train_manifest_ko.json" \
  ++model.validation_ds.manifest_filepath="${DATA_DIR}/test_manifest_ko.json" \
  ++model.optim.sched.d_model=1024 \
  ++trainer.devices=1 \
  ++trainer.max_epochs=200 \
  ++trainer.precision=bf16 \
  ++model.train_ds.batch_duration=200 \
  ++model.optim.name="adamw" \
  ++model.optim.lr=0.02 \
  ++model.optim.weight_decay=0.001 \
  ++model.optim.sched.warmup_steps=100 \
  ++exp_manager.exp_dir=${DATA_DIR}/checkpoints
```

**Key changes from demo notebook:**
- `max_epochs`: 1 → 200 (full training)
- `optim.lr`: 0.1 → 0.02 (smaller LR per NVIDIA guidance)
- `limit_train_batches` 제거 (전체 학습)
- `exp_manager.version`, `use_datetime_version` 제거 (기본값 사용)
- `batch_duration`은 OOM 발생 시 동적으로 감소

### Core Evaluation Command

```bash
python ${NEMO_DIR}/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
  model_path=${CKPT_PATH} \
  dataset_manifest="${DATA_DIR}/test_manifest_ko.json" \
  target_lang=ko-KR \
  att_context_size="[56,3]" \
  decoder_type=rnnt \
  pad_and_drop_preencoded=true \
  batch_size=8 \
  cuda=0 \
  strip_lang_tags=false
```

**Key change:** `target_lang=ko-KR` 명시 (auto 대신), CER metric 사용 (한국어는 WER 대신 CER).

## 3. Project Structure

```
finetune-kor-nemotron-asr/
├── SPEC.md                          # 이 문서
├── CLAUDE.md                        # 프로젝트 가이드 (기존)
├── asr-finetune-kor.ipynb           # 메인 fine-tuning 노트북 (신규)
├── README.md                        # 실행 방법, 선행 조건, 결과 요약
├── data/                            # 데이터 디렉토리 (RunPod에서 마운트)
│   ├── raw/                         # 원본 한국어 오디오 파일
│   └── processed/                   # 변환된 WAV + manifest JSON
├── checkpoints/                     # 학습된 .nemo 체크포인트 출력
├── scripts/                         # 보조 스크립트
│   ├── build_manifest.py            # 매니페스트 생성 (재사용 가능)
│   ├── compute_cer.py               # CER 계산 유틸리티
│   └── tts_augment.py               # TTS 데이터 증강 (선택)
└── configs/                         # NeMo 설정 오버라이드
    └── override.yaml                # Korean-specific 하이퍼파라미터 오버라이드
```

### Key Variables (Environment)

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | 데이터 및 체크포인트 루트 | `./data` |
| `NEMO_DIR` | NeMo 클론 경로 | `./NeMo` |
| `HF_CKPT` | HuggingFace pretrained .nemo 모델 경로 | **필수 설정 필요** |
| `BATCH_DURATION` | 배치 크기 (OOM 시 동적 감소) | `200` |
| `MAX_EPOCHS` | 학습 에포크 수 | `200` |

## 4. Code Style

### Notebook Conventions
- 셀 순서는 위 Cell Map을 따른다 (선형 실행 가능해야 함).
- 각 섹션 앞에 Markdown 셀로 목적과 예상 출력을 설명한다.
- 모든 경로는 `os.path.join` 또는 `pathlib.Path` 사용, 하드코딩 금지.
- 오류 발생 시 의미 있는 메시지를 출력하고 중단한다 (silent failure 금지).
- GPU 메모리 체크 셀을 포함하여 RunPod 환경 상태를 확인할 수 있게 한다 (`nvidia-smi`).

### Python Guidelines (from CLAUDE.md)
- 복잡하지 않게: 요청된 기능만 구현, 추상화는 실제 중복이 발생할 때만 도입.
- 기존 노트북의 구조와 셀 순서를 최대한 유지 (NVIDIA 공식 튜토리얼과의 일관성).
- 한국어 주석은 필요한 경우에만, 변수명과 함수명은 영어 사용.

## 5. Testing & Evaluation Strategy

### 평가 파이프라인
1. **내부 평가**: 학습 완료 후 `speech_to_text_cache_aware_streaming_infer.py`로 test manifest에 대해 추론 실행
2. **CER 계산**: `scripts/compute_cer.py`로 한국어 Character Error Rate 산출
3. **기준 비교**: 모델 카드의 `7.12` (FLEURS ko-KR, 1.12s chunk, LangID)와 비교

### 평가 매트릭
| Metric | Description | Target |
|--------|-------------|--------|
| CER (Character Error Rate) | 한국어 음소/문자 단위 오류율 | ≤ 7.12 |
| IL (Insertions + Deletions + Substitutions) | CER 구성 요소 breakdown | - |
| RTF (Real-Time Factor) | 추론 속도 | ≤ 0.1 (streaming) |

### 검증 체크리스트
- [ ] 학습 후 `.nemo` 체크포인트 파일이 생성되는가
- [ ] 학습 loss가 감소 추세를 보이는가 (NaN 없음)
- [ ] 평가 manifest의 모든 오디오 파일이 존재하는가
- [ ] CER이 모델 카드 기준과 비교 가능한 수치인가
- [ ] OOM 없이 전체 학습이 완료되는가
- [ ] `target_lang=ko-KR`로 LangID 모드 추론 결과가 auto-detect보다 좋은가

### 언어 혼합 검증
- 학습 manifest에서 한국어 비율이 70~80%인지 확인하는 assertion 포함
- 다른 언어 manifest의 `lang` 필드가 올바르게 설정되었는지 확인

## 6. Boundaries

### Always Do
- RunPod GPU 환경(Pod)에서 실행 가능하도록 모든 경로를 환경 변수로 제어
- NeMo `main` 브랜치 사용 (최신 기능 + 버그 수정)
- `bf16` precision 사용 (메모리 효율 + 속도)
- 각 오디오 클립에 `lang` 및 `target_lang` 태그 명시
- Pretrained 토크나이저 재사용 (데이터 < 50 hrs 기준)

### Ask First
- 학습 데이터셋 선택 (AI Hub, 자체 데이터, FLEURS 등)
- TTS 증강 적용 여부 및 대상 도메인 (의료, 법률, 콜센터 등)
- 언어 혼합 비율 조정 (20~30% 범위 내에서)
- 학습률, 배치 크기 등 주요 하이퍼파라미터 변경
- `.nemo` 모델을 HuggingFace에 업로드할지 여부

### Never Do
- **Word boosting / N-gram LM fusion은 이번 스코프에서 제외** (fine-tuning에 집중)
- 토크나이저를 처음부터 학습하지 않음 (데이터 부족)
- 모델 배포/서빙(Riva NIM, Triton)은 다루지 않음
- 멀티 GPU 학습은 다루지 않음 (단일 GPU 기준)
- 원본 데이터를 직접 수정하지 않음 (복사본으로 작업)

### RunPod Constraints
- Pod 시작 시 NeMo 및 의존성 설치 필요 (사전 빌드된 이미지 없음)
- 데이터는 RunPod 볼륨 또는 `/workspace`에 저장
- 체크포인트는 주기적으로 HuggingFace Hub 또는 외부 스토리지에 백업
- Spot instance 중단을 대비한 체크포인트 재개(resume) 로직 포함

## References
- [NVIDIA Riva Fine-Tuning Tutorial (Official)](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb)
- [HuggingFace Discussion #11 — Fine-Tuning Tips](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/discussions/11)
- [Model Card — Evaluation Datasets](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b#evaluation-datasets)
- [NeMo GitHub](https://github.com/NVIDIA/NeMo)
