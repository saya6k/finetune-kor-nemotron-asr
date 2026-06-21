#!/usr/bin/env python3
"""
Wrapper around speech_to_text_finetune.py that freezes the lower N encoder
layers before the optimizer is created, reducing catastrophic forgetting.

All Hydra/NeMo args pass through unchanged. Replace the finetune script path
with this file in the docker run command.

Env:
  FREEZE_ENCODER_LAYERS  Number of lower encoder layers to freeze (default: 8)
"""
import os
import sys
import runpy

FREEZE_N = int(os.environ.get("FREEZE_ENCODER_LAYERS", "8"))

from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt

_orig_setup = EncDecRNNTBPEModelWithPrompt.setup_optimizer_param_groups

def _setup_with_freeze(self):
    _orig_setup(self)
    n_frozen = 0
    for layer in self.encoder.layers[:FREEZE_N]:
        for p in layer.parameters():
            p.requires_grad_(False)
            n_frozen += 1
    trainable = sum(p.requires_grad for p in self.parameters())
    total = sum(1 for _ in self.parameters())
    print(
        f"[freeze] encoder.layers[0:{FREEZE_N}] frozen ({n_frozen} tensors) | "
        f"trainable: {trainable:,}/{total:,} param tensors",
        flush=True,
    )

EncDecRNNTBPEModelWithPrompt.setup_optimizer_param_groups = _setup_with_freeze
print(f"[freeze] Patched: will freeze lower {FREEZE_N} of 24 encoder layers", flush=True)

nemo_dir = os.environ.get("NEMO_DIR", "/opt/NeMo")
script = os.path.join(nemo_dir, "examples", "asr", "speech_to_text_finetune.py")
runpy.run_path(script, run_name="__main__")
