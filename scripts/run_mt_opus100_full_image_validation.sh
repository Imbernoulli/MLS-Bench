#!/usr/bin/env bash
set -euo pipefail

echo "MT_FULL_START timestamp=$(date -Is) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<'PY'
import gc

import torch
from transformers import AutoModelForSeq2SeqLM

print(
    f"MT_ENV torch={torch.__version__} cuda={torch.cuda.is_available()} "
    f"ngpu={torch.cuda.device_count()}",
    flush=True,
)
for model_name in ("opus-mt-de-en", "opus-mt-fr-en", "opus-mt-ru-en"):
    model_path = f"/data/machine-translation/models/{model_name}"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    print(
        f"MT_MODEL name={model_name} params={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    del model
    gc.collect()
PY

cd /workspace/machine-translation
for direction in de_en fr_en ru_en; do
    export MT_DIR="${direction}"
    echo "MT_SETTING_START direction=${direction}"
    python harness_beam.py --solution solution/beam.py --seed 42
    echo "MT_SETTING_DONE direction=${direction}"
done

echo "MT_FULL_DONE timestamp=$(date -Is)"
