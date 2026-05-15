#!/bin/bash
# Evaluation script for cv-diffusion-efficiency — SDXL

WORKDIR_BASE="${OUTPUT_DIR:-examples/workdir}"
METHOD=${METHOD:-"ddim_cfg++"}
CFG_GUIDANCE=${CFG_GUIDANCE:-0.6}
SEED=${SEED:-42}
if [ -z "${NGPU:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]; then
        IFS=',' read -ra _MLSBENCH_GPUS <<< "$CUDA_VISIBLE_DEVICES"
        NGPU=0
        for _gpu in "${_MLSBENCH_GPUS[@]}"; do
            if [ -n "$_gpu" ]; then
                NGPU=$((NGPU + 1))
            fi
        done
    elif command -v python >/dev/null 2>&1; then
        NGPU=$(python - <<'PY_NGPU'
import torch
print(torch.cuda.device_count())
PY_NGPU
)
    else
        NGPU=1
    fi
fi
NGPU=${NGPU:-1}
if [ "$NGPU" -lt 1 ]; then
    NGPU=1
fi
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

torchrun --nproc_per_node=$NGPU --master_port=$MASTER_PORT batch_eval.py \
    --model sdxl \
    --method "$METHOD" \
    --cfg_guidance "$CFG_GUIDANCE" \
    --NFE 20 \
    --seed "$SEED" \
    --workdir "$WORKDIR_BASE/eval_sdxl_${SEED}"
