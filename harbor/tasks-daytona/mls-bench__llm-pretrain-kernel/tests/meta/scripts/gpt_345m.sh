#!/bin/bash
# GPT-2 Medium (24L/16H/1024D, ~355M total params) on ~7.1B tokens (D=20N Chinchilla).
# H100 DDP, BSZ=32 per GPU per backward, GA=16 (divided by world_size internally).
N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
# Pre-clean stale outputs from a previous test iteration: custom_pretrain.py
# writes these only at the successful end of training and the eval step only
# checks that they exist, so a leftover checkpoint from an earlier run would
# be silently scored if this training run fails.
rm -f "${OUTPUT_DIR:-out}/ckpt_${ENV:-model}.pt" "${OUTPUT_DIR:-out}/model_source_${ENV:-model}.py"
N_LAYER=24 N_HEAD=16 N_EMBD=1024 \
MAX_ITERS=${MAX_ITERS:-13535} EVAL_INTERVAL=${EVAL_INTERVAL:-1000} \
BATCH_SIZE=${BATCH_SIZE:-32} GRAD_ACCUM=${GRAD_ACCUM:-16} LEARNING_RATE=${LEARNING_RATE:-3e-4} \
torchrun --nproc_per_node=${N_GPU} --standalone custom_pretrain.py
rc=$?
# Fixed MLP-kernel throughput benchmark (mlp_speedup = t_reference / t_submission,
# unfused PyTorch GELU MLP vs the submitted fused_mlp_forward, same GPU).
if [ "${rc}" -eq 0 ]; then
    python3 "$(dirname "${BASH_SOURCE[0]}")/mlp_bench.py" --source custom_pretrain.py
fi
exit "${rc}"
