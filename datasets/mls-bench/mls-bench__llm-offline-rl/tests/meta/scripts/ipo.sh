#!/bin/bash
# IPO baseline (Azar et al. 2023): squared-loss preference optimization with
# reference model — robust to deterministic preference labels.
set -e
export WANDB_MODE=disabled
export DISABLE_VERSION_CHECK=1

torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_port=$((29400 + ${SLURM_JOB_ID:-$$} % 10000)) \
    src/train.py \
    --model_name_or_path /models/Qwen2.5-Math-1.5B-Instruct \
    --trust_remote_code true \
    --stage dpo \
    --do_train \
    --finetuning_type full \
    --pref_beta 0.1 \
    --pref_loss ipo \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --dataset math_step_dpo \
    --template qwen \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir ${OUTPUT_DIR:-${SAVE_PATH:-/tmp/saves}/qwen-math-1.5b/ipo} \
    --seed ${SEED:-42} \
    --logging_steps 10 \
    --save_steps 99999 \
    --plot_loss true \
    --overwrite_output_dir true \
    --save_only_model true \
    --report_to none \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5.0e-7 \
    --num_train_epochs 4.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000
