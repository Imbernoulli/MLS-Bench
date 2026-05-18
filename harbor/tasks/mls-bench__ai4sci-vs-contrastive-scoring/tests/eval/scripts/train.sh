#!/bin/bash
# Train the custom scoring model with end-to-end fine-tuning of backbones.
# Uses unicore-train with custom_vs_model and custom_vs_loss.
#
# NOTE: HypSeek paper trains backbones jointly (not frozen). Freezing the
# backbones collapsed absolute metrics ~4x below paper values AND inverted
# the hyperbolic-vs-Euclidean baseline ordering, because projection heads
# alone could not adapt features to the target geometry. We now train
# end-to-end on a single GPU using the paper's hyperparameters.
#
# LOCAL-GRADIENT-AVERAGING: the upstream training script used for this task was
# run with n_gpu=4 and update_freq=1. Since three_hybrid_loss.py computes loss
# per rank, the single-GPU benchmark uses update_freq=4 to approximate the same
# effective optimizer batch size at batch_size=24.
# With update_freq=1 the hyperbolic+cone baseline received ~4x noisier
# gradients than the paper setting and failed to converge past the Euclidean
# ablation. Also pass --learn-curv so curvature adapts jointly with the
# scoring head.


data_path="/data/vs_data"
# IMPORTANT: save_dir must contain the substring "no_similar_protein" to trigger
# protein-similarity filtering in vendor/external_packages/HypSeek/unimol/tasks/
# train_task.py:566,578,593 (the upstream gate is `if "no_similar_protein" in
# self.args.save_dir`). Without this substring the training set leaks DUD-E /
# DEKOIS / LIT-PCBA homologous proteins (via the protein cluster expansion at
# lines 567-571), which inflates downstream DUD-E AUC/BEDROC well past paper
# values. The actual filtering threshold is controlled by --protein-similarity
# -thres=1.0 below.
save_dir="${OUTPUT_DIR}/checkpoints_no_similar_protein"
tmp_save_dir="${OUTPUT_DIR}/tmp"
tsb_dir="${OUTPUT_DIR}/tensorboard"
mkdir -p "${save_dir}" "${tmp_save_dir}" "${tsb_dir}"

finetune_mol_model="/data/pretrain/mol_pre_no_h_220816.pt"
finetune_pocket_model="/data/pretrain/pocket_pre_220816.pt"

n_gpu=1
batch_size=24
batch_size_valid=32
epoch=50
warmup=0.06
lr=1e-4
# update_freq=4 approximates the upstream 4-GPU local-loss effective batch.
update_freq=4

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
UNICORE_TRAIN=$(command -v unicore-train)

CUDA_VISIBLE_DEVICES=0 python ${UNICORE_TRAIN} ${data_path} \
    --user-dir ./unimol \
    --task train_task \
    --arch custom_vs_model \
    --loss custom_vs_loss \
    --train-subset train \
    --valid-subset valid \
    --valid-set CASF \
    --num-workers 0 \
    --ddp-backend c10d \
    --max-pocket-atoms 256 \
    --optimizer adam \
    --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-8 \
    --clip-norm 1.0 \
    --lr-scheduler polynomial_decay \
    --lr ${lr} \
    --warmup-ratio ${warmup} \
    --max-epoch ${epoch} \
    --batch-size ${batch_size} \
    --batch-size-valid ${batch_size_valid} \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --update-freq ${update_freq} \
    --seed ${SEED:-1} \
    --tensorboard-logdir ${tsb_dir} \
    --log-interval 100 \
    --log-format simple \
    --validate-interval 1 \
    --all-gather-list-size 2048000 \
    --save-dir ${save_dir} \
    --tmp-save-dir ${tmp_save_dir} \
    --keep-best-checkpoints 3 \
    --keep-last-epochs 3 \
    --find-unused-parameters \
    --finetune-pocket-model ${finetune_pocket_model} \
    --finetune-mol-model ${finetune_mol_model} \
    --max-lignum 16 \
    --best-checkpoint-metric valid_bedroc \
    --maximize-best-checkpoint-metric \
    --protein-similarity-thres 1.0 \
    --learn-curv \
    2>&1
