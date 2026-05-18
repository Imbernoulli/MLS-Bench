cd /workspace
python pytorch-vision/bench/poison/run_poison_robust.py \
    --arch mobilenetv2 \
    --dataset fmnist \
    --data-root /data/fmnist \
    --poison-fraction 0.15 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --seed "${SEED:-42}"
