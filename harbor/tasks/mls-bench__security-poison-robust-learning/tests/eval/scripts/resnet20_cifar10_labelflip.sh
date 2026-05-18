cd /workspace
python pytorch-vision/bench/poison/run_poison_robust.py \
    --arch resnet20 \
    --dataset cifar10 \
    --data-root /data/cifar \
    --poison-fraction 0.10 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --seed "${SEED:-42}"
