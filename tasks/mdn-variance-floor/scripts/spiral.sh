#!/bin/bash
# mdn-variance-floor: build a mixture density network from the agent's choice of the
# HEAD-SIDE VARIANCE FLOOR (sigma_eps) (solution/variance_floor.py -> build_mdn), train it for a FIXED budget on
# the spiral multimodal conditional p(y|x), then report exact held-out mixture
# NLL (nats, lower is better).
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=mdn-variance-floor rc=%s\\n" "$rc"; fi' EXIT
# The task config runs settings serially on one allocated GPU. Inherit the
# verifier-provided CUDA_VISIBLE_DEVICES instead of selecting a host GPU here.
cd /workspace/mdn-density

python harness_mdn.py \
    --task mdn-variance-floor \
    --solution solution/variance_floor.py \
    --surface variance_floor \
    --target spiral \
    --seed ${SEED:-42} \
    --steps 4000 \
    --batch-size 512 \
    --lr 1e-3 \
    --n-train 20000 \
    --n-test 20000
