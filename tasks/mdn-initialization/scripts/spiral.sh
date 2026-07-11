#!/bin/bash
# mdn-initialization: build a mixture density network from the agent's choice of the
# INITIALISATION (starting sigma scale) (solution/initialization.py -> build_mdn), train it for a FIXED budget on
# the spiral multimodal conditional p(y|x), then report exact held-out mixture
# NLL (nats, lower is better).
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=mdn-initialization rc=%s\\n" "$rc"; fi' EXIT
# The task config runs settings serially on one allocated GPU. Inherit the
# verifier-provided CUDA_VISIBLE_DEVICES instead of selecting a host GPU here.
cd /workspace/mdn-density

python harness_mdn.py \
    --task mdn-initialization \
    --solution solution/initialization.py \
    --surface initial_sigma \
    --target spiral \
    --seed ${SEED:-42} \
    --steps 4000 \
    --batch-size 512 \
    --lr 1e-3 \
    --n-train 20000 \
    --n-test 20000
