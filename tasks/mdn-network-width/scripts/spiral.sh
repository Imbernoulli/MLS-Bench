#!/bin/bash
# mdn-network-width: build a mixture density network from the agent's choice of the
# TRUNK WIDTH (hidden units) (solution/network_width.py -> build_mdn), train it for a FIXED budget on
# the spiral multimodal conditional p(y|x), then report exact held-out mixture
# NLL (nats, lower is better).
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=mdn-network-width rc=%s\\n" "$rc"; fi' EXIT
# The task config runs settings serially on one allocated GPU. Inherit the
# verifier-provided CUDA_VISIBLE_DEVICES instead of selecting a host GPU here.
cd /workspace/mdn-density

python harness_mdn.py \
    --task mdn-network-width \
    --solution solution/network_width.py \
    --surface network_width \
    --target spiral \
    --seed ${SEED:-42} \
    --steps 4000 \
    --batch-size 512 \
    --lr 1e-3 \
    --n-train 20000 \
    --n-test 20000
