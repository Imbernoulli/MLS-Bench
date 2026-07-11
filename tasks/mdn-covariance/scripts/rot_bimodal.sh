#!/bin/bash
# mdn-covariance: build a mixture density network from the agent's choice of the
# COMPONENT COVARIANCE STRUCTURE (diag vs full) (solution/covariance.py -> build_mdn), train it for a FIXED budget on
# the rot_bimodal multimodal conditional p(y|x), then report exact held-out mixture
# NLL (nats, lower is better).
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=mdn-covariance rc=%s\\n" "$rc"; fi' EXIT
# The task config runs settings serially on one allocated GPU. Inherit the
# verifier-provided CUDA_VISIBLE_DEVICES instead of selecting a host GPU here.
cd /workspace/mdn-density

python harness_mdn.py \
    --task mdn-covariance \
    --solution solution/covariance.py \
    --surface covariance \
    --target rot_bimodal \
    --seed ${SEED:-42} \
    --steps 4000 \
    --batch-size 512 \
    --lr 1e-3 \
    --n-train 20000 \
    --n-test 20000
