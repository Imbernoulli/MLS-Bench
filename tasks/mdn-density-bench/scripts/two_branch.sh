#!/bin/bash
# mdn-density-bench setting two_branch: build the agent's ONE MDN design
# (solution/density_bench.py -> build_mdn), train it for a FIXED budget on the two_branch
# multimodal conditional p(y|x), report exact held-out mixture NLL.
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=mdn-density-bench rc=%s\\n" "$rc"; fi' EXIT
# The task config runs settings serially on one allocated GPU. Inherit the
# verifier-provided CUDA_VISIBLE_DEVICES instead of selecting a host GPU here.
cd /workspace/mdn-density

python harness_mdn.py \
    --task mdn-density-bench \
    --solution solution/density_bench.py \
    --surface density_family \
    --target two_branch \
    --seed ${SEED:-42} \
    --steps 4000 \
    --batch-size 512 \
    --lr 1e-3 \
    --n-train 20000 \
    --n-test 20000
