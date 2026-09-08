#!/bin/bash
# Fix the shared expert-demo cache race (atomic publish + flock-serialized
# generate-or-load) in the workspace scaffold before running; strict no-op
# if the fix is already present. See _runtime_patch.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_runtime_patch.sh"
python custom_irl.py \
    --env-id HalfCheetah-v4 \
    --seed ${SEED:-42} \
    --total-timesteps 1000000
