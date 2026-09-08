#!/bin/bash
# Train humanoid locomotion policy with custom algorithm and export JIT policy
set -e

# Diagnose — and where possible repair — the CUDA user-space driver before
# IsaacGym touches the GPU. On Hopper, PhysX has to JIT its compute_86 PTX to
# sm_90, and pre-570 user-space drivers segfault doing it (issue #59).
# shellcheck source=gpu_env_preflight.sh
source "$(dirname "${BASH_SOURCE[0]}")/gpu_env_preflight.sh"

cd /workspace

# Use SEED (injected by MLS-Bench framework) for reproducibility
SEED=${SEED:-42}

echo "Training with seed: $SEED"

# Seed-scoped experiment dir. --experiment_name overrides
# XBotLCfgPPO.runner.experiment_name (helpers.update_cfg_from_args), so the
# training run dir, play.py's checkpoint lookup (load_run=-1) and the policy
# export all land under this per-seed root. Without it, concurrent seeds share
# logs/XBot_ppo: play.py would pick the LATEST run (possibly another seed's)
# and every seed would overwrite the same exported/policies/policy_1.pt.
EXPERIMENT_NAME="XBot_ppo_s${SEED}"
POLICY_DIR="humanoid-gym/logs/${EXPERIMENT_NAME}/exported"

# Training always runs. An earlier revision skipped training whenever
# $OUTPUT_DIR already held a policy, so eval-only re-runs could avoid the ~9h
# training cost — but OUTPUT_DIR is fixed for a whole agent run, so every test
# after the first silently re-scored the iteration-1 policy and discarded all
# later agent edits. Baselines run in isolated one-shot workspaces and never
# benefited from that cache anyway. For deliberate manual eval-only re-runs,
# set MLSB_REUSE_CACHED_POLICY=1 explicitly (never set by the harness).
if [ "${MLSB_REUSE_CACHED_POLICY:-0}" = "1" ] && [ -n "$OUTPUT_DIR" ] \
        && [ -f "$OUTPUT_DIR/exported/policies/policy_1.pt" ]; then
    echo "MLSB_REUSE_CACHED_POLICY=1: reusing cached policy at $OUTPUT_DIR/exported/policies/policy_1.pt — skipping training."
    exit 0
fi

# Clear stale artifacts up front so nothing from a previous test iteration can
# be scored if this run fails: the evals read $OUTPUT_DIR/exported first, and
# a leftover workspace export could otherwise feed their fallback path.
if [ -n "$OUTPUT_DIR" ]; then
    rm -rf "$OUTPUT_DIR/exported"
fi
rm -rf "humanoid-gym/logs/${EXPERIMENT_NAME}"

# Train the policy with custom algorithm (PPO, ActorCritic, RolloutStorage).
# Do not override max_iterations; XBotLCfgPPO.runner sets the official recipe.
python humanoid-gym/humanoid/scripts/train.py \
    --task humanoid_custom \
    --num_envs 4096 \
    --headless \
    --experiment_name "$EXPERIMENT_NAME" \
    --seed $SEED

# Export policy as JIT (play.py does this automatically at line 80-81).
# play.py crashes at line 108 on a string + None concat when saving a video,
# but only AFTER the policy export. Tolerate that trailing error so set -e
# doesn't abort the script before the export check below. The experiment dir
# was cleared above, so load_run=-1 can only resolve to the run just trained.
echo "Exporting policy as JIT..."
python humanoid-gym/humanoid/scripts/play.py \
    --task humanoid_custom \
    --experiment_name "$EXPERIMENT_NAME" \
    --headless --run_name "" || echo "play.py post-export error ignored (export verified below)"

# The || above must not mask a genuinely failed export: require the policy.
if [ ! -f "$POLICY_DIR/policies/policy_1.pt" ]; then
    echo "ERROR: training/export did not produce $POLICY_DIR/policies/policy_1.pt" >&2
    exit 1
fi

# Copy exported policy to OUTPUT_DIR if specified (injected by MLS-Bench framework)
# LEGGED_GYM_ROOT_DIR = /workspace/humanoid-gym, so logs are at humanoid-gym/logs/
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    # Fresh, non-nesting copy: cp -r into an existing exported/ would create
    # exported/exported/ and leave the previous policy_1.pt in place.
    rm -rf "$OUTPUT_DIR/exported"
    cp -r "$POLICY_DIR" "$OUTPUT_DIR/"
    echo "Exported policy saved to $OUTPUT_DIR/exported/"
fi

echo "Training and export complete!"
