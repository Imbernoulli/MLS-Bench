#!/bin/bash
# Train humanoid locomotion policy with custom algorithm and export JIT policy
set -e

cd /workspace

# Use SEED (injected by MLS-Bench framework) for reproducibility
SEED=${SEED:-42}

echo "Training with seed: $SEED"

# Skip retraining if the OUTPUT_DIR already has a cached policy from a prior run.
# This lets us re-run only the eval phase without paying for the ~9h training again.
if [ -n "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/exported/policies/policy_1.pt" ]; then
    echo "Cached policy at $OUTPUT_DIR/exported/policies/policy_1.pt — skipping training."
    exit 0
fi

# Train the policy with custom algorithm (PPO, ActorCritic, RolloutStorage).
# Do not override max_iterations; XBotLCfgPPO.runner sets the official recipe.
python humanoid-gym/humanoid/scripts/train.py \
    --task humanoid_custom \
    --num_envs 4096 \
    --headless \
    --seed $SEED

# Export policy as JIT (play.py does this automatically at line 80-81).
# play.py crashes at line 108 on a string + None concat when saving a video,
# but only AFTER the policy export. Tolerate that trailing error so set -e
# doesn't abort the script before the OUTPUT_DIR copy below.
echo "Exporting policy as JIT..."
python humanoid-gym/humanoid/scripts/play.py \
    --task humanoid_custom \
    --headless --run_name "" || echo "play.py post-export error ignored (policy already exported)"

# Copy exported policy to OUTPUT_DIR if specified (injected by MLS-Bench framework)
# LEGGED_GYM_ROOT_DIR = /workspace/humanoid-gym, so logs are at humanoid-gym/logs/
POLICY_DIR="humanoid-gym/logs/XBot_ppo/exported"
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
    if [ -d "$POLICY_DIR" ]; then
        cp -r $POLICY_DIR $OUTPUT_DIR/
        echo "Exported policy saved to $OUTPUT_DIR/exported/"
    else
        echo "Warning: Exported policy not found at $POLICY_DIR"
    fi
fi

echo "Training and export complete!"
