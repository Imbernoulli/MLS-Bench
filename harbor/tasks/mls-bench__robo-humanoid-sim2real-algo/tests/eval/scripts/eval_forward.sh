#!/bin/bash
# Evaluate on forward-only commands (minimal lateral/turning)

# Diagnose — and where possible repair — the CUDA user-space driver before
# IsaacGym touches the GPU. On Hopper, PhysX has to JIT its compute_86 PTX to
# sm_90, and pre-570 user-space drivers segfault doing it (issue #47).
# shellcheck source=gpu_env_preflight.sh
source "$(dirname "${BASH_SOURCE[0]}")/gpu_env_preflight.sh"

cd /workspace

# Set command ranges for forward-only commands
export EVAL_VX_MIN="0.3"
export EVAL_VX_MAX="1.0"
export EVAL_VY_MIN="-0.1"
export EVAL_VY_MAX="0.1"
export EVAL_DYAW_MIN="-0.2"
export EVAL_DYAW_MAX="0.2"

echo "Evaluating on forward-only commands..."
echo "Command ranges: vx=[$EVAL_VX_MIN, $EVAL_VX_MAX], vy=[$EVAL_VY_MIN, $EVAL_VY_MAX], dyaw=[$EVAL_DYAW_MIN, $EVAL_DYAW_MAX]"

# Run evaluation (num_commands and eval_duration are read from env vars in the script)
export NUM_COMMANDS=100
export EVAL_DURATION=10.0
python _task/scripts/eval_sim2sim_diverse.py
