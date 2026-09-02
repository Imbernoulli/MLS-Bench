#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_env_preflight.sh"
cd /workspace
export NUM_COMMANDS=3
export EVAL_DURATION=0.2
export EVAL_VX_MIN=0.8 EVAL_VX_MAX=1.5 EVAL_VY_MIN=-0.5 EVAL_VY_MAX=0.5 EVAL_DYAW_MIN=-0.8 EVAL_DYAW_MAX=0.8
echo "SMOKE override: NUM_COMMANDS=${NUM_COMMANDS} EVAL_DURATION=${EVAL_DURATION} (runtime only)"
python _task/scripts/eval_sim2sim_diverse.py
