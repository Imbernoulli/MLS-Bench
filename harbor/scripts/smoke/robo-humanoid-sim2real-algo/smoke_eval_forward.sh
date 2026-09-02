#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_env_preflight.sh"
cd /workspace
export NUM_COMMANDS=3
export EVAL_DURATION=0.2
export EVAL_VX_MIN=0.3 EVAL_VX_MAX=1.0 EVAL_VY_MIN=-0.1 EVAL_VY_MAX=0.1 EVAL_DYAW_MIN=-0.2 EVAL_DYAW_MAX=0.2
echo "SMOKE override: NUM_COMMANDS=${NUM_COMMANDS} EVAL_DURATION=${EVAL_DURATION} (runtime only)"
python _task/scripts/eval_sim2sim_diverse.py
