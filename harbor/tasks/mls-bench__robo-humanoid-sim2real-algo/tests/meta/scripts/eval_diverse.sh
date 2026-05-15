#!/bin/bash
# Evaluate on diverse mixed commands (forward/backward, lateral, turning)

cd /workspace

# Set command ranges for diverse commands
export EVAL_VX_MIN="-0.5"
export EVAL_VX_MAX="1.0"
export EVAL_VY_MIN="-0.4"
export EVAL_VY_MAX="0.4"
export EVAL_DYAW_MIN="-0.5"
export EVAL_DYAW_MAX="0.5"

echo "Evaluating on diverse commands..."
echo "Command ranges: vx=[$EVAL_VX_MIN, $EVAL_VX_MAX], vy=[$EVAL_VY_MIN, $EVAL_VY_MAX], dyaw=[$EVAL_DYAW_MIN, $EVAL_DYAW_MAX]"

# Run evaluation (num_commands and eval_duration are read from env vars in the script)
export NUM_COMMANDS=100
export EVAL_DURATION=10.0
python _task/scripts/eval_sim2sim_diverse.py
