#!/bin/bash
# Evaluate on high-speed commands (stress test)

cd /workspace

# Set command ranges for high-speed commands
export EVAL_VX_MIN="0.8"
export EVAL_VX_MAX="1.5"
export EVAL_VY_MIN="-0.5"
export EVAL_VY_MAX="0.5"
export EVAL_DYAW_MIN="-0.8"
export EVAL_DYAW_MAX="0.8"

echo "Evaluating on high-speed commands..."
echo "Command ranges: vx=[$EVAL_VX_MIN, $EVAL_VX_MAX], vy=[$EVAL_VY_MIN, $EVAL_VY_MAX], dyaw=[$EVAL_DYAW_MIN, $EVAL_DYAW_MAX]"

# Run evaluation (num_commands and eval_duration are read from env vars in the script)
export NUM_COMMANDS=100
export EVAL_DURATION=10.0
python _task/scripts/eval_sim2sim_diverse.py
