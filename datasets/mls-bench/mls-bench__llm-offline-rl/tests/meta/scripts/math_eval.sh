#!/bin/bash
# Math evaluation on GSM8K, MATH-500, AIME-2024 (judge-free, MathRuler grader).
# Runs inside the MathRuler container; splits live at ./data/{aime,gsm8k,math}_splits.
set -e
cd /workspace/MathRuler

python "${TASK_DIR:-/workspace/_task}/scripts/math_eval.py"
