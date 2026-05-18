#!/bin/bash
ulimit -n 65536 2>/dev/null || true
SCRIPT_DIR="$(dirname "$0")"
python -u "$SCRIPT_DIR/run_workflow.py" --instruments csi100 --experiment-name csi100_stock_prediction
