#!/bin/bash
ulimit -n 65536 2>/dev/null || true
SCRIPT_DIR="$(dirname "$0")"
python -u "$SCRIPT_DIR/run_workflow.py" \
    --fit-start 2010-01-01 --fit-end 2016-12-31 \
    --train-start 2010-01-01 --train-end 2016-12-31 \
    --val-start 2017-01-01 --val-end 2018-12-31 \
    --test-start 2019-01-01 --test-end 2020-08-01 \
    --experiment-name csi300_recent_stock_prediction
