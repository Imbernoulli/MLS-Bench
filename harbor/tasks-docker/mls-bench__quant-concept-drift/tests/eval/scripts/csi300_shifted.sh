#!/bin/bash
ulimit -n 65536 2>/dev/null || true
SCRIPT_DIR="$(dirname "$0")"
python -u "$SCRIPT_DIR/run_workflow.py" \
    --fit-start 2008-01-01 --fit-end 2013-12-31 \
    --train-start 2008-01-01 --train-end 2013-12-31 \
    --val-start 2014-01-01 --val-end 2015-12-31 \
    --test-start 2016-01-01 --test-end 2018-12-31 \
    --experiment-name csi300_shifted_concept_drift
