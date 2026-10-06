#!/bin/bash
ulimit -n 65536 2>/dev/null || true
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
# Under Harbor the test-period data reach the eval from tests/ only: rebuild
# the full cn_data (prints nothing natively, where the image has it all).
PROVIDER="$(python "$SCRIPT_DIR/_withheld_qlib.py" /root/.qlib/qlib_data/cn_data)" || exit 1
python -u "$SCRIPT_DIR/run_workflow.py" \
    --fit-start 2008-01-01 --fit-end 2013-12-31 \
    --train-start 2008-01-01 --train-end 2013-12-31 \
    --val-start 2014-01-01 --val-end 2015-12-31 \
    --test-start 2016-01-01 --test-end 2018-12-31 \
    --experiment-name csi300_shifted_concept_drift \
    --handler-start 2008-01-01 --handler-end 2020-08-01 \
    ${PROVIDER:+--provider-uri "$PROVIDER"}
