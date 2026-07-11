#!/bin/bash
set -euo pipefail
cd /workspace/prompt-optimization-lab
python harness_induce.py \\
    --solution solution/exemplar.py \\
    --surface select_exemplars \\
    --dataset sst2 \\
    --seed ${SEED:-42}
