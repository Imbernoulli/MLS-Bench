#!/bin/bash
set -euo pipefail
cd /workspace/prompt-optimization-lab
python harness_induce.py \\
    --solution solution/meta_prompt.py \\
    --surface meta_prompt \\
    --dataset sst2 \\
    --seed ${SEED:-42}
