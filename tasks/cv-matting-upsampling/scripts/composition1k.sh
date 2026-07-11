#!/bin/bash
# Train once on the full Composition-1K protocol and evaluate all three trimap settings.
set -euo pipefail
cd /workspace/image-matting
python harness.py \
    --data-root ${MATTING_DATA_ROOT:-/data/image-matting/composition1k} \
    --task-id cv-matting-upsampling \
    --surface upsampling \
    --solution solution/upsampling.py \
    --iters 100000 \
    --seed ${SEED:-42}
