#!/bin/bash
# Drop this label's best_model from a previous agent iteration so an
# eval that never improves on it cannot silently score the old model.
rm -f "${OUTPUT_DIR}/${ENV}/best_model.pt"
python custom_molprop.py \
    --dataset bace --data-dir /data/molecular_property_prediction \
    --epochs 60 --batch-size 32 --lr 1e-4 \
    --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}/${ENV}
