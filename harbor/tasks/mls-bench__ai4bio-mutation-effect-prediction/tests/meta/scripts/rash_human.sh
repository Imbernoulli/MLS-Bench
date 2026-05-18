#!/bin/bash
# Run mutation effect prediction on RASH_HUMAN (GTPase, Activity)
cd /workspace

python ProteinGym/custom_mutation_pred.py \
    --assay-id RASH_HUMAN_Bandaru_2017 \
    --data-dir /data/esm2_embeddings \
    --cv-dir /data/proteingym/cv_folds \
    --epochs 200 --batch-size 64 --lr 1e-3 --weight-decay 0.05 \
    --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}/${ENV}
