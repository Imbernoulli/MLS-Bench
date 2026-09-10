#!/bin/bash
# Run mutation effect prediction on BLAT_ECOLX (beta-lactamase, OrganismalFitness)
cd /workspace

python ProteinGym/custom_mutation_pred.py \
    --assay-id BLAT_ECOLX_Firnberg_2014 \
    --data-dir /data/esm2_embeddings \
    --cv-dir /data/proteingym/cv_folds \
    --epochs 200 --batch-size 64 --lr 1e-3 --weight-decay 0.05 \
    --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}/${ENV}
