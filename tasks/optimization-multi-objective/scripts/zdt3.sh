#!/bin/bash
set -e
cd /workspace

ENV=p1 SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u deap/custom_moea.py
