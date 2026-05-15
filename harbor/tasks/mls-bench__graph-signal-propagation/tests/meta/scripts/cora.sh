#!/bin/bash
set -e
cd /workspace/ChebNetII/main

ENV=cora SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u custom_filter.py
