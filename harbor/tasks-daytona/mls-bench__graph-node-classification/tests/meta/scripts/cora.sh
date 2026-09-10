#!/bin/bash
set -e
cd /workspace

ENV=Cora SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u pytorch-geometric/custom_nodecls.py
