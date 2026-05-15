#!/bin/bash
set -e
cd /workspace

ENV=PubMed SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u pytorch-geometric/custom_nodecls.py
