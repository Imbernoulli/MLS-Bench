#!/bin/bash
# Run clustering benchmark on isotropic Gaussian blobs dataset
cd /workspace
python "/tests/eval/_inputgen/apply.py" "ml-clustering-algorithm" /workspace
ENV=1c67c4b89b56 python scikit-learn/custom_clustering.py
