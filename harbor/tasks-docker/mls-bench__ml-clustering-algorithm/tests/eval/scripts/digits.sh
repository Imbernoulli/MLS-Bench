#!/bin/bash
# Run clustering benchmark on sklearn Digits (real-world) dataset
cd /workspace
python "/tests/eval/_inputgen/apply.py" "ml-clustering-algorithm" /workspace
ENV=b971a9aa56ba python scikit-learn/custom_clustering.py
