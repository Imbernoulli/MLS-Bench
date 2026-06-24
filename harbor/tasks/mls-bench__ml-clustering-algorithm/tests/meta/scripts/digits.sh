#!/bin/bash
# Run clustering benchmark on sklearn Digits (real-world) dataset
cd /workspace
python "/tests/eval/_inputgen/apply.py" "ml-clustering-algorithm" /workspace
python scikit-learn/custom_clustering.py
