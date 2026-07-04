#!/bin/bash
# Run clustering benchmark on two-moons (non-convex) dataset
cd /workspace
python "/tests/eval/_inputgen/apply.py" "ml-clustering-algorithm" /workspace
ENV=ff4ae96cbf1e python scikit-learn/custom_clustering.py
