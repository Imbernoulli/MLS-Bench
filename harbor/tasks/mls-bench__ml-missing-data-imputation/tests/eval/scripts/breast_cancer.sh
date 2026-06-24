#!/bin/bash
# Run missing data imputation benchmark on Breast Cancer Wisconsin dataset
cd /workspace
python "/tests/eval/_inputgen/apply.py" "ml-missing-data-imputation" /workspace
python scikit-learn/custom_imputation.py
