#!/bin/bash
# Run missing data imputation benchmark on Wine dataset
cd /workspace
python "/tests/eval/_inputgen/apply.py" "ml-missing-data-imputation" /workspace
python scikit-learn/custom_imputation.py
