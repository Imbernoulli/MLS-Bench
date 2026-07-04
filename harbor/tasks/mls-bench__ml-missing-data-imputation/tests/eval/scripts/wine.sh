#!/bin/bash
# Run missing data imputation benchmark on Wine dataset
cd /workspace
python "/tests/eval/_inputgen/apply.py" "ml-missing-data-imputation" /workspace
ENV=625f10bf8048 python scikit-learn/custom_imputation.py
