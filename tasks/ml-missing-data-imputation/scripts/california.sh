#!/bin/bash
# Run missing data imputation benchmark on California Housing dataset
cd /workspace
ENV=e587ae0bcf72 python scikit-learn/custom_imputation.py
