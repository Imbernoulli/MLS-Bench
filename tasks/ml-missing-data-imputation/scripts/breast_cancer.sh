#!/bin/bash
# Run missing data imputation benchmark on Breast Cancer Wisconsin dataset
cd /workspace
ENV=1b70fece810f python scikit-learn/custom_imputation.py
