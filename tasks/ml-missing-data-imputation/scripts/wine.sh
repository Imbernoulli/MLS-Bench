#!/bin/bash
# Run missing data imputation benchmark on Wine dataset
cd /workspace
ENV=625f10bf8048 python scikit-learn/custom_imputation.py
