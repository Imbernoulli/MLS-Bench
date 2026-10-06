"""Verifier-only data for ts-exogenous-forecast (no generator lives here).

The adapter stages holdout/<task>/ into the bundle's tests/meta/ only when this
file exists, and tests/ is mounted at verification time and never during the
agent's session. What this directory carries is the scored test portion of the
three TSLib CSVs (data/withheld/*_tail.csv.gz) plus the sha256 of each full
file; gen_withheld.py documents how the split was cut and from which image.
scripts/_withheld.sh rebuilds each full CSV for the eval and refuses to run if
the rebuilt file does not hash to the original.
"""
