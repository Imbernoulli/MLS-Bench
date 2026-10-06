"""Verifier-only data for quant-concept-drift (no generator lives here).

The adapter stages holdout/<task>/ into the bundle's tests/meta/ only when this
file exists, and tests/ is mounted at verification time and never during the
agent's session. This directory carries the qlib cn_data of the scored universe
(CSI300/CSI100 constituents + SH000300) after 2015-12-31 -- the test periods of
all three settings -- as tails of each .day.bin plus the full calendar and
index files, with the sha256 of every restored file; gen_withheld.py records
the cut and the image digest. scripts/_withheld_qlib.py rebuilds the full
cn_data for the eval.
"""
