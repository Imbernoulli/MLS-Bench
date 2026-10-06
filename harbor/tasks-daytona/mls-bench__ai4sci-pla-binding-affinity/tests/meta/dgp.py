"""Verifier-only data for ai4sci-pla-binding-affinity (no generator lives here).

The adapter stages holdout/<task>/ into the bundle's tests/meta/ only when this
file exists, and tests/ is mounted at verification time and never during the
agent's session. This directory carries the PDBbind test labels of the three
scored test sets (data/withheld/pla_test_labels.json, float32 values in file
order) and a manifest with each original file's sha256 and a fingerprint of its
non-label content; gen_withheld.py records how they were extracted and from
which image digest. scripts/_withheld_pla.py re-inserts the labels for the eval.
"""
