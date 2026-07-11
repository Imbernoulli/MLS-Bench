"""Factorized-prior baseline for compress-entropy-model.

Balle 2018 fully-factorized prior (entropy bottleneck on y, no side information).
Reference: compressai.models.FactorizedPrior.
"""

_FILE = "compressai/solution/entropy_model.py"

_CONTENT = '    return "factorized"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
