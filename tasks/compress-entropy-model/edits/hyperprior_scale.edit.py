"""Scale-hyperprior baseline for compress-entropy-model.

Balle 2018 scale hyperprior: a hyper-latent z predicts a per-element Gaussian scale
(zero mean) for y. Reference: compressai.models.ScaleHyperprior.
"""

_FILE = "compressai/solution/entropy_model.py"

_CONTENT = '    return "hyperprior_scale"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
