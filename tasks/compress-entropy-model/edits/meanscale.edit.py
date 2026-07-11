"""Mean-scale-hyperprior baseline for compress-entropy-model.

Minnen 2018 mean-scale hyperprior: z predicts both the mean and the scale of each y
element. Reference: compressai.models.MeanScaleHyperprior.
"""

_FILE = "compressai/solution/entropy_model.py"

_CONTENT = '    return "meanscale"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
