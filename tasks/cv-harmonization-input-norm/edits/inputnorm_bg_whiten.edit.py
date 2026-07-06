"""cv-harmonization-input-norm baseline: bg_whiten (naive background-whitening — corrupts input (weak)).
Reference: vendor/image-harmonization/baselines/inputnorm_bg_whiten.py
"""

_FILE = "image-harmonization/solution/inputnorm.py"

_CONTENT = 'def get_input_norm():\n    return "bg_whiten"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 21, "end_line": 23, "content": _CONTENT},
]
