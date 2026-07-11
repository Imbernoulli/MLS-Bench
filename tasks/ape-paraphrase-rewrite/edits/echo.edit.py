"""Echo baseline edit for ape-paraphrase-rewrite: replace the editable
function in prompt-optimization-lab/solution/rewrite.py (lines 14-17).
Reference: vendor/prompt-optimization-lab/baselines/
"""

_FILE = "prompt-optimization-lab/solution/rewrite.py"

_CONTENT = r'''def rewrite(seed, ctx):
    # Weak: echo the seed unchanged — no paraphrase, so dev-selection just keeps the
    # seed and you cannot improve over the plain fixed instruction.
    return [seed]'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 17, "content": _CONTENT},
]
