"""Generic baseline edit for ape-candidate-generation: replace the editable
function in prompt-optimization-lab/solution/propose.py (lines 14-17).
Reference: vendor/prompt-optimization-lab/baselines/
"""

_FILE = "prompt-optimization-lab/solution/propose.py"

_CONTENT = r'''def propose(ctx):
    # Weak: a single generic instruction — fixed dev-selection has nothing useful
    # to pick, so it returns a vague prompt that lands near the class prior.
    return ["Classify the text."]'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 17, "content": _CONTENT},
]
