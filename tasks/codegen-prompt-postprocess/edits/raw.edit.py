"""Unmeasured direct-output candidate for full-protocol calibration.

Reference surface: vendor/code-generation-lab/solution/policy_postprocess.py
"""

_FILE = "code-generation-lab/solution/policy_postprocess.py"

_CONTENT = '''def build_prompt(problem):
    return f"Write a Python function.\\n\\n{problem['prompt']}"


def postprocess(raw_text, entry_point):
    return raw_text'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 16, "content": _CONTENT},
]
