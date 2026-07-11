"""Unmeasured structured-prompt candidate for full-protocol calibration.

Reference surface: vendor/code-generation-lab/solution/policy_docstring.py
"""

_FILE = "code-generation-lab/solution/policy_docstring.py"

_CONTENT = 'def build_prompt(problem):\n    ep = problem["entry_point"]\n    return (\n        "Implement the Python function `" + ep + "` that satisfies the "\n        "specification below. Requirements:\\n"\n        "- Define exactly one top-level function named `" + ep + "`.\\n"\n        "- Match the interface stated by the specification.\\n"\n        "- Return only Python source, without example usage or explanation.\\n\\n"\n        + problem["prompt"]\n    )'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 10, "end_line": 11, "content": _CONTENT},
]
