"""Unmeasured two-demonstration candidate for full-protocol calibration.

Reference surface: vendor/code-generation-lab/solution/policy_fewshot.py
"""

_FILE = "code-generation-lab/solution/policy_fewshot.py"

_CONTENT = 'def fewshot():\n    return (\n        "Task: Return the larger of two integers a and b.\\n"\n        "```python\\ndef max2(a, b):\\n    return a if a > b else b\\n```\\n\\n"\n        "Task: Return whether string s is empty.\\n"\n        "```python\\ndef is_empty(s):\\n    return len(s) == 0\\n```"\n    )'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 10, "end_line": 11, "content": _CONTENT},
]
