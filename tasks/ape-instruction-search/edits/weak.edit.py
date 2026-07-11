"""Weak baseline: EMPTY (zero) instruction — no automatic prompt optimization.

Return the empty instruction: the frozen LM executes with no task guidance and sits
near the class prior. This is the naive floor an APE search must beat.
Reference: vendor/prompt-optimization-lab/solution/search.py (default).
"""

_FILE = "prompt-optimization-lab/solution/search.py"

_CONTENT = '''def optimize(ctx) -> str:
    # Empty / zero instruction: no task guidance at all.
    return ""'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 46, "end_line": 48, "content": _CONTENT},
]
