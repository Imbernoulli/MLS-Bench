"""Weak reference: single arbitrary exemplar."""

_FILE = "prompt-optimization-lab/solution/exemplar.py"
_CONTENT = '''def select_exemplars(pool, ctx):
    # One arbitrary example gives reverse-mode induction narrow evidence.
    return [pool[0]]'''
OPS = [{"op": "replace", "file": _FILE, "start_line": 14, "end_line": 17, "content": _CONTENT}]
