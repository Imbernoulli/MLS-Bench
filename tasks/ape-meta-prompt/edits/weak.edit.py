"""Weak reference: vague induction request."""

_FILE = "prompt-optimization-lab/solution/meta_prompt.py"
_CONTENT = '''def meta_prompt(examples, ctx):
    # Vague induction request with no output contract.
    return "Say something about these examples."'''
OPS = [{"op": "replace", "file": _FILE, "start_line": 14, "end_line": 17, "content": _CONTENT}]
