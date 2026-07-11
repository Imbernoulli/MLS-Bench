"""Unmeasured full-protocol candidate: Python AST structure as the key."""

_FILE = "code-generation-lab/solution/policy_consensus.py"
_CONTENT = '''def canonical(program):
    """Normalize valid Python through its abstract syntax tree."""
    import ast
    try:
        return ("ast", ast.dump(ast.parse(program or ""), include_attributes=False))
    except SyntaxError:
        return ("syntax-error", (program or "").strip())'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 7, "content": _CONTENT},
]
