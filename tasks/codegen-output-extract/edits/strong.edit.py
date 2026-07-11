"""Unmeasured fenced-trim candidate for full-protocol calibration.

Reference surface: vendor/code-generation-lab/solution/policy_extract.py
"""

_FILE = "code-generation-lab/solution/policy_extract.py"

_CONTENT = '''def extract(raw_text, entry_point):
    import ast

    marker = "def " + entry_point
    pieces = raw_text.split("```")
    code = next((piece for piece in pieces if marker in piece), raw_text)
    if code.startswith("python\\n"):
        code = code[7:]
    elif code.startswith("py\\n"):
        code = code[3:]
    lines = code.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\\n".join(lines[:end]).strip()
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == entry_point
            for node in tree.body
        ):
            return candidate
    return code.strip()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 10, "end_line": 11, "content": _CONTENT},
]
