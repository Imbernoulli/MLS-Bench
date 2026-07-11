"""Unmeasured fenced-trim candidate for full-protocol calibration.

Reference surface: vendor/code-generation-lab/solution/policy_postprocess.py
"""

_FILE = "code-generation-lab/solution/policy_postprocess.py"

_CONTENT = '''def build_prompt(problem):
    return (
        "Complete the following Python function. Return only Python source, "
        "without example usage or explanation.\\n\\n" + problem["prompt"]
    )


def postprocess(raw_text, entry_point):
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
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 16, "content": _CONTENT},
]
