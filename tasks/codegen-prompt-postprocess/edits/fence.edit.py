"""Unmeasured fenced-extraction candidate for full-protocol calibration.

Reference surface: vendor/code-generation-lab/solution/policy_postprocess.py
"""

_FILE = "code-generation-lab/solution/policy_postprocess.py"

_CONTENT = '''def build_prompt(problem):
    return (
        "Complete the following Python function. Put the complete function in a "
        "single ```python code block.\\n\\n" + problem["prompt"]
    )


def postprocess(raw_text, entry_point):
    marker = "def " + entry_point
    pieces = raw_text.split("```")
    code = next((piece for piece in pieces if marker in piece), raw_text)
    if code.startswith("python\\n"):
        code = code[7:]
    elif code.startswith("py\\n"):
        code = code[3:]
    return code.strip()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 16, "content": _CONTENT},
]
