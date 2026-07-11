"""Unmeasured full-protocol candidate: task-and-candidate reprompt."""

_FILE = "code-generation-lab/solution/policy_repair.py"
_CONTENT = '''def build_repair_prompt(problem, program, error, round_index):
    return (
        "Review the candidate function and return a corrected complete function "
        "inside one Python code block.\\n\\n"
        f"Task:\\n{problem['prompt']}\\n\\nCandidate:\\n```python\\n{program}\\n```"
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 9, "end_line": 15, "content": _CONTENT},
]
