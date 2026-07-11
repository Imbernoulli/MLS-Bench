"""Strong reference candidate: structured reverse-mode template."""

_FILE = "prompt-optimization-lab/solution/meta_prompt.py"
_CONTENT = '''def meta_prompt(examples, ctx):
    # Structured reverse-mode induction contract.
    return ("Read the labeled input/output examples and infer ONE concise "
            "instruction that maps each input to exactly one of: {labels}. "
            "Output only that instruction.\\n\\n{demo}\\n\\nThe instruction was:")'''
OPS = [{"op": "replace", "file": _FILE, "start_line": 14, "end_line": 17, "content": _CONTENT}]
