"""WEAK baseline (cd-numeric-answer): answer-only over-constraining.

Constrain the ENTIRE generation to a bare integer from the first token. Always
structurally valid, but forces the model to commit to a number before any
reasoning -> low accuracy on GSM8K. This is the "over-constrained, valid but
often wrong" failure the task is about.
Reference surface: vendor/constrained-decoding-lab/solution/decoder_numeric.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_numeric.py"

_CONTENT = '''def build_decoder(question: str, tok):
    prompt = (
        "Answer the math problem with the final integer only.\\n\\n"
        f"Problem: {question}\\nAnswer: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        answer_regex=r"[ ]?-?[0-9]{1,7}",
        max_answer_tokens=10,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 35, "end_line": 47, "content": _CONTENT},
]
