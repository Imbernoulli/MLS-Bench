"""WEAK baseline (cd-numeric-trigger): trigger is a single space, so the constraint fires after the first token of reasoning -> ~answer-only, low accuracy.

Reference surface: vendor/constrained-decoding-lab/solution/decoder_trigger.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_trigger.py"

_CONTENT = r'''def build_decoder(question: str, tok):
    prompt = (
        "Solve the math problem step by step, then give the final integer.\n\n"
        f"Problem: {question}\nSolution: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        preamble_regex=r"[\s\S]*",
        trigger=" ",
        answer_regex=r"[ ]?-?[0-9]{1,7}",
        max_answer_tokens=10,
        max_free_tokens=256,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 18, "end_line": 30, "content": _CONTENT},
]
