"""MID baseline (cd-numeric-json): reason-then-constrain with a LOOSE JSON regex (optional closing brace) and a short reasoning budget.

Reference surface: vendor/constrained-decoding-lab/solution/decoder_json.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_json.py"

_CONTENT = r'''def build_decoder(question: str, tok):
    prompt = (
        "Solve the math problem step by step. After your reasoning, write "
        "'#### ' then a JSON object {\"answer\": <int>}.\n\n"
        f"Problem: {question}\nSolution: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        preamble_regex=r"[\s\S]*",
        trigger="#### ",
        answer_regex=r'\{[ ]?"?answer"?[ ]?:[ ]?-?[0-9]{1,7}[ ]?\}?',
        max_answer_tokens=20,
        max_free_tokens=96,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 18, "end_line": 28, "content": _CONTENT},
]
