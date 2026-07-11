"""STRONG baseline (cd-numeric-json): reason freely, then constrain the answer to a robust JSON object {"answer": <int>}.

Reference surface: vendor/constrained-decoding-lab/solution/decoder_json.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_json.py"

_CONTENT = r'''def build_decoder(question: str, tok):
    prompt = (
        "Solve the math problem step by step. After your reasoning, write "
        "'#### ' then a JSON object {\"answer\": <int>} with the final integer.\n\n"
        f"Problem: {question}\nSolution: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        preamble_regex=r"[\s\S]*",
        trigger="#### ",
        answer_regex=r'\{[ ]?"answer"[ ]?:[ ]?-?[0-9]{1,7}[ ]?\}',
        max_answer_tokens=24,
        max_free_tokens=256,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 18, "end_line": 28, "content": _CONTENT},
]
