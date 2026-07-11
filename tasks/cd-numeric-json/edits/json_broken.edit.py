"""WEAK baseline (cd-numeric-json): force a JSON object from the FIRST token (no reasoning). Over-constrained and brittle -> low accuracy.

Reference surface: vendor/constrained-decoding-lab/solution/decoder_json.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_json.py"

_CONTENT = r'''def build_decoder(question: str, tok):
    prompt = (
        "Answer the math problem. Output ONLY a JSON object with the integer "
        "answer and nothing else.\n\n"
        f"Problem: {question}\nJSON: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        answer_regex=r'\{"answer":-?[0-9]{1,7}\}',
        max_answer_tokens=16,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 18, "end_line": 28, "content": _CONTENT},
]
