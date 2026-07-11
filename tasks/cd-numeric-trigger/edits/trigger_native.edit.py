"""STRONG baseline (cd-numeric-trigger): trigger '#### ' (GSM8K-native delimiter, emitted after reasoning).

Reference surface: vendor/constrained-decoding-lab/solution/decoder_trigger.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_trigger.py"

_CONTENT = r'''def build_decoder(question: str, tok):
    prompt = (
        "Solve the math problem step by step. After your reasoning, write "
        "'#### ' followed by the final integer answer.\n\n"
        f"Problem: {question}\nSolution: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        preamble_regex=r"[\s\S]*",
        trigger="#### ",
        answer_regex=r"[ ]?-?[0-9]{1,7}",
        max_answer_tokens=10,
        max_free_tokens=256,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 18, "end_line": 30, "content": _CONTENT},
]
