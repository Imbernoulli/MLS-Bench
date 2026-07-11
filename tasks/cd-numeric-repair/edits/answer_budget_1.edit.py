"""One-token answer region; multi-token integers can exhaust the budget."""

_FILE = "constrained-decoding-lab/solution/decoder_repair.py"

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
        answer_regex=r"[ ]?-?[0-9]{1,10}",
        max_answer_tokens=1,
        max_free_tokens=256,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 20, "end_line": 33, "content": _CONTENT},
]
