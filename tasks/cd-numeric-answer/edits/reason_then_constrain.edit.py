"""STRONG baseline (cd-numeric-answer): reason-first, constrain only the answer.

Let the model produce a free-form chain-of-thought, then emit a trigger token
("####", the GSM8K answer delimiter), after which a tight integer constraint
kicks in for the final answer only. Preserves reasoning -> high accuracy, while
still guaranteeing the final answer is a structurally valid integer.
Reference surface: vendor/constrained-decoding-lab/solution/decoder_numeric.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_numeric.py"

_CONTENT = '''def build_decoder(question: str, tok):
    prompt = (
        "Solve the math problem step by step. After your reasoning, write "
        "'#### ' followed by the final integer answer.\\n\\n"
        f"Problem: {question}\\nSolution: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        preamble_regex=r"[\\s\\S]*",
        trigger="#### ",
        answer_regex=r"[ ]?-?[0-9]{1,7}",
        max_answer_tokens=10,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 35, "end_line": 47, "content": _CONTENT},
]
