"""MID baseline (cd-numeric-answer): partial reasoning budget then constrain.

Same step-by-step prompt as the strong baseline, but the FREE reasoning region
is capped SHORT (max_free_tokens=64) before the answer is constrained. The model
gets only partial chain-of-thought, so accuracy lands between the answer-only
(no reasoning) and full reason-then-constrain baselines. Demonstrates that HOW
MANY reasoning tokens you allow before constraining is a real, measurable design
axis (too small starves reasoning; too large wastes compute).
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
        max_free_tokens=192,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 35, "end_line": 47, "content": _CONTENT},
]
