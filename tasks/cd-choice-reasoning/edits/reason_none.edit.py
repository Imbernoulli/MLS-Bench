"""No-preamble candidate: emit a label directly from the FSM alternation.

Reference surface: vendor/constrained-decoding-lab/solution/decoder_choice_reasoning.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_choice_reasoning.py"

_CONTENT = r'''def build_decoder(text: str, labels, tok):
    prompt = (
        "Classify the news topic. Reply with exactly one of: World, Sports, "
        "Business, Sci/Tech.\n\n"
        f"News: {text}\nTopic: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        answer_regex=r"World|Sports|Business|Sci/Tech",
        max_answer_tokens=8,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 18, "end_line": 28, "content": _CONTENT},
]
