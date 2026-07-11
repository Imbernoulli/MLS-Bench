"""Short-preamble candidate: reason briefly, then FSM-emit the label.

Reference surface: vendor/constrained-decoding-lab/solution/decoder_choice_reasoning.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_choice_reasoning.py"

_CONTENT = r'''def build_decoder(text: str, labels, tok):
    prompt = (
        "Read the news snippet, think briefly about its topic, then write "
        "'Topic: ' followed by exactly one of World, Sports, Business, Sci/Tech.\n\n"
        f"News: {text}\nThought: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        preamble_regex=r"[\s\S]*",
        trigger="Topic: ",
        answer_regex=r"World|Sports|Business|Sci/Tech",
        max_answer_tokens=8,
        max_free_tokens=64,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 18, "end_line": 28, "content": _CONTENT},
]
