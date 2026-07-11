"""Long-preamble candidate: use class descriptions before FSM label emission.

Reference surface: vendor/constrained-decoding-lab/solution/decoder_choice_reasoning.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_choice_reasoning.py"

_CONTENT = r'''def build_decoder(text: str, labels, tok):
    prompt = (
        "You are a news topic classifier. Read the snippet, reason about which "
        "of World (international/politics), Sports (games/athletes), Business "
        "(companies/markets), Sci/Tech (science/technology) it fits, then write "
        "'Topic: ' followed by exactly that one label.\n\n"
        f"News: {text}\nReasoning: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        preamble_regex=r"[\s\S]*",
        trigger="Topic: ",
        answer_regex=r"World|Sports|Business|Sci/Tech",
        max_answer_tokens=8,
        max_free_tokens=192,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 18, "end_line": 28, "content": _CONTENT},
]
