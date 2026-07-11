"""Semantic-alias baseline (cd-choice-verbalizer): readable aliases mapped one-to-one to canonical labels.

Reference surface: vendor/constrained-decoding-lab/solution/decoder_choice_verbalizer.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_choice_verbalizer.py"

_CONTENT = r'''def build_decoder(text: str, labels, tok):
    prompt = (
        "You are a news topic classifier. Pick exactly one topic from: "
        "International, Athletics, Markets, Technology.\n\n"
        f"News: {text}\nTopic:"
    )
    return common.DecodeSpec(
        prompt=prompt,
        choices=["International", "Athletics", "Markets", "Technology"],
        choice_labels=list(labels),
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 19, "end_line": 29, "content": _CONTENT},
]
