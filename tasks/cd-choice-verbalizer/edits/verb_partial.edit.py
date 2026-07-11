"""MID baseline (cd-choice-verbalizer): compact code verbalizers with a complete one-to-one label map.

Reference surface: vendor/constrained-decoding-lab/solution/decoder_choice_verbalizer.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_choice_verbalizer.py"

_CONTENT = r'''def build_decoder(text: str, labels, tok):
    prompt = (
        "Classify the topic using this codebook: A = World, B = Sports, "
        "C = Business, D = Sci/Tech. Reply with one code.\n\n"
        f"News: {text}\nTopic:"
    )
    return common.DecodeSpec(
        prompt=prompt,
        choices=["A", "B", "C", "D"],
        choice_labels=list(labels),
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 19, "end_line": 29, "content": _CONTENT},
]
