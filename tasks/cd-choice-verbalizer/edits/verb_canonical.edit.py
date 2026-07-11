"""STRONG baseline (cd-choice-verbalizer): choices == the canonical gold labels (exact), matching the proven constrained classifier.

Reference surface: vendor/constrained-decoding-lab/solution/decoder_choice_verbalizer.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_choice_verbalizer.py"

_CONTENT = r'''def build_decoder(text: str, labels, tok):
    label_list = ", ".join(labels)
    prompt = (
        "You are a news topic classifier. Read the news snippet and reply with "
        f"exactly one topic from this list: {label_list}. "
        "World = international/politics, Sports = games and athletes, "
        "Business = companies/markets/economy, Sci/Tech = science and "
        "technology.\n\n"
        f"News: {text}\nTopic:"
    )
    return common.DecodeSpec(
        prompt=prompt,
        choices=list(labels),
        choice_labels=list(labels),
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 19, "end_line": 29, "content": _CONTENT},
]
