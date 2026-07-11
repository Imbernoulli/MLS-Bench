"""STRONG baseline (cd-forced-choice): constrain directly to the label set.

Pass the fixed label set as `choices`. The harness commits to the label with
the highest length-normalized logprob under the model -> ALWAYS structurally
valid, and correctness tracks the model's real preference over the labels. This
is the classic "auto-correcting" benefit of constraining a classifier to its
label set: validity 1.0 and accuracy well above majority class.
Reference surface: vendor/constrained-decoding-lab/solution/decoder_choice.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_choice.py"

_CONTENT = '''def build_decoder(text: str, labels, tok):
    label_list = ", ".join(labels)
    prompt = (
        "You are a news topic classifier. Read the news snippet and reply with "
        f"exactly one topic from this list: {label_list}. "
        "World = international/politics, Sports = games and athletes, "
        "Business = companies/markets/economy, Sci/Tech = science and "
        "technology.\\n\\n"
        f"News: {text}\\nTopic:"
    )
    return common.DecodeSpec(
        prompt=prompt,
        choices=list(labels),
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 48, "content": _CONTENT},
]
