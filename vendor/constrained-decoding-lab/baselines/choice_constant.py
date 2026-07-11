"""DEGENERATE baseline (cd-forced-choice): always emit one constant label.

Constrain to a SINGLE label (the first one) regardless of the input. Output is
always structurally VALID (validity = 1.0) but the classifier is meaningless, so
accuracy collapses to that label's frequency (majority-/one-class rate). This
baseline exists to prove the metric is un-gameable: a constant-valid decoder
must score LOW, well below the real constrained classifier.
Reference surface: vendor/constrained-decoding-lab/solution/decoder_choice.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_choice.py"

_CONTENT = '''def build_decoder(text: str, labels, tok):
    # Ignore the text entirely; always return the first label.
    prompt = (
        f"Text: {text}\\nLabel: "
    )
    return common.DecodeSpec(
        prompt=prompt,
        choices=[list(labels)[0]],
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 48, "content": _CONTENT},
]
