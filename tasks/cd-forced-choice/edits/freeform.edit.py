"""WEAK baseline (cd-forced-choice): unconstrained free-form generation.

Ask for the label but do NOT constrain to the label set. The model frequently
emits extra words / punctuation / a wrong casing, so the exact-label match often
fails -> low validity AND low accuracy. This is the "malformed/verbose naive
decode" the task improves on.
Reference surface: vendor/constrained-decoding-lab/solution/decoder_choice.py
"""

_FILE = "constrained-decoding-lab/solution/decoder_choice.py"

_CONTENT = '''    label_list = ", ".join(labels)
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
        answer_regex=r"[^\\n]{1,40}",
        max_answer_tokens=16,
    )'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 48, "content": _CONTENT},
]
