"""Strong baseline: generous encoder-side input budget (reads every source in full)."""


def build_max_input_tokens() -> int:
    return 160
