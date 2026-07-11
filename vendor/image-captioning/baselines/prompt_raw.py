"""Design the Text Prompt / Caption Formatting — strong baseline (raw).

Reference implementation for the caption-prompt-format surface (prompt_prefix). See tasks/caption-prompt-format/edits/raw.edit.py.
"""

def prompt_prefix():
    # No textual prompt: condition on the visual prefix only.
    return ""


def format_caption(c):
    # Use the reference caption verbatim (just stripped of surrounding space).
    return c.strip()
