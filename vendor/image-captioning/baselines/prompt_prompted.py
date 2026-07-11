"""Design the Text Prompt / Caption Formatting — weak baseline (prompted).

Reference implementation for the caption-prompt-format surface (prompt_prefix). See tasks/caption-prompt-format/edits/prompted.edit.py.
"""

def prompt_prefix():
    # Naive fixed visual prompt shipped with the scaffold.
    return "a photo of "


def format_caption(c):
    # Verbatim caption (kept as-is so the prompt is the isolated variable).
    return c.strip()
