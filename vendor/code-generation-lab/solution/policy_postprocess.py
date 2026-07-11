"""Agent-editable prompt and output policy for codegen-prompt-postprocess.

`build_prompt(problem)` receives exactly `prompt` and `entry_point` and returns a
non-empty user message of at most 1024 chat-template tokens.
`postprocess(raw_text, entry_point)` receives one of three deterministic views of
the same completion and returns Python source. All three views are scored.
"""
from __future__ import annotations

# >>> EDITABLE REGION (the agent writes the functions below) >>>
def build_prompt(problem):
    return problem["prompt"]


def postprocess(raw_text, entry_point):
    return raw_text
# <<< END EDITABLE REGION <<<
