"""Agent-editable prompt policy for codegen-docstring-design.

`build_prompt(problem)` receives exactly two fields: `prompt` and `entry_point`.
It must return a non-empty user message. After applying the frozen chat template,
the verifier rejects messages longer than 1024 tokenizer tokens.
"""
from __future__ import annotations

# >>> EDITABLE REGION (the agent writes the function below) >>>
def build_prompt(problem):
    return problem["prompt"]
# <<< END EDITABLE REGION <<<
