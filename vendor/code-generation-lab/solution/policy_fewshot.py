"""Agent-editable demonstration policy for codegen-fewshot-priming.

`fewshot()` is called exactly once for the complete evaluation and receives no
problem data. It returns a run-wide demonstration prefix, possibly empty, capped
at 256 tokenizer tokens. The harness appends each problem after this fixed prefix.
"""
from __future__ import annotations

# >>> EDITABLE REGION (the agent writes the function below) >>>
def fewshot():
    return ""
# <<< END EDITABLE REGION <<<
