"""Agent-editable output policy for codegen-output-extract.

`extract(raw_text, entry_point)` receives only the raw model completion and the
required function name. It must return Python source text. It never receives the
problem prompt, setup code, or any assertions.
"""
from __future__ import annotations

# >>> EDITABLE REGION (the agent writes the function below) >>>
def extract(raw_text, entry_point):
    return raw_text
# <<< END EDITABLE REGION <<<
