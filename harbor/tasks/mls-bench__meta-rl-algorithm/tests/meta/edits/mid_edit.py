"""Mid-edit operations for the meta-rl-algorithm task.

Applied to the oyster workspace after pre_edit, before the agent starts.
Creates:
  - custom_meta_rl.py: Self-contained meta-RL algorithm template
"""

from pathlib import Path

_TEMPLATE = (Path(__file__).parent / "custom_meta_rl_template.py").read_text()

# -- Mid-edit operations --
OPS = [
    {
        "op": "create",
        "file": "oyster/custom_meta_rl.py",
        "content": _TEMPLATE,
    },
]
