"""Mid-edit operations for the meta-rl task.

Applied to the oyster workspace after pre_edit, before the agent starts.
Creates:
  - custom_encoder.py: Editable context encoder module
  - launch_custom.py: Fixed experiment launcher using custom encoder
"""

from pathlib import Path

_ENCODER_TEMPLATE = (Path(__file__).parent / "custom_encoder_template.py").read_text()
_LAUNCHER_TEMPLATE = (Path(__file__).parent / "launch_custom_template.py").read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "oyster/custom_encoder.py",
        "content": _ENCODER_TEMPLATE,
    },
    {
        "op": "create",
        "file": "oyster/launch_custom.py",
        "content": _LAUNCHER_TEMPLATE,
    },
]
