"""Mid-edit operations for the marl-centralized-critic task.

Applied to the epymarl workspace after pre_edit, before the agent starts.
Creates custom_critic.py — the agent's editable centralized critic — from
custom_template.py. The pre_edit already registers "custom_critic" in
critics/__init__.py via try/except, so once this file exists the registry
picks it up at import time.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "epymarl/src/modules/critics/custom_critic.py",
        "content": _CUSTOM_PY,
    },
]
