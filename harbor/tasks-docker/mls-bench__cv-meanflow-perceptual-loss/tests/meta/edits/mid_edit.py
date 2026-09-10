"""Mid-edit operations for cv-meanflow-perceptual-loss task.

Applied to the alphaflow-main workspace after mid-edit of cv-meanflow-training,
before the agent starts. Replaces lines 429-439 in custom_train_perceptual.py with
a stub so the agent must implement their own perceptual loss function.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_TEMPLATE = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "alphaflow-main/custom_train_perceptual.py",
        "content": _CUSTOM_TEMPLATE,
    },
]
