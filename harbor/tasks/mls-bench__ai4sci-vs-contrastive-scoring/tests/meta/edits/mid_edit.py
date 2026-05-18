"""Mid-edit: create the editable template and fixed wrappers for vs-contrastive-scoring."""

from pathlib import Path

_DIR = Path(__file__).parent

_SCORING_TEMPLATE = (_DIR / "custom_template.py").read_text()
_MODEL_WRAPPER = (_DIR / "custom_vs_model.py").read_text()
_LOSS_WRAPPER = (_DIR / "custom_vs_loss.py").read_text()

OPS = [
    # 1. Create the editable scoring module
    {
        "op": "create",
        "file": "HypSeek/unimol/custom_scoring.py",
        "content": _SCORING_TEMPLATE,
    },
    # 2. Create the model wrapper (FIXED — imports CustomScoring)
    {
        "op": "create",
        "file": "HypSeek/unimol/models/custom_vs_model.py",
        "content": _MODEL_WRAPPER,
    },
    # 3. Create the loss wrapper (FIXED — delegates to CustomScoring)
    {
        "op": "create",
        "file": "HypSeek/unimol/losses/custom_vs_loss.py",
        "content": _LOSS_WRAPPER,
    },
]
