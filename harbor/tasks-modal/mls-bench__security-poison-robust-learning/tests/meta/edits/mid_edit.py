"""Mid-edit operations for security-poison-robust-learning."""

from pathlib import Path

_HERE = Path(__file__).parent

OPS = [
    {
        "op": "create",
        "file": "pytorch-vision/bench/poison/run_poison_robust.py",
        "content": (_HERE / "run_poison_robust_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "pytorch-vision/bench/poison/custom_robust_loss.py",
        "content": (_HERE / "custom_robust_loss_template.py").read_text(),
    },
]
