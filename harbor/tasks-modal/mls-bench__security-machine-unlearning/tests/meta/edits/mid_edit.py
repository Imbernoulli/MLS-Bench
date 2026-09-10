"""Mid-edit operations for security-machine-unlearning."""

from pathlib import Path

_HERE = Path(__file__).parent

OPS = [
    {
        "op": "create",
        "file": "pytorch-vision/bench/unlearning/run_unlearning.py",
        "content": (_HERE / "run_unlearning_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "pytorch-vision/bench/unlearning/custom_unlearning.py",
        "content": (_HERE / "custom_unlearning_template.py").read_text(),
    },
]
