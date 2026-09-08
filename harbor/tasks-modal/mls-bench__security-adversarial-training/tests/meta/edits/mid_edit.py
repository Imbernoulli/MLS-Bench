"""Mid-edit operations for security-adversarial-training.

Creates bench/ training scaffold inside torchattacks workspace:
  bench/run_adv_train.py    - training and evaluation harness
  bench/models.py           - model architecture definitions
  bench/custom_adv_train.py - agent-editable adversarial training method
"""

from pathlib import Path

_HERE = Path(__file__).parent

OPS = [
    {
        "op": "create",
        "file": "torchattacks/bench/run_adv_train.py",
        "content": (_HERE / "run_adv_train_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "torchattacks/bench/models.py",
        "content": (_HERE / "models_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "torchattacks/bench/custom_adv_train.py",
        "content": (_HERE / "custom_adv_train_template.py").read_text(),
    },
]
