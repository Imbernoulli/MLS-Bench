"""Mid-edit operations for security-adversarial-attack-white-box-linf.

Creates bench/ evaluation scaffold inside torchattacks workspace:
  bench/run_eval.py       - evaluation harness
  bench/custom_attack.py  - agent-editable attack entrypoint
"""

from pathlib import Path

_HERE = Path(__file__).parent

OPS = [
    {
        "op": "create",
        "file": "torchattacks/bench/run_eval.py",
        "content": (_HERE / "run_eval_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "torchattacks/bench/custom_attack.py",
        "content": (_HERE / "custom_attack_template.py").read_text(),
    },
]
