"""Mid-edit operations for security-backdoor-defense."""

from pathlib import Path

_HERE = Path(__file__).parent

OPS = [
    {
        "op": "create",
        "file": "pytorch-vision/bench/backdoor/run_backdoor_defense.py",
        "content": (_HERE / "run_backdoor_defense_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "pytorch-vision/bench/backdoor/custom_backdoor_defense.py",
        "content": (_HERE / "custom_backdoor_defense_template.py").read_text(),
    },
]
