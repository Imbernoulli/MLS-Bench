"""Mid-edit operations for security-membership-inference-defense."""

from pathlib import Path

_HERE = Path(__file__).parent

OPS = [
    {
        "op": "create",
        "file": "pytorch-vision/run_membership_defense.py",
        "content": (_HERE / "run_membership_defense_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "pytorch-vision/custom_membership_defense.py",
        "content": (_HERE / "custom_membership_defense_template.py").read_text(),
    },
]
