"""Mid-edit: create the editable scaling-law template and observed train trials."""

from pathlib import Path

_ROOT = Path(__file__).parent.parent
_TEMPLATE = Path(__file__).parent / "custom_template.py"
_CONTENT = _TEMPLATE.read_text()

_OBSERVED_TRIALS = {
    "scaling-law-lab/observed_trials/sld_vocab_train.jsonl": (
        _ROOT / "data" / "sld_vocab_train.jsonl"
    ).read_text(),
    "scaling-law-lab/observed_trials/sld_lrbsz_train.jsonl": (
        _ROOT / "data" / "sld_lrbsz_train.jsonl"
    ).read_text(),
    "scaling-law-lab/observed_trials/sld_dataconstrained_train.jsonl": (
        _ROOT / "data" / "sld_dataconstrained_train.jsonl"
    ).read_text(),
}

OPS = [
    {
        "op": "create",
        "file": "scaling-law-lab/custom_scaling_law.py",
        "content": _CONTENT,
    },
    *[
        {
            "op": "create",
            "file": path,
            "content": content,
        }
        for path, content in _OBSERVED_TRIALS.items()
    ],
]
