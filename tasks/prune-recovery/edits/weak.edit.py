"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/recovery.py"

_CONTENT = 'def recover(model, masked_finetune, cfg):\n    # Consume the full fixed budget with a conservative learning rate.\n    masked_finetune(epochs=cfg["epochs"], lr=min(float(cfg["lr"]), 0.0025))\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 16, "end_line": 18, "content": _CONTENT},
]
