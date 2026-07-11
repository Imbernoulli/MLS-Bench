"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/recovery.py"

_CONTENT = 'def recover(model, masked_finetune, cfg):\n    # Standard 160-epoch recovery fine-tune of the surviving weights (mask re-applied).\n    masked_finetune(epochs=cfg.get("epochs", 160), lr=cfg.get("lr", 1e-2))\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 16, "end_line": 18, "content": _CONTENT},
]
