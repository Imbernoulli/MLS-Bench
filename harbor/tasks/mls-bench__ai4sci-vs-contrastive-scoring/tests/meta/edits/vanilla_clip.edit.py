"""Vanilla CLIP contrastive loss baseline (DrugCLIP-style).

Euclidean L2-normalized dot product with symmetric in-batch softmax.
This is the simplest contrastive approach for virtual screening.

Reference: DrugCLIP (NeurIPS 2023) — Gao et al.
    vendor/external_packages/HypSeek/unimol/losses/three_hybrid_loss.py (simplified)
"""

_FILE = "HypSeek/unimol/custom_scoring.py"

_CONTENT = open(__file__.replace("vanilla_clip.edit.py", "custom_template.py")).read()

# The default template IS the vanilla CLIP baseline — no changes needed.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 1,
        "end_line": -1,
        "content": _CONTENT,
    },
]
