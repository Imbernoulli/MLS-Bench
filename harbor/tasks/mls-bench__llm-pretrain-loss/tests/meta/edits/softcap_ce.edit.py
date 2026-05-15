"""Softcapped cross-entropy baseline (medium).

Applies a sigmoid-based softcap to logits before computing cross-entropy,
preventing extreme logit values. Inspired by Gemma 2 and modded-nanogpt.

Reference: KellerJordan/modded-nanogpt PR #199
"""

_FILE = "nanoGPT/custom_pretrain.py"

_SOFTCAP_CE = """\
def compute_loss(logits, targets):
    \"\"\"Cross-entropy with logit softcapping via sigmoid.\"\"\"
    # Softcap: maps logits through A * sigmoid((logits + B) / C)
    # Prevents extreme logit magnitudes while preserving ranking
    # Constants from modded-nanogpt PR #199
    A, B, C = 23.0, 5.0, 7.5
    capped_logits = A * torch.sigmoid((logits.float() + B) / C)
    return F.cross_entropy(
        capped_logits.view(-1, capped_logits.size(-1)), targets.view(-1),
        ignore_index=-1
    )
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 188,
        "end_line": 191,
        "content": _SOFTCAP_CE,
    },
]
