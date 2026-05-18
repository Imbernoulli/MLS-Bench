"""Z-loss regularization baseline (strongest).

Adds a penalty on the log of the sum of exponentials (log-partition function),
which stabilizes training by preventing logit magnitudes from growing too large.

Reference: Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways" (2022)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_Z_LOSS = """\
def compute_loss(logits, targets):
    \"\"\"Cross-entropy with z-loss regularization.\"\"\"
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)
    ce_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)
    # Z-loss: penalize large log-partition values
    log_z = torch.logsumexp(flat_logits, dim=-1)
    # Only compute z-loss for non-ignored positions
    mask = flat_targets != -1
    z_loss = (log_z[mask] ** 2).mean()
    return ce_loss + 1e-4 * z_loss
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 188,
        "end_line": 191,
        "content": _Z_LOSS,
    },
]
