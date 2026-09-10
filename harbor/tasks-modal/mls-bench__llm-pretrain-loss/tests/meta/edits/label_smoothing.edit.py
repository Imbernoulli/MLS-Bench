"""Label smoothing cross-entropy baseline (basic).

Distributes a small fraction of probability mass uniformly across all tokens,
reducing overconfidence and improving generalization. Uses a task-local
eps=0.05 setting for this short-run language-modeling harness.

Reference: Szegedy et al., "Rethinking the Inception Architecture" (2016)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_LABEL_SMOOTHING = """\
def compute_loss(logits, targets):
    \"\"\"Cross-entropy with label smoothing (eps=0.05) during training only.

    Label smoothing is applied only when gradients are enabled (training).
    During evaluation (@torch.no_grad()), standard cross-entropy is used
    so that val_loss remains comparable across methods.
    \"\"\"
    smoothing = 0.05 if torch.is_grad_enabled() else 0.0
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=-1, label_smoothing=smoothing
    )
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 188,
        "end_line": 191,
        "content": _LABEL_SMOOTHING,
    },
]
