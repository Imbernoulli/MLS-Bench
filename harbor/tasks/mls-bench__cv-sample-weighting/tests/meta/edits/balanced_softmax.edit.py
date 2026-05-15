"""Balanced Softmax-inspired log-prior weighting baseline.

Derives per-class weights from the Balanced Softmax calibration principle:
the log-prior shift log(pi_c) is expressed through the task's class-weight API
as log-frequency weights, avoiding the extreme ratios of raw inverse frequency.

Reference: Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual
Recognition" (NeurIPS 2020)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_weighting.py"

_CONTENT = """\
def compute_class_weights(class_counts, num_classes, config):
    \"\"\"Balanced Softmax-inspired log-prior weighting (Ren et al., NeurIPS 2020).

    Balanced Softmax adjusts logits by subtracting log(pi_c). In this task's
    class-weighting interface, we use weights derived from the log-frequency gap:
    weight[c] = 1 + alpha * (log(n_max) - log(n_c)), alpha chosen so that
    the max/min weight ratio is moderate (~5:1).
    This provides a log-scale reweighting that is gentler than inverse
    frequency but still informed by class prior.
    \"\"\"
    log_counts = torch.log(class_counts.float())
    log_max = log_counts.max()
    # Log-gap weights: classes further from max count get higher weight
    gap = log_max - log_counts  # 0 for most frequent, log(ratio) for rarest
    # Scale so the rarest class gets weight ~5x the most frequent
    max_gap = gap.max()
    if max_gap > 0:
        weights = 1.0 + 4.0 * (gap / max_gap)
    else:
        weights = torch.ones(num_classes)
    # Normalize so weights sum to num_classes
    weights = weights / weights.sum() * num_classes
    return weights
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 164,
        "end_line": 195,
        "content": _CONTENT,
    },
]
