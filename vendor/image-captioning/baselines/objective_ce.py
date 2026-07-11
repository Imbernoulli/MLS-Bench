"""Plain cross-entropy training objective baseline.

Standard next-token cross-entropy over the caption tokens (pad ignored). Trains
the mapping network to maximise the likelihood of the reference caption. The
reference / weak baseline for the caption-objective task.
"""
import torch.nn.functional as F


def caption_loss(logits, targets, pad_id, prefix_len):
    B, L, V = logits.shape
    T = targets.shape[1]
    logits_cap = logits[:, prefix_len - 1: prefix_len - 1 + T, :]
    return F.cross_entropy(
        logits_cap.reshape(-1, V), targets.reshape(-1), ignore_index=pad_id
    )
