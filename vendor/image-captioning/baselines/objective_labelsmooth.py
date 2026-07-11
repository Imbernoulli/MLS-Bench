"""Label-smoothed cross-entropy training objective baseline.

Next-token cross-entropy with label smoothing (eps=0.1): the target distribution
puts (1-eps) on the gold token and eps spread over the vocabulary. Smoothing
regularises the decoder, curbs over-confident repetition and produces better-
calibrated, more fluent captions than plain CE, a documented captioning trick
(Vaswani et al. 2017 for MT; used in most captioning systems). The stronger
baseline for the caption-objective task. Padded positions are masked out.
"""
import torch
import torch.nn.functional as F


def caption_loss(logits, targets, pad_id, prefix_len, eps=0.1):
    B, L, V = logits.shape
    T = targets.shape[1]
    logits_cap = logits[:, prefix_len - 1: prefix_len - 1 + T, :].reshape(-1, V)
    tgt = targets.reshape(-1)
    mask = tgt != pad_id
    if mask.sum() == 0:
        return logits_cap.sum() * 0.0
    logp = F.log_softmax(logits_cap, dim=-1)
    nll = -logp.gather(1, tgt.clamp(min=0).unsqueeze(1)).squeeze(1)
    smooth = -logp.mean(dim=-1)
    loss = (1 - eps) * nll + eps * smooth
    return (loss * mask).sum() / mask.sum()
