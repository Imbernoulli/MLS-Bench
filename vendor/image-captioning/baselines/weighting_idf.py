"""Design the Per-Token Loss Weighting — weak baseline (idf).

Reference implementation for the caption-token-weighting surface (token_weights). See tasks/caption-token-weighting/edits/idf.edit.py.
"""
import torch


def token_weights(targets, pad_id, ctx):
    # Up-weight RARE content tokens by raw caption-IDF (ctx["idf"]: token ->
    # log(N/df)), mean-normalised per batch; pad positions are zeroed.
    idf = ctx.get("idf", {})
    if not idf:
        return torch.ones_like(targets, dtype=torch.float)
    hi = int(targets.max().item()) + 1
    # NOTE: lut must live on targets' device (targets are CUDA in the harness).
    lut = torch.tensor([idf.get(i, 0.0) for i in range(hi)],
                       dtype=torch.float, device=targets.device)
    w = lut[targets.clamp(min=0)]
    m = (targets != pad_id).float()
    w = w * m
    mean = (w.sum() / m.sum().clamp_min(1.0)).clamp_min(1e-6)
    return w / mean
