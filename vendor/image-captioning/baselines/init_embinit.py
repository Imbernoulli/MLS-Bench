"""Design the Mapping-Network Initialization — strong baseline (embinit).

Reference implementation for the caption-mapping-init surface (init_mapping). See tasks/caption-mapping-init/edits/embinit.edit.py.
"""
import torch


def init_mapping(mapping, ctx):
    # Start the prefix near the MEAN caption-token embedding: zero the output
    # layer's weights and set its bias to the mean caption embedding tiled across
    # the prefix, so the frozen GPT-2 begins decoding in-distribution.
    mean = ctx.get("mean_cap_embed", None)
    if mean is None:
        return None
    prefix_len = ctx["prefix_len"]
    bias = mean.to(torch.float32).reshape(-1).repeat(prefix_len)
    with torch.no_grad():
        fc2 = mapping.fc2
        fc2.weight.zero_()
        fc2.bias.zero_()
        n = min(fc2.bias.numel(), bias.numel())
        fc2.bias[:n].copy_(bias[:n])
    return None
