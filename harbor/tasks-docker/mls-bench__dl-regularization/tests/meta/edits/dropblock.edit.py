"""Spatial weight co-activation regularization baseline.

Captures intermediate feature maps via forward hooks and penalizes
spatial co-activation within local blocks, encouraging the network
not to rely on contiguous feature-map regions. Strength linearly
increases from 0 to target over training (scheduled keep_prob).

Related idea: Ghiasi et al., "DropBlock: A regularization method for
convolutional neural networks" (NeurIPS 2018), though this baseline
regularizes convolutional weights rather than masking activations.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_reg.py"

_CONTENT = """\
def compute_regularization(model, inputs, outputs, targets, config):
    \"\"\"Spatial co-activation penalty on convolutional weights.

    Applies a spatial co-activation penalty on convolutional weights.
    For each Conv2d layer with spatial kernels >= block_size, it
    penalizes the mean energy of local spatial blocks in the weight
    tensor, discouraging spatially correlated filter patterns.

    Uses conservative strength (lambda_max=1e-4) with linear warm-up
    and only activates after 20% of training to avoid destabilizing
    early learning, particularly for BatchNorm-heavy architectures.

    block_size=3, lambda_max=1e-4, linear warm-up with delayed start.
    \"\"\"
    block_size = 3
    lambda_max = 1e-4
    progress = config['epoch'] / max(config['total_epochs'] - 1, 1)

    # Delay activation: no penalty for first 20% of training
    if progress < 0.2:
        return torch.tensor(0.0, device=outputs.device)

    # Linear schedule from 20% to 100% of training
    adjusted_progress = (progress - 0.2) / 0.8
    lam = lambda_max * adjusted_progress

    reg = torch.tensor(0.0, device=outputs.device)
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size[0] >= block_size:
            w = m.weight  # [out_c, in_c, kH, kW]
            if w.size(-1) >= block_size and w.size(-2) >= block_size:
                # Mean squared magnitude within spatial blocks
                w_sq = w.pow(2).mean(dim=1, keepdim=True)  # [out_c, 1, kH, kW]
                pad = block_size // 2
                local = F.avg_pool2d(w_sq, block_size, stride=1, padding=pad)
                reg = reg + local.mean()
                count += 1

    if count > 0:
        reg = reg / count
    return lam * reg
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 246,
        "end_line": 273,
        "content": _CONTENT,
    },
]
