"""Fixup-inspired residual initialization baseline.

Initialization scheme designed for residual networks that accounts for the
accumulation of variance across residual blocks. Scales the last convolution
in each residual block by L^(-0.5) where L is the number of residual blocks,
keeping the variance of residual outputs controlled at initialization.

For non-residual networks (VGG), applies Kaiming normal as the base strategy
since there are no residual branches to scale.

Related method: Zhang et al., "Fixup Initialization: Residual Learning Without
Normalization" (ICLR 2019). This baseline keeps BatchNorm layers in the model.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_init.py"

_CONTENT = """\
def initialize_weights(model, config):
    \"\"\"Fixup-inspired residual scaling with zero-gamma BatchNorm.

    For ResNets: Kaiming normal for all Conv2d, then scale the last conv in
    each residual block by n_blocks^(-0.5) to control variance accumulation.
    Zero-initialize the last BN in each block (Goyal et al., 2017).
    For VGG: Kaiming normal (no residual branches to scale).
    Linear: small normal init with zero bias.
    \"\"\"
    arch = config['arch']
    is_resnet = arch.startswith('resnet')

    # Phase 1: standard Kaiming init for all layers
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Phase 2: Fixup-inspired residual branch scaling for ResNets
    if is_resnet:
        n_blocks = sum(1 for m in model.modules() if isinstance(m, BasicBlock))
        fixup_scale = n_blocks ** (-0.5)
        for m in model.modules():
            if isinstance(m, BasicBlock):
                # Scale the last conv (conv2) in each residual block
                m.conv2.weight.data.mul_(fixup_scale)
                # Zero-init the last BN so residual branch starts near identity
                nn.init.constant_(m.bn2.weight, 0)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 228,
        "end_line": 261,
        "content": _CONTENT,
    },
]
