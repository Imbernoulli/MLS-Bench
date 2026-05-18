"""ZerO Init (Zero-shot initialization) baseline.

Initializes Conv2d weights using partial identity (Hadamard-like) matrices
and zeros the second convolution in each residual block, enabling
identity-like signal propagation at initialization. This allows training
very deep networks without special learning rate tuning.

Reference: Zhao et al., "ZerO Initialization: Initializing Neural Networks
with only Zeros and Ones" (TMLR 2022)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_init.py"

_CONTENT = """\
def initialize_weights(model, config):
    \"\"\"ZerO-style initialization (Zhao et al., 2022).

    Phase 1: Kaiming normal for all Conv2d and Linear layers, standard BN init.
    Phase 2: For residual networks, zero-init the last BN (bn2) in each
    BasicBlock so the residual branch output is zero at init: f(x) = x + 0 = x.
    For non-residual networks (VGG), Phase 1 alone is sufficient since there
    are no residual branches to zero out.
    \"\"\"
    # Phase 1: Standard Kaiming init for all layers
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

    # Phase 2: Zero-init last BN in each residual block (ZerO init)
    # This makes residual branch output zero at init: f(x) = x + 0*g(x) = x
    # Only applies to models with BasicBlock (ResNets); VGG/MobileNet skip this.
    for m in model.modules():
        if isinstance(m, BasicBlock):
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
