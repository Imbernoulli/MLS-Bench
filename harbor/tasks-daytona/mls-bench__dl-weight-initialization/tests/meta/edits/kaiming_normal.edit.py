"""Kaiming/He normal initialization baseline.

Standard initialization for networks with ReLU activations. Draws Conv2d weights
from N(0, sqrt(2/fan_out)) to preserve variance in the forward pass.

Reference: He et al., "Delving Deep into Rectifiers: Surpassing Human-Level
Performance on ImageNet Classification" (ICCV 2015)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_init.py"

_CONTENT = """\
def initialize_weights(model, config):
    \"\"\"Kaiming/He normal initialization (fan_out, ReLU).

    Conv2d: N(0, sqrt(2/fan_out)) — preserves forward-pass variance with ReLU.
    BatchNorm2d: weight=1, bias=0.
    Linear: N(0, sqrt(2/fan_in)).
    \"\"\"
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
