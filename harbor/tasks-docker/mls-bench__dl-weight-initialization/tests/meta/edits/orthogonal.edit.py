"""Orthogonal initialization baseline.

Initializes weight matrices as (semi-)orthogonal matrices, which preserves
gradient norms during backpropagation and enables training of very deep networks.
Achieves dynamical isometry when combined with appropriate nonlinearities.

Reference: Saxe et al., "Exact solutions to the nonlinear dynamics of learning
in deep linear neural networks" (ICLR 2014)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_init.py"

_CONTENT = """\
def initialize_weights(model, config):
    \"\"\"Orthogonal initialization.

    Conv2d & Linear: orthogonal matrix (gain=sqrt(2) for ReLU).
    BatchNorm2d: weight=1, bias=0.
    \"\"\"
    gain = nn.init.calculate_gain('relu')
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=gain)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
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
