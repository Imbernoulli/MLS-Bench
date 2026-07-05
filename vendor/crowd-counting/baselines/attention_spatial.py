"""Good baseline for cv-count-attention: spatial ATTENTION gate.

A lightweight spatial-attention module predicts a per-pixel gate in [0,1] and multiplies
the features by it, learning to SUPPRESS the unannotated distractor clutter and focus
density mass on real objects -> lower counting MAE. This mirrors attention-based crowd
counters (SCAR, ADCrowdNet, SFANet) that add spatial/segmentation attention to a VGG
backbone.
"""


def build_attention(cin):
    import torch.nn as nn

    class SpatialAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Sequential(
                nn.Conv2d(cin, cin // 2, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(cin // 2, 1, 1), nn.Sigmoid())

        def forward(self, x):
            return x * self.gate(x)   # per-pixel gated features

    return SpatialAttention()
