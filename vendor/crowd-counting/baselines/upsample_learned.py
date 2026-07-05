"""Good baseline for cv-count-upsample: learned UPSAMPLING decoder (finer output).

A learned transposed-conv decoder upsamples the stride-8 features to stride 4 (a finer
density map) with a refinement conv. Higher output resolution lets nearby objects
occupy separate cells, so dense scenes are resolved better -> lower counting MAE. This
mirrors the finer-resolution decoders of TEDnet / SANet, which report lower MAE from
higher-quality maps. (The count is still the map's integral; DENSITY_SCALE unchanged.)
"""


def build_decoder(cin):
    import torch.nn as nn

    class UpDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.ConvTranspose2d(cin, cin, 2, stride=2)
            self.refine = nn.Sequential(
                nn.Conv2d(cin, cin, 3, padding=1), nn.ReLU(True))

        def forward(self, x):
            return self.refine(self.up(x))

    return UpDecoder()
