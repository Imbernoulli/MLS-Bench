"""Good baseline for cv-count-multiscale: MULTI-SCALE context aggregation (CAN-style pyramid). Handles scale variation -> lower MAE. Ref: vendor/crowd-counting/baselines/multiscale_context.py"""

_FILE = "crowd-counting/solution/multiscale.py"

_CONTENT = '    import torch\n    import torch.nn as nn\n    import torch.nn.functional as F\n\n    class ContextModule(nn.Module):\n        def __init__(self, scales=(2, 4, 8)):\n            super().__init__()\n            self.scales = scales\n            self.projs = nn.ModuleList(\n                [nn.Conv2d(cin, cin, 1) for _ in scales])\n            self.fuse = nn.Conv2d(cin * (len(scales) + 1), cin, 1)\n\n        def forward(self, x):\n            h, w = x.shape[-2:]\n            feats = [x]\n            for s, proj in zip(self.scales, self.projs):\n                p = F.adaptive_avg_pool2d(x, output_size=max(1, min(h, w) // s))\n                p = proj(p)\n                feats.append(F.interpolate(p, size=(h, w), mode="bilinear",\n                                           align_corners=False))\n            # RESIDUAL fusion: add multi-scale context back to the base features so the\n            # module starts near-identity and converges fast.\n            return F.relu(x + self.fuse(torch.cat(feats, dim=1)))\n\n    return ContextModule()'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 41, "end_line": 43, "content": _CONTENT},
]
