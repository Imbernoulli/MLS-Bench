import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from layers.Basic import MLP
from layers.Embedding import unified_pos_embedding


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Custom'
        self.args = args

        # Input encoding: spatial coords (3D) + features (7D) -> hidden_dim
        self.encoder = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                           n_layers=0, res=False, act=args.act)

        # TODO: Define your custom model architecture here.
        # This model operates on UNSTRUCTURED 3D point clouds (car meshes).
        # Each mesh has variable number of points (~5000-10000).
        # Batch size is always 1.
        # args.geotype = 'unstructured'
        #
        # You can use:
        # - Graph neural networks (edge_index available via geo parameter)
        # - Point cloud methods (PointNet-style global pooling)
        # - Transformer-based approaches (self-attention on all points)
        # - Physics-aware methods (Transolver-style slicing)
        #
        # Reference models: PointNet (global pooling), GraphSAGE (message passing),
        # Graph_UNet (multi-scale graph), Transolver (physics attention)

        # Output projection: hidden_dim -> out_dim (velocity xyz + pressure)
        self.decoder = MLP(args.n_hidden, args.n_hidden * 2, args.out_dim,
                           n_layers=0, res=False, act=args.act)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx, T=None, geo=None):
        # x: (1, N, 3) spatial coords, fx: (1, N, 7) features
        # geo: edge_index tensor if graph connectivity is needed (can be None)
        z = torch.cat((x, fx), dim=-1)  # (1, N, 10)
        z = self.encoder(z)  # (1, N, n_hidden)

        # TODO: Implement your custom forward pass here.
        # Input z has shape (1, N, n_hidden) where N varies per mesh.
        # Output should have shape (1, N, out_dim) where out_dim=4.

        out = self.decoder(z)  # (1, N, 4)
        return out


# =====================================================================
# CONFIG_OVERRIDES: per-method hyperparameter overrides
# =====================================================================
# Override widths/capacities that depend on the model family.
# Allowed keys: n_hidden (int), slice_num (int).
# Defaults follow the baseline shell scripts (n_hidden=128, slice_num=32),
# matching the GraphSAGE configuration in Neural-Solver-Library/scripts/DesignBench/car/.
# Other paper settings (for reference): PointNet=16, Transolver=256, Graph_UNet=16, GNOT=256.
CONFIG_OVERRIDES = {}


# =====================================================================
# FIXED: Parameter budget check — do not modify below this line
# =====================================================================
_orig_init = Model.__init__

def _patched_init(self, args):
    _orig_init(self, args)
    _total = sum(p.numel() for p in self.parameters())
    print(f"Total params: {_total:,} (task budget enforced by budget_check.py)")

Model.__init__ = _patched_init
