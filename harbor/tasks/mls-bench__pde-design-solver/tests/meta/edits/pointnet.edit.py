"""PointNet baseline — rigorous codebase edit ops.
Replaces Custom.py model body with PointNet implementation
(faithful to vendor/external_packages/Neural-Solver-Library/models/PointNet.py).
Sets n_hidden=16 to match the paper-faithful script
vendor/external_packages/Neural-Solver-Library/scripts/DesignBench/car/PointNet.sh.
"""

_FILE = "Neural-Solver-Library/models/Custom.py"

_CONTENT = """\
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from layers.Embedding import unified_pos_embedding
from layers.Basic import MLP


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Custom'

        self.in_block = MLP(args.n_hidden, args.n_hidden * 2, args.n_hidden * 2, n_layers=0, res=False,
                            act=args.act)
        self.max_block = MLP(args.n_hidden * 2, args.n_hidden * 8, args.n_hidden * 32, n_layers=0, res=False,
                             act=args.act)

        self.out_block = MLP(args.n_hidden * (2 + 32), args.n_hidden * 16, args.n_hidden * 4, n_layers=0, res=False,
                             act=args.act)

        self.encoder = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden, n_layers=0, res=False,
                           act=args.act)
        self.decoder = MLP(args.n_hidden, args.n_hidden * 2, args.out_dim, n_layers=0, res=False, act=args.act)

        self.fcfinal = nn.Linear(args.n_hidden * 4, args.n_hidden)

    def forward(self, x, fx, T=None, geo=None):
        if geo is None:
            raise ValueError('Please provide edge index for Graph Neural Networks')
        z, batch = torch.cat((x, fx), dim=-1).float().squeeze(0), torch.zeros([x.shape[1]]).cuda().long()

        z = self.encoder(z)
        z = self.in_block(z)

        global_coef = self.max_block(z)
        global_coef = nng.global_max_pool(global_coef, batch=batch)
        nb_points = torch.zeros(global_coef.shape[0], device=z.device)

        for i in range(batch.max() + 1):
            nb_points[i] = (batch == i).sum()
        nb_points = nb_points.long()
        global_coef = torch.repeat_interleave(global_coef, nb_points, dim=0)

        z = torch.cat([z, global_coef], dim=1)
        z = self.out_block(z)
        z = self.fcfinal(z)
        z = self.decoder(z)

        return z.unsqueeze(0)
"""

_CONFIG_OVERRIDES = "CONFIG_OVERRIDES = {'n_hidden': 16}\n"

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 74,
        "end_line": 74,
        "content": _CONFIG_OVERRIDES,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 1,
        "end_line": 64,
        "content": _CONTENT,
    },
]
