import torch
import torch.nn as nn
import torch_geometric.nn as nng
from layers.Basic import MLP


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'GraphSAGE'

        self.nb_hidden_layers = args.n_layers
        self.size_hidden_layers = args.n_hidden
        self.bn_bool = True
        self.activation = nn.ReLU()

        self.encoder = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden, n_layers=0, res=False,
                           act=args.act)
        self.decoder = MLP(args.n_hidden, args.n_hidden * 2, args.out_dim, n_layers=0, res=False, act=args.act)

        self.in_layer = nng.SAGEConv(
            in_channels=args.n_hidden,
            out_channels=self.size_hidden_layers
        )

        self.hidden_layers = nn.ModuleList()
        for n in range(self.nb_hidden_layers - 1):
            self.hidden_layers.append(nng.SAGEConv(
                in_channels=self.size_hidden_layers,
                out_channels=self.size_hidden_layers
            ))

        self.out_layer = nng.SAGEConv(
            in_channels=self.size_hidden_layers,
            out_channels=self.size_hidden_layers
        )

        if self.bn_bool:
            self.bn = nn.ModuleList()
            for n in range(self.nb_hidden_layers):
                self.bn.append(nn.BatchNorm1d(self.size_hidden_layers, track_running_stats=False))

    def forward(self, x, fx, T=None, geo=None):
        if geo is None:
            raise ValueError('Please provide edge index for Graph Neural Networks')
        z, edge_index = torch.cat((x, fx), dim=-1).squeeze(0), geo
        z = self.encoder(z)
        z = self.in_layer(z, edge_index)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)

        for n in range(self.nb_hidden_layers - 1):
            z = self.hidden_layers[n](z, edge_index)
            if self.bn_bool:
                z = self.bn[n + 1](z)
            z = self.activation(z)
        z = self.out_layer(z, edge_index)
        z = self.decoder(z)
        return z.unsqueeze(0)
