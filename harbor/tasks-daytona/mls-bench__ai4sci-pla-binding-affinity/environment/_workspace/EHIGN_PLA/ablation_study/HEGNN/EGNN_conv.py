"""
A DGL version of EGNN. The original EGNN can be found in https://github.com/vgsatorras/egnn
According to the EGNN paper, updating coordinates is unnecessary for those tasks that do not need to predict vectors.
"""
import torch as th
from torch import nn
import dgl.function as fn

class EGNNConv(nn.Module):

    def __init__(self, input_dim, hidden_dim, edge_dim=1):
        super(EGNNConv, self).__init__()

        self.edge_mlp_u = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU())
        self.edge_mlp_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU())
        self.edge_mlp_e = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU())
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))

    def edge_update(self, edges):
        # This is equal to the concat operation but saves computational cost greatly
        return {'m': edges.src['hn'] + edges.dst['hn'] + edges.data['he']}

    def forward(self, graph, node_feat, edge_feat):
        with graph.local_scope():
            feat_src, feat_dst = node_feat[0], node_feat[1]

            graph.srcdata['hn'] = self.edge_mlp_u(feat_src)
            graph.dstdata['hn'] = self.edge_mlp_v(feat_dst)
            graph.edata['he'] = self.edge_mlp_e(edge_feat)
            graph.update_all(self.edge_update, fn.sum('m', 'un'))

            rst = th.cat([feat_dst, graph.dstdata['un']], dim=-1)
            rst = self.node_mlp(rst)

            return rst
