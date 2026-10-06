import torch as th
import torch.nn.functional as F
from torch import nn
from dgl.utils import expand_as_pair
import dgl.function as fn

class CIGConv(nn.Module):

    def __init__(self, input_dim, output_dim, drop=0.1):
        super(CIGConv, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(drop),
            nn.LeakyReLU(),
            nn.BatchNorm1d(output_dim)
        )

    def message(self, edges):
        return {'m': F.relu(edges.src['hn'] + edges.data['he'])}

    def forward(self, graph, node_feat, edge_feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(node_feat, graph)
            graph.srcdata['hn'] = feat_src
            graph.edata['he'] = edge_feat
            graph.update_all(self.message, fn.sum('m', 'neigh'))
            rst = feat_dst + graph.dstdata['neigh']
            rst = self.mlp(rst)

            return rst
