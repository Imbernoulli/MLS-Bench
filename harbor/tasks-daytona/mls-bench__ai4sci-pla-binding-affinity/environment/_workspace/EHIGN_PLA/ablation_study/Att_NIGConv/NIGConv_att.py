import torch
from torch import nn
from torch.nn import functional as F
import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import edge_softmax

class NIGConvAtt(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 bias=True):
        super(NIGConvAtt, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_dst = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_edge = nn.Linear(self._in_dst_feats, out_feats)

        self.prj_src = nn.Linear(in_feats, out_feats)
        self.prj_dst = nn.Linear(in_feats, out_feats)
        self.prj_edge = nn.Linear(in_feats, out_feats)
        self.lin_att = nn.Sequential(
            nn.PReLU(),
            nn.Linear(out_feats, 1)
        )

        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def get_weight(self, edges):
        w = edges.src['hw'] + edges.dst['hw'] + edges.data['ew']
        w = self.lin_att(w)

        return {'w' : w}
    
    def apply_scores(self, edges):
        return {'l' : edges.data['a'] * edges.data['e'] * edges.src['h'] * edges.dst['h']}

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            graph.srcdata['hw'] = self.prj_src(feat_src)
            graph.dstdata['hw'] = self.prj_dst(feat_dst)
            graph.edata['ew'] = self.prj_edge(edge_weight)
            graph.apply_edges(self.get_weight)
            # compute the attention scores.
            scores = edge_softmax(graph, graph.edata['w'])
            
            # message passing
            graph.edata['a'] = scores
            graph.edata['e'] = self.fc_edge(edge_weight)
            graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
            graph.dstdata['h'] = self.fc_dst(feat_dst)
            # employ attention scores
            graph.apply_edges(self.apply_scores)
            # use sum aggregation
            graph.update_all(fn.copy_edge('l', 'm'), fn.sum('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)
  
            rst = self.fc_self(h_self) + h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            return rst
