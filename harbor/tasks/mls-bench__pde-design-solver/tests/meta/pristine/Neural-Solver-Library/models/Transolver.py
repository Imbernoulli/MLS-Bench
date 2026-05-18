import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from layers.Basic import MLP
from layers.Embedding import timestep_embedding, unified_pos_embedding
from layers.Physics_Attention import Physics_Attention_Irregular_Mesh
from layers.Physics_Attention import Physics_Attention_Structured_Mesh_1D
from layers.Physics_Attention import Physics_Attention_Structured_Mesh_2D
from layers.Physics_Attention import Physics_Attention_Structured_Mesh_3D

PHYSICS_ATTENTION = {
    'unstructured': Physics_Attention_Irregular_Mesh,
    'structured_1D': Physics_Attention_Structured_Mesh_1D,
    'structured_2D': Physics_Attention_Structured_Mesh_2D,
    'structured_3D': Physics_Attention_Structured_Mesh_3D
}


class Transolver_block(nn.Module):
    """Transolver encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            geotype='unstructured',
            shapelist=None
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.Attn = PHYSICS_ATTENTION[geotype](hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                               dropout=dropout, slice_num=slice_num, shapelist=shapelist)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Transolver'
        self.args = args
        ## embedding
        if args.unified_pos and args.geotype != 'unstructured':  # only for structured mesh
            self.pos = unified_pos_embedding(args.shapelist, args.ref)
            self.preprocess = MLP(args.fun_dim + args.ref ** len(args.shapelist), args.n_hidden * 2,
                                  args.n_hidden, n_layers=0, res=False, act=args.act)
        else:
            self.preprocess = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                                  n_layers=0, res=False, act=args.act)
        if args.time_input:
            self.time_fc = nn.Sequential(nn.Linear(args.n_hidden, args.n_hidden), nn.SiLU(),
                                         nn.Linear(args.n_hidden, args.n_hidden))

        ## models
        self.blocks = nn.ModuleList([Transolver_block(num_heads=args.n_heads, hidden_dim=args.n_hidden,
                                                      dropout=args.dropout,
                                                      act=args.act,
                                                      mlp_ratio=args.mlp_ratio,
                                                      out_dim=args.out_dim,
                                                      slice_num=args.slice_num,
                                                      last_layer=(_ == args.n_layers - 1),
                                                      geotype=args.geotype,
                                                      shapelist=args.shapelist)
                                     for _ in range(args.n_layers)])
        self.placeholder = nn.Parameter((1 / (args.n_hidden)) * torch.rand(args.n_hidden, dtype=torch.float))
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

    def structured_geo(self, x, fx, T=None):
        if self.args.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)
        return fx

    def unstructured_geo(self, x, fx, T=None):
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)
        return fx

    def forward(self, x, fx, T=None, geo=None):
        if self.args.geotype == 'unstructured':
            return self.unstructured_geo(x, fx, T)
        else:
            return self.structured_geo(x, fx, T)
