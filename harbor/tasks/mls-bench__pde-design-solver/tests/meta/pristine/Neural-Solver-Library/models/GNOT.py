import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import repeat, rearrange
from torch.nn import functional as F
from layers.Basic import MLP, LinearAttention, ACTIVATION
from layers.Embedding import timestep_embedding, unified_pos_embedding

class GNOT_block(nn.Module):
    """Transformer encoder block in MOE style."""

    def __init__(self, num_heads: int,
                 hidden_dim: int,
                 dropout: float,
                 act='gelu',
                 mlp_ratio=4,
                 space_dim=2,
                 n_experts=3):
        super(GNOT_block, self).__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)

        self.selfattn = LinearAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout)
        self.crossattn = LinearAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout)
        self.resid_drop1 = nn.Dropout(dropout)
        self.resid_drop2 = nn.Dropout(dropout)

        ## MLP in MOE
        self.n_experts = n_experts
        if act in ACTIVATION.keys():
            self.act = ACTIVATION[act]
        self.moe_mlp1 = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            self.act(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        ) for _ in range(self.n_experts)])

        self.moe_mlp2 = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            self.act(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        ) for _ in range(self.n_experts)])

        self.gatenet = nn.Sequential(
            nn.Linear(space_dim, hidden_dim * mlp_ratio),
            self.act(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim * mlp_ratio),
            self.act(),
            nn.Linear(hidden_dim * mlp_ratio, self.n_experts)
        )

    def forward(self, x, y, pos):
        ## point-wise gate for moe
        gate_score = F.softmax(self.gatenet(pos), dim=-1).unsqueeze(2)
        ## cross attention between geo and physics observation
        x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln2(y)))
        ## moe mlp
        x_moe1 = torch.stack([self.moe_mlp1[i](x) for i in range(self.n_experts)], dim=-1)
        x_moe1 = (gate_score * x_moe1).sum(dim=-1, keepdim=False)
        x = x + self.ln3(x_moe1)
        ## self attention among geo
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        ## moe mlp
        x_moe2 = torch.stack([self.moe_mlp2[i](x) for i in range(self.n_experts)], dim=-1)
        x_moe2 = (gate_score * x_moe2).sum(dim=-1, keepdim=False)
        x = x + self.ln5(x_moe2)
        return x


class Model(nn.Module):
    ## GNOT: Transformer in MOE style
    def __init__(self, args, n_experts=3):
        super(Model, self).__init__()
        self.__name__ = 'GNOT'
        self.args = args
        ## embedding
        if args.unified_pos and args.geotype != 'unstructured':  # only for structured mesh
            self.pos = unified_pos_embedding(args.shapelist, args.ref)
            self.preprocess_x = MLP(args.ref ** len(args.shapelist), args.n_hidden * 2,
                                    args.n_hidden, n_layers=0, res=False, act=args.act)
            self.preprocess_z = MLP(args.fun_dim + args.ref ** len(args.shapelist), args.n_hidden * 2,
                                    args.n_hidden, n_layers=0, res=False, act=args.act)
        else:
            self.preprocess_x = MLP(args.space_dim, args.n_hidden * 2, args.n_hidden,
                                    n_layers=0, res=False, act=args.act)
            self.preprocess_z = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                                    n_layers=0, res=False, act=args.act)
        if args.time_input:
            self.time_fc = nn.Sequential(nn.Linear(args.n_hidden, args.n_hidden), nn.SiLU(),
                                         nn.Linear(args.n_hidden, args.n_hidden))

        ## models
        self.blocks = nn.ModuleList([GNOT_block(num_heads=args.n_heads,
                                                hidden_dim=args.n_hidden,
                                                dropout=args.dropout,
                                                act=args.act,
                                                mlp_ratio=args.mlp_ratio,
                                                space_dim=args.space_dim,
                                                n_experts=n_experts)
                                     for _ in range(args.n_layers)])
        self.placeholder = nn.Parameter((1 / (args.n_hidden)) * torch.rand(args.n_hidden, dtype=torch.float))
        # projectors
        self.fc1 = nn.Linear(args.n_hidden, args.n_hidden * 2)
        self.fc2 = nn.Linear(args.n_hidden * 2, args.out_dim)
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
        pos = x
        if self.args.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess_z(fx)
        else:
            fx = self.preprocess_z(x)
        fx = fx + self.placeholder[None, None, :]
        x = self.preprocess_x(x)
        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(x, fx, pos)
        fx = self.fc1(fx)
        fx = F.gelu(fx)
        fx = self.fc2(fx)
        return fx
