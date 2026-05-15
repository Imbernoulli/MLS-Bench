"""SchNet baseline — Continuous-filter convolution with RBF distance expansion.
Uses radial basis function distance encoding for all edge types on heterogeneous graphs.
HeteroGraphConv pattern: parallel compute per edge type, sum aggregate per node type.
Reference: EHIGN_PLA/ablation_study/HSchNet/
"""

_FILE = "EHIGN_PLA/custom_pla.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — SchNet: RBF Distance-based Heterogeneous GNN
# =====================================================================

class RBFExpansion(nn.Module):
    \"\"\"Radial basis function expansion of distances.\"\"\"
    def __init__(self, low=0.0, high=6.0, gap=0.1):
        super().__init__()
        centers = torch.arange(low, high, gap)
        self.register_buffer('centers', centers)
        self.register_buffer('width', torch.tensor(gap))

    @property
    def num_features(self):
        return self.centers.size(0)

    def forward(self, dist):
        return torch.exp(-0.5 * ((dist - self.centers) / self.width) ** 2)


class CFConv(nn.Module):
    \"\"\"Continuous-filter convolution (SchNet interaction block).
    filter_net(rbf) * node_proj(src), sum aggregation, residual, output MLP.
    \"\"\"
    def __init__(self, node_dim, rbf_dim, hidden_dim):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(rbf_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_src, x_dst, edge_index, rbf_feat, num_dst):
        src, dst = edge_index
        W = self.filter_net(rbf_feat)
        msg = self.node_proj(x_src[src]) * W
        agg = torch.zeros(num_dst, msg.size(-1), device=msg.device)
        agg.index_add_(0, dst, msg)
        return x_dst + self.output(agg)


class FC(nn.Module):
    \"\"\"Fully connected prediction head.\"\"\"
    def __init__(self, d_in, d_hidden, n_layers, dropout, n_out):
        super().__init__()
        layers = []
        for j in range(n_layers):
            if j == 0:
                layers += [nn.Linear(d_in, d_hidden), nn.Dropout(dropout),
                           nn.LeakyReLU(), nn.BatchNorm1d(d_hidden)]
            if j == n_layers - 1:
                layers.append(nn.Linear(d_hidden, n_out))
            else:
                layers += [nn.Linear(d_hidden, d_hidden), nn.Dropout(dropout),
                           nn.LeakyReLU(), nn.BatchNorm1d(d_hidden)]
        self.layers = nn.ModuleList(layers)

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h


class AffinityModel(nn.Module):
    \"\"\"SchNet-based heterogeneous GNN for binding affinity.

    Uses RBF distance expansion and continuous-filter convolution for all edge types.
    HeteroGraphConv pattern: parallel compute, sum aggregate per node type.
    Dual bidirectional prediction with attention-based bias correction.
    \"\"\"
    def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
        super().__init__()
        H = 256
        num_layers = 3
        self.rbf = RBFExpansion(high=6.0, gap=0.1)
        rbf_dim = self.rbf.num_features

        self.lin_node_l = nn.Linear(lig_dim, H)
        self.lin_node_p = nn.Linear(poc_dim, H)

        self.cf_l = nn.ModuleList([CFConv(H, rbf_dim, H) for _ in range(num_layers)])
        self.cf_p = nn.ModuleList([CFConv(H, rbf_dim, H) for _ in range(num_layers)])
        self.cf_lp = nn.ModuleList([CFConv(H, rbf_dim, H) for _ in range(num_layers)])
        self.cf_pl = nn.ModuleList([CFConv(H, rbf_dim, H) for _ in range(num_layers)])

        # Readout via inter-molecular interaction scoring
        self.prj_lp_src = nn.Linear(H, H)
        self.prj_lp_dst = nn.Linear(H, H)
        self.prj_lp_edge = nn.Linear(rbf_dim, H)
        self.fc_lp = nn.Linear(H, 1)
        self.prj_pl_src = nn.Linear(H, H)
        self.prj_pl_dst = nn.Linear(H, H)
        self.prj_pl_edge = nn.Linear(rbf_dim, H)
        self.fc_pl = nn.Linear(H, 1)

        # Bias correction (L->P) with attention
        self.bc_lp_prj_src = nn.Linear(H, H)
        self.bc_lp_prj_dst = nn.Linear(H, H)
        self.bc_lp_prj_edge = nn.Linear(rbf_dim, H)
        self.bc_lp_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_lp_w_src = nn.Linear(H, H)
        self.bc_lp_w_dst = nn.Linear(H, H)
        self.bc_lp_w_edge = nn.Linear(rbf_dim, H)
        self.bc_lp_fc = FC(H, 200, 2, 0.1, 1)

        # Bias correction (P->L) with attention
        self.bc_pl_prj_src = nn.Linear(H, H)
        self.bc_pl_prj_dst = nn.Linear(H, H)
        self.bc_pl_prj_edge = nn.Linear(rbf_dim, H)
        self.bc_pl_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_pl_w_src = nn.Linear(H, H)
        self.bc_pl_w_dst = nn.Linear(H, H)
        self.bc_pl_w_edge = nn.Linear(rbf_dim, H)
        self.bc_pl_fc = FC(H, 200, 2, 0.1, 1)

    def _get_rbf(self, edge_attr):
        dist = edge_attr[:, -1:] * 10
        return self.rbf(dist)

    def _edge_softmax(self, scores, batch_idx, num_graphs):
        max_scores = torch.zeros(num_graphs, 1, device=scores.device).fill_(-1e9)
        max_scores.index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
        exp_scores = torch.exp(scores - max_scores[batch_idx])
        sum_exp = torch.zeros(num_graphs, 1, device=scores.device)
        sum_exp.index_add_(0, batch_idx, exp_scores)
        return exp_scores / sum_exp[batch_idx].clamp(min=1e-8)

    def forward(self, batch: PLABatch) -> torch.Tensor:
        B = batch.labels.size(0)
        lig_h = self.lin_node_l(batch.lig_x)
        poc_h = self.lin_node_p(batch.poc_x)

        lig_rbf = self._get_rbf(batch.lig_edge_attr)
        poc_rbf = self._get_rbf(batch.poc_edge_attr)
        lp_rbf = self._get_rbf(batch.l2p_edge_attr) if batch.l2p_edge_attr.size(0) > 0 else None
        pl_rbf = self._get_rbf(batch.p2l_edge_attr) if batch.p2l_edge_attr.size(0) > 0 else None

        # HeteroGraphConv pattern: parallel compute, sum aggregate
        for i in range(len(self.cf_l)):
            lig_in, poc_in = lig_h, poc_h

            lig_intra = self.cf_l[i](lig_in, lig_in, batch.lig_edge_index, lig_rbf, lig_in.size(0))
            poc_intra = self.cf_p[i](poc_in, poc_in, batch.poc_edge_index, poc_rbf, poc_in.size(0))

            lig_inter = torch.zeros_like(lig_in)
            poc_inter = torch.zeros_like(poc_in)
            if lp_rbf is not None and batch.l2p_edge_index.size(1) > 0:
                poc_inter = self.cf_lp[i](lig_in, poc_in, batch.l2p_edge_index, lp_rbf, poc_in.size(0))
            if pl_rbf is not None and batch.p2l_edge_index.size(1) > 0:
                lig_inter = self.cf_pl[i](poc_in, lig_in, batch.p2l_edge_index, pl_rbf, lig_in.size(0))

            lig_h = lig_intra + lig_inter
            poc_h = poc_intra + poc_inter

        # Scoring (L->P)
        l2p_src, l2p_dst = batch.l2p_edge_index
        i_lp = self.prj_lp_edge(lp_rbf) * self.prj_lp_src(lig_h)[l2p_src] * self.prj_lp_dst(poc_h)[l2p_dst]
        logit_lp = self.fc_lp(i_lp)
        pred_lp = torch.zeros(B, 1, device=logit_lp.device)
        pred_lp.index_add_(0, batch.inter_batch, logit_lp)

        # Scoring (P->L)
        p2l_src, p2l_dst = batch.p2l_edge_index
        p2l_batch = batch.lig_batch[p2l_dst]
        i_pl = self.prj_pl_edge(pl_rbf) * self.prj_pl_src(poc_h)[p2l_src] * self.prj_pl_dst(lig_h)[p2l_dst]
        logit_pl = self.fc_pl(i_pl)
        pred_pl = torch.zeros(B, 1, device=logit_pl.device)
        pred_pl.index_add_(0, p2l_batch, logit_pl)

        # Bias correction (L->P) with attention
        w_lp = self.bc_lp_prj_src(lig_h)[l2p_src] + self.bc_lp_prj_dst(poc_h)[l2p_dst] + self.bc_lp_prj_edge(lp_rbf)
        a_lp = self._edge_softmax(self.bc_lp_att(w_lp), batch.inter_batch, B)
        s_lp = a_lp * self.bc_lp_w_edge(lp_rbf) * self.bc_lp_w_src(lig_h)[l2p_src] * self.bc_lp_w_dst(poc_h)[l2p_dst]
        bias_lp_agg = torch.zeros(B, s_lp.size(-1), device=s_lp.device)
        bias_lp_agg.index_add_(0, batch.inter_batch, s_lp)
        bias_lp = self.bc_lp_fc(bias_lp_agg)

        # Bias correction (P->L) with attention
        w_pl = self.bc_pl_prj_src(poc_h)[p2l_src] + self.bc_pl_prj_dst(lig_h)[p2l_dst] + self.bc_pl_prj_edge(pl_rbf)
        a_pl = self._edge_softmax(self.bc_pl_att(w_pl), p2l_batch, B)
        s_pl = a_pl * self.bc_pl_w_edge(pl_rbf) * self.bc_pl_w_src(poc_h)[p2l_src] * self.bc_pl_w_dst(lig_h)[p2l_dst]
        bias_pl_agg = torch.zeros(B, s_pl.size(-1), device=s_pl.device)
        bias_pl_agg.index_add_(0, p2l_batch, s_pl)
        bias_pl = self.bc_pl_fc(bias_pl_agg)

        pred = ((pred_lp - bias_lp) + (pred_pl - bias_pl)) / 2
        return pred.squeeze(-1)

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 101,
        "end_line": 191,
        "content": _CONTENT,
    },
]
