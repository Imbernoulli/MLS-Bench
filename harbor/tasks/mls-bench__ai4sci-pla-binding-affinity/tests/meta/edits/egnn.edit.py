"""EGNN baseline — E(n) Equivariant Graph Neural Network.
Uses distance-based scalar edge features with equivariant message passing.
HeteroGraphConv pattern: parallel compute per edge type, sum aggregate per node type.
Reference: EHIGN_PLA/ablation_study/HEGNN/
"""

_FILE = "EHIGN_PLA/custom_pla.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — EGNN: Equivariant Graph Neural Network
# =====================================================================

class EGNNConv(nn.Module):
    \"\"\"E(n)-equivariant message passing layer using distance as edge feature.
    Message: mlp_u(src) + mlp_v(dst) + mlp_e(dist), sum aggregation,
    then node_mlp(cat[dst, agg]).
    \"\"\"
    def __init__(self, input_dim, hidden_dim, edge_dim=1):
        super().__init__()
        self.edge_mlp_u = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.edge_mlp_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.edge_mlp_e = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x_src, x_dst, edge_index, edge_feat, num_dst):
        src, dst = edge_index
        msg = self.edge_mlp_u(x_src[src]) + self.edge_mlp_v(x_dst[dst]) + self.edge_mlp_e(edge_feat)
        agg = torch.zeros(num_dst, msg.size(-1), device=msg.device)
        agg.index_add_(0, dst, msg)
        return self.node_mlp(torch.cat([x_dst, agg], dim=-1))


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
    \"\"\"EGNN-based heterogeneous model for binding affinity.

    Uses E(n)-equivariant message passing with distance as scalar edge feature.
    HeteroGraphConv pattern: parallel compute, sum aggregate per node type.
    Dual bidirectional prediction with attention-based bias correction.
    \"\"\"
    def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
        super().__init__()
        H = 256
        num_layers = 3

        self.lin_node_l = nn.Linear(lig_dim, H)
        self.lin_node_p = nn.Linear(poc_dim, H)

        # EGNN layers for all 4 edge types (using distance as 1-dim edge feat)
        self.egnn_l = nn.ModuleList([EGNNConv(H, H, edge_dim=1) for _ in range(num_layers)])
        self.egnn_p = nn.ModuleList([EGNNConv(H, H, edge_dim=1) for _ in range(num_layers)])
        self.egnn_lp = nn.ModuleList([EGNNConv(H, H, edge_dim=1) for _ in range(num_layers)])
        self.egnn_pl = nn.ModuleList([EGNNConv(H, H, edge_dim=1) for _ in range(num_layers)])

        # Interaction scoring (with 1-dim distance edge features)
        self.prj_lp_src = nn.Linear(H, H)
        self.prj_lp_dst = nn.Linear(H, H)
        self.prj_lp_edge = nn.Linear(1, H)
        self.fc_lp = nn.Linear(H, 1)
        self.prj_pl_src = nn.Linear(H, H)
        self.prj_pl_dst = nn.Linear(H, H)
        self.prj_pl_edge = nn.Linear(1, H)
        self.fc_pl = nn.Linear(H, 1)

        # Bias correction (L->P)
        self.bc_lp_prj_src = nn.Linear(H, H)
        self.bc_lp_prj_dst = nn.Linear(H, H)
        self.bc_lp_prj_edge = nn.Linear(1, H)
        self.bc_lp_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_lp_w_src = nn.Linear(H, H)
        self.bc_lp_w_dst = nn.Linear(H, H)
        self.bc_lp_w_edge = nn.Linear(1, H)
        self.bc_lp_fc = FC(H, 200, 2, 0.1, 1)

        # Bias correction (P->L)
        self.bc_pl_prj_src = nn.Linear(H, H)
        self.bc_pl_prj_dst = nn.Linear(H, H)
        self.bc_pl_prj_edge = nn.Linear(1, H)
        self.bc_pl_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_pl_w_src = nn.Linear(H, H)
        self.bc_pl_w_dst = nn.Linear(H, H)
        self.bc_pl_w_edge = nn.Linear(1, H)
        self.bc_pl_fc = FC(H, 200, 2, 0.1, 1)

    def _get_dist(self, edge_attr):
        # Last dim is L2 distance * 0.1, rescale to angstroms
        return edge_attr[:, -1:] * 10

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

        lig_dist = self._get_dist(batch.lig_edge_attr)
        poc_dist = self._get_dist(batch.poc_edge_attr)
        lp_dist = self._get_dist(batch.l2p_edge_attr) if batch.l2p_edge_attr.size(0) > 0 else None
        pl_dist = self._get_dist(batch.p2l_edge_attr) if batch.p2l_edge_attr.size(0) > 0 else None

        # HeteroGraphConv pattern: parallel compute, sum aggregate
        for i in range(len(self.egnn_l)):
            lig_in, poc_in = lig_h, poc_h

            lig_intra = self.egnn_l[i](lig_in, lig_in, batch.lig_edge_index, lig_dist, lig_in.size(0))
            poc_intra = self.egnn_p[i](poc_in, poc_in, batch.poc_edge_index, poc_dist, poc_in.size(0))

            lig_inter = torch.zeros_like(lig_in)
            poc_inter = torch.zeros_like(poc_in)
            if lp_dist is not None and batch.l2p_edge_index.size(1) > 0:
                poc_inter = self.egnn_lp[i](lig_in, poc_in, batch.l2p_edge_index, lp_dist, poc_in.size(0))
            if pl_dist is not None and batch.p2l_edge_index.size(1) > 0:
                lig_inter = self.egnn_pl[i](poc_in, lig_in, batch.p2l_edge_index, pl_dist, lig_in.size(0))

            lig_h = lig_intra + lig_inter
            poc_h = poc_intra + poc_inter

        # Atom-atom affinities (L->P) with edge features
        l2p_src, l2p_dst = batch.l2p_edge_index
        i_lp = self.prj_lp_edge(lp_dist) * self.prj_lp_src(lig_h)[l2p_src] * self.prj_lp_dst(poc_h)[l2p_dst]
        logit_lp = self.fc_lp(i_lp)
        pred_lp = torch.zeros(B, 1, device=logit_lp.device)
        pred_lp.index_add_(0, batch.inter_batch, logit_lp)

        # Atom-atom affinities (P->L) with edge features
        p2l_src, p2l_dst = batch.p2l_edge_index
        p2l_batch = batch.lig_batch[p2l_dst]
        i_pl = self.prj_pl_edge(pl_dist) * self.prj_pl_src(poc_h)[p2l_src] * self.prj_pl_dst(lig_h)[p2l_dst]
        logit_pl = self.fc_pl(i_pl)
        pred_pl = torch.zeros(B, 1, device=logit_pl.device)
        pred_pl.index_add_(0, p2l_batch, logit_pl)

        # Bias correction (L->P) with attention
        w_lp = self.bc_lp_prj_src(lig_h)[l2p_src] + self.bc_lp_prj_dst(poc_h)[l2p_dst] + self.bc_lp_prj_edge(lp_dist)
        a_lp = self._edge_softmax(self.bc_lp_att(w_lp), batch.inter_batch, B)
        s_lp = a_lp * self.bc_lp_w_edge(lp_dist) * self.bc_lp_w_src(lig_h)[l2p_src] * self.bc_lp_w_dst(poc_h)[l2p_dst]
        bias_lp_agg = torch.zeros(B, s_lp.size(-1), device=s_lp.device)
        bias_lp_agg.index_add_(0, batch.inter_batch, s_lp)
        bias_lp = self.bc_lp_fc(bias_lp_agg)

        # Bias correction (P->L) with attention
        w_pl = self.bc_pl_prj_src(poc_h)[p2l_src] + self.bc_pl_prj_dst(lig_h)[p2l_dst] + self.bc_pl_prj_edge(pl_dist)
        a_pl = self._edge_softmax(self.bc_pl_att(w_pl), p2l_batch, B)
        s_pl = a_pl * self.bc_pl_w_edge(pl_dist) * self.bc_pl_w_src(poc_h)[p2l_src] * self.bc_pl_w_dst(lig_h)[p2l_dst]
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
