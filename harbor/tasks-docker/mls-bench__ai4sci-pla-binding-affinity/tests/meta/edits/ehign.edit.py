"""EHIGN baseline — Edge-enhanced Heterogeneous Interaction Graph Network.
Replaces editable section with heterogeneous graph convolution (CIG intra + NIG inter),
dual bidirectional prediction, and attention-based bias correction.
Reference: EHIGN_PLA (guaguabujianle/EHIGN_PLA)
"""

_FILE = "EHIGN_PLA/custom_pla.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — EHIGN: Heterogeneous Interaction Graph Network
# =====================================================================

class CIGConv(nn.Module):
    \"\"\"Covalent Interaction Graph Convolution (intra-molecular).
    Message: ReLU(src + edge_feat), sum aggregation, residual, MLP.
    \"\"\"
    def __init__(self, input_dim, output_dim, drop=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(drop),
            nn.LeakyReLU(),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        msg = F.relu(x[src] + edge_attr)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        rst = x + agg  # residual
        return self.mlp(rst)


class NIGConv(nn.Module):
    \"\"\"Non-covalent Interaction Graph Convolution (inter-molecular).
    Uses edge weights as multiplicative gates on source features, mean aggregation.
    Matches original: when in_feats == out_feats, fc_neigh applied AFTER aggregation.
    \"\"\"
    def __init__(self, in_feats, out_feats, feat_drop=0.0):
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(in_feats, out_feats, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_feats))
        nn.init.xavier_uniform_(self.fc_self.weight)
        nn.init.xavier_uniform_(self.fc_neigh.weight)

    def forward(self, x_src, x_dst, edge_index, edge_weight, num_dst):
        x_src = self.feat_drop(x_src)
        x_dst = self.feat_drop(x_dst)
        src, dst = edge_index
        # Edge-weighted messages: src_feat * edge_weight (element-wise)
        msg = x_src[src] * edge_weight
        # Mean aggregation
        agg = torch.zeros(num_dst, msg.size(-1), device=msg.device)
        count = torch.zeros(num_dst, 1, device=msg.device)
        agg.index_add_(0, dst, msg)
        count.index_add_(0, dst, torch.ones(src.size(0), 1, device=src.device))
        h_neigh = self.fc_neigh(agg / count.clamp(min=1))
        return self.fc_self(x_dst) + h_neigh + self.bias


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
    \"\"\"EHIGN: Edge-enhanced Heterogeneous Interaction Graph Network.

    Uses CIGConv for intra-molecular and NIGConv for inter-molecular message passing.
    HeteroGraphConv pattern: all edge types computed in parallel, outputs summed per node type.
    Dual bidirectional prediction with attention-based bias correction.
    \"\"\"
    def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
        super().__init__()
        H = 256
        num_layers = 3
        self.lin_node_l = nn.Linear(lig_dim, H)
        self.lin_node_p = nn.Linear(poc_dim, H)
        self.lin_edge_ll = nn.Linear(intra_edge_dim, H)
        self.lin_edge_pp = nn.Linear(intra_edge_dim, H)
        self.lin_edge_lp = nn.Linear(inter_edge_dim, H)
        self.lin_edge_pl = nn.Linear(inter_edge_dim, H)

        self.cig_l = nn.ModuleList([CIGConv(H, H) for _ in range(num_layers)])
        self.cig_p = nn.ModuleList([CIGConv(H, H) for _ in range(num_layers)])
        self.nig_lp = nn.ModuleList([NIGConv(H, H, 0.1) for _ in range(num_layers)])
        self.nig_pl = nn.ModuleList([NIGConv(H, H, 0.1) for _ in range(num_layers)])

        # Atom-atom affinity heads
        self.prj_lp_src = nn.Linear(H, H)
        self.prj_lp_dst = nn.Linear(H, H)
        self.prj_lp_edge = nn.Linear(H, H)
        self.fc_lp = nn.Linear(H, 1)
        self.prj_pl_src = nn.Linear(H, H)
        self.prj_pl_dst = nn.Linear(H, H)
        self.prj_pl_edge = nn.Linear(H, H)
        self.fc_pl = nn.Linear(H, 1)

        # Bias correction (L->P direction)
        self.bc_lp_prj_src = nn.Linear(H, H)
        self.bc_lp_prj_dst = nn.Linear(H, H)
        self.bc_lp_prj_edge = nn.Linear(H, H)
        self.bc_lp_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_lp_w_src = nn.Linear(H, H)
        self.bc_lp_w_dst = nn.Linear(H, H)
        self.bc_lp_w_edge = nn.Linear(H, H)
        self.bc_lp_fc = FC(H, 200, 2, 0.1, 1)

        # Bias correction (P->L direction)
        self.bc_pl_prj_src = nn.Linear(H, H)
        self.bc_pl_prj_dst = nn.Linear(H, H)
        self.bc_pl_prj_edge = nn.Linear(H, H)
        self.bc_pl_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_pl_w_src = nn.Linear(H, H)
        self.bc_pl_w_dst = nn.Linear(H, H)
        self.bc_pl_w_edge = nn.Linear(H, H)
        self.bc_pl_fc = FC(H, 200, 2, 0.1, 1)

    def _edge_softmax(self, scores, batch_idx, num_graphs):
        max_scores = torch.zeros(num_graphs, 1, device=scores.device).fill_(-1e9)
        max_scores.index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
        exp_scores = torch.exp(scores - max_scores[batch_idx])
        sum_exp = torch.zeros(num_graphs, 1, device=scores.device)
        sum_exp.index_add_(0, batch_idx, exp_scores)
        return exp_scores / sum_exp[batch_idx].clamp(min=1e-8)

    def _forward_heads(self, batch: PLABatch):
        \"\"\"Compute both dual prediction heads. Returns (pred_lp, pred_pl) each [B].\"\"\"
        B = batch.labels.size(0)
        # Project features
        lig_h = self.lin_node_l(batch.lig_x)
        poc_h = self.lin_node_p(batch.poc_x)
        lig_e = self.lin_edge_ll(batch.lig_edge_attr)
        poc_e = self.lin_edge_pp(batch.poc_edge_attr)
        lp_e = self.lin_edge_lp(batch.l2p_edge_attr)
        pl_e = self.lin_edge_pl(batch.p2l_edge_attr)

        # Message passing: HeteroGraphConv pattern — parallel compute, sum aggregate
        for i in range(len(self.cig_l)):
            # Save inputs (all convs use same input features)
            lig_in, poc_in = lig_h, poc_h

            # Intra-molecular (CIGConv has internal residual)
            lig_intra = self.cig_l[i](lig_in, batch.lig_edge_index, lig_e)
            poc_intra = self.cig_p[i](poc_in, batch.poc_edge_index, poc_e)

            # Inter-molecular (NIGConv with edge weights)
            lig_inter = torch.zeros_like(lig_in)
            poc_inter = torch.zeros_like(poc_in)
            if batch.l2p_edge_index.size(1) > 0:
                poc_inter = self.nig_lp[i](lig_in, poc_in, batch.l2p_edge_index, lp_e, poc_in.size(0))
            if batch.p2l_edge_index.size(1) > 0:
                lig_inter = self.nig_pl[i](poc_in, lig_in, batch.p2l_edge_index, pl_e, lig_in.size(0))

            # Sum aggregation per destination node type
            lig_h = lig_intra + lig_inter
            poc_h = poc_intra + poc_inter

        # Atom-atom affinities (L->P)
        l2p_src, l2p_dst = batch.l2p_edge_index
        i_lp = self.prj_lp_edge(lp_e) * self.prj_lp_src(lig_h)[l2p_src] * self.prj_lp_dst(poc_h)[l2p_dst]
        logit_lp = self.fc_lp(i_lp)
        pred_lp = torch.zeros(B, 1, device=logit_lp.device)
        pred_lp.index_add_(0, batch.inter_batch, logit_lp)

        # Atom-atom affinities (P->L)
        p2l_src, p2l_dst = batch.p2l_edge_index
        p2l_batch = batch.lig_batch[p2l_dst]
        i_pl = self.prj_pl_edge(pl_e) * self.prj_pl_src(poc_h)[p2l_src] * self.prj_pl_dst(lig_h)[p2l_dst]
        logit_pl = self.fc_pl(i_pl)
        pred_pl = torch.zeros(B, 1, device=logit_pl.device)
        pred_pl.index_add_(0, p2l_batch, logit_pl)

        # Bias correction (L->P)
        w_lp = self.bc_lp_prj_src(lig_h)[l2p_src] + self.bc_lp_prj_dst(poc_h)[l2p_dst] + self.bc_lp_prj_edge(lp_e)
        a_lp = self._edge_softmax(self.bc_lp_att(w_lp), batch.inter_batch, B)
        s_lp = a_lp * self.bc_lp_w_edge(lp_e) * self.bc_lp_w_src(lig_h)[l2p_src] * self.bc_lp_w_dst(poc_h)[l2p_dst]
        bias_lp_agg = torch.zeros(B, s_lp.size(-1), device=s_lp.device)
        bias_lp_agg.index_add_(0, batch.inter_batch, s_lp)
        bias_lp = self.bc_lp_fc(bias_lp_agg)

        # Bias correction (P->L)
        w_pl = self.bc_pl_prj_src(poc_h)[p2l_src] + self.bc_pl_prj_dst(lig_h)[p2l_dst] + self.bc_pl_prj_edge(pl_e)
        a_pl = self._edge_softmax(self.bc_pl_att(w_pl), p2l_batch, B)
        s_pl = a_pl * self.bc_pl_w_edge(pl_e) * self.bc_pl_w_src(poc_h)[p2l_src] * self.bc_pl_w_dst(lig_h)[p2l_dst]
        bias_pl_agg = torch.zeros(B, s_pl.size(-1), device=s_pl.device)
        bias_pl_agg.index_add_(0, p2l_batch, s_pl)
        bias_pl = self.bc_pl_fc(bias_pl_agg)

        pred_lp_final = (pred_lp - bias_lp).squeeze(-1)
        pred_pl_final = (pred_pl - bias_pl).squeeze(-1)
        return pred_lp_final, pred_pl_final

    def forward(self, batch: PLABatch) -> torch.Tensor:
        pred_lp, pred_pl = self._forward_heads(batch)
        return (pred_lp + pred_pl) / 2

    def compute_loss(self, batch: PLABatch, labels: torch.Tensor) -> torch.Tensor:
        \"\"\"EHIGN 3-term dual-head loss (paper: guaguabujianle/EHIGN_PLA train.py#L852):
            loss = (MSE(pred_lp, y) + MSE(pred_pl, y) + MSE(pred_lp, pred_pl)) / 3
        The third term is a consistency regularizer between the two bidirectional heads.
        \"\"\"
        pred_lp, pred_pl = self._forward_heads(batch)
        loss = (F.mse_loss(pred_lp, labels)
                + F.mse_loss(pred_pl, labels)
                + F.mse_loss(pred_lp, pred_pl)) / 3
        return loss

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
