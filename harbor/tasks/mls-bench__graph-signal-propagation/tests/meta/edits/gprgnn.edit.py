"""GPRGNN baseline -- Generalized PageRank GNN (Chien et al., 2021).

Learns polynomial filter coefficients in the monomial basis via gradient
descent. Initialized with uniform coefficients (dataset-agnostic) so the
model works well on both homophilic and heterophilic graphs without
requiring per-dataset alpha tuning.

Reference: Chien et al., "Adaptive Universal Generalized PageRank Graph
Neural Network" (ICLR 2021)
Reference code: github.com/jianhao2016/GPRGNN
"""

_FILE = "ChebNetII/main/custom_filter.py"

_GPRGNN = """\
class CustomProp(MessagePassing):
    \"\"\"GPR propagation: learnable polynomial in the monomial basis.

    Filter: h(A) = sum_{k=0}^{K} gamma_k * A^k
    where A is the GCN-normalized adjacency and gamma_k are learnable.

    Initialized with uniform coefficients (1/(K+1)) so the filter starts
    as an equal-weight average of all hops. This is dataset-agnostic and
    lets the optimizer freely learn both low-pass (homophilic) and
    high-pass (heterophilic) filters.
    \"\"\"

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.alpha = alpha
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        # Uniform initialization for dataset-agnostic starting point.
        nn.init.constant_(self.temp, 1.0 / (self.K + 1))

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )
        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            hidden = hidden + self.temp[k + 1] * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    \"\"\"GPRGNN: Generalized PageRank GNN (Chien et al., 2021).

    MLP encoder + learnable monomial polynomial filter.
    \"\"\"

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = CustomProp(K, alpha)
        self.dropout = dropout
        self.dprate = 0.0  # GPRGNN paper: no propagation dropout
        # Override training hyperparams (read by template's training loop)
        self.custom_lr = 0.05
        self.custom_wd = 0.0005
        self.custom_prop_lr = 0.05  # same lr for filter coefficients
        self.custom_prop_wd = 0.0

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 211,
        "end_line": 308,
        "content": _GPRGNN,
    },
]
