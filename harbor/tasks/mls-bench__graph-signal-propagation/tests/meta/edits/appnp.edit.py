"""APPNP baseline -- Approximate Personalized Propagation of Neural Predictions.

Uses K-step power iteration of the personalized PageRank matrix with
teleport probability alpha. MLP encoder + APPNP propagation.

Reference: Klicpera et al., 2019 (ICLR)
Reference code: vendor/external_packages/ChebNetII/main/models.py (APPNP_Net)
"""

_FILE = "ChebNetII/main/custom_filter.py"

_APPNP = """\
class CustomProp(MessagePassing):
    \"\"\"Placeholder -- APPNP uses PyG's built-in APPNP propagation.\"\"\"

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, edge_weight=None):
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    \"\"\"APPNP: Approximate Personalized PageRank propagation (Klicpera et al., 2019).

    Architecture: MLP encoder followed by K-step power iteration of the
    personalized PageRank matrix with teleport probability alpha.
    The filter is fixed (not learned) -- only the MLP weights are trained.
    \"\"\"

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = PyGAPPNP(K, alpha)
        self.dropout = dropout
        # APPNP paper hyperparameters (Klicpera et al., 2019)
        self.custom_lr = 0.01
        self.custom_wd = 5e-4
        self.custom_prop_lr = 0.01
        self.custom_prop_wd = 0.0

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 211,
        "end_line": 308,
        "content": _APPNP,
    },
]
