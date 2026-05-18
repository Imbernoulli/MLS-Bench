"""SGC baseline -- Simplified Graph Convolution (Wu et al., 2019).

Removes nonlinearities between GCN layers, reducing to a single linear
transformation followed by K-hop propagation. Equivalent to GPRGNN with
monomial basis where only the K-th coefficient is 1.

Reference: Wu et al., "Simplifying Graph Convolutional Networks" (ICML 2019)
"""

_FILE = "ChebNetII/main/custom_filter.py"

_SGC = """\
class CustomProp(MessagePassing):
    \"\"\"SGC propagation: K-hop aggregation with GCN normalization.

    Applies the normalized adjacency matrix K times: (D^{-1/2}AD^{-1/2})^K x.
    No learnable parameters in the propagation layer.
    \"\"\"

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )
        for _ in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    \"\"\"SGC: Simplified Graph Convolution (Wu et al., 2019).

    Linear transformation followed by K-hop propagation.
    No nonlinearities between propagation steps.
    \"\"\"

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, num_classes)
        # SGC paper default: K=2 (K=10 is too aggressive on heterophilic graphs)
        self.prop = CustomProp(2)
        self.dropout = dropout
        self.dprate = dprate
        # SGC paper hyperparameters (Wu et al., 2019)
        self.custom_lr = 0.2
        self.custom_wd = 5e-6
        self.custom_prop_lr = 0.2
        self.custom_prop_wd = 5e-6

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
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
        "content": _SGC,
    },
]
