"""BernNet baseline -- Bernstein Polynomial Graph Filter (He et al., 2021).

Learns spectral filter via Bernstein polynomial approximation on the
graph Laplacian spectrum. Uses non-negative coefficients (via ReLU)
to ensure filter controllability.

Reference: He et al., "BernNet: Learning Arbitrary Graph Spectral Filters
via Bernstein Approximation" (NeurIPS 2021)
Reference code: github.com/ivam-he/BernNet
"""

_FILE = "ChebNetII/main/custom_filter.py"

_BERNNET = """\
class CustomProp(MessagePassing):
    \"\"\"Bernstein polynomial propagation layer.

    Filter: h(L) = sum_{k=0}^{K} theta_k * C(K,k)/2^K * L^k * (2I-L)^{K-k}
    where theta_k = ReLU(learnable), C(K,k) is binomial coefficient,
    and L is the symmetric normalized Laplacian.
    \"\"\"

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)

        # L = I - D^{-1/2}AD^{-1/2}
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim)
        )
        # 2I - L
        edge_index2, norm2 = add_self_loops(
            edge_index1, -norm1, fill_value=2.0,
            num_nodes=x.size(self.node_dim)
        )

        # Compute (2I-L)^k * x for k = 0, ..., K
        tmp = [x]
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        # Bernstein basis evaluation
        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            # Apply L^{i+1}
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    \"\"\"BernNet: Bernstein polynomial graph filter (He et al., 2021).

    MLP encoder + Bernstein polynomial propagation.
    \"\"\"

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = CustomProp(K)
        self.dropout = dropout
        self.dprate = dprate

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
        "content": _BERNNET,
    },
]
