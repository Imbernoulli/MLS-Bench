"""ChebNetII baseline -- Chebyshev Interpolation Graph Filter (He et al., 2022).

Uses Chebyshev interpolation (not approximation) to learn spectral filters,
avoiding the Runge phenomenon that plagues original ChebNet. Coefficients
are constrained via ReLU and converted from interpolation nodes to
Chebyshev polynomial coefficients.

Reference: He et al., "Convolutional Neural Networks on Graphs with
Chebyshev Approximation, Revisited" (NeurIPS 2022)
Reference code: vendor/external_packages/ChebNetII/main/models.py (ChebNetII)
"""

_FILE = "ChebNetII/main/custom_filter.py"

_CHEBNETII = """\
class CustomProp(MessagePassing):
    \"\"\"ChebNetII propagation: Chebyshev interpolation filter.

    Learns filter values at Chebyshev interpolation nodes, then converts
    to Chebyshev polynomial coefficients. Uses ReLU to ensure non-negative
    interpolation values.

    Filter: h(L_tilde) = sum_{k=0}^{K} c_k * T_k(L_tilde)
    where L_tilde = L - I (shifted Laplacian), T_k is the k-th Chebyshev
    polynomial, and c_k are computed from interpolation values via DCT-like transform.
    \"\"\"

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

    def forward(self, x, edge_index, edge_weight=None):
        coe_tmp = F.relu(self.temp)
        coe = coe_tmp.clone()

        # Convert interpolation values to Chebyshev coefficients
        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        # L = I - D^{-1/2}AD^{-1/2}
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim)
        )
        # L_tilde = L - I (shifted to [-1, 1] range)
        edge_index_tilde, norm_tilde = add_self_loops(
            edge_index1, norm1, fill_value=-1.0,
            num_nodes=x.size(self.node_dim)
        )

        # Chebyshev recurrence: T_0(x)=x, T_1(x)=x, T_{k+1}=2xT_k - T_{k-1}
        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)

        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    \"\"\"ChebNetII: Chebyshev interpolation graph filter (He et al., 2022).

    MLP encoder + ChebNetII propagation with Chebyshev interpolation.
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
        "content": _CHEBNETII,
    },
]
