"""JacobiConv baseline -- Jacobi Polynomial Graph Filter (Wang & Zhang, 2022).

Linear spectral GNN using Jacobi polynomials as the filter basis. Uses
learnable polynomial coefficients with the three-term recurrence of
Jacobi polynomials P_k^{(a,b)}(x). Default parameters a=b=1.

Reference: Wang & Zhang, "How Powerful are Spectral Graph Neural Networks"
(ICML 2022)
Reference code: github.com/GraphPKU/JacobiConv
"""

_FILE = "ChebNetII/main/custom_filter.py"

_JACOBICONV = """\
class CustomProp(MessagePassing):
    \"\"\"Jacobi polynomial propagation layer.

    Filter: h(L_tilde) = sum_{k=0}^{K} gamma_k * P_k^{(a,b)}(L_tilde)
    where P_k^{(a,b)} are Jacobi polynomials with parameters a, b,
    L_tilde = L - I (shifted Laplacian), and gamma_k are learnable.

    Jacobi polynomials generalize Chebyshev (a=b=0), Legendre (a=b=0.5),
    and other orthogonal polynomial families.
    \"\"\"

    def __init__(self, K, alpha=0.1, a=1.0, b=1.0, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.a = a
        self.b = b
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize uniformly
        self.temp.data.fill_(1.0 / (self.K + 1))

    def forward(self, x, edge_index, edge_weight=None):
        # L = I - D^{-1/2}AD^{-1/2}
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim)
        )
        # L_tilde = L - I (shifted to [-1, 1])
        edge_index_tilde, norm_tilde = add_self_loops(
            edge_index1, norm1, fill_value=-1.0,
            num_nodes=x.size(self.node_dim)
        )

        a, b = self.a, self.b

        # Jacobi three-term recurrence
        # P_0^{(a,b)}(x) = 1
        Px_0 = x
        out = self.temp[0] * Px_0

        if self.K >= 1:
            # P_1^{(a,b)}(x) = (a+1) + (a+b+2)/2 * (x-1)
            #                 = ((a-b)/2) + ((a+b+2)/2) * x
            # Using matrix form: P_1 = c1 * L_tilde @ x + c0 * x
            c0 = (a - b) / 2.0
            c1 = (a + b + 2.0) / 2.0
            Px_1_prop = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)
            Px_1 = c1 * Px_1_prop + c0 * x
            out = out + self.temp[1] * Px_1

        for k in range(2, self.K + 1):
            # Three-term recurrence coefficients for Jacobi polynomials
            k_f = float(k)
            denom1 = 2.0 * k_f * (k_f + a + b) * (2.0 * k_f + a + b - 2.0)
            A_k = ((2.0 * k_f + a + b - 1.0) * (a * a - b * b)) / denom1
            B_k = ((2.0 * k_f + a + b - 1.0) * (2.0 * k_f + a + b - 2.0) * (2.0 * k_f + a + b)) / denom1
            C_k = (2.0 * (k_f - 1.0 + a) * (k_f - 1.0 + b) * (2.0 * k_f + a + b)) / denom1

            # P_k(x) = (A_k + B_k * x) * P_{k-1}(x) - C_k * P_{k-2}(x)
            Px_1_prop = self.propagate(edge_index_tilde, x=Px_1, norm=norm_tilde, size=None)
            Px_2 = (A_k * Px_1 + B_k * Px_1_prop) - C_k * Px_0
            out = out + self.temp[k] * Px_2
            Px_0, Px_1 = Px_1, Px_2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    \"\"\"JacobiConv: Jacobi polynomial graph filter (Wang & Zhang, 2022).

    MLP encoder + Jacobi polynomial propagation with learnable coefficients.
    \"\"\"

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = CustomProp(K, alpha, a=1.0, b=1.0)
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
        "content": _JACOBICONV,
    },
]
