"""GCN baseline -- Graph Convolutional Network (Kipf & Welling, 2017).

Two-layer GCN with spectral first-order approximation filter.
Uses GCNConv from PyG which implements the renormalization trick.

Reference: vendor/external_packages/ChebNetII/main/models.py (GCN_Net)
"""

_FILE = "ChebNetII/main/custom_filter.py"

_GCN = """\
class CustomProp(MessagePassing):
    \"\"\"Placeholder -- not used by GCN (GCNConv handles propagation internally).\"\"\"

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
    \"\"\"GCN: two-layer Graph Convolutional Network (Kipf & Welling, 2017).

    Uses the first-order Chebyshev approximation of spectral graph convolutions,
    which amounts to a single-hop neighborhood aggregation with symmetric
    normalization and renormalization trick.
    \"\"\"

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, num_classes)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 211,
        "end_line": 308,
        "content": _GCN,
    },
]
