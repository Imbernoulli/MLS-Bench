"""SortPool (DGCNN) readout baseline.

Sorts node embeddings by their last-channel WL color value and applies
1D convolution over the sorted sequence. From "An End-to-End Deep Learning
Architecture for Graph Classification" (Zhang et al., AAAI 2018).

Reference: torch_geometric.nn.aggr.SortAggregation / global_sort_pool
Reported: MUTAG ~85.8, PROTEINS ~75.5, NCI1 ~74.4
"""

_FILE = "pytorch-geometric/custom_graph_cls.py"

_CONTENT = """\
class GraphReadout(nn.Module):
    \"\"\"SortPool Readout (Zhang et al., 2018).

    Sorts nodes by their last-dimension value (WL color proxy),
    truncates/pads to fixed size k, then applies 1D convolution.
    \"\"\"

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k = 30  # Number of nodes to keep after sorting

        # 1D conv over sorted node sequence
        self.conv1d = nn.Conv1d(1, 16, kernel_size=hidden_dim, stride=hidden_dim)
        self.fc = nn.Linear(16 * self.k, hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch, layer_outputs):
        from torch_geometric.nn import global_sort_pool
        # global_sort_pool sorts by last channel, pads/truncates to k
        sorted_x = global_sort_pool(x, batch, self.k)  # [B, k * hidden_dim]
        sorted_x = sorted_x.unsqueeze(1)  # [B, 1, k * hidden_dim]
        out = F.relu(self.conv1d(sorted_x))  # [B, 16, k]
        out = out.view(out.size(0), -1)  # [B, 16 * k]
        return self.fc(out)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 41,
        "end_line": 81,
        "content": _CONTENT,
    },
]
