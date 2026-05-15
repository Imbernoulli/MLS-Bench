"""GraphRNN baseline for graph-generation.

Autoregressive graph generation using GRU-based node and edge-level RNNs.
Generates graphs node-by-node in BFS ordering, predicting edge connections
to previously generated nodes at each step.

Reference: You et al., "GraphRNN: Generating Realistic Graphs with an
Autoregressive Model" (ICML 2018)
"""

_FILE = "pytorch-geometric/custom_graphgen.py"

_CONTENT = """\
# The agent should modify the GraphGenerator class below.
# The class must implement:
#   - __init__(self, max_nodes, **kwargs): initialize model parameters
#   - train_step(self, adj, node_counts) -> dict: one training step, returns loss dict
#   - sample(self, n_samples, device) -> (adj_matrices, node_counts):
#       generate n_samples graphs, return adjacency tensors and node count tensors
#
# The model receives adjacency matrices [B, max_nodes, max_nodes] and node counts [B].
# It should generate adjacency matrices of similar structure.
# ============================================================================

class GraphGenerator(nn.Module):
    \"\"\"GraphRNN: Autoregressive graph generation with GRU.

    Uses a graph-level GRU to maintain state across node additions,
    and an edge-level GRU to predict edges to previous nodes.

    Reference: You et al., ICML 2018.
    \"\"\"

    def __init__(self, max_nodes, hidden_dim=128, edge_hidden_dim=16,
                 num_layers=4, lr=1e-3, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim

        # Graph-level RNN: predicts initial hidden state for edge RNN
        self.graph_rnn = nn.GRU(
            input_size=max_nodes,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Edge-level RNN: predicts edges to previous nodes autoregressively
        self.edge_rnn = nn.GRU(
            input_size=1,
            hidden_size=edge_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Map graph RNN hidden to edge RNN initial hidden
        self.hidden_map = nn.Linear(hidden_dim, edge_hidden_dim * num_layers)
        self.edge_output = nn.Linear(edge_hidden_dim, 1)
        self.num_layers = num_layers
        self.edge_hidden_dim = edge_hidden_dim

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _get_bfs_seq(self, adj, node_count):
        \"\"\"Convert adjacency matrix to BFS-ordered edge sequence.\"\"\"
        n = int(node_count)
        A = adj[:n, :n].cpu().numpy()

        # BFS from node 0
        visited = [False] * n
        order = []
        queue = [0]
        visited[0] = True
        while queue:
            v = queue.pop(0)
            order.append(v)
            neighbors = sorted(np.where(A[v] > 0.5)[0])
            for u in neighbors:
                if not visited[u]:
                    visited[u] = True
                    queue.append(u)
        # Add any unvisited nodes
        for i in range(n):
            if not visited[i]:
                order.append(i)

        # Reorder adjacency to BFS order
        perm = np.array(order)
        A_bfs = A[np.ix_(perm, perm)]
        return A_bfs, n

    def train_step(self, adj, node_counts):
        \"\"\"Train on a batch of adjacency matrices.\"\"\"
        self.train()
        self.optimizer.zero_grad()
        B = adj.shape[0]
        device = adj.device
        total_loss = 0.0

        for b in range(B):
            A_bfs, n = self._get_bfs_seq(adj[b], node_counts[b])
            if n < 2:
                continue

            # Build sequences: for each node i (from 1 to n-1),
            # the target is edges to nodes 0..i-1
            max_prev = min(n - 1, self.max_nodes)

            # Graph-level input: row of adjacency (padded)
            graph_input = torch.zeros(1, n - 1, self.max_nodes, device=device)
            for i in range(1, n):
                row = A_bfs[i, :i]
                padded = np.zeros(self.max_nodes)
                padded[:len(row)] = row
                graph_input[0, i - 1] = torch.tensor(padded, dtype=torch.float32, device=device)

            # Run graph RNN
            graph_out, _ = self.graph_rnn(graph_input)  # [1, n-1, hidden]

            step_loss = 0.0
            n_steps = 0
            for i in range(1, n):
                # Target edges for node i to nodes 0..i-1
                target = torch.tensor(A_bfs[i, :i], dtype=torch.float32, device=device)

                # Edge RNN initial hidden from graph RNN output
                h_graph = graph_out[0, i - 1]  # [hidden]
                h_edge = self.hidden_map(h_graph)  # [edge_hidden * num_layers]
                h_edge = h_edge.view(self.num_layers, 1, self.edge_hidden_dim)

                # Autoregressive edge prediction
                edge_input = torch.zeros(1, i, 1, device=device)
                if i > 1:
                    edge_input[0, 1:, 0] = target[:i - 1]  # Teacher forcing

                edge_out, _ = self.edge_rnn(edge_input, h_edge)  # [1, i, edge_hidden]
                edge_logits = self.edge_output(edge_out).squeeze(-1)  # [1, i]

                step_loss += F.binary_cross_entropy_with_logits(
                    edge_logits[0], target, reduction="sum"
                )
                n_steps += i

            if n_steps > 0:
                total_loss += step_loss / n_steps

        loss = total_loss / max(B, 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": loss.item()}

    def sample(self, n_samples, device):
        \"\"\"Generate graphs autoregressively.\"\"\"
        self.eval()
        all_adjs = []
        all_counts = []

        with torch.no_grad():
            for _ in range(n_samples):
                adj = np.zeros((self.max_nodes, self.max_nodes))
                h_graph = torch.zeros(self.graph_rnn.num_layers, 1, self.hidden_dim, device=device)

                n_nodes = 1  # Start with 1 node
                for i in range(1, self.max_nodes):
                    # Graph RNN step
                    row_input = torch.zeros(1, 1, self.max_nodes, device=device)
                    if i > 0:
                        row_tensor = torch.tensor(adj[i - 1], dtype=torch.float32, device=device)
                        row_input[0, 0] = row_tensor

                    graph_out, h_graph = self.graph_rnn(row_input, h_graph)

                    # Edge RNN
                    h_edge_init = self.hidden_map(graph_out[0, 0])
                    h_edge = h_edge_init.view(self.num_layers, 1, self.edge_hidden_dim)

                    edges = []
                    edge_in = torch.zeros(1, 1, 1, device=device)
                    for j in range(i):
                        edge_out, h_edge = self.edge_rnn(edge_in, h_edge)
                        logit = self.edge_output(edge_out[0, 0])
                        prob = torch.sigmoid(logit).item()
                        edge = 1.0 if random.random() < prob else 0.0
                        edges.append(edge)
                        edge_in = torch.tensor([[[edge]]], device=device)

                    # Check if this node has any edges (termination condition)
                    if sum(edges) == 0 and i > 2:
                        break

                    for j, e in enumerate(edges):
                        adj[i, j] = e
                        adj[j, i] = e
                    n_nodes = i + 1

                all_adjs.append(adj)
                all_counts.append(n_nodes)

        adjs = torch.tensor(np.array(all_adjs), dtype=torch.float32, device=device)
        counts = torch.tensor(all_counts, dtype=torch.long, device=device)
        return adjs, counts

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 446,
        "end_line": 590,
        "content": _CONTENT,
    },
]
