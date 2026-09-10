# Graph Neural Network: Node Classification Message Passing

## Research Question
Design a novel **message-passing mechanism** for graph neural networks that
improves node-classification performance across citation network benchmarks.

## Background
Graph neural networks learn node representations by iteratively aggregating
information from neighboring nodes through message passing. The core design
choices are:

- **Message construction**: how to compute messages from source to target
  nodes (e.g., linear transform, attention-weighted, edge-conditioned).
- **Aggregation**: how to combine incoming messages (e.g., sum, mean, max,
  attention-weighted).
- **Update**: how to integrate aggregated messages with the node's own
  representation (residual, gated, concatenation, ...).

Classic approaches include GCN (symmetric normalization), GAT (attention-based
weighting), and GraphSAGE (mean aggregation with self/neighbor separation).
Recent advances include Graph Transformers (GPS) that combine local message
passing with global self-attention, and methods like NAGphormer that use
multi-hop tokenization with Transformer encoders.

## Task
Modify the `CustomMessagePassingLayer` class and `CustomGNN` model in
`custom_nodecls.py` to implement a novel message-passing mechanism. Your
implementation must work within PyTorch Geometric's `MessagePassing` framework.

```python
class CustomMessagePassingLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # learnable parameters and layers
        ...

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # x: [num_nodes, in_channels], edge_index: [2, num_edges]
        # returns [num_nodes, out_channels]
        ...

    def message(self, x_j: Tensor, ...) -> Tensor:
        # per-edge message computation
        ...


class CustomGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        ...

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # returns logits [num_nodes, out_channels]
        ...
```

Available PyG utilities:
- `MessagePassing` base class: `self.propagate(edge_index, ...)` orchestrates
  message / aggregate / update.
- `add_self_loops(edge_index)`: add self-loop edges.
- `degree(index, num_nodes)`: compute node degrees.
- `softmax(src, index)`: sparse softmax over edges.
- Reference convolution layers: `GCNConv`, `GATConv`, `SAGEConv`
  (imported but read-only).

## Evaluation
Trained and evaluated on three citation networks (semi-supervised node
classification with standard Planetoid splits):

| Label    | Nodes  | Edges  | Classes | Features |
|----------|--------|--------|---------|----------|
| Cora     | 2,708  | 5,429  | 7       | 1,433    |
| CiteSeer | 3,327  | 4,732  | 6       | 3,703    |
| PubMed   | 19,717 | 44,338 | 3       | 500      |

Fixed training pipeline: 200 epochs with early stopping (patience=50), Adam,
`lr=0.01`, `weight_decay=5e-4`.

Metrics: test accuracy and macro F1, both higher-is-better.

The research contribution should be the GNN propagation/model design rather
than changing the data split, loss target, or evaluation protocol.
