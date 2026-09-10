# Graph Link Prediction

## Research Question
Design a novel link-prediction method for graphs. The goal is to learn an
encoder that maps nodes to embeddings and a decoder that scores candidate
edges, such that the model accurately predicts missing or future links across
diverse graph types.

## Background
Link prediction is a fundamental graph-learning task: given a partially
observed graph, predict which unobserved edges are likely to exist. It has
applications in social networks (friend recommendation), citation networks
(paper recommendation), knowledge graph completion, and biological interaction
prediction.

Classical approaches:
- **GCN + dot-product decoder**: a GCN encodes nodes and the dot product of
  embeddings scores edges. Simple but often competitive.
- **VGAE** (Variational Graph Auto-Encoder): a probabilistic GCN encoder with
  KL regularization and inner-product decoder. Kipf & Welling, "Variational
  Graph Auto-Encoders," 2016 (arXiv:1611.07308).
- **node2vec**: random-walk based embeddings with biased walks balancing BFS
  and DFS. Grover & Leskovec, KDD 2016 (arXiv:1607.00653).

Recent SOTA methods exploit richer structural information:
- **SEAL** extracts k-hop enclosing subgraphs per edge and uses the DRNL
  labelling trick + GNN for edge classification. Zhang & Chen, "Link Prediction
  Based on Graph Neural Networks," NeurIPS 2018 (arXiv:1802.09691).
- **Neo-GNN** learns neighborhood-overlap features from the adjacency matrix
  to augment GNN predictions. Yun, Kim, Lee, Kang & Kim, NeurIPS 2021
  (arXiv:2206.04216).
- **BUDDY / ELPH** uses subgraph sketching with HyperLogLog and MinHash for
  scalable structural information. Chamberlain, Shirobokov, Rossi, Frasca,
  Markovich, Hammerla, Bronstein & Hansmire, "Graph Neural Networks for Link
  Prediction with Subgraph Sketching," ICLR 2023 (arXiv:2209.15486).

## What to Implement
Implement the `LinkPredictor` class in `custom_linkpred.py`:

```python
class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        ...
    def encode(self, x, edge_index):
        # returns [N, hidden_channels]
        ...
    def decode(self, z_src, z_dst):
        # returns [num_edges] -- scores for given source/dest embeddings
        ...
    def forward(self, x, edge_index, edge_label_index):
        # returns [num_edges] -- end-to-end forward pass
        ...
```

Input format:
- `x`: node features `[N, in_channels]`. Feature dimension varies by dataset.
- `edge_index`: training graph edges `[2, E_train]` in COO format
  (undirected).
- `edge_label_index`: candidate edges to score `[2, num_candidates]`.

Available PyG modules (pre-installed): any of `GCNConv`, `SAGEConv`, `GATConv`,
`GINConv`, `GraphConv`, `MessagePassing`, global pooling, `torch_geometric.utils`
(e.g. `negative_sampling`, `to_undirected`, `degree`),
`torch_geometric.nn`, `torch_geometric.transforms`.

## Evaluation
Datasets:

| Label         | Nodes   | Edges     | Features | Split / metric set      |
|---------------|---------|-----------|----------|-------------------------|
| Cora          | 2,708   | 10,556    | 1,433    | 85/5/10 link split; AUC, MRR, Hits@20 |
| CiteSeer      | 3,327   | 9,104     | 3,703    | 85/5/10 link split; AUC, MRR, Hits@20 |
| ogbl-collab   | 235,868 | 1,285,465 | 128      | Official OGB split; Hits@50, MRR      |

All metrics are higher-is-better.

The scientific contribution may improve the encoder, the edge decoder, or the
structural features used for candidate edges. The method should avoid assuming
a fixed feature dimension or graph size and should work for undirected
training graphs with positive and sampled negative candidate edges.
