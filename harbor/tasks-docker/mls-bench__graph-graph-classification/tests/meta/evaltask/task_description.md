# Graph-Level Readout / Pooling for Graph Classification

## Research Question
Design a novel **graph-level readout (pooling) mechanism** that aggregates node
representations from a fixed message-passing backbone into a graph-level
embedding for graph classification, improving accuracy and generalization
across diverse molecular and biological graph datasets.

## Background
Graph classification requires mapping a variable-size graph to a fixed-size
vector for downstream prediction. The standard approach uses simple
permutation-invariant operations (sum, mean, max) over node embeddings, but
these discard structural information and treat all nodes equally. Notable
prior work:

- **Sum / Mean / Max readout** (basic). Xu, Hu, Leskovec & Jegelka, "How
  Powerful are Graph Neural Networks?", ICLR 2019 (arXiv:1810.00826) shows
  sum readout is most expressive among basic operations and motivates GIN.
- **SortPooling** (Zhang, Cui, Neumann & Chen, "An End-to-End Deep Learning
  Architecture for Graph Classification," AAAI 2018) sorts nodes by structural
  role via WL colors and applies a 1-D convolution.
- **Set2Set** (Vinyals, Bengio & Kudlur, "Order Matters: Sequence to sequence
  for sets," ICLR 2016; arXiv:1511.06391) uses LSTM-based attention over the
  node set.
- **SAGPool** (Lee, Lee & Kang, "Self-Attention Graph Pooling," ICML 2019;
  arXiv:1904.08082) computes self-attention scores for hierarchical top-k node
  selection.
- **DiffPool** (Ying, You, Morris, Ren, Hamilton & Leskovec, "Hierarchical
  Graph Representation Learning with Differentiable Pooling," NeurIPS 2018;
  arXiv:1806.08804) learns differentiable soft cluster assignments for
  hierarchical coarsening.
- **GMT** (Baek, Kang & Hwang, "Accurate Learning of Graph Representations
  with Graph Multiset Pooling," ICLR 2021; arXiv:2102.11533) is a multi-head
  attention based global pooling layer.

There is substantial room to improve graph readout by combining attention,
multi-scale aggregation, structural encodings, or learned pooling strategies.

## What You Can Modify
The `GraphReadout` class in `custom_graph_cls.py`. It receives node embeddings
from a fixed GIN backbone and must produce graph-level embeddings.

You may modify:
- The aggregation function (sum, mean, max, attention, learned weights, ...).
- Hierarchical coarsening (cluster, pool, repeat).
- How to combine multi-layer GNN outputs (jumping knowledge, concatenation,
  attention).
- Self-attention or cross-attention mechanisms over nodes.
- Structural encoding or positional information in the readout.
- Any combination of the above.

Constraints / interface:
- Input: `x` `[N_total, hidden_dim]`, `edge_index` `[2, E_total]`, `batch`
  `[N_total]`, `layer_outputs` list of `[N_total, hidden_dim]`.
- Output: `[B, output_dim]` tensor; set `self.output_dim` in `__init__`.
- Must handle variable graph sizes within a batch.
- Must be permutation equivariant / invariant as appropriate.
- Available imports: `torch`, `torch.nn`, `torch.nn.functional`,
  `torch_geometric.nn`, `torch_geometric.utils`.

## Evaluation
Datasets:
- **MUTAG** (188 graphs, 2 classes, molecular mutagenicity).
- **PROTEINS** (1113 graphs, 2 classes, protein enzyme classification).
- **NCI1** (4110 graphs, 2 classes, chemical compound activity).

Fixed pipeline:
- GNN backbone: 5-layer GIN (`hidden_dim=64`), fixed.
- Optimizer: Adam (`lr=0.01`), cosine annealing, 350 epochs per fold.
- Evaluation: 10-fold stratified cross-validation; report mean test accuracy
  and macro F1.

Metrics: test accuracy and macro F1, both higher-is-better.

A useful method should handle batches of graphs with different sizes, preserve
permutation invariance at the graph level, and generalize across small
molecular graphs and larger bio/chemical graph collections.
