# Graph Generation Model Design

## Research Question
Design a novel generative model architecture for **unconditional graph
generation** that produces realistic graph structures matching the statistical
properties of a training distribution.

## Background
Graph generation is a fundamental problem with applications in drug discovery,
social network modeling, and materials science. The goal is to learn a
distribution over a set of graphs and generate new graphs that are
statistically indistinguishable from the training data. Existing approaches
span several paradigms:

- **Autoregressive**: GraphRNN generates graphs node-by-node with RNNs (You,
  Ying, Ren, Hamilton & Leskovec, "GraphRNN: Generating Realistic Graphs with
  Deep Auto-regressive Models," ICML 2018; arXiv:1802.08773); GRAN uses graph
  attention for one-shot block generation (Liao et al., "Efficient Graph
  Generation with Graph Recurrent Attention Networks," NeurIPS 2019;
  arXiv:1910.00760).
- **VAE-based**: GraphVAE encodes graphs into latent space and decodes a
  probabilistic adjacency matrix of fixed maximum size (Simonovsky & Komodakis,
  "GraphVAE: Towards Generation of Small Graphs Using Variational
  Autoencoders," 2018; arXiv:1802.03480).
- **Flow-based**: MoFlow uses normalizing flows for invertible molecular graph
  generation (Zang & Wang, KDD 2020; arXiv:2006.10137).
- **Score-based / diffusion**: GDSS applies score-based SDEs to graph
  generation (Jo, Lee & Hwang, ICML 2022; arXiv:2202.02514); DiGress uses
  discrete denoising diffusion (Vignac, Krawczuk, Siraudin, Wang, Cevher &
  Frossard, ICLR 2023; arXiv:2209.14734).

Evaluation uses Maximum Mean Discrepancy (MMD) between graph statistics
(degree, clustering, orbits) of generated and reference graphs.

## What You Can Modify
The `GraphGenerator` class in `custom_graphgen.py`. This class must implement:

1. `__init__(self, max_nodes, **kwargs)`: initialize model parameters and
   optimizer.
2. `train_step(self, adj, node_counts) -> dict`: one training step on a batch
   of adjacency matrices. Must return a dict containing at least
   `'loss'` (float).
3. `sample(self, n_samples, device) -> (adj, node_counts)`:
   - `adj`: Tensor `[n_samples, max_nodes, max_nodes]` -- binary symmetric
     adjacency matrices, no self-loops.
   - `node_counts`: Tensor `[n_samples]` -- number of nodes per graph
     (minimum 2).

Input adjacency matrices are binary, symmetric, zero-diagonal, and padded to
`max_nodes`. You may define helper classes/functions in the editable region.
The optimizer should be created in `__init__` and stepped in `train_step`.

Available imports inside the editable region: `torch`, `torch.nn`,
`torch.nn.functional`, `torch.optim`, `numpy`, `math`.

## Evaluation
Datasets:
- `community_small`: 100 synthetic 2-community graphs (12-20 nodes).
- `ego_small`: 200 ego graphs from Citeseer (4-18 nodes).
- `enzymes`: 587 protein structure graphs from BRENDA (10-125 nodes).

Fixed pipeline (shared by all baselines and the agent):
- 500 epochs, batch size 32, single GPU. (This is reduced from the 3000 epochs
  used in some published setups so that all methods fit the per-task compute
  budget; the same schedule is used for every method.)
- Multiple seeds for statistical reliability.

Metrics (all lower is better):
- `mmd_degree`: MMD of degree distributions.
- `mmd_clustering`: MMD of clustering-coefficient distributions.
- `mmd_orbit`: MMD of 4-orbit count distributions.
- `mmd_avg`: average of the three MMD metrics.

Suitable contributions may be autoregressive, latent-variable, diffusion-like,
energy-based, score-based, or otherwise structured, provided they can train
within the fixed budget and sample valid undirected graphs without relying on
the evaluation labels.
