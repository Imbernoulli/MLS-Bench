# MLS-Bench: graph-generation

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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-geometric/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-geometric/custom_graphgen.py`
- editable lines **446–590**




## Readable Context


### `pytorch-geometric/custom_graphgen.py`  [EDITABLE — lines 446–590 only]

```python
     1: """Graph Generation Benchmark.
     2: 
     3: Train a generative model on small graph datasets and evaluate using MMD statistics.
     4: 
     5: FIXED: Dataset loading/generation, graph statistics computation, MMD evaluation,
     6:        training loop orchestration, argument parsing.
     7: EDITABLE: GraphGenerator class (the generative model).
     8: 
     9: Usage:
    10:     python pytorch-geometric/custom_graphgen.py --dataset community_small --seed 42
    11: """
    12: 
    13: import argparse
    14: import math
    15: import os
    16: import random
    17: import warnings
    18: from collections import defaultdict
    19: from typing import List, Tuple, Optional
    20: 
    21: import numpy as np
    22: import torch
    23: import torch.nn as nn
    24: import torch.nn.functional as F
    25: import torch.optim as optim
    26: from torch.utils.data import Dataset, DataLoader
    27: 
    28: warnings.filterwarnings("ignore")
    29: 
    30: 
    31: # ============================================================================
    32: # Dataset Generation & Loading (FIXED)
    33: # ============================================================================
    34: 
    35: def generate_community_small(n_graphs=100, min_nodes=12, max_nodes=20):
    36:     """Generate community_small dataset: 2-community graphs.
    37: 
    38:     Each graph has 2 communities connected by a few inter-community edges.
    39:     Uses the small-community setup common in GraphRNN/GDSS-style benchmarks.
    40:     """
    41:     try:
    42:         import networkx as nx
    43:     except ImportError:
    44:         raise ImportError("networkx is required: pip install networkx")
    45: 
    46:     graphs = []
    47:     for _ in range(n_graphs):
    48:         n = random.randint(min_nodes, max_nodes)
    49:         n1 = n // 2
    50:         n2 = n - n1
    51:         G = nx.planted_partition_graph(2, n // 2, p_in=0.7, p_out=0.05)
    52:         G = G.to_undirected()
    53:         G.remove_edges_from(nx.selfloop_edges(G))
    54:         graphs.append(G)
    55:     return graphs
    56: 
    57: 
    58: def generate_ego_small(n_max=200):
    59:     """Generate ego_small dataset: small ego graphs from Citeseer.
    60: 
    61:     Uses PyG's Planetoid(Citeseer) and extracts ego graphs of 1-hop neighborhoods.
    62:     Returns up to n_max graphs with 4-18 nodes.
    63:     """
    64:     try:
    65:         import networkx as nx
    66:         from torch_geometric.datasets import Planetoid
    67:         from torch_geometric.utils import to_networkx
    68:     except ImportError:
    69:         raise ImportError("torch_geometric and networkx required")
    70: 
    71:     dataset = Planetoid(root=os.environ.get("DATA_ROOT", "/data") + "/Planetoid", name="CiteSeer")
    72:     data = dataset[0]
    73:     G_full = to_networkx(data, to_undirected=True)
    74: 
    75:     graphs = []
    76:     nodes = list(G_full.nodes())
    77:     random.shuffle(nodes)
    78:     for node in nodes:
    79:         ego = nx.ego_graph(G_full, node, radius=1)
    80:         if 4 <= ego.number_of_nodes() <= 18:
    81:             # Relabel to consecutive integers
    82:             ego = nx.convert_node_labels_to_integers(ego)
    83:             ego.remove_edges_from(nx.selfloop_edges(ego))
    84:             graphs.append(ego)
    85:         if len(graphs) >= n_max:
    86:             break
    87:     return graphs
    88: 
    89: 
    90: def load_enzymes(n_max=587):
    91:     """Load ENZYMES dataset from TUDataset.
    92: 
    93:     Protein tertiary structure graphs, 587 graphs, 10-125 nodes.
    94:     """
    95:     try:
    96:         import networkx as nx
    97:         from torch_geometric.datasets import TUDataset
    98:         from torch_geometric.utils import to_networkx
    99:     except ImportError:
   100:         raise ImportError("torch_geometric and networkx required")
   101: 
   102:     dataset = TUDataset(root=os.environ.get("DATA_ROOT", "/data") + "/TUDataset", name="ENZYMES")
   103:     graphs = []
   104:     for i in range(min(len(dataset), n_max)):
   105:         data = dataset[i]
   106:         G = to_networkx(data, to_undirected=True)
   107:         G = nx.convert_node_labels_to_integers(G)
   108:         G.remove_edges_from(nx.selfloop_edges(G))
   109:         if G.number_of_nodes() >= 2:
   110:             graphs.append(G)
   111:     return graphs
   112: 
   113: 
   114: def load_dataset(name: str):
   115:     """Load a graph dataset by name. Returns list of networkx graphs."""
   116:     if name == "community_small":
   117:         return generate_community_small()
   118:     elif name == "ego_small":
   119:         return generate_ego_small()
   120:     elif name == "enzymes":
   121:         return load_enzymes()
   122:     else:
   123:         raise ValueError(f"Unknown dataset: {name}")
   124: 
   125: 
   126: # ============================================================================
   127: # Graph Representation Utilities (FIXED)
   128: # ============================================================================
   129: 
   130: def graphs_to_adj(graphs, max_nodes=None):
   131:     """Convert networkx graphs to padded adjacency matrices.
   132: 
   133:     Returns:
   134:         adjs: Tensor [N, max_nodes, max_nodes] (binary adjacency)
   135:         node_counts: Tensor [N] (actual number of nodes per graph)
   136:     """
   137:     if max_nodes is None:
   138:         max_nodes = max(G.number_of_nodes() for G in graphs)
   139:     adjs = []
   140:     node_counts = []
   141:     for G in graphs:
   142:         n = G.number_of_nodes()
   143:         A = np.zeros((max_nodes, max_nodes), dtype=np.float32)
   144:         for u, v in G.edges():
   145:             if u < max_nodes and v < max_nodes:
   146:                 A[u, v] = 1.0
   147:                 A[v, u] = 1.0
   148:         adjs.append(A)
   149:         node_counts.append(n)
   150:     return torch.tensor(np.array(adjs)), torch.tensor(node_counts, dtype=torch.long)
   151: 
   152: 
   153: def adj_to_graphs(adjs, node_counts):
   154:     """Convert adjacency matrices back to networkx graphs.
   155: 
   156:     Args:
   157:         adjs: Tensor or ndarray [N, max_nodes, max_nodes]
   158:         node_counts: Tensor or list [N]
   159: 
   160:     Returns:
   161:         List of networkx graphs.
   162:     """
   163:     import networkx as nx
   164:     if isinstance(adjs, torch.Tensor):
   165:         adjs = adjs.detach().cpu().numpy()
   166:     if isinstance(node_counts, torch.Tensor):
   167:         node_counts = node_counts.cpu().tolist()
   168: 
   169:     graphs = []
   170:     for A, n in zip(adjs, node_counts):
   171:         n = int(n)
   172:         G = nx.Graph()
   173:         G.add_nodes_from(range(n))
   174:         A_sub = A[:n, :n]
   175:         # Threshold at 0.5 for probabilistic outputs
   176:         A_bin = (A_sub > 0.5).astype(int)
   177:         # Make symmetric and remove self-loops
   178:         A_sym = np.maximum(A_bin, A_bin.T)
   179:         np.fill_diagonal(A_sym, 0)
   180:         for i in range(n):
   181:             for j in range(i + 1, n):
   182:                 if A_sym[i, j]:
   183:                     G.add_edge(i, j)
   184:         graphs.append(G)
   185:     return graphs
   186: 
   187: 
   188: class GraphDataset(Dataset):
   189:     """Dataset of adjacency matrices for graph generation."""
   190: 
   191:     def __init__(self, adjs, node_counts):
   192:         self.adjs = adjs        # [N, max_nodes, max_nodes]
   193:         self.node_counts = node_counts  # [N]
   194: 
   195:     def __len__(self):
   196:         return len(self.adjs)
   197: 
   198:     def __getitem__(self, idx):
   199:         return self.adjs[idx], self.node_counts[idx]
   200: 
   201: 
   202: # ============================================================================
   203: # Graph Statistics & MMD Evaluation (FIXED)
   204: # ============================================================================
   205: 
   206: def degree_stats(G):
   207:     """Degree sequence of graph (sorted)."""
   208:     return sorted([d for _, d in G.degree()], reverse=True)
   209: 
   210: 
   211: def clustering_stats(G):
   212:     """Clustering coefficient distribution."""
   213:     import networkx as nx
   214:     cc = nx.clustering(G)
   215:     return sorted(cc.values(), reverse=True)
   216: 
   217: 
   218: def orbit_stats(G):
   219:     """4-orbit count distribution.
   220: 
   221:     Returns an approximate per-graph orbit feature vector in a GDSS-style format:
   222:     a single feature vector per graph (mean orbit count per node for several
   223:     orbit types). Unlike degree/clustering, the orbit MMD in GDSS is computed
   224:     directly on these raw per-graph vectors with a Gaussian kernel (is_hist=False),
   225:     NOT histogrammed. Returning a numpy vector here signals this to compute_mmd.
   226:     """
   227:     import networkx as nx
   228:     n = G.number_of_nodes()
   229:     # 4 orbit-like counts per node: triangles, 2-paths, 3-paths, 4-cliques-participation
   230:     if n < 1:
   231:         return np.zeros(4, dtype=np.float64)
   232: 
   233:     triangles = np.zeros(n, dtype=np.float64)
   234:     two_paths = np.zeros(n, dtype=np.float64)
   235:     three_paths = np.zeros(n, dtype=np.float64)
   236:     four_cliques = np.zeros(n, dtype=np.float64)
   237: 
   238:     nodes = list(G.nodes())
   239:     idx = {v: i for i, v in enumerate(nodes)}
   240:     adj = {v: set(G.neighbors(v)) for v in nodes}
   241: 
   242:     for v in nodes:
   243:         Nv = adj[v]
   244:         # triangles through v
   245:         nbrs = list(Nv)
   246:         for i in range(len(nbrs)):
   247:             for j in range(i + 1, len(nbrs)):
   248:                 if nbrs[j] in adj[nbrs[i]]:
   249:                     triangles[idx[v]] += 1
   250:         # 2-paths (v -- u -- w, w != v, w not adj to v)
   251:         for u in Nv:
   252:             for w in adj[u]:
   253:                 if w != v and w not in Nv:
   254:                     two_paths[idx[v]] += 1
   255:         # 3-paths starting at v (v-u-w-x, distinct, no shortcut to v)
   256:         for u in Nv:
   257:             for w in adj[u]:
   258:                 if w == v or w in Nv:
   259:                     continue
   260:                 for x in adj[w]:
   261:                     if x != v and x != u and x not in Nv:
   262:                         three_paths[idx[v]] += 1
   263:         # 4-clique participation
   264:         for i in range(len(nbrs)):
   265:             for j in range(i + 1, len(nbrs)):
   266:                 if nbrs[j] not in adj[nbrs[i]]:
   267:                     continue
   268:                 for k in range(j + 1, len(nbrs)):
   269:                     if nbrs[k] in adj[nbrs[i]] and nbrs[k] in adj[nbrs[j]]:
   270:                         four_cliques[idx[v]] += 1
   271: 
   272:     # Per-graph feature: mean count across nodes for each orbit type
   273:     vec = np.array([
   274:         triangles.mean(),
   275:         two_paths.mean(),
   276:         three_paths.mean(),
   277:         four_cliques.mean(),
   278:     ], dtype=np.float64)
   279:     return vec
   280: 
   281: 
   282: def _to_histogram(values, n_bins=100):
   283:     """Convert a list of values to a normalized histogram (probability distribution).
   284: 
   285:     This is the standard approach used in graph generation benchmarks (GraphRNN,
   286:     GRAN, GDSS) to make statistics comparable across graphs of different sizes.
   287:     """
   288:     values = np.array(values, dtype=np.float64)
   289:     if len(values) == 0:
   290:         return np.zeros(n_bins)
   291:     # Use fixed bin edges across the full range
   292:     hist, _ = np.histogram(values, bins=n_bins, range=(values.min(), max(values.max(), values.min() + 1e-8)), density=False)
   293:     total = hist.sum()
   294:     if total > 0:
   295:         hist = hist / total
   296:     return hist.astype(np.float64)
   297: 
   298: 
   299: def _gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
   300:     """Gaussian kernel with an Earth Mover's Distance inner metric.
   301: 
   302:     GDSS-style Gaussian kernel over a 1-D EMD distance between probability
   303:     histograms. Falls back to simple L1 when scipy is unavailable.
   304:     """
   305:     x = np.asarray(x, dtype=np.float64)
   306:     y = np.asarray(y, dtype=np.float64)
   307:     # Try scipy's Wasserstein (EMD) on 1-D distributions
   308:     try:
   309:         from scipy.stats import wasserstein_distance
   310:         support = np.arange(len(x)) * distance_scaling
   311:         d = wasserstein_distance(support, support, u_weights=x, v_weights=y)
   312:     except Exception:
   313:         # Fallback: L1 / total variation times distance_scaling as a proxy
   314:         d = 0.5 * np.sum(np.abs(x - y)) * distance_scaling
   315:     return float(np.exp(-(d * d) / (2.0 * sigma * sigma)))
   316: 
   317: 
   318: def _gaussian(x, y, sigma=1.0):
   319:     x = np.asarray(x, dtype=np.float64)
   320:     y = np.asarray(y, dtype=np.float64)
   321:     d2 = float(np.sum((x - y) ** 2))
   322:     return float(np.exp(-d2 / (2.0 * sigma * sigma)))
   323: 
   324: 
   325: def compute_mmd(samples1, samples2, kernel="gaussian_emd", sigma=1.0, is_hist=True):
   326:     """Compute MMD between two sets of graph statistics (biased estimator).
   327: 
   328:     Uses GDSS-style graph-statistic MMD conventions:
   329:       - For degree/clustering: each sample is a 1-D list of per-node values
   330:         which is converted to a normalized histogram, then MMD uses a Gaussian-EMD
   331:         kernel with sigma=1.0.
   332:       - For orbit: each sample is already a per-graph feature vector (is_hist=False),
   333:         MMD uses a plain Gaussian kernel with sigma=30.0.
   334:     """
   335:     # Pick kernel
   336:     if kernel == "gaussian_emd":
   337:         kfn = lambda a, b: _gaussian_emd(a, b, sigma=sigma)
   338:     else:
   339:         kfn = lambda a, b: _gaussian(a, b, sigma=sigma)
   340: 
   341:     if is_hist:
   342:         # Build histograms with integer bins over the global value range.
   343:         # Degree/clustering: degrees are integers; clustering is in [0,1].
   344:         # Use fixed bins: for degree, bins = max_degree+1; for clustering,
   345:         # bins = 100 over [0,1]. We pick bins based on value type.
   346:         all_vals = [v for s in list(samples1) + list(samples2) for v in s]
   347:         if len(all_vals) == 0:
   348:             return 0.0
   349:         vmin = float(np.min(all_vals))
   350:         vmax = float(np.max(all_vals))
   351:         if vmax <= 1.0 and vmin >= 0.0:
   352:             # Clustering coefficient in [0, 1]
   353:             n_bins = 100
   354:             bin_range = (0.0, 1.0)
   355:         else:
   356:             # Degree-like integer values
   357:             n_bins = int(np.ceil(vmax)) + 1
   358:             n_bins = max(n_bins, 2)
   359:             bin_range = (0.0, float(n_bins))
   360: 
   361:         def to_hist(values):
   362:             values = np.asarray(values, dtype=np.float64)
   363:             if len(values) == 0:
   364:                 return np.zeros(n_bins, dtype=np.float64)
   365:             hist, _ = np.histogram(values, bins=n_bins, range=bin_range, density=False)
   366:             s = hist.sum()
   367:             if s > 0:
   368:                 hist = hist.astype(np.float64) / s
   369:             return hist.astype(np.float64)
   370: 
   371:         vecs1 = [to_hist(s) for s in samples1]
   372:         vecs2 = [to_hist(s) for s in samples2]
   373:     else:
   374:         vecs1 = [np.asarray(s, dtype=np.float64) for s in samples1]
   375:         vecs2 = [np.asarray(s, dtype=np.float64) for s in samples2]
   376: 
   377:     n = len(vecs1)
   378:     m = len(vecs2)
   379:     if n == 0 or m == 0:
   380:         return 0.0
   381: 
   382:     xx = 0.0
   383:     for i in range(n):
   384:         for j in range(n):
   385:             xx += kfn(vecs1[i], vecs1[j])
   386:     xx /= n * n
   387: 
   388:     yy = 0.0
   389:     for i in range(m):
   390:         for j in range(m):
   391:             yy += kfn(vecs2[i], vecs2[j])
   392:     yy /= m * m
   393: 
   394:     xy = 0.0
   395:     for i in range(n):
   396:         for j in range(m):
   397:             xy += kfn(vecs1[i], vecs2[j])
   398:     xy /= n * m
   399: 
   400:     return xx + yy - 2 * xy
   401: 
   402: 
   403: def evaluate_graphs(gen_graphs, ref_graphs, n_eval=None):
   404:     """Evaluate generated graphs against reference graphs using MMD.
   405: 
   406:     Returns dict with mmd_degree, mmd_clustering, mmd_orbit.
   407:     """
   408:     if n_eval is not None:
   409:         gen_graphs = gen_graphs[:n_eval]
   410: 
   411:     # Filter out empty graphs
   412:     gen_graphs = [G for G in gen_graphs if G.number_of_nodes() > 0 and G.number_of_edges() > 0]
   413:     ref_graphs = [G for G in ref_graphs if G.number_of_nodes() > 0 and G.number_of_edges() > 0]
   414: 
   415:     if len(gen_graphs) == 0:
   416:         return {"mmd_degree": 10.0, "mmd_clustering": 10.0, "mmd_orbit": 10.0}
   417: 
   418:     # Compute statistics for reference graphs
   419:     ref_degree = [degree_stats(G) for G in ref_graphs]
   420:     ref_cluster = [clustering_stats(G) for G in ref_graphs]
   421:     ref_orbit = [orbit_stats(G) for G in ref_graphs]
   422: 
   423:     # Compute statistics for generated graphs
   424:     gen_degree = [degree_stats(G) for G in gen_graphs]
   425:     gen_cluster = [clustering_stats(G) for G in gen_graphs]
   426:     gen_orbit = [orbit_stats(G) for G in gen_graphs]
   427: 
   428:     # Compute MMD for each statistic (GDSS conventions)
   429:     # degree: histogrammed integer degrees, gaussian_emd kernel, sigma=1.0
   430:     mmd_deg = compute_mmd(ref_degree, gen_degree, kernel="gaussian_emd", sigma=1.0, is_hist=True)
   431:     # clustering: histogrammed over [0,1], gaussian_emd kernel, sigma=1.0 / bins=100
   432:     mmd_clus = compute_mmd(ref_cluster, gen_cluster, kernel="gaussian_emd", sigma=1.0 / 10.0, is_hist=True)
   433:     # orbit: raw per-graph orbit count vectors, gaussian kernel, sigma=30.0
   434:     mmd_orb = compute_mmd(ref_orbit, gen_orbit, kernel="gaussian", sigma=30.0, is_hist=False)
   435: 
   436:     return {
   437:         "mmd_degree": float(mmd_deg),
   438:         "mmd_clustering": float(mmd_clus),
   439:         "mmd_orbit": float(mmd_orb),
   440:     }
   441: 
   442: 
   443: # ============================================================================
   444: # EDITABLE REGION START (lines 446-590)
   445: # ============================================================================
   446: # The agent should modify the GraphGenerator class below.
   447: # The class must implement:
   448: #   - __init__(self, max_nodes, **kwargs): initialize model parameters
   449: #   - train_step(self, adj, node_counts) -> dict: one training step, returns loss dict
   450: #   - sample(self, n_samples, device) -> (adj_matrices, node_counts):
   451: #       generate n_samples graphs, return adjacency tensors and node count tensors
   452: #
   453: # The model receives adjacency matrices [B, max_nodes, max_nodes] and node counts [B].
   454: # It should generate adjacency matrices of similar structure.
   455: # ============================================================================
   456: 
   457: class GraphGenerator(nn.Module):
   458:     """Generative model for graphs.
   459: 
   460:     This is a simple VAE-based baseline that encodes graphs into a latent space
   461:     and decodes to adjacency matrices. The agent should replace this with a
   462:     better generative model.
   463: 
   464:     Design space includes:
   465:       - Autoregressive models (node-by-node or edge-by-edge generation)
   466:       - One-shot models (generate full adjacency at once)
   467:       - Score-based / diffusion models (iterative denoising)
   468:       - Flow-based models (invertible transformations)
   469:       - VAE variants with graph-aware encoders/decoders
   470: 
   471:     Args:
   472:         max_nodes: Maximum number of nodes in generated graphs.
   473:         hidden_dim: Hidden dimension for encoder/decoder networks.
   474:         latent_dim: Latent space dimension.
   475:         lr: Learning rate for the optimizer.
   476:     """
   477: 
   478:     def __init__(self, max_nodes, hidden_dim=256, latent_dim=64, lr=1e-3, **kwargs):
   479:         super().__init__()
   480:         self.max_nodes = max_nodes
   481:         self.hidden_dim = hidden_dim
   482:         self.latent_dim = latent_dim
   483:         adj_size = max_nodes * max_nodes
   484: 
   485:         # Encoder: adjacency -> latent
   486:         self.encoder = nn.Sequential(
   487:             nn.Linear(adj_size, hidden_dim),
   488:             nn.ReLU(),
   489:             nn.Linear(hidden_dim, hidden_dim),
   490:             nn.ReLU(),
   491:         )
   492:         self.fc_mu = nn.Linear(hidden_dim, latent_dim)
   493:         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
   494: 
   495:         # Decoder: latent -> adjacency
   496:         self.decoder = nn.Sequential(
   497:             nn.Linear(latent_dim, hidden_dim),
   498:             nn.ReLU(),
   499:             nn.Linear(hidden_dim, hidden_dim),
   500:             nn.ReLU(),

[truncated: showing at most 500 lines / 60000 bytes from pytorch-geometric/custom_graphgen.py]
```

## Parameter Budget

This task enforces a parameter-count cap. Your edits will be rejected if
the resulting model exceeds **1.05×** the strongest
baseline's parameter count. The check runs automatically inside the eval
scripts — you don't need to invoke it.

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `graphvae` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-geometric/custom_graphgen.py`:

```python
Lines 446–578:
   443: # ============================================================================
   444: # EDITABLE REGION START (lines 446-590)
   445: # ============================================================================
   446: # The agent should modify the GraphGenerator class below.
   447: # The class must implement:
   448: #   - __init__(self, max_nodes, **kwargs): initialize model parameters
   449: #   - train_step(self, adj, node_counts) -> dict: one training step, returns loss dict
   450: #   - sample(self, n_samples, device) -> (adj_matrices, node_counts):
   451: #       generate n_samples graphs, return adjacency tensors and node count tensors
   452: #
   453: # The model receives adjacency matrices [B, max_nodes, max_nodes] and node counts [B].
   454: # It should generate adjacency matrices of similar structure.
   455: # ============================================================================
   456: 
   457: class GCNLayer(nn.Module):
   458:     """Simple GCN layer: X' = D^{-1/2} A_hat D^{-1/2} X W."""
   459: 
   460:     def __init__(self, in_features, out_features):
   461:         super().__init__()
   462:         self.linear = nn.Linear(in_features, out_features)
   463: 
   464:     def forward(self, x, adj):
   465:         # Add self-loops
   466:         I = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
   467:         A_hat = adj + I
   468:         # Degree normalization
   469:         D = A_hat.sum(dim=-1, keepdim=True).clamp(min=1)
   470:         D_inv_sqrt = 1.0 / torch.sqrt(D)
   471:         A_norm = A_hat * D_inv_sqrt * D_inv_sqrt.transpose(-1, -2)
   472:         out = torch.bmm(A_norm, x)
   473:         return self.linear(out)
   474: 
   475: 
   476: class GraphGenerator(nn.Module):
   477:     """GraphVAE: Variational Autoencoder for graph generation.
   478: 
   479:     Uses GCN encoder to produce graph-level latent representation,
   480:     and MLP decoder to produce adjacency matrix probabilities.
   481: 
   482:     Reference: Simonovsky & Komodakis, arXiv:1802.03480 / ICANN 2018.
   483:     """
   484: 
   485:     def __init__(self, max_nodes, hidden_dim=256, latent_dim=64, lr=1e-3, **kwargs):
   486:         super().__init__()
   487:         self.max_nodes = max_nodes
   488:         self.latent_dim = latent_dim
   489:         adj_size = max_nodes * max_nodes
   490: 
   491:         # GCN encoder
   492:         self.gcn1 = GCNLayer(max_nodes, hidden_dim)
   493:         self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
   494:         self.fc_mu = nn.Linear(hidden_dim, latent_dim)
   495:         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
   496: 
   497:         # MLP decoder
   498:         self.decoder = nn.Sequential(
   499:             nn.Linear(latent_dim, hidden_dim),
   500:             nn.ReLU(),
   501:             nn.Linear(hidden_dim, hidden_dim),
   502:             nn.ReLU(),
   503:             nn.Linear(hidden_dim, adj_size),
   504:         )
   505: 
   506:         # Node existence predictor
   507:         self.node_pred = nn.Sequential(
   508:             nn.Linear(latent_dim, hidden_dim // 2),
   509:             nn.ReLU(),
   510:             nn.Linear(hidden_dim // 2, max_nodes),
   511:         )
   512: 
   513:         self.optimizer = optim.Adam(self.parameters(), lr=lr)
   514: 
   515:     def encode(self, adj):
   516:         B, N, _ = adj.shape
   517:         # Use identity as node features (one-hot position)
   518:         x = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
   519:         h = F.relu(self.gcn1(x, adj))
   520:         h = F.relu(self.gcn2(h, adj))
   521:         # Graph-level readout (mean pooling)
   522:         h_graph = h.mean(dim=1)  # [B, hidden]
   523:         return self.fc_mu(h_graph), self.fc_logvar(h_graph)
   524: 
   525:     def reparameterize(self, mu, logvar):
   526:         std = torch.exp(0.5 * logvar)
   527:         eps = torch.randn_like(std)
   528:         return mu + eps * std
   529: 
   530:     def decode(self, z):
   531:         B = z.shape[0]
   532:         logits = self.decoder(z).view(B, self.max_nodes, self.max_nodes)
   533:         # Symmetrize
   534:         logits = (logits + logits.transpose(1, 2)) / 2
   535:         # Zero diagonal
   536:         mask = 1 - torch.eye(self.max_nodes, device=z.device).unsqueeze(0)
   537:         return logits * mask
   538: 
   539:     def train_step(self, adj, node_counts):
   540:         self.train()
   541:         self.optimizer.zero_grad()
   542: 
   543:         mu, logvar = self.encode(adj)
   544:         z = self.reparameterize(mu, logvar)
   545:         adj_logits = self.decode(z)
   546:         node_logits = self.node_pred(z)  # [B, max_nodes]
   547: 
   548:         # Reconstruction loss
   549:         recon_loss = F.binary_cross_entropy_with_logits(adj_logits, adj, reduction="mean")
   550: 
   551:         # Node existence loss
   552:         node_target = (adj.sum(dim=-1) > 0).float()  # [B, max_nodes]
   553:         node_loss = F.binary_cross_entropy_with_logits(node_logits, node_target, reduction="mean")
   554: 
   555:         # KL divergence
   556:         kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
   557: 
   558:         loss = recon_loss + 0.5 * node_loss + 0.001 * kl_loss
   559:         loss.backward()
   560:         self.optimizer.step()
   561: 
   562:         return {"loss": loss.item(), "recon": recon_loss.item(), "kl": kl_loss.item()}
   563: 
   564:     def sample(self, n_samples, device):
   565:         self.eval()
   566:         with torch.no_grad():
   567:             z = torch.randn(n_samples, self.latent_dim, device=device)
   568:             adj_logits = self.decode(z)
   569:             adj = (torch.sigmoid(adj_logits) > 0.5).float()
   570:             node_logits = self.node_pred(z)
   571:             node_probs = torch.sigmoid(node_logits)
   572:             node_mask = (node_probs > 0.5).float()
   573:             # Mask adjacency by existing nodes
   574:             adj = adj * node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
   575:             node_counts = node_mask.sum(dim=-1).long()
   576:             node_counts = torch.clamp(node_counts, min=2)
   577:         return adj, node_counts
   578: 
   579: # ============================================================================
   580: # EDITABLE REGION END (lines 446-590)
   581: # ============================================================================
```

### `gran` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-geometric/custom_graphgen.py`:

```python
Lines 446–637:
   443: # ============================================================================
   444: # EDITABLE REGION START (lines 446-590)
   445: # ============================================================================
   446: # The agent should modify the GraphGenerator class below.
   447: # The class must implement:
   448: #   - __init__(self, max_nodes, **kwargs): initialize model parameters
   449: #   - train_step(self, adj, node_counts) -> dict: one training step, returns loss dict
   450: #   - sample(self, n_samples, device) -> (adj_matrices, node_counts):
   451: #       generate n_samples graphs, return adjacency tensors and node count tensors
   452: #
   453: # The model receives adjacency matrices [B, max_nodes, max_nodes] and node counts [B].
   454: # It should generate adjacency matrices of similar structure.
   455: # ============================================================================
   456: 
   457: class AttentionBlock(nn.Module):
   458:     """Multi-head attention for graph nodes."""
   459: 
   460:     def __init__(self, dim, n_heads=4):
   461:         super().__init__()
   462:         self.n_heads = n_heads
   463:         self.head_dim = dim // n_heads
   464:         self.qkv = nn.Linear(dim, 3 * dim)
   465:         self.proj = nn.Linear(dim, dim)
   466:         self.norm = nn.LayerNorm(dim)
   467: 
   468:     def forward(self, x, mask=None):
   469:         B, N, C = x.shape
   470:         qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
   471:         qkv = qkv.permute(2, 0, 3, 1, 4)
   472:         q, k, v = qkv[0], qkv[1], qkv[2]
   473:         attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
   474:         if mask is not None:
   475:             attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float('-inf'))
   476:         attn = F.softmax(attn, dim=-1)
   477:         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
   478:         return self.norm(x + self.proj(out))
   479: 
   480: 
   481: class GRANBlock(nn.Module):
   482:     """GRAN message passing block with attention and edge prediction."""
   483: 
   484:     def __init__(self, node_dim, edge_dim=1, n_heads=4):
   485:         super().__init__()
   486:         self.attn = AttentionBlock(node_dim, n_heads)
   487:         self.edge_mlp = nn.Sequential(
   488:             nn.Linear(2 * node_dim + edge_dim, node_dim),
   489:             nn.ReLU(),
   490:             nn.Linear(node_dim, edge_dim),
   491:         )
   492:         self.node_mlp = nn.Sequential(
   493:             nn.Linear(node_dim + edge_dim, node_dim),
   494:             nn.ReLU(),
   495:             nn.Linear(node_dim, node_dim),
   496:         )
   497:         self.norm = nn.LayerNorm(node_dim)
   498: 
   499:     def forward(self, node_feat, edge_feat, mask=None):
   500:         B, N, D = node_feat.shape
   501:         # Attention-based node update
   502:         node_feat = self.attn(node_feat, mask)
   503: 
   504:         # Edge update
   505:         ni = node_feat.unsqueeze(2).expand(-1, -1, N, -1)
   506:         nj = node_feat.unsqueeze(1).expand(-1, N, -1, -1)
   507:         edge_input = torch.cat([ni, nj, edge_feat], dim=-1)
   508:         edge_feat = edge_feat + self.edge_mlp(edge_input)
   509: 
   510:         # Aggregate edge info to nodes
   511:         if mask is not None:
   512:             edge_agg = (edge_feat * mask.unsqueeze(-1).unsqueeze(-1).float()).sum(dim=2)
   513:         else:
   514:             edge_agg = edge_feat.mean(dim=2)
   515:         node_input = torch.cat([node_feat, edge_agg], dim=-1)
   516:         node_feat = self.norm(node_feat + self.node_mlp(node_input))
   517: 
   518:         return node_feat, edge_feat
   519: 
   520: 
   521: class GraphGenerator(nn.Module):
   522:     """GRAN: Graph Recurrent Attention Network.
   523: 
   524:     Iteratively refines node and edge representations using attention-based
   525:     message passing, then predicts edge probabilities for graph generation.
   526: 
   527:     Reference: Liao et al., NeurIPS 2019.
   528:     """
   529: 
   530:     def __init__(self, max_nodes, hidden_dim=128, n_layers=3, n_heads=4,
   531:                  n_refine_steps=5, lr=1e-3, **kwargs):
   532:         super().__init__()
   533:         self.max_nodes = max_nodes
   534:         self.hidden_dim = hidden_dim
   535:         self.n_refine_steps = n_refine_steps
   536: 
   537:         # Node embedding
   538:         self.node_embed = nn.Linear(max_nodes, hidden_dim)
   539: 
   540:         # GRAN blocks (shared across refinement steps)
   541:         self.blocks = nn.ModuleList([
   542:             GRANBlock(hidden_dim, edge_dim=1, n_heads=n_heads)
   543:             for _ in range(n_layers)
   544:         ])
   545: 
   546:         # Final edge prediction
   547:         self.edge_pred = nn.Sequential(
   548:             nn.Linear(2 * hidden_dim + 1, hidden_dim),
   549:             nn.ReLU(),
   550:             nn.Linear(hidden_dim, 1),
   551:         )
   552: 
   553:         # Node existence prediction
   554:         self.node_pred = nn.Sequential(
   555:             nn.Linear(hidden_dim, hidden_dim // 2),
   556:             nn.ReLU(),
   557:             nn.Linear(hidden_dim // 2, 1),
   558:         )
   559: 
   560:         self.optimizer = optim.Adam(self.parameters(), lr=lr)
   561: 
   562:     def _forward(self, adj, node_mask=None):
   563:         B, N, _ = adj.shape
   564:         device = adj.device
   565: 
   566:         # Initial node features (identity-based)
   567:         x = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
   568:         node_feat = F.relu(self.node_embed(x))  # [B, N, hidden]
   569: 
   570:         # Initial edge features from adjacency
   571:         edge_feat = adj.unsqueeze(-1)  # [B, N, N, 1]
   572: 
   573:         # Iterative refinement
   574:         for block in self.blocks:
   575:             node_feat, edge_feat = block(node_feat, edge_feat, node_mask)
   576: 
   577:         # Predict edges
   578:         ni = node_feat.unsqueeze(2).expand(-1, -1, N, -1)
   579:         nj = node_feat.unsqueeze(1).expand(-1, N, -1, -1)
   580:         edge_input = torch.cat([ni, nj, edge_feat], dim=-1)
   581:         edge_logits = self.edge_pred(edge_input).squeeze(-1)  # [B, N, N]
   582: 
   583:         # Symmetrize and remove self-loops
   584:         edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
   585:         diag_mask = 1 - torch.eye(N, device=device).unsqueeze(0)
   586:         edge_logits = edge_logits * diag_mask
   587: 
   588:         # Node existence
   589:         node_logits = self.node_pred(node_feat).squeeze(-1)  # [B, N]
   590: 
   591:         return edge_logits, node_logits
   592: 
   593:     def train_step(self, adj, node_counts):
   594:         self.train()
   595:         self.optimizer.zero_grad()
   596: 
   597:         edge_logits, node_logits = self._forward(adj)
   598: 
   599:         # Edge loss
   600:         edge_loss = F.binary_cross_entropy_with_logits(edge_logits, adj, reduction="mean")
   601: 
   602:         # Node existence loss
   603:         node_target = (adj.sum(dim=-1) > 0).float()
   604:         node_loss = F.binary_cross_entropy_with_logits(node_logits, node_target, reduction="mean")
   605: 
   606:         loss = edge_loss + 0.5 * node_loss
   607:         loss.backward()
   608:         torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
   609:         self.optimizer.step()
   610: 
   611:         return {"loss": loss.item(), "edge_loss": edge_loss.item()}
   612: 
   613:     def sample(self, n_samples, device):
   614:         self.eval()
   615:         with torch.no_grad():
   616:             # Start from random sparse adjacency (not zeros) to provide edge signal.
   617:             # Starting from zeros leads to empty output: node aggregation is zero ->
   618:             # node predictor predicts all nodes absent -> empty graph.
   619:             p_init = 0.3
   620:             adj = (torch.rand(n_samples, self.max_nodes, self.max_nodes, device=device) < p_init).float()
   621:             adj = torch.triu(adj, diagonal=1)
   622:             adj = adj + adj.transpose(1, 2)
   623: 
   624:             for step in range(self.n_refine_steps):
   625:                 edge_logits, node_logits = self._forward(adj)
   626:                 edge_probs = torch.sigmoid(edge_logits)
   627:                 adj = (torch.rand_like(edge_probs) < edge_probs).float()
   628:                 adj = torch.triu(adj, diagonal=1)
   629:                 adj = adj + adj.transpose(1, 2)
   630: 
   631:             # Derive node counts from connectivity (node predictor is unreliable
   632:             # when initialized from random adjacency at inference time).
   633:             node_counts = (adj.sum(dim=-1) > 0).long().sum(dim=-1)
   634:             node_counts = torch.clamp(node_counts, min=2)
   635: 
   636:         return adj, node_counts
   637: 
   638: # ============================================================================
   639: # EDITABLE REGION END (lines 446-590)
   640: # ============================================================================
```

### `digress` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-geometric/custom_graphgen.py`:

```python
Lines 446–671:
   443: # ============================================================================
   444: # EDITABLE REGION START (lines 446-590)
   445: # ============================================================================
   446: # The agent should modify the GraphGenerator class below.
   447: # The class must implement:
   448: #   - __init__(self, max_nodes, **kwargs): initialize model parameters
   449: #   - train_step(self, adj, node_counts) -> dict: one training step, returns loss dict
   450: #   - sample(self, n_samples, device) -> (adj_matrices, node_counts):
   451: #       generate n_samples graphs, return adjacency tensors and node count tensors
   452: #
   453: # The model receives adjacency matrices [B, max_nodes, max_nodes] and node counts [B].
   454: # It should generate adjacency matrices of similar structure.
   455: # ============================================================================
   456: 
   457: class GraphTransformerLayer(nn.Module):
   458:     """Graph transformer layer with edge-aware attention."""
   459: 
   460:     def __init__(self, dim, n_heads=4, ff_dim=None):
   461:         super().__init__()
   462:         ff_dim = ff_dim or 4 * dim
   463:         self.n_heads = n_heads
   464:         self.head_dim = dim // n_heads
   465: 
   466:         self.q = nn.Linear(dim, dim)
   467:         self.k = nn.Linear(dim, dim)
   468:         self.v = nn.Linear(dim, dim)
   469:         self.edge_bias = nn.Linear(1, n_heads)
   470:         self.proj = nn.Linear(dim, dim)
   471: 
   472:         self.norm1 = nn.LayerNorm(dim)
   473:         self.norm2 = nn.LayerNorm(dim)
   474:         self.ff = nn.Sequential(
   475:             nn.Linear(dim, ff_dim),
   476:             nn.GELU(),
   477:             nn.Linear(ff_dim, dim),
   478:         )
   479: 
   480:     def forward(self, x, adj):
   481:         B, N, C = x.shape
   482:         # Multi-head attention with edge bias
   483:         q = self.q(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
   484:         k = self.k(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
   485:         v = self.v(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
   486: 
   487:         attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
   488: 
   489:         # Edge bias
   490:         edge_b = self.edge_bias(adj.unsqueeze(-1))  # [B, N, N, n_heads]
   491:         attn = attn + edge_b.permute(0, 3, 1, 2)
   492: 
   493:         attn = F.softmax(attn, dim=-1)
   494:         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
   495:         x = self.norm1(x + self.proj(out))
   496:         x = self.norm2(x + self.ff(x))
   497:         return x
   498: 
   499: 
   500: class DiscreteDenoiser(nn.Module):
   501:     """Denoiser network for discrete adjacency diffusion."""
   502: 
   503:     def __init__(self, max_nodes, hidden_dim=128, n_layers=4, n_heads=4):
   504:         super().__init__()
   505:         self.node_embed = nn.Linear(max_nodes, hidden_dim)
   506:         self.time_embed = nn.Sequential(
   507:             nn.Linear(1, hidden_dim),
   508:             nn.SiLU(),
   509:             nn.Linear(hidden_dim, hidden_dim),
   510:         )
   511:         self.layers = nn.ModuleList([
   512:             GraphTransformerLayer(hidden_dim, n_heads)
   513:             for _ in range(n_layers)
   514:         ])
   515:         self.edge_pred = nn.Sequential(
   516:             nn.Linear(2 * hidden_dim + 1, hidden_dim),
   517:             nn.ReLU(),
   518:             nn.Linear(hidden_dim, 1),
   519:         )
   520:         self.node_pred = nn.Sequential(
   521:             nn.Linear(hidden_dim, hidden_dim // 2),
   522:             nn.ReLU(),
   523:             nn.Linear(hidden_dim // 2, 1),
   524:         )
   525: 
   526:     def forward(self, adj_noisy, t):
   527:         B, N, _ = adj_noisy.shape
   528:         device = adj_noisy.device
   529: 
   530:         # Node features
   531:         x = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
   532:         x = self.node_embed(x)
   533: 
   534:         # Add time conditioning
   535:         if t.dim() == 1:
   536:             t = t.unsqueeze(-1)
   537:         t_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, hidden]
   538:         x = x + t_emb
   539: 
   540:         # Graph transformer layers
   541:         for layer in self.layers:
   542:             x = layer(x, adj_noisy)
   543: 
   544:         # Edge prediction
   545:         ni = x.unsqueeze(2).expand(-1, -1, N, -1)
   546:         nj = x.unsqueeze(1).expand(-1, N, -1, -1)
   547:         edge_input = torch.cat([ni, nj, adj_noisy.unsqueeze(-1)], dim=-1)
   548:         edge_logits = self.edge_pred(edge_input).squeeze(-1)
   549:         edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
   550:         mask = 1 - torch.eye(N, device=device).unsqueeze(0)
   551:         edge_logits = edge_logits * mask
   552: 
   553:         # Node prediction
   554:         node_logits = self.node_pred(x).squeeze(-1)
   555: 
   556:         return edge_logits, node_logits
   557: 
   558: 
   559: class GraphGenerator(nn.Module):
   560:     """DiGress: Discrete denoising diffusion for graphs.
   561: 
   562:     Uses a discrete corruption process (edge flipping) and a graph
   563:     transformer denoiser to predict the clean graph.
   564: 
   565:     Reference: Vignac et al., ICLR 2023.
   566:     """
   567: 
   568:     def __init__(self, max_nodes, hidden_dim=128, n_layers=4, n_heads=4,
   569:                  n_diffusion_steps=50, lr=2e-4, **kwargs):
   570:         super().__init__()
   571:         self.max_nodes = max_nodes
   572:         self.n_steps = n_diffusion_steps
   573: 
   574:         # Beta schedule: cosine schedule for discrete diffusion
   575:         steps = torch.arange(n_diffusion_steps + 1, dtype=torch.float64)
   576:         alpha_bar = torch.cos((steps / n_diffusion_steps + 0.008) / 1.008 * math.pi / 2) ** 2
   577:         alpha_bar = alpha_bar / alpha_bar[0]
   578:         betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
   579:         betas = torch.clamp(betas, max=0.999)
   580:         self.register_buffer("betas", betas.float())
   581:         self.register_buffer("alpha_bar", alpha_bar[1:].float())
   582: 
   583:         self.denoiser = DiscreteDenoiser(max_nodes, hidden_dim, n_layers, n_heads)
   584:         self.optimizer = optim.Adam(self.denoiser.parameters(), lr=lr)
   585: 
   586:     def _corrupt(self, adj, t_idx):
   587:         """Discrete corruption: flip edges with probability depending on t."""
   588:         B = adj.shape[0]
   589:         device = adj.device
   590: 
   591:         # Flip probability = 0.5 * (1 - alpha_bar_t)
   592:         alpha_bar_t = self.alpha_bar[t_idx].view(B, 1, 1)
   593:         flip_prob = 0.5 * (1 - alpha_bar_t)
   594: 
   595:         # Sample flip mask
   596:         flip_mask = (torch.rand_like(adj) < flip_prob).float()
   597:         # Make symmetric
   598:         flip_mask = torch.triu(flip_mask, diagonal=1)
   599:         flip_mask = flip_mask + flip_mask.transpose(1, 2)
   600: 
   601:         # Apply flips: XOR with flip mask
   602:         adj_noisy = torch.abs(adj - flip_mask)
   603:         return adj_noisy
   604: 
   605:     def train_step(self, adj, node_counts):
   606:         self.train()
   607:         self.optimizer.zero_grad()
   608:         B = adj.shape[0]
   609:         device = adj.device
   610: 
   611:         # Sample random timestep
   612:         t_idx = torch.randint(0, self.n_steps, (B,), device=device)
   613: 
   614:         # Corrupt adjacency
   615:         adj_noisy = self._corrupt(adj, t_idx)
   616: 
   617:         # Predict clean adjacency
   618:         t_float = t_idx.float() / self.n_steps
   619:         edge_logits, node_logits = self.denoiser(adj_noisy, t_float)
   620: 
   621:         # Cross-entropy loss to predict original clean graph
   622:         edge_loss = F.binary_cross_entropy_with_logits(edge_logits, adj, reduction="mean")
   623: 
   624:         # Node existence loss
   625:         node_target = (adj.sum(dim=-1) > 0).float()
   626:         node_loss = F.binary_cross_entropy_with_logits(node_logits, node_target, reduction="mean")
   627: 
   628:         loss = edge_loss + 0.5 * node_loss
   629:         loss.backward()
   630:         torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
   631:         self.optimizer.step()
   632: 
   633:         return {"loss": loss.item(), "edge_loss": edge_loss.item()}
   634: 
   635:     def sample(self, n_samples, device):
   636:         """Generate graphs via iterative discrete denoising."""
   637:         self.eval()
   638:         N = self.max_nodes
   639:         mask = 1 - torch.eye(N, device=device).unsqueeze(0)
   640: 
   641:         with torch.no_grad():
   642:             # Start from random binary adjacency (Bernoulli(0.5))
   643:             adj = (torch.rand(n_samples, N, N, device=device) > 0.5).float()
   644:             adj = torch.triu(adj, diagonal=1)
   645:             adj = adj + adj.transpose(1, 2)
   646: 
   647:             for step in range(self.n_steps - 1, -1, -1):
   648:                 t_float = torch.ones(n_samples, device=device) * (step / self.n_steps)
   649:                 edge_logits, node_logits = self.denoiser(adj, t_float)
   650:                 edge_probs = torch.sigmoid(edge_logits)
   651: 
   652:                 if step > 0:
   653:                     # Sample with some noise
   654:                     adj = (torch.rand_like(edge_probs) < edge_probs).float()
   655:                 else:
   656:                     # Final step: use threshold
   657:                     adj = (edge_probs > 0.5).float()
   658: 
   659:                 # Ensure symmetry and no self-loops
   660:                 adj = torch.triu(adj, diagonal=1)
   661:                 adj = adj + adj.transpose(1, 2)
   662: 
   663:             # Node counts from predictor
   664:             node_probs = torch.sigmoid(node_logits)
   665:             node_mask_pred = (node_probs > 0.5).float()
   666:             adj = adj * node_mask_pred.unsqueeze(-1) * node_mask_pred.unsqueeze(-2)
   667:             node_counts = node_mask_pred.sum(dim=-1).long()
   668:             node_counts = torch.clamp(node_counts, min=2)
   669: 
   670:         return adj, node_counts
   671: 
   672: # ============================================================================
   673: # EDITABLE REGION END (lines 446-590)
   674: # ============================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
