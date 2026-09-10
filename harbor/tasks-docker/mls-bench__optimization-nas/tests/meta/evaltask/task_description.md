# Sample-Efficient Neural Architecture Search

## Objective
Design and implement a novel **sample-efficient** NAS optimizer that discovers high-performing architectures in the NAS-Bench-201 search space under a **strict query budget**. Your code goes in the `NASOptimizer` class in `custom_nas_search.py`. Three reference implementations (Random Search, REA, and a BANANAS-style predictor-guided search) are provided as read-only.

## Research Question
With only **K = 30 architecture evaluations**, how can a search strategy maximize the expected accuracy of the best-found architecture?

This is the regime in which real-world NAS is actually hard: the full benchmark contains 15,625 architectures, but the agent can only query 30 of them, so naïve enumeration is impossible and algorithmic differences are load-bearing. Sample-efficient NAS has been studied by BANANAS (White, Neiswanger, and Savani, AAAI 2021; arXiv:1910.11858), NPENAS (Wei, Niu, Chen, and Wang, IEEE TNNLS, 2022), and NAS-Bench-Suite (White et al., 2022) and consistently shows a measurable gap between random search, regularized evolution, and predictor-guided methods at K ≤ 50.

## Search Space
- NAS-Bench-201 cell: 4 nodes, 6 edges, 5 operations per edge (Dong and Yang, "NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search", ICLR 2020; arXiv:2001.00326).
- Operations: `skip_connect, none, nor_conv_3x3, nor_conv_1x1, avg_pool_3x3`.
- 5^6 = 15,625 architectures total.
- An architecture is represented as a list of 6 integers in `[0, 4]`.

## Evaluation Protocol
- Datasets: CIFAR-10, CIFAR-100, ImageNet16-120 (three separate settings).
- **Query budget: `NAS_EPOCHS = 30` validation queries per dataset per seed** (the harness enforces this; exceeding it aborts the run).
- Metric: **test accuracy of the final returned architecture** on the NAS-Bench-201 test split (one extra query at the end, not counted against the budget).
- Seeds: `{0, 1, 2, 3, 4}`. Report mean ± std across seeds — at K = 30, variance is non-trivial.

## What Counts as a Contribution
Acceptable research directions (this list is not exhaustive):
- **Better acquisition functions**: e.g. UCB / EI over a learned predictor, Thompson sampling, information-theoretic criteria.
- **Better surrogate models**: GPs on path-encoded architectures, GNN predictors, MLP ensembles, zero-cost proxy hybrids (Mellor, Turner, Storkey, and Crowley, "Neural Architecture Search without Training", ICML 2021; Abdelfattah, Mehrotra, Dudziak, and Lane, "Zero-Cost Proxies for Lightweight NAS", ICLR 2021).
- **Smarter exploration–exploitation mixing**: local search around the Pareto front, portfolio methods, warm-started evolution.
- **Encoding choices**: adjacency vs path encoding (White, Neiswanger, Nolen, and Savani, "A Study on Encodings for Neural Architecture Search", NeurIPS 2020 showed path encoding substantially improves predictor accuracy at low K).

What does **not** count:
- Increasing the effective budget (e.g. re-querying the same architecture, wrapping queries, etc.). The harness counts every call to `api.query_val_accuracy` and will terminate after `K = 30`.
- Hard-coding known good architectures from NAS-Bench-201 literature.

## Baselines (paper-cited reference implementations, all under the same K = 30 budget)

| Name | Strategy |
|------|----------|
| `random_search` | Uniform sampling over valid architectures. |
| `rea` | Regularized Evolution (Real, Aggarwal, Huang, and Le, AAAI 2019; arXiv:1802.01548) with tournament selection (paper-default `S = 10`, `population_size = 20`) and 1-edge mutation. |
| `bananas` | Predictor-guided: MLP ensemble over path encodings, pick candidate with highest predicted val_acc (White, Neiswanger, and Savani, AAAI 2021; arXiv:1910.11858). Paper-default 5-MLP ensemble, 100 mutation candidates per acquisition. |
