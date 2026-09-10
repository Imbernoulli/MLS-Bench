# Unsupervised Anomaly Detection Algorithm Design

## Research Question
Design a novel unsupervised anomaly detection algorithm for tabular data that generalizes across datasets with different sample counts, dimensionality, and anomaly rates. The contribution is the *scoring rule* — how to model normal structure on standardized tabular features and assign higher scores to deviating points — using only unlabeled features at fit time.

## Background
Unsupervised anomaly detection identifies rare or unusual samples without labels during training. No single method dominates across dataset characteristics; promising designs combine density, isolation, distance, projection, ensemble, or robust-statistics ideas.

Reference baselines:
- **Isolation Forest (iForest)** — Liu, Ting, Zhou, ICDM 2008 ([paper](https://ieeexplore.ieee.org/document/4781136)). Tree-based isolation: anomalies are isolated with shorter random-partition path lengths. Default hyperparameters: 100 trees, sub-sample size 256.
- **Local Outlier Factor (LOF)** — Breunig, Kriegel, Ng, Sander, SIGMOD 2000. Density-based: ratio of a point's local reachability density to that of its k-nearest neighbors. Default `n_neighbors=20`.
- **One-Class SVM (OCSVM)** — Schölkopf, Platt, Shawe-Taylor, Smola, Williamson, 2001. Boundary-based: RBF kernel with `nu` controlling outlier fraction.
- **ECOD (Empirical Cumulative-distribution Outlier Detection)** — Li, Zhao, Hu, Botta, Ionescu, Chen, TKDE 2022 ([arXiv:2201.00382](https://arxiv.org/abs/2201.00382)). Per-dimension empirical CDFs; aggregate (negative) log tail probabilities across dimensions. Parameter-free.
- **COPOD (Copula-Based Outlier Detection)** — Li, Zhao, Botta, Ionescu, Hu, ICDM 2020 ([arXiv:2009.09463](https://arxiv.org/abs/2009.09463)). Empirical copula on per-dimension marginals; uses left/right/skewness-corrected tail probabilities. Parameter-free.

## Implementation Contract
Implement `CustomAnomalyDetector` in `custom_anomaly.py`:

```python
class CustomAnomalyDetector:
    def __init__(self):
        # Initialize hyperparameters and internal state
        ...

    def fit(self, X):
        # X: numpy array (n_samples, n_features), already standardized
        # (zero mean, unit variance). No labels used.
        return self

    def decision_function(self, X):
        # Return anomaly scores: numpy array (n_samples,)
        # Higher = more anomalous.
        return scores
```

Available libraries: `numpy`, `scipy` (linear algebra, statistics, spatial, optimization), `scikit-learn` (PCA, KDE, NearestNeighbors, GaussianMixture, ...), `pyod` (IForest, LOF, OCSVM, ECOD, COPOD, KNN, HBOS, PCA, LODA, SUOD, ...).

## Fixed Pipeline & Evaluation
Datasets (from ADBench / ODDS):
- **Cardio** — 1,831 samples, 21 features, ~9.6% anomalies (cardiotocography).
- **Thyroid** — 3,772 samples, 6 features, ~2.5% anomalies.
- **Satellite** — 6,435 samples, 36 features, ~31.6% anomalies (Landsat).
- **Shuttle** — 49,097 samples, 9 features, ~7.2% anomalies (NASA shuttle).

Protocol: 60/40 stratified train/test split (standard ADBench/ECOD protocol). Detector fits on train features without labels; scores are computed for test features.

Metrics (higher is better):
- **AUROC** — area under ROC curve (ranking quality).
- **F1** — F1 score at the optimal contamination threshold (decision quality after thresholding).
