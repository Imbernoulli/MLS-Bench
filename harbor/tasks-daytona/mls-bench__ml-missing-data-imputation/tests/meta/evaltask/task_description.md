# Missing Data Imputation

## Research Question
Design a tabular missing-data imputation method that achieves low reconstruction error and preserves downstream predictive performance across diverse datasets. The contribution is the *imputer itself*: how feature dependencies are exploited, how imputations are iterated/refined, and how completed values are produced from data containing NaNs.

## Background
Missing data is ubiquitous. Mean/median imputation ignores feature correlations; iterative predictive methods exploit them.

Reference baselines:
- **Mean imputation** — replace NaNs in each column with the column mean from training data.
- **k-Nearest Neighbors imputation** — Troyanskaya et al., Bioinformatics 2001. For each missing entry, average over the `k` most similar rows (computed on observed features). Default `n_neighbors=5`.
- **MICE (Multivariate Imputation by Chained Equations)** — van Buuren & Groothuis-Oudshoorn, JSS 2011 ([paper](https://www.jstatsoft.org/v45/i03/)). Iterative: at each round and for each variable with missingness, fit a regression of that variable on all others (using the latest imputations) and replace its missing values with predictions. `sklearn.impute.IterativeImputer` is the de-facto MICE implementation; default `max_iter=10`.
- **MissForest** — Stekhoven & Bühlmann, Bioinformatics 2012 ([paper](https://academic.oup.com/bioinformatics/article/28/1/112/219101)). Iterative random-forest-based imputation; same chained-equations skeleton as MICE but uses a Random Forest as the per-variable predictor. Handles mixed-type data and complex interactions.
- **GAIN (Generative Adversarial Imputation Nets)** — Yoon, Jordon, van der Schaar, ICML 2018 ([arXiv:1806.02920](https://arxiv.org/abs/1806.02920)). GAN-based: generator imputes missing entries conditional on observed ones; discriminator tries to identify which entries were imputed; a hint mechanism reveals partial mask information.

## Implementation Contract
Implement `CustomImputer` in `scikit-learn/custom_imputation.py`:

```python
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42, max_iter=10):
        ...

    def fit(self, X, y=None):
        # X: numpy array (n_samples, n_features) with NaN for missing values.
        # Learn imputation model. Must NOT use test labels.
        return self

    def transform(self, X):
        # X: numpy array (n_samples, n_features) with NaN.
        # Return: numpy array of the same shape with NO NaNs (finite values).
        return X_imputed
```

Available libraries: `numpy`, `scipy`, `scikit-learn` (all submodules: `sklearn.impute`, `sklearn.ensemble`, `sklearn.neighbors`, ...).

## Fixed Pipeline & Evaluation
Datasets, all with **20% MCAR (Missing Completely At Random)** corruption:
- **Breast Cancer Wisconsin** — 569 samples, 30 features, binary classification.
- **Wine** — 178 samples, 13 features, 3-class classification.
- **California Housing** — 5,000 samples, 8 features, regression.

Metrics:
- **RMSE** — root mean squared error between imputed and true values on the masked entries (lower is better).
- **downstream_score** — accuracy (breast_cancer, wine) or R² (california) of a `GradientBoosting` model trained on the imputed data (higher is better).
