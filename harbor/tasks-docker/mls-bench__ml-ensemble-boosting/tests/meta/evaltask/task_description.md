# Ensemble Boosting Strategy Design

## Research Question
Design a novel sample-weighting and update strategy for boosting that improves over standard methods (AdaBoost, gradient boosting, XGBoost-style Newton update) across both classification and regression tasks. The contribution is the *strategy itself* (how sample weights are initialized and updated, what pseudo-targets each weak learner fits, how each learner is weighted), with shallow decision trees as the fixed weak learner.

## Background
Boosting builds an ensemble of weak learners sequentially, each round trying to correct errors left by previous rounds. Key design axes:
- **Pseudo-target computation**: original labels (AdaBoost), negative gradients (gradient boosting), Newton-step targets using second-order information (XGBoost).
- **Learner weighting**: from weighted error (AdaBoost), fixed at 1.0 with learning rate shrinkage (gradient boosting), via line search / Newton optimization (XGBoost).
- **Sample reweighting**: exponential reweighting of misclassified samples (AdaBoost) vs. uniform weights with pseudo-residual fitting (gradient methods).

Reference baselines:
- **AdaBoost** — Freund & Schapire, JCSS 1997 ([paper](https://www.sciencedirect.com/science/article/pii/S002200009791504X)). Exponential loss; alpha = `0.5 * log((1-err)/err)`; multiplicative reweighting `w_i *= exp(alpha * 1[y_i ≠ h(x_i)])` (binary classification).
- **Gradient boosting** — Friedman, Annals of Statistics 2001. Fit each new tree to the negative gradient of the loss at current predictions; constant learner weight 1.0 with global learning-rate shrinkage (here `lr=0.1`).
- **XGBoost-style (second-order)** — Chen & Guestrin, KDD 2016 ([arXiv:1603.02754](https://arxiv.org/abs/1603.02754)). Use both gradient `g` and Hessian `h` of the loss; pseudo-targets and leaf values follow the Newton step `-g/h`.

## Implementation Contract
Modify `BoostingStrategy` in `scikit-learn/custom_boosting.py`:

```python
class BoostingStrategy:
    def init_weights(self, n_samples):
        # Initialize sample weights (should sum to 1).
        ...

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        # Pseudo-targets the next weak learner will fit.
        ...

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                               sample_weights, round_idx):
        # Alpha for the just-fitted learner.
        ...

    def update_weights(self, sample_weights, learner, X, y,
                       pseudo_targets, alpha, round_idx):
        # Sample weights for the next round.
        ...
```

Available context: true labels, current ensemble predictions, sample weights, fitted learner (`learner.predict(X)`), round index, config dict with dataset metadata. Available imports in the FIXED section: `numpy`, `sklearn.tree`, `sklearn.metrics`, `sklearn.datasets`, `sklearn.model_selection`.

## Fixed Pipeline & Evaluation
- 200 boosting rounds, base learner = `DecisionTree(max_depth=3)`, learning rate `0.1`, 80/20 train/test split.
- Datasets:
  - **Breast Cancer Wisconsin** — classification, 569 samples, 30 features → metric `test_accuracy` (higher is better).
  - **Diabetes** — regression, 442 samples, 10 features → metric `test_rmse` (lower is better).
  - **California Housing** — regression, 20,640 samples, 8 features → metric `test_rmse` (lower is better).
