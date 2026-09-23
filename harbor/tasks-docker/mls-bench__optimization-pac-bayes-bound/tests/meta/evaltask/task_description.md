# Task: PAC-Bayes Generalization Bound Optimization

## Research Question
Design a tighter PAC-Bayes generalization bound by optimizing the bound formulation, prior/posterior parameterization, and KL divergence estimation for stochastic neural networks.

## Background
PAC-Bayes theory provides non-vacuous generalization bounds for stochastic classifiers (McAllester, "Some PAC-Bayesian Theorems", *Machine Learning* 37, 1999; Catoni, "PAC-Bayesian Supervised Classification", IMS Lecture Notes Vol. 56, 2007). Given a prior distribution `P` over hypotheses (chosen before seeing data) and a posterior `Q` (learned from data), PAC-Bayes bounds certify that with high probability `1 - delta`, the true risk of a stochastic classifier sampled from `Q` is bounded.

The first non-vacuous PAC-Bayes bounds for over-parameterized neural networks were obtained by Dziugaite and Roy, "Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data" (UAI 2017; arXiv:1703.11008), with subsequent tighter constructions in Pérez-Ortiz, Rivasplata, Shawe-Taylor, and Szepesvári, "Tighter Risk Certificates for Neural Networks" (JMLR 2021; arXiv:2007.12911).

The key components of a PAC-Bayes bound are:
- **Empirical risk**: estimated loss of the stochastic predictor on training data.
- **KL divergence**: `KL(Q || P)` measuring complexity of the posterior relative to the prior.
- **Bound formula**: how these terms combine to yield the final certificate.

Standard bounds include:
- **McAllester / Maurer**: `risk + sqrt(KL_term / (2n))` — simple but loose.
- **Catoni / Lambda**: `risk / (1 - lam/2) + KL_term / (n * lam * (1 - lam/2))` — tighter with tuned `lam` (Catoni, 2007).
- **Quadratic / inverted-kl**: `(sqrt(risk + KL_term) + sqrt(KL_term))^2` — better at low risk; PAC-Bayes-kl inversion (Seeger, 2002; Maurer, 2004) is provably the tightest.

The bound can be further tightened through:
- Optimizing the bound functional form (beyond classical inequalities).
- Better training objectives that minimize the bound directly.
- Improved risk certificate evaluation (e.g., PAC-Bayes-kl inversion).
- Data-dependent prior construction.
- Tighter KL estimation or alternative divergence measures.

## What to Implement
Implement the `BoundOptimizer` class in `custom_pac_bayes.py`. You must implement:
1. `compute_bound(empirical_risk, kl, n, delta)` — the PAC-Bayes bound formula.
2. `train_step(model, data, target, device, n_bound, delta)` — training objective.
3. `compute_risk_certificate(model, bound_loader, device, delta, mc_samples)` — your own certificate evaluation, printed for reference only.

The scored metrics (`risk_certificate`, `test_error`, `kl_divergence`, `ce_bound`, `empirical_01_risk`) are computed by fixed code (`evaluate_certificate` in `custom_pac_bayes.py`) from the trained posterior's parameters and the prior fixed before posterior training: the MC majority-vote 0-1 risk on the bound set, the closed-form `KL(Q || P)`, the PAC-Bayes-kl inversion `kl^-1(risk, (KL + log(2 sqrt(n) / delta)) / n)` as `risk_certificate`, and the McAllester bound on the cross-entropy (log-probabilities floored at `log(1e-5)`) as `ce_bound`. Nothing `compute_risk_certificate` or `compute_bound` returns enters the score. The prior (its mean and `prior_sigma`) must not be modified; a run that modifies it fails.

## Interface
- `model(x, sample=True/False)`: stochastic forward pass (`sample=True`) or posterior mean (`sample=False`).
- `get_total_kl(model)`: sum of KL divergence across all probabilistic layers.
- `inv_kl(q, c)`: binary KL inversion — find `p` such that `KL(Ber(q) || Ber(p)) = c`.
- `compute_01_risk(model, loader, device, mc_samples)`: MC estimate of 0-1 risk.
- Available losses: `F.nll_loss`, `F.cross_entropy` on log-softmax outputs.

## Evaluation
The bound optimizer is tested on three settings:
1. **MNIST-FCN**: 4-layer fully connected network (`784-600-600-600-10`) on MNIST.
2. **MNIST-CNN**: 4-layer CNN (2 conv + 2 fc) on MNIST.
3. **FashionMNIST-CNN**: same CNN architecture on FashionMNIST.

**Primary metric**: `risk_certificate` (0-1 loss PAC-Bayes bound) — **lower is better** (tighter bound). Test error, KL divergence, cross-entropy-style bound, and empirical 0-1 risk are also recorded.

Training uses data-dependent priors: 50% of training data trains a deterministic prior, 50% evaluates the bound (Pérez-Ortiz et al., 2021).

## Baselines (paper-cited bound formulations)
- **mcallester** — McAllester / Maurer bound: `risk + sqrt((KL + log(2 sqrt(n) / delta)) / (2n))` (McAllester, *Machine Learning* 1999; Maurer, "A Note on the PAC-Bayesian Theorem", arXiv:cs/0411099, 2004).
- **catoni** — Catoni's lambda bound (Catoni, IMS Lecture Notes 56, 2007); paper-default `lambda` tuned over a small grid.
- **quadratic** — quadratic / kl-inversion bound (Seeger, 2002; Maurer, 2004); standard inverted-kl form `(sqrt(risk + KL_term) + sqrt(KL_term))^2`.
