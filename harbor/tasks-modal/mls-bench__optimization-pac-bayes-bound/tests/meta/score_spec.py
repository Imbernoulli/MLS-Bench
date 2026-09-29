"""Score spec for optimization-pac-bayes-bound.

KL(Q||P) is reported but not scored on its own: it already enters both bounds
(risk_certificate and ce_bound). As a separate objective it paid a posterior
that does not move from the prior (KL = 0, a no-op train_step) more than the
best baseline (0.344 vs 0.339 under the round-2 anchors)."""
from mlsbench.scoring.dsl import *

# mnist-fcn setting
term("risk_certificate_mnist_fcn",
    col("risk_certificate_mnist-fcn").lower().id()
    .bounded_power(bound=0.0))

term("test_error_mnist_fcn",
    col("test_error_mnist-fcn").lower().id()
    .bounded_power(bound=0.0))

term("ce_bound_mnist_fcn",
    col("ce_bound_mnist-fcn").lower().id()
    .bounded_power(bound=0.0))

term("empirical_01_risk_mnist_fcn",
    col("empirical_01_risk_mnist-fcn").lower().id()
    .bounded_power(bound=0.0))

# mnist-cnn setting
term("risk_certificate_mnist_cnn",
    col("risk_certificate_mnist-cnn").lower().id()
    .bounded_power(bound=0.0))

term("test_error_mnist_cnn",
    col("test_error_mnist-cnn").lower().id()
    .bounded_power(bound=0.0))

term("ce_bound_mnist_cnn",
    col("ce_bound_mnist-cnn").lower().id()
    .bounded_power(bound=0.0))

term("empirical_01_risk_mnist_cnn",
    col("empirical_01_risk_mnist-cnn").lower().id()
    .bounded_power(bound=0.0))

# fmnist-cnn setting
term("risk_certificate_fmnist_cnn",
    col("risk_certificate_fmnist-cnn").lower().id()
    .bounded_power(bound=0.0))

term("test_error_fmnist_cnn",
    col("test_error_fmnist-cnn").lower().id()
    .bounded_power(bound=0.0))

term("ce_bound_fmnist_cnn",
    col("ce_bound_fmnist-cnn").lower().id()
    .bounded_power(bound=0.0))

term("empirical_01_risk_fmnist_cnn",
    col("empirical_01_risk_fmnist-cnn").lower().id()
    .bounded_power(bound=0.0))

setting("mnist-fcn", weighted_mean(
    ("risk_certificate_mnist_fcn", 1.0),
    ("test_error_mnist_fcn", 1.0),
    ("ce_bound_mnist_fcn", 1.0),
    ("empirical_01_risk_mnist_fcn", 1.0),
))
setting("mnist-cnn", weighted_mean(
    ("risk_certificate_mnist_cnn", 1.0),
    ("test_error_mnist_cnn", 1.0),
    ("ce_bound_mnist_cnn", 1.0),
    ("empirical_01_risk_mnist_cnn", 1.0),
))
setting("fmnist-cnn", weighted_mean(
    ("risk_certificate_fmnist_cnn", 1.0),
    ("test_error_fmnist_cnn", 1.0),
    ("ce_bound_fmnist_cnn", 1.0),
    ("empirical_01_risk_fmnist_cnn", 1.0),
))

task(gmean("mnist-fcn", "mnist-cnn", "fmnist-cnn"))
