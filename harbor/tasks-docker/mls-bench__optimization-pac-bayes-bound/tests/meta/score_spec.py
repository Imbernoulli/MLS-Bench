"""Score spec for optimization-pac-bayes-bound."""
from mlsbench.scoring.dsl import *

# mnist-fcn setting
term("risk_certificate_mnist_fcn",
    col("risk_certificate_mnist-fcn").lower().id()
    .bounded_power(bound=0.0))

term("test_error_mnist_fcn",
    col("test_error_mnist-fcn").lower().id()
    .bounded_power(bound=0.0))

term("kl_divergence_mnist_fcn",
    col("kl_divergence_mnist-fcn").lower().id()
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

term("kl_divergence_mnist_cnn",
    col("kl_divergence_mnist-cnn").lower().id()
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

term("kl_divergence_fmnist_cnn",
    col("kl_divergence_fmnist-cnn").lower().id()
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
    ("kl_divergence_mnist_fcn", 1.0),
    ("ce_bound_mnist_fcn", 1.0),
    ("empirical_01_risk_mnist_fcn", 1.0),
))
setting("mnist-cnn", weighted_mean(
    ("risk_certificate_mnist_cnn", 1.0),
    ("test_error_mnist_cnn", 1.0),
    ("kl_divergence_mnist_cnn", 1.0),
    ("ce_bound_mnist_cnn", 1.0),
    ("empirical_01_risk_mnist_cnn", 1.0),
))
setting("fmnist-cnn", weighted_mean(
    ("risk_certificate_fmnist_cnn", 1.0),
    ("test_error_fmnist_cnn", 1.0),
    ("kl_divergence_fmnist_cnn", 1.0),
    ("ce_bound_fmnist_cnn", 1.0),
    ("empirical_01_risk_fmnist_cnn", 1.0),
))

task(gmean("mnist-fcn", "mnist-cnn", "fmnist-cnn"))
