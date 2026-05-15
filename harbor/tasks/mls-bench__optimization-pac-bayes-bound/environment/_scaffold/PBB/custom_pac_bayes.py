"""PAC-Bayes Bound Optimization — custom template.

This script trains a stochastic neural network by minimizing a PAC-Bayes
bound and then evaluates the tightness of the resulting risk certificate.

The agent edits the EDITABLE section (BoundOptimizer class) which controls:
  1. How the PAC-Bayes bound is computed from empirical risk + KL divergence
  2. How the posterior distribution is optimized (training objective)
  3. How the final risk certificate is evaluated

Fixed sections handle data loading, model architecture, stochastic layers,
and the outer training loop.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms

# ================================================================
# FIXED — Stochastic layers and model architectures (do not modify)
# ================================================================


class Gaussian:
    """Gaussian weight distribution for variational inference."""

    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps

    def log_prob(self, x):
        return (
            -0.5 * math.log(2 * math.pi)
            - torch.log(self.sigma)
            - 0.5 * ((x - self.mu) / self.sigma) ** 2
        )


class ProbLinear(nn.Module):
    """Probabilistic linear layer with Gaussian weights and data-dependent prior."""

    def __init__(self, in_features, out_features, prior_sigma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize posterior rho so that initial sigma matches prior_sigma
        # sigma = log(1 + exp(rho)), so rho = log(exp(sigma) - 1)
        rho_init = math.log(math.exp(prior_sigma) - 1.0)

        # Posterior parameters (learnable)
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        )
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features).fill_(rho_init)
        )
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), rho_init))

        # Prior (fixed, data-dependent: set via set_prior_mu)
        self.prior_sigma = prior_sigma
        self.register_buffer("weight_prior_mu",
                             torch.zeros(out_features, in_features))
        self.register_buffer("bias_prior_mu", torch.zeros(out_features))

        self._kl = 0.0

    def set_prior_mu(self, weight_mu, bias_mu):
        """Set the prior mean from a trained deterministic model."""
        self.weight_prior_mu.copy_(weight_mu.data)
        self.bias_prior_mu.copy_(bias_mu.data)

    def forward(self, x, sample=True):
        if sample:
            w_posterior = Gaussian(self.weight_mu, self.weight_rho)
            b_posterior = Gaussian(self.bias_mu, self.bias_rho)
            weight = w_posterior.sample()
            bias = b_posterior.sample()

            # KL divergence: KL(q(w) || p(w)) with data-dependent prior
            # Prior is N(prior_mu, prior_sigma^2), posterior is N(mu, sigma^2)
            # Analytic KL for diagonal Gaussians
            q_sigma_w = w_posterior.sigma
            q_sigma_b = b_posterior.sigma
            p_var = self.prior_sigma ** 2

            kl_w = (0.5 * (
                (q_sigma_w ** 2 + (self.weight_mu - self.weight_prior_mu) ** 2) / p_var
                - 1.0
                + math.log(p_var) - 2.0 * torch.log(q_sigma_w)
            )).sum()

            kl_b = (0.5 * (
                (q_sigma_b ** 2 + (self.bias_mu - self.bias_prior_mu) ** 2) / p_var
                - 1.0
                + math.log(p_var) - 2.0 * torch.log(q_sigma_b)
            )).sum()

            self._kl = kl_w + kl_b
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            self._kl = 0.0

        return F.linear(x, weight, bias)


class ProbConv2d(nn.Module):
    """Probabilistic 2D convolution with Gaussian weights and data-dependent prior."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, prior_sigma=0.1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.prior_sigma = prior_sigma

        # Initialize posterior rho so that initial sigma matches prior_sigma
        rho_init = math.log(math.exp(prior_sigma) - 1.0)

        self.weight_mu = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
            .uniform_(-0.2, 0.2)
        )
        self.weight_rho = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
            .fill_(rho_init)
        )
        self.bias_mu = nn.Parameter(torch.zeros(out_channels))
        self.bias_rho = nn.Parameter(torch.full((out_channels,), rho_init))

        # Data-dependent prior mean (set via set_prior_mu)
        self.register_buffer("weight_prior_mu",
                             torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.register_buffer("bias_prior_mu", torch.zeros(out_channels))
        self._kl = 0.0

    def set_prior_mu(self, weight_mu, bias_mu):
        """Set the prior mean from a trained deterministic model."""
        self.weight_prior_mu.copy_(weight_mu.data)
        self.bias_prior_mu.copy_(bias_mu.data)

    def forward(self, x, sample=True):
        if sample:
            w_post = Gaussian(self.weight_mu, self.weight_rho)
            b_post = Gaussian(self.bias_mu, self.bias_rho)
            weight = w_post.sample()
            bias = b_post.sample()

            # Analytic KL with data-dependent prior N(prior_mu, prior_sigma^2)
            q_sigma_w = w_post.sigma
            q_sigma_b = b_post.sigma
            p_var = self.prior_sigma ** 2

            kl_w = (0.5 * (
                (q_sigma_w ** 2 + (self.weight_mu - self.weight_prior_mu) ** 2) / p_var
                - 1.0
                + math.log(p_var) - 2.0 * torch.log(q_sigma_w)
            )).sum()

            kl_b = (0.5 * (
                (q_sigma_b ** 2 + (self.bias_mu - self.bias_prior_mu) ** 2) / p_var
                - 1.0
                + math.log(p_var) - 2.0 * torch.log(q_sigma_b)
            )).sum()

            self._kl = kl_w + kl_b
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            self._kl = 0.0

        return F.conv2d(x, weight, bias, self.stride, self.padding)


def get_total_kl(model):
    """Sum KL divergence across all probabilistic layers."""
    kl = 0.0
    for m in model.modules():
        if hasattr(m, "_kl"):
            kl = kl + m._kl
    return kl


class StochasticFCN(nn.Module):
    """4-layer fully connected stochastic network for MNIST (28x28)."""

    def __init__(self, prior_sigma=0.1):
        super().__init__()
        self.fc1 = ProbLinear(784, 600, prior_sigma)
        self.fc2 = ProbLinear(600, 600, prior_sigma)
        self.fc3 = ProbLinear(600, 600, prior_sigma)
        self.fc4 = ProbLinear(600, 10, prior_sigma)

    def forward(self, x, sample=True):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x, sample))
        x = F.relu(self.fc2(x, sample))
        x = F.relu(self.fc3(x, sample))
        return self.fc4(x, sample)


class StochasticCNN(nn.Module):
    """4-layer CNN stochastic network (2 conv + 2 fc)."""

    def __init__(self, in_channels=1, num_classes=10, prior_sigma=0.1):
        super().__init__()
        self.conv1 = ProbConv2d(in_channels, 32, 3, padding=1, prior_sigma=prior_sigma)
        self.conv2 = ProbConv2d(32, 64, 3, padding=1, prior_sigma=prior_sigma)
        self.in_channels = in_channels
        # Compute flattened size after two 2x2 max pools
        if in_channels == 1:
            self._flat_size = 64 * 7 * 7  # MNIST/FashionMNIST: 28->14->7
        else:
            self._flat_size = 64 * 8 * 8  # CIFAR-10: 32->16->8
        self.fc1 = ProbLinear(self._flat_size, 256, prior_sigma)
        self.fc2 = ProbLinear(256, num_classes, prior_sigma)

    def forward(self, x, sample=True):
        x = F.relu(self.conv1(x, sample))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x, sample))
        return self.fc2(x, sample)


class DeterministicFCN(nn.Module):
    """Deterministic FCN for prior training."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 600)
        self.fc4 = nn.Linear(600, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DeterministicCNN(nn.Module):
    """Deterministic CNN for prior training."""

    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        if in_channels == 1:
            flat_size = 64 * 7 * 7
        else:
            flat_size = 64 * 8 * 8
        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.in_channels = in_channels

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ================================================================
# FIXED — Data loading utilities (do not modify)
# ================================================================


def load_dataset(name, data_dir="/workspace/data"):
    """Load dataset with standard normalization."""
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train = torchvision.datasets.MNIST(data_dir, train=True, download=False,
                                           transform=transform)
        test = torchvision.datasets.MNIST(data_dir, train=False, download=False,
                                          transform=transform)
        in_channels, num_classes = 1, 10
    elif name == "fashionmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train = torchvision.datasets.FashionMNIST(data_dir, train=True,
                                                   download=False, transform=transform)
        test = torchvision.datasets.FashionMNIST(data_dir, train=False,
                                                  download=False, transform=transform)
        in_channels, num_classes = 1, 10
    elif name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        train = torchvision.datasets.CIFAR10(data_dir, train=True, download=False,
                                             transform=transform)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=False,
                                            transform=transform)
        in_channels, num_classes = 3, 10
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, test, in_channels, num_classes


def split_data_for_prior(train_dataset, prior_frac=0.5, seed=42):
    """Split training data into prior-training set and bound-evaluation set."""
    n = len(train_dataset)
    n_prior = int(n * prior_frac)
    n_bound = n - n_prior
    gen = torch.Generator().manual_seed(seed)
    prior_set, bound_set = random_split(train_dataset, [n_prior, n_bound],
                                        generator=gen)
    return prior_set, bound_set


# ================================================================
# FIXED — Utility functions (do not modify)
# ================================================================


def inv_kl(q, c):
    """Compute the inverse KL: find the largest p such that KL(q||p) <= c.

    Uses binary search. KL(q||p) = q*log(q/p) + (1-q)*log((1-q)/(1-p)).
    Returns p >= q such that KL(q||p) = c.
    """
    if c < 0:
        raise ValueError("c must be non-negative")
    if q >= 1.0:
        return 1.0
    if c == 0:
        return q

    lo, hi = q, 1.0 - 1e-10
    for _ in range(64):
        mid = (lo + hi) / 2.0
        kl_val = _kl_bernoulli(q, mid)
        if kl_val < c:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _kl_bernoulli(q, p):
    """Binary KL divergence: KL(Ber(q) || Ber(p))."""
    if q < 1e-12:
        return -math.log(1 - p + 1e-12)
    if q > 1 - 1e-12:
        return -math.log(p + 1e-12)
    return q * math.log(q / (p + 1e-12)) + (1 - q) * math.log(
        (1 - q) / (1 - p + 1e-12)
    )


def compute_01_risk(model, loader, device, mc_samples=100):
    """Compute stochastic 0-1 risk via Monte Carlo sampling."""
    model.eval()
    total_wrong = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            votes = torch.zeros(batch_size, 10, device=device)

            for _ in range(mc_samples):
                logits = model(data, sample=True)
                preds = logits.argmax(dim=1)
                votes.scatter_add_(1, preds.unsqueeze(1),
                                   torch.ones(batch_size, 1, device=device))

            final_preds = votes.argmax(dim=1)
            total_wrong += (final_preds != target).sum().item()
            total_samples += batch_size

    return total_wrong / total_samples


def compute_test_error(model, loader, device):
    """Compute deterministic test error using posterior mean."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data, sample=False)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 1.0 - correct / total


def transfer_weights_to_stochastic(det_model, stoch_model):
    """Initialize stochastic model's posterior means and prior means from deterministic model."""
    det_state = det_model.state_dict()
    stoch_state = stoch_model.state_dict()

    mapping = {}
    for det_key in det_state:
        # Map fc1.weight -> fc1.weight_mu, fc1.bias -> fc1.bias_mu
        parts = det_key.rsplit(".", 1)
        if len(parts) == 2:
            prefix, suffix = parts
            mu_key = f"{prefix}.{suffix}_mu"
            if mu_key in stoch_state:
                mapping[det_key] = mu_key

    for det_key, stoch_key in mapping.items():
        stoch_state[stoch_key] = det_state[det_key]

    stoch_model.load_state_dict(stoch_state)

    # Set prior means on each probabilistic layer
    det_modules = dict(det_model.named_modules())
    for name, module in stoch_model.named_modules():
        if hasattr(module, "set_prior_mu") and name in det_modules:
            det_mod = det_modules[name]
            module.set_prior_mu(det_mod.weight, det_mod.bias)


# ================================================================
# EDITABLE SECTION — BoundOptimizer class (lines 460 to 604)
# The agent modifies this section to design tighter PAC-Bayes bounds.
# ================================================================


class BoundOptimizer:
    """PAC-Bayes bound computation and posterior optimization.

    This class controls:
    1. compute_bound(): How the generalization bound is computed from
       empirical risk and KL divergence.
    2. train_step(): The training objective for posterior optimization.
    3. compute_risk_certificate(): Final bound evaluation after training.

    The training pipeline calls these methods. The goal is to achieve
    the tightest (lowest) risk certificate on the 0-1 loss.

    Available information:
    - n_bound: number of samples in the bound-evaluation set
    - delta: confidence parameter (default 0.025)
    - kl: KL divergence between posterior and prior KL(Q||P)
    - empirical_risk: estimated loss on bound-evaluation set
    - inv_kl(q, c): binary KL inversion (find p s.t. KL(q||p)=c)

    Interface contract:
    - compute_bound(empirical_risk, kl, n, delta) -> bound_value (float tensor)
    - train_step(model, data, target, device, n_bound, delta) -> loss (float tensor)
    - compute_risk_certificate(model, bound_loader, device, delta, mc_samples)
        -> (risk_cert_01, metrics_dict)
    """

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.1,
                 pmin=1e-5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin

    def compute_bound(self, empirical_risk, kl, n, delta):
        """Compute PAC-Bayes upper bound on true risk.

        Default: McAllester/Maurer bound (fclassic).
        B(Q,S) = empirical_risk + sqrt((KL(Q||P) + log(2*sqrt(n)/delta)) / (2n))

        Args:
            empirical_risk: estimated risk on bound data (tensor)
            kl: KL divergence between posterior and prior (tensor)
            n: number of bound-evaluation samples
            delta: confidence parameter

        Returns:
            bound_value: upper bound on true risk (tensor)
        """
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
        bound = empirical_risk + torch.sqrt(kl_term)
        return bound

    def train_step(self, model, data, target, device, n_bound, delta):
        """Compute training loss (PAC-Bayes objective to minimize).

        Default: McAllester bound with NLL surrogate.

        Args:
            model: stochastic neural network
            data: input batch (already on device)
            target: label batch (already on device)
            device: torch device
            n_bound: number of bound-evaluation samples
            delta: confidence parameter

        Returns:
            loss: scalar tensor to backpropagate
        """
        output = model(data, sample=True)
        # Bounded cross-entropy as surrogate for 0-1 loss
        log_probs = F.log_softmax(output, dim=1)
        log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
        nll = F.nll_loss(log_probs, target)

        kl = get_total_kl(model)
        bound = self.compute_bound(nll, kl, n_bound, delta)
        return bound

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 mc_samples=1000):
        """Evaluate final PAC-Bayes risk certificate after training.

        Computes:
        1. Empirical 0-1 risk via MC sampling on the bound-evaluation set
        2. KL divergence between posterior and prior
        3. PAC-Bayes-kl bound inversion for the final certificate

        Args:
            model: trained stochastic model
            bound_loader: DataLoader for bound-evaluation set
            device: torch device
            delta: confidence parameter
            mc_samples: number of MC samples per input

        Returns:
            (risk_cert_01, metrics_dict)
        """
        model.eval()
        n_bound = len(bound_loader.dataset)

        # 1. Compute empirical 0-1 risk via MC sampling
        emp_risk_01 = compute_01_risk(model, bound_loader, device,
                                      mc_samples=mc_samples)

        # 2. Compute NLL-based empirical risk for the CE bound
        total_nll = 0.0
        total_samples = 0
        kl_total = None
        with torch.no_grad():
            for data, target in bound_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, sample=True)
                log_probs = F.log_softmax(output, dim=1)
                log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
                nll = F.nll_loss(log_probs, target, reduction="sum")
                total_nll += nll.item()
                total_samples += target.size(0)
                if kl_total is None:
                    kl_total = get_total_kl(model)

        emp_nll = total_nll / total_samples

        # 3. Get KL from a single forward pass
        with torch.no_grad():
            dummy_data = next(iter(bound_loader))[0][:1].to(device)
            model(dummy_data, sample=True)
            kl = get_total_kl(model).item()

        # 4. PAC-Bayes-kl bound inversion for 0-1 loss certificate
        c = (kl + math.log(2.0 * math.sqrt(n_bound) / delta)) / n_bound
        risk_cert_01 = inv_kl(emp_risk_01, c)

        # 5. Compute the direct bound from compute_bound for CE risk
        emp_nll_t = torch.tensor(emp_nll)
        kl_t = torch.tensor(kl)
        ce_bound = self.compute_bound(emp_nll_t, kl_t, n_bound, delta).item()

        metrics = {
            "empirical_01_risk": emp_risk_01,
            "empirical_nll": emp_nll,
            "kl_divergence": kl,
            "ce_bound": ce_bound,
        }

        return risk_cert_01, metrics


# ================================================================
# FIXED — Training pipeline and evaluation (do not modify)
# ================================================================


def train_prior(det_model, train_loader, device, epochs=20, lr=0.005):
    """Train a deterministic model as the data-dependent prior."""
    optimizer = optim.SGD(det_model.parameters(), lr=lr, momentum=0.99)
    det_model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = det_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        if (epoch + 1) % 5 == 0:
            acc = correct / total
            avg_loss = total_loss / total
            print(f"TRAIN_METRICS prior_epoch={epoch+1} loss={avg_loss:.6f} "
                  f"accuracy={acc:.4f}", flush=True)


def train_posterior(stoch_model, bound_optimizer, train_loader, device,
                    epochs=50, n_bound=30000, delta=0.025):
    """Train the stochastic model by minimizing the PAC-Bayes bound."""
    optimizer = optim.SGD(
        stoch_model.parameters(),
        lr=bound_optimizer.learning_rate,
        momentum=bound_optimizer.momentum,
    )
    stoch_model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = bound_optimizer.train_step(
                stoch_model, data, target, device, n_bound, delta
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / total_samples
            with torch.no_grad():
                dummy = next(iter(train_loader))[0][:1].to(device)
                stoch_model(dummy, sample=True)
                kl = get_total_kl(stoch_model).item()
            print(f"TRAIN_METRICS posterior_epoch={epoch+1} "
                  f"train_obj={avg_loss:.6f} kl={kl:.2f}",
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description="PAC-Bayes Bound Optimization")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashionmnist", "cifar10"])
    parser.add_argument("--model", type=str, default="fcn",
                        choices=["fcn", "cnn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--prior-epochs", type=int, default=20)
    parser.add_argument("--posterior-epochs", type=int, default=50)
    parser.add_argument("--prior-frac", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.025)
    parser.add_argument("--prior-sigma", type=float, default=0.03)
    parser.add_argument("--mc-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--data-dir", type=str, default="/workspace/data")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {args.dataset}, Model: {args.model}", flush=True)

    # Load data
    train_dataset, test_dataset, in_channels, num_classes = load_dataset(
        args.dataset, args.data_dir
    )

    # Split: prior training set + bound evaluation set
    prior_set, bound_set = split_data_for_prior(
        train_dataset, prior_frac=args.prior_frac, seed=args.seed
    )

    prior_loader = DataLoader(prior_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    bound_loader = DataLoader(bound_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    n_bound = len(bound_set)
    print(f"Prior set: {len(prior_set)}, Bound set: {n_bound}, "
          f"Test set: {len(test_dataset)}", flush=True)

    # Step 1: Train data-dependent prior
    print("\n--- Training data-dependent prior ---", flush=True)
    if args.model == "fcn":
        det_model = DeterministicFCN().to(device)
    else:
        det_model = DeterministicCNN(in_channels, num_classes).to(device)
    train_prior(det_model, prior_loader, device, epochs=args.prior_epochs)

    # Step 2: Initialize stochastic model from prior
    if args.model == "fcn":
        stoch_model = StochasticFCN(prior_sigma=args.prior_sigma).to(device)
    else:
        stoch_model = StochasticCNN(in_channels, num_classes,
                                     prior_sigma=args.prior_sigma).to(device)
    transfer_weights_to_stochastic(det_model, stoch_model)

    # Step 3: Create bound optimizer and train posterior
    bound_optimizer = BoundOptimizer(
        learning_rate=0.001,
        momentum=0.95,
        prior_sigma=args.prior_sigma,
        pmin=1e-5,
    )

    print("\n--- Training posterior (minimizing PAC-Bayes bound) ---", flush=True)
    train_posterior(
        stoch_model, bound_optimizer, bound_loader, device,
        epochs=args.posterior_epochs,
        n_bound=n_bound,
        delta=args.delta,
    )

    # Step 4: Evaluate risk certificate
    print("\n--- Evaluating risk certificate ---", flush=True)

    # Use a separate loader without shuffling for evaluation
    eval_bound_loader = DataLoader(bound_set, batch_size=args.batch_size,
                                   shuffle=False, num_workers=2, pin_memory=True)

    risk_cert_01, metrics = bound_optimizer.compute_risk_certificate(
        stoch_model, eval_bound_loader, device,
        delta=args.delta,
        mc_samples=args.mc_samples,
    )

    # Also compute test error for reference
    test_error = compute_test_error(stoch_model, test_loader, device)

    print(f"\n--- Results ---", flush=True)
    print(f"Empirical 0-1 risk (bound set): {metrics['empirical_01_risk']:.6f}",
          flush=True)
    print(f"KL divergence: {metrics['kl_divergence']:.2f}", flush=True)
    print(f"CE bound: {metrics['ce_bound']:.6f}", flush=True)
    print(f"Risk certificate (0-1 loss): {risk_cert_01:.6f}", flush=True)
    print(f"Test error (posterior mean): {test_error:.6f}", flush=True)

    # Output metrics for parser
    print(f"TEST_METRICS risk_certificate={risk_cert_01:.6f}", flush=True)
    print(f"TEST_METRICS test_error={test_error:.6f}", flush=True)
    print(f"TEST_METRICS kl_divergence={metrics['kl_divergence']:.2f}", flush=True)
    print(f"TEST_METRICS ce_bound={metrics['ce_bound']:.6f}", flush=True)
    print(f"TEST_METRICS empirical_01_risk={metrics['empirical_01_risk']:.6f}",
          flush=True)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(stoch_model.state_dict(),
               os.path.join(args.output_dir, "model.pt"))
    print(f"\nModel saved to {args.output_dir}/model.pt", flush=True)


if __name__ == "__main__":
    main()
