# MLS-Bench: optimization-pac-bayes-bound

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
3. `compute_risk_certificate(model, bound_loader, device, delta, mc_samples)` — final certificate evaluation.

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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/PBB/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `PBB/custom_pac_bayes.py`
- editable lines **460–604**


Other files you may **read** for context (do not modify):
- `PBB/pbb/models.py`
- `PBB/pbb/utils.py`


## Readable Context


### `PBB/custom_pac_bayes.py`  [EDITABLE — lines 460–604 only]

```python
     1: """PAC-Bayes Bound Optimization — custom template.
     2: 
     3: This script trains a stochastic neural network by minimizing a PAC-Bayes
     4: bound and then evaluates the tightness of the resulting risk certificate.
     5: 
     6: The agent edits the EDITABLE section (BoundOptimizer class) which controls:
     7:   1. How the PAC-Bayes bound is computed from empirical risk + KL divergence
     8:   2. How the posterior distribution is optimized (training objective)
     9:   3. How the final risk certificate is evaluated
    10: 
    11: Fixed sections handle data loading, model architecture, stochastic layers,
    12: and the outer training loop.
    13: """
    14: 
    15: import argparse
    16: import math
    17: import os
    18: import sys
    19: import time
    20: 
    21: import numpy as np
    22: import torch
    23: import torch.nn as nn
    24: import torch.nn.functional as F
    25: import torch.optim as optim
    26: from torch.utils.data import DataLoader, Subset, random_split
    27: import torchvision
    28: import torchvision.transforms as transforms
    29: 
    30: # ================================================================
    31: # FIXED — Stochastic layers and model architectures (do not modify)
    32: # ================================================================
    33: 
    34: 
    35: class Gaussian:
    36:     """Gaussian weight distribution for variational inference."""
    37: 
    38:     def __init__(self, mu, rho):
    39:         self.mu = mu
    40:         self.rho = rho
    41: 
    42:     @property
    43:     def sigma(self):
    44:         return torch.log1p(torch.exp(self.rho))
    45: 
    46:     def sample(self):
    47:         eps = torch.randn_like(self.mu)
    48:         return self.mu + self.sigma * eps
    49: 
    50:     def log_prob(self, x):
    51:         return (
    52:             -0.5 * math.log(2 * math.pi)
    53:             - torch.log(self.sigma)
    54:             - 0.5 * ((x - self.mu) / self.sigma) ** 2
    55:         )
    56: 
    57: 
    58: class ProbLinear(nn.Module):
    59:     """Probabilistic linear layer with Gaussian weights and data-dependent prior."""
    60: 
    61:     def __init__(self, in_features, out_features, prior_sigma=0.1):
    62:         super().__init__()
    63:         self.in_features = in_features
    64:         self.out_features = out_features
    65: 
    66:         # Initialize posterior rho so that initial sigma matches prior_sigma
    67:         # sigma = log(1 + exp(rho)), so rho = log(exp(sigma) - 1)
    68:         rho_init = math.log(math.exp(prior_sigma) - 1.0)
    69: 
    70:         # Posterior parameters (learnable)
    71:         self.weight_mu = nn.Parameter(
    72:             torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
    73:         )
    74:         self.weight_rho = nn.Parameter(
    75:             torch.empty(out_features, in_features).fill_(rho_init)
    76:         )
    77:         self.bias_mu = nn.Parameter(torch.zeros(out_features))
    78:         self.bias_rho = nn.Parameter(torch.full((out_features,), rho_init))
    79: 
    80:         # Prior (fixed, data-dependent: set via set_prior_mu)
    81:         self.prior_sigma = prior_sigma
    82:         self.register_buffer("weight_prior_mu",
    83:                              torch.zeros(out_features, in_features))
    84:         self.register_buffer("bias_prior_mu", torch.zeros(out_features))
    85: 
    86:         self._kl = 0.0
    87: 
    88:     def set_prior_mu(self, weight_mu, bias_mu):
    89:         """Set the prior mean from a trained deterministic model."""
    90:         self.weight_prior_mu.copy_(weight_mu.data)
    91:         self.bias_prior_mu.copy_(bias_mu.data)
    92: 
    93:     def forward(self, x, sample=True):
    94:         if sample:
    95:             w_posterior = Gaussian(self.weight_mu, self.weight_rho)
    96:             b_posterior = Gaussian(self.bias_mu, self.bias_rho)
    97:             weight = w_posterior.sample()
    98:             bias = b_posterior.sample()
    99: 
   100:             # KL divergence: KL(q(w) || p(w)) with data-dependent prior
   101:             # Prior is N(prior_mu, prior_sigma^2), posterior is N(mu, sigma^2)
   102:             # Analytic KL for diagonal Gaussians
   103:             q_sigma_w = w_posterior.sigma
   104:             q_sigma_b = b_posterior.sigma
   105:             p_var = self.prior_sigma ** 2
   106: 
   107:             kl_w = (0.5 * (
   108:                 (q_sigma_w ** 2 + (self.weight_mu - self.weight_prior_mu) ** 2) / p_var
   109:                 - 1.0
   110:                 + math.log(p_var) - 2.0 * torch.log(q_sigma_w)
   111:             )).sum()
   112: 
   113:             kl_b = (0.5 * (
   114:                 (q_sigma_b ** 2 + (self.bias_mu - self.bias_prior_mu) ** 2) / p_var
   115:                 - 1.0
   116:                 + math.log(p_var) - 2.0 * torch.log(q_sigma_b)
   117:             )).sum()
   118: 
   119:             self._kl = kl_w + kl_b
   120:         else:
   121:             weight = self.weight_mu
   122:             bias = self.bias_mu
   123:             self._kl = 0.0
   124: 
   125:         return F.linear(x, weight, bias)
   126: 
   127: 
   128: class ProbConv2d(nn.Module):
   129:     """Probabilistic 2D convolution with Gaussian weights and data-dependent prior."""
   130: 
   131:     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
   132:                  padding=0, prior_sigma=0.1):
   133:         super().__init__()
   134:         self.stride = stride
   135:         self.padding = padding
   136:         self.prior_sigma = prior_sigma
   137: 
   138:         # Initialize posterior rho so that initial sigma matches prior_sigma
   139:         rho_init = math.log(math.exp(prior_sigma) - 1.0)
   140: 
   141:         self.weight_mu = nn.Parameter(
   142:             torch.empty(out_channels, in_channels, kernel_size, kernel_size)
   143:             .uniform_(-0.2, 0.2)
   144:         )
   145:         self.weight_rho = nn.Parameter(
   146:             torch.empty(out_channels, in_channels, kernel_size, kernel_size)
   147:             .fill_(rho_init)
   148:         )
   149:         self.bias_mu = nn.Parameter(torch.zeros(out_channels))
   150:         self.bias_rho = nn.Parameter(torch.full((out_channels,), rho_init))
   151: 
   152:         # Data-dependent prior mean (set via set_prior_mu)
   153:         self.register_buffer("weight_prior_mu",
   154:                              torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
   155:         self.register_buffer("bias_prior_mu", torch.zeros(out_channels))
   156:         self._kl = 0.0
   157: 
   158:     def set_prior_mu(self, weight_mu, bias_mu):
   159:         """Set the prior mean from a trained deterministic model."""
   160:         self.weight_prior_mu.copy_(weight_mu.data)
   161:         self.bias_prior_mu.copy_(bias_mu.data)
   162: 
   163:     def forward(self, x, sample=True):
   164:         if sample:
   165:             w_post = Gaussian(self.weight_mu, self.weight_rho)
   166:             b_post = Gaussian(self.bias_mu, self.bias_rho)
   167:             weight = w_post.sample()
   168:             bias = b_post.sample()
   169: 
   170:             # Analytic KL with data-dependent prior N(prior_mu, prior_sigma^2)
   171:             q_sigma_w = w_post.sigma
   172:             q_sigma_b = b_post.sigma
   173:             p_var = self.prior_sigma ** 2
   174: 
   175:             kl_w = (0.5 * (
   176:                 (q_sigma_w ** 2 + (self.weight_mu - self.weight_prior_mu) ** 2) / p_var
   177:                 - 1.0
   178:                 + math.log(p_var) - 2.0 * torch.log(q_sigma_w)
   179:             )).sum()
   180: 
   181:             kl_b = (0.5 * (
   182:                 (q_sigma_b ** 2 + (self.bias_mu - self.bias_prior_mu) ** 2) / p_var
   183:                 - 1.0
   184:                 + math.log(p_var) - 2.0 * torch.log(q_sigma_b)
   185:             )).sum()
   186: 
   187:             self._kl = kl_w + kl_b
   188:         else:
   189:             weight = self.weight_mu
   190:             bias = self.bias_mu
   191:             self._kl = 0.0
   192: 
   193:         return F.conv2d(x, weight, bias, self.stride, self.padding)
   194: 
   195: 
   196: def get_total_kl(model):
   197:     """Sum KL divergence across all probabilistic layers."""
   198:     kl = 0.0
   199:     for m in model.modules():
   200:         if hasattr(m, "_kl"):
   201:             kl = kl + m._kl
   202:     return kl
   203: 
   204: 
   205: class StochasticFCN(nn.Module):
   206:     """4-layer fully connected stochastic network for MNIST (28x28)."""
   207: 
   208:     def __init__(self, prior_sigma=0.1):
   209:         super().__init__()
   210:         self.fc1 = ProbLinear(784, 600, prior_sigma)
   211:         self.fc2 = ProbLinear(600, 600, prior_sigma)
   212:         self.fc3 = ProbLinear(600, 600, prior_sigma)
   213:         self.fc4 = ProbLinear(600, 10, prior_sigma)
   214: 
   215:     def forward(self, x, sample=True):
   216:         x = x.view(x.size(0), -1)
   217:         x = F.relu(self.fc1(x, sample))
   218:         x = F.relu(self.fc2(x, sample))
   219:         x = F.relu(self.fc3(x, sample))
   220:         return self.fc4(x, sample)
   221: 
   222: 
   223: class StochasticCNN(nn.Module):
   224:     """4-layer CNN stochastic network (2 conv + 2 fc)."""
   225: 
   226:     def __init__(self, in_channels=1, num_classes=10, prior_sigma=0.1):
   227:         super().__init__()
   228:         self.conv1 = ProbConv2d(in_channels, 32, 3, padding=1, prior_sigma=prior_sigma)
   229:         self.conv2 = ProbConv2d(32, 64, 3, padding=1, prior_sigma=prior_sigma)
   230:         self.in_channels = in_channels
   231:         # Compute flattened size after two 2x2 max pools
   232:         if in_channels == 1:
   233:             self._flat_size = 64 * 7 * 7  # MNIST/FashionMNIST: 28->14->7
   234:         else:
   235:             self._flat_size = 64 * 8 * 8  # CIFAR-10: 32->16->8
   236:         self.fc1 = ProbLinear(self._flat_size, 256, prior_sigma)
   237:         self.fc2 = ProbLinear(256, num_classes, prior_sigma)
   238: 
   239:     def forward(self, x, sample=True):
   240:         x = F.relu(self.conv1(x, sample))
   241:         x = F.max_pool2d(x, 2)
   242:         x = F.relu(self.conv2(x, sample))
   243:         x = F.max_pool2d(x, 2)
   244:         x = x.view(x.size(0), -1)
   245:         x = F.relu(self.fc1(x, sample))
   246:         return self.fc2(x, sample)
   247: 
   248: 
   249: class DeterministicFCN(nn.Module):
   250:     """Deterministic FCN for prior training."""
   251: 
   252:     def __init__(self):
   253:         super().__init__()
   254:         self.fc1 = nn.Linear(784, 600)
   255:         self.fc2 = nn.Linear(600, 600)
   256:         self.fc3 = nn.Linear(600, 600)
   257:         self.fc4 = nn.Linear(600, 10)
   258: 
   259:     def forward(self, x):
   260:         x = x.view(x.size(0), -1)
   261:         x = F.relu(self.fc1(x))
   262:         x = F.relu(self.fc2(x))
   263:         x = F.relu(self.fc3(x))
   264:         return self.fc4(x)
   265: 
   266: 
   267: class DeterministicCNN(nn.Module):
   268:     """Deterministic CNN for prior training."""
   269: 
   270:     def __init__(self, in_channels=1, num_classes=10):
   271:         super().__init__()
   272:         self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
   273:         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
   274:         if in_channels == 1:
   275:             flat_size = 64 * 7 * 7
   276:         else:
   277:             flat_size = 64 * 8 * 8
   278:         self.fc1 = nn.Linear(flat_size, 256)
   279:         self.fc2 = nn.Linear(256, num_classes)
   280:         self.in_channels = in_channels
   281: 
   282:     def forward(self, x):
   283:         x = F.relu(self.conv1(x))
   284:         x = F.max_pool2d(x, 2)
   285:         x = F.relu(self.conv2(x))
   286:         x = F.max_pool2d(x, 2)
   287:         x = x.view(x.size(0), -1)
   288:         x = F.relu(self.fc1(x))
   289:         return self.fc2(x)
   290: 
   291: 
   292: # ================================================================
   293: # FIXED — Data loading utilities (do not modify)
   294: # ================================================================
   295: 
   296: 
   297: def load_dataset(name, data_dir="/workspace/data"):
   298:     """Load dataset with standard normalization."""
   299:     if name == "mnist":
   300:         transform = transforms.Compose([
   301:             transforms.ToTensor(),
   302:             transforms.Normalize((0.1307,), (0.3081,)),
   303:         ])
   304:         train = torchvision.datasets.MNIST(data_dir, train=True, download=False,
   305:                                            transform=transform)
   306:         test = torchvision.datasets.MNIST(data_dir, train=False, download=False,
   307:                                           transform=transform)
   308:         in_channels, num_classes = 1, 10
   309:     elif name == "fashionmnist":
   310:         transform = transforms.Compose([
   311:             transforms.ToTensor(),
   312:             transforms.Normalize((0.2860,), (0.3530,)),
   313:         ])
   314:         train = torchvision.datasets.FashionMNIST(data_dir, train=True,
   315:                                                    download=False, transform=transform)
   316:         test = torchvision.datasets.FashionMNIST(data_dir, train=False,
   317:                                                   download=False, transform=transform)
   318:         in_channels, num_classes = 1, 10
   319:     elif name == "cifar10":
   320:         transform = transforms.Compose([
   321:             transforms.ToTensor(),
   322:             transforms.Normalize((0.4914, 0.4822, 0.4465),
   323:                                  (0.2470, 0.2435, 0.2616)),
   324:         ])
   325:         train = torchvision.datasets.CIFAR10(data_dir, train=True, download=False,
   326:                                              transform=transform)
   327:         test = torchvision.datasets.CIFAR10(data_dir, train=False, download=False,
   328:                                             transform=transform)
   329:         in_channels, num_classes = 3, 10
   330:     else:
   331:         raise ValueError(f"Unknown dataset: {name}")
   332:     return train, test, in_channels, num_classes
   333: 
   334: 
   335: def split_data_for_prior(train_dataset, prior_frac=0.5, seed=42):
   336:     """Split training data into prior-training set and bound-evaluation set."""
   337:     n = len(train_dataset)
   338:     n_prior = int(n * prior_frac)
   339:     n_bound = n - n_prior
   340:     gen = torch.Generator().manual_seed(seed)
   341:     prior_set, bound_set = random_split(train_dataset, [n_prior, n_bound],
   342:                                         generator=gen)
   343:     return prior_set, bound_set
   344: 
   345: 
   346: # ================================================================
   347: # FIXED — Utility functions (do not modify)
   348: # ================================================================
   349: 
   350: 
   351: def inv_kl(q, c):
   352:     """Compute the inverse KL: find the largest p such that KL(q||p) <= c.
   353: 
   354:     Uses binary search. KL(q||p) = q*log(q/p) + (1-q)*log((1-q)/(1-p)).
   355:     Returns p >= q such that KL(q||p) = c.
   356:     """
   357:     if c < 0:
   358:         raise ValueError("c must be non-negative")
   359:     if q >= 1.0:
   360:         return 1.0
   361:     if c == 0:
   362:         return q
   363: 
   364:     lo, hi = q, 1.0 - 1e-10
   365:     for _ in range(64):
   366:         mid = (lo + hi) / 2.0
   367:         kl_val = _kl_bernoulli(q, mid)
   368:         if kl_val < c:
   369:             lo = mid
   370:         else:
   371:             hi = mid
   372:     return (lo + hi) / 2.0
   373: 
   374: 
   375: def _kl_bernoulli(q, p):
   376:     """Binary KL divergence: KL(Ber(q) || Ber(p))."""
   377:     if q < 1e-12:
   378:         return -math.log(1 - p + 1e-12)
   379:     if q > 1 - 1e-12:
   380:         return -math.log(p + 1e-12)
   381:     return q * math.log(q / (p + 1e-12)) + (1 - q) * math.log(
   382:         (1 - q) / (1 - p + 1e-12)
   383:     )
   384: 
   385: 
   386: def compute_01_risk(model, loader, device, mc_samples=100):
   387:     """Compute stochastic 0-1 risk via Monte Carlo sampling."""
   388:     model.eval()
   389:     total_wrong = 0
   390:     total_samples = 0
   391: 
   392:     with torch.no_grad():
   393:         for data, target in loader:
   394:             data, target = data.to(device), target.to(device)
   395:             batch_size = data.size(0)
   396:             votes = torch.zeros(batch_size, 10, device=device)
   397: 
   398:             for _ in range(mc_samples):
   399:                 logits = model(data, sample=True)
   400:                 preds = logits.argmax(dim=1)
   401:                 votes.scatter_add_(1, preds.unsqueeze(1),
   402:                                    torch.ones(batch_size, 1, device=device))
   403: 
   404:             final_preds = votes.argmax(dim=1)
   405:             total_wrong += (final_preds != target).sum().item()
   406:             total_samples += batch_size
   407: 
   408:     return total_wrong / total_samples
   409: 
   410: 
   411: def compute_test_error(model, loader, device):
   412:     """Compute deterministic test error using posterior mean."""
   413:     model.eval()
   414:     correct = 0
   415:     total = 0
   416:     with torch.no_grad():
   417:         for data, target in loader:
   418:             data, target = data.to(device), target.to(device)
   419:             output = model(data, sample=False)
   420:             pred = output.argmax(dim=1)
   421:             correct += (pred == target).sum().item()
   422:             total += target.size(0)
   423:     return 1.0 - correct / total
   424: 
   425: 
   426: def transfer_weights_to_stochastic(det_model, stoch_model):
   427:     """Initialize stochastic model's posterior means and prior means from deterministic model."""
   428:     det_state = det_model.state_dict()
   429:     stoch_state = stoch_model.state_dict()
   430: 
   431:     mapping = {}
   432:     for det_key in det_state:
   433:         # Map fc1.weight -> fc1.weight_mu, fc1.bias -> fc1.bias_mu
   434:         parts = det_key.rsplit(".", 1)
   435:         if len(parts) == 2:
   436:             prefix, suffix = parts
   437:             mu_key = f"{prefix}.{suffix}_mu"
   438:             if mu_key in stoch_state:
   439:                 mapping[det_key] = mu_key
   440: 
   441:     for det_key, stoch_key in mapping.items():
   442:         stoch_state[stoch_key] = det_state[det_key]
   443: 
   444:     stoch_model.load_state_dict(stoch_state)
   445: 
   446:     # Set prior means on each probabilistic layer
   447:     det_modules = dict(det_model.named_modules())
   448:     for name, module in stoch_model.named_modules():
   449:         if hasattr(module, "set_prior_mu") and name in det_modules:
   450:             det_mod = det_modules[name]
   451:             module.set_prior_mu(det_mod.weight, det_mod.bias)
   452: 
   453: 
   454: # ================================================================
   455: # EDITABLE SECTION — BoundOptimizer class (lines 460 to 604)
   456: # The agent modifies this section to design tighter PAC-Bayes bounds.
   457: # ================================================================
   458: 
   459: 
   460: class BoundOptimizer:
   461:     """PAC-Bayes bound computation and posterior optimization.
   462: 
   463:     This class controls:
   464:     1. compute_bound(): How the generalization bound is computed from
   465:        empirical risk and KL divergence.
   466:     2. train_step(): The training objective for posterior optimization.
   467:     3. compute_risk_certificate(): Final bound evaluation after training.
   468: 
   469:     The training pipeline calls these methods. The goal is to achieve
   470:     the tightest (lowest) risk certificate on the 0-1 loss.
   471: 
   472:     Available information:
   473:     - n_bound: number of samples in the bound-evaluation set
   474:     - delta: confidence parameter (default 0.025)
   475:     - kl: KL divergence between posterior and prior KL(Q||P)
   476:     - empirical_risk: estimated loss on bound-evaluation set
   477:     - inv_kl(q, c): binary KL inversion (find p s.t. KL(q||p)=c)
   478: 
   479:     Interface contract:
   480:     - compute_bound(empirical_risk, kl, n, delta) -> bound_value (float tensor)
   481:     - train_step(model, data, target, device, n_bound, delta) -> loss (float tensor)
   482:     - compute_risk_certificate(model, bound_loader, device, delta, mc_samples)
   483:         -> (risk_cert_01, metrics_dict)
   484:     """
   485: 
   486:     def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.1,
   487:                  pmin=1e-5):
   488:         self.learning_rate = learning_rate
   489:         self.momentum = momentum
   490:         self.prior_sigma = prior_sigma
   491:         self.pmin = pmin
   492: 
   493:     def compute_bound(self, empirical_risk, kl, n, delta):
   494:         """Compute PAC-Bayes upper bound on true risk.
   495: 
   496:         Default: McAllester/Maurer bound (fclassic).
   497:         B(Q,S) = empirical_risk + sqrt((KL(Q||P) + log(2*sqrt(n)/delta)) / (2n))
   498: 
   499:         Args:
   500:             empirical_risk: estimated risk on bound data (tensor)

[truncated: showing at most 500 lines / 60000 bytes from PBB/custom_pac_bayes.py]
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **mnist-fcn** — wall-clock budget `0:59:00`, compute share `0.33`
- **mnist-cnn** — wall-clock budget `0:59:00`, compute share `0.33`
- **fmnist-cnn** — wall-clock budget `0:59:00`, compute share `0.33`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `mcallester` baseline — editable region  [READ-ONLY — reference implementation]

In `PBB/custom_pac_bayes.py`:

```python
Lines 460–538:
   457: # ================================================================
   458: 
   459: 
   460: class BoundOptimizer:
   461:     """McAllester/Maurer PAC-Bayes bound (fclassic).
   462: 
   463:     Classic bound: emp_risk + sqrt((KL + log(2*sqrt(n)/delta)) / (2n))
   464:     Training objective: same functional form with NLL surrogate for 0-1 loss.
   465:     Certificate: PAC-Bayes-kl inversion on 0-1 risk.
   466:     """
   467: 
   468:     def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
   469:                  pmin=1e-5):
   470:         self.learning_rate = learning_rate
   471:         self.momentum = momentum
   472:         self.prior_sigma = prior_sigma
   473:         self.pmin = pmin
   474: 
   475:     def compute_bound(self, empirical_risk, kl, n, delta):
   476:         """McAllester/Maurer bound."""
   477:         kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
   478:         bound = empirical_risk + torch.sqrt(kl_term)
   479:         return bound
   480: 
   481:     def train_step(self, model, data, target, device, n_bound, delta):
   482:         """Training objective: McAllester bound with NLL surrogate."""
   483:         output = model(data, sample=True)
   484:         log_probs = F.log_softmax(output, dim=1)
   485:         log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
   486:         nll = F.nll_loss(log_probs, target)
   487: 
   488:         kl = get_total_kl(model)
   489:         bound = self.compute_bound(nll, kl, n_bound, delta)
   490:         return bound
   491: 
   492:     def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
   493:                                  mc_samples=1000):
   494:         """Evaluate McAllester risk certificate with PAC-Bayes-kl inversion."""
   495:         model.eval()
   496:         n_bound = len(bound_loader.dataset)
   497: 
   498:         # 1. Empirical 0-1 risk via MC sampling
   499:         emp_risk_01 = compute_01_risk(model, bound_loader, device,
   500:                                       mc_samples=mc_samples)
   501: 
   502:         # 2. NLL-based empirical risk
   503:         total_nll = 0.0
   504:         total_samples = 0
   505:         with torch.no_grad():
   506:             for data, target in bound_loader:
   507:                 data, target = data.to(device), target.to(device)
   508:                 output = model(data, sample=True)
   509:                 log_probs = F.log_softmax(output, dim=1)
   510:                 log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
   511:                 nll = F.nll_loss(log_probs, target, reduction="sum")
   512:                 total_nll += nll.item()
   513:                 total_samples += target.size(0)
   514:         emp_nll = total_nll / total_samples
   515: 
   516:         # 3. KL divergence
   517:         with torch.no_grad():
   518:             dummy_data = next(iter(bound_loader))[0][:1].to(device)
   519:             model(dummy_data, sample=True)
   520:             kl = get_total_kl(model).item()
   521: 
   522:         # 4. PAC-Bayes-kl inversion for 0-1 loss certificate
   523:         c = (kl + math.log(2.0 * math.sqrt(n_bound) / delta)) / n_bound
   524:         risk_cert_01 = inv_kl(emp_risk_01, c)
   525: 
   526:         # 5. CE bound
   527:         emp_nll_t = torch.tensor(emp_nll)
   528:         kl_t = torch.tensor(kl)
   529:         ce_bound = self.compute_bound(emp_nll_t, kl_t, n_bound, delta).item()
   530: 
   531:         metrics = {
   532:             "empirical_01_risk": emp_risk_01,
   533:             "empirical_nll": emp_nll,
   534:             "kl_divergence": kl,
   535:             "ce_bound": ce_bound,
   536:         }
   537: 
   538:         return risk_cert_01, metrics
   539: 
   540: 
   541: # ================================================================
```

### `catoni` baseline — editable region  [READ-ONLY — reference implementation]

In `PBB/custom_pac_bayes.py`:

```python
Lines 460–576:
   457: # ================================================================
   458: 
   459: 
   460: class BoundOptimizer:
   461:     """Catoni/Lambda PAC-Bayes bound (flamb).
   462: 
   463:     Bound: emp_risk / (1 - lam/2) + (KL + log(2*sqrt(n)/delta)) / (n*lam*(1 - lam/2))
   464:     Lambda is a learnable parameter optimized jointly with the posterior.
   465:     Tighter than McAllester when lambda is well-tuned.
   466:     """
   467: 
   468:     def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
   469:                  pmin=1e-5, initial_lambda=0.5, lambda_lr=0.01):
   470:         self.learning_rate = learning_rate
   471:         self.momentum = momentum
   472:         self.prior_sigma = prior_sigma
   473:         self.pmin = pmin
   474:         # Lambda parameter for the Catoni bound (learnable)
   475:         self._lambda_param = torch.tensor(initial_lambda, requires_grad=True)
   476:         self.lambda_lr = lambda_lr
   477:         self._lambda_optimizer = None
   478: 
   479:     def _get_lambda(self):
   480:         """Get clamped lambda value in (0, 2)."""
   481:         return torch.clamp(self._lambda_param, min=0.01, max=1.99)
   482: 
   483:     def _ensure_lambda_optimizer(self):
   484:         if self._lambda_optimizer is None:
   485:             self._lambda_optimizer = torch.optim.SGD(
   486:                 [self._lambda_param], lr=self.lambda_lr
   487:             )
   488: 
   489:     def compute_bound(self, empirical_risk, kl, n, delta):
   490:         """Catoni/Lambda bound."""
   491:         lam = self._get_lambda()
   492:         kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (
   493:             n * lam * (1.0 - lam / 2.0)
   494:         )
   495:         bound = empirical_risk / (1.0 - lam / 2.0) + kl_term
   496:         return bound
   497: 
   498:     def train_step(self, model, data, target, device, n_bound, delta):
   499:         """Training objective: Catoni/lambda bound with joint lambda optimization."""
   500:         # Ensure lambda is on correct device
   501:         if self._lambda_param.device != device:
   502:             self._lambda_param = self._lambda_param.to(device).detach().requires_grad_(True)
   503:             self._lambda_optimizer = None
   504:         self._ensure_lambda_optimizer()
   505: 
   506:         output = model(data, sample=True)
   507:         log_probs = F.log_softmax(output, dim=1)
   508:         log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
   509:         nll = F.nll_loss(log_probs, target)
   510: 
   511:         kl = get_total_kl(model)
   512: 
   513:         # Update lambda on a detached copy — the outer loop's optimizer.step()
   514:         # only knows about posterior params, so lambda would stay frozen at
   515:         # init without this explicit step. Before the fix, lambda=1.0 caused
   516:         # the Catoni bound to double the KL contribution (1-lam/2=0.5), which
   517:         # forced KL to grow to ~10x McAllester's value.
   518:         self._lambda_optimizer.zero_grad()
   519:         lam = self._get_lambda()
   520:         lam_bound = nll.detach() / (1.0 - lam / 2.0) + (
   521:             kl.detach() + math.log(2.0 * math.sqrt(n_bound) / delta)
   522:         ) / (n_bound * lam * (1.0 - lam / 2.0))
   523:         lam_bound.backward()
   524:         self._lambda_optimizer.step()
   525: 
   526:         bound = self.compute_bound(nll, kl, n_bound, delta)
   527:         return bound
   528: 
   529:     def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
   530:                                  mc_samples=1000):
   531:         """Evaluate Catoni risk certificate with PAC-Bayes-kl inversion."""
   532:         model.eval()
   533:         n_bound = len(bound_loader.dataset)
   534: 
   535:         # 1. Empirical 0-1 risk via MC sampling
   536:         emp_risk_01 = compute_01_risk(model, bound_loader, device,
   537:                                       mc_samples=mc_samples)
   538: 
   539:         # 2. NLL-based empirical risk
   540:         total_nll = 0.0
   541:         total_samples = 0
   542:         with torch.no_grad():
   543:             for data, target in bound_loader:
   544:                 data, target = data.to(device), target.to(device)
   545:                 output = model(data, sample=True)
   546:                 log_probs = F.log_softmax(output, dim=1)
   547:                 log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
   548:                 nll = F.nll_loss(log_probs, target, reduction="sum")
   549:                 total_nll += nll.item()
   550:                 total_samples += target.size(0)
   551:         emp_nll = total_nll / total_samples
   552: 
   553:         # 3. KL divergence
   554:         with torch.no_grad():
   555:             dummy_data = next(iter(bound_loader))[0][:1].to(device)
   556:             model(dummy_data, sample=True)
   557:             kl = get_total_kl(model).item()
   558: 
   559:         # 4. PAC-Bayes-kl inversion for 0-1 loss certificate
   560:         c = (kl + math.log(2.0 * math.sqrt(n_bound) / delta)) / n_bound
   561:         risk_cert_01 = inv_kl(emp_risk_01, c)
   562: 
   563:         # 5. CE bound using Catoni formula
   564:         emp_nll_t = torch.tensor(emp_nll)
   565:         kl_t = torch.tensor(kl)
   566:         ce_bound = self.compute_bound(emp_nll_t, kl_t, n_bound, delta).item()
   567: 
   568:         metrics = {
   569:             "empirical_01_risk": emp_risk_01,
   570:             "empirical_nll": emp_nll,
   571:             "kl_divergence": kl,
   572:             "ce_bound": ce_bound,
   573:             "lambda": self._get_lambda().item(),
   574:         }
   575: 
   576:         return risk_cert_01, metrics
   577: 
   578: 
   579: # ================================================================
```

### `quadratic` baseline — editable region  [READ-ONLY — reference implementation]

In `PBB/custom_pac_bayes.py`:

```python
Lines 460–551:
   457: # ================================================================
   458: 
   459: 
   460: class BoundOptimizer:
   461:     """Quadratic PAC-Bayes bound (fquad, Rivasplata 2019 / Perez-Ortiz 2021).
   462: 
   463:     Bound: (sqrt(emp_risk + kl_term) + sqrt(kl_term))^2
   464:     where kl_term = (KL + log(2*sqrt(n)/delta)) / (2n)
   465: 
   466:     Tighter than McAllester when empirical risk is low.
   467:     """
   468: 
   469:     def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
   470:                  pmin=1e-5):
   471:         self.learning_rate = learning_rate
   472:         self.momentum = momentum
   473:         self.prior_sigma = prior_sigma
   474:         self.pmin = pmin
   475:         # PBB's loss-bounding constant: maps unbounded NLL into [0,1] via
   476:         # ell_tilde = NLL / log(1/pmin).  See Perez-Ortiz 2021 Sec 5.
   477:         self._loss_scale = 1.0 / math.log(1.0 / self.pmin)
   478: 
   479:     def compute_bound(self, empirical_risk, kl, n, delta):
   480:         """Quadratic PAC-Bayes bound (fquad)."""
   481:         kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
   482:         # Ensure non-negative under sqrt
   483:         inner = torch.clamp(empirical_risk + kl_term, min=0.0)
   484:         kl_term_clamped = torch.clamp(kl_term, min=0.0)
   485:         bound = (torch.sqrt(inner) + torch.sqrt(kl_term_clamped)) ** 2
   486:         return bound
   487: 
   488:     def train_step(self, model, data, target, device, n_bound, delta):
   489:         """Training objective: bounded NLL passed through the fquad formula.
   490: 
   491:         The NLL is rescaled by 1/log(1/pmin) so that the surrogate loss lies in
   492:         [0,1], matching the PBB reference implementation. This is essential
   493:         for fquad to actually be tighter than fclassic in practice.
   494:         """
   495:         output = model(data, sample=True)
   496:         log_probs = F.log_softmax(output, dim=1)
   497:         log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
   498:         # Bounded NLL surrogate, in [0, 1]
   499:         nll = F.nll_loss(log_probs, target) * self._loss_scale
   500: 
   501:         kl = get_total_kl(model)
   502:         bound = self.compute_bound(nll, kl, n_bound, delta)
   503:         return bound
   504: 
   505:     def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
   506:                                  mc_samples=1000):
   507:         """Evaluate quadratic risk certificate with PAC-Bayes-kl inversion."""
   508:         model.eval()
   509:         n_bound = len(bound_loader.dataset)
   510: 
   511:         # 1. Empirical 0-1 risk via MC sampling
   512:         emp_risk_01 = compute_01_risk(model, bound_loader, device,
   513:                                       mc_samples=mc_samples)
   514: 
   515:         # 2. Bounded NLL empirical risk (same scaling as training)
   516:         total_nll = 0.0
   517:         total_samples = 0
   518:         with torch.no_grad():
   519:             for data, target in bound_loader:
   520:                 data, target = data.to(device), target.to(device)
   521:                 output = model(data, sample=True)
   522:                 log_probs = F.log_softmax(output, dim=1)
   523:                 log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
   524:                 nll = F.nll_loss(log_probs, target, reduction="sum")
   525:                 total_nll += nll.item()
   526:                 total_samples += target.size(0)
   527:         emp_nll_bounded = (total_nll / total_samples) * self._loss_scale
   528: 
   529:         # 3. KL divergence
   530:         with torch.no_grad():
   531:             dummy_data = next(iter(bound_loader))[0][:1].to(device)
   532:             model(dummy_data, sample=True)
   533:             kl = get_total_kl(model).item()
   534: 
   535:         # 4. PAC-Bayes-kl inversion for 0-1 loss certificate
   536:         c = (kl + math.log(2.0 * math.sqrt(n_bound) / delta)) / n_bound
   537:         risk_cert_01 = inv_kl(emp_risk_01, c)
   538: 
   539:         # 5. Quadratic bound on bounded CE risk (in [0,1])
   540:         emp_nll_t = torch.tensor(emp_nll_bounded)
   541:         kl_t = torch.tensor(kl)
   542:         ce_bound = self.compute_bound(emp_nll_t, kl_t, n_bound, delta).item()
   543: 
   544:         metrics = {
   545:             "empirical_01_risk": emp_risk_01,
   546:             "empirical_nll": emp_nll_bounded,
   547:             "kl_divergence": kl,
   548:             "ce_bound": ce_bound,
   549:         }
   550: 
   551:         return risk_cert_01, metrics
   552: 
   553: 
   554: # ================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
