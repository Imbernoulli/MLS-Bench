# Federated Learning Aggregation Strategy Design

## Research Question
Design a federated-learning strategy that converges faster and to higher test accuracy under heterogeneous (non-IID) client data. The contribution is the *aggregation rule*, optionally together with the client-selection rule and the client-side local-update correction that the `Strategy` interface below exposes (a proximal term, control variates, a regularizer, ...). The simulation harness, data partitions, models, the client population, communication rounds, and evaluation are fixed. Two budgets are handed to the strategy rather than enforced: `select_clients` receives the per-round participation (`num_to_select`), and `client_local_train` receives the reference local budget (`local_epochs`, `local_lr`, `local_batch_size`). A strategy may change which clients it picks and how the local update is computed (regularizers, corrections, an adaptive local step size), but is expected to select `num_to_select` clients and keep the local compute at that budget — the harness checks neither, and a submission that uses more clients or trains longer per round is not comparable to the baselines.

## Background
Federated Learning (FL) trains a shared global model across many clients without centralizing data. Under non-IID client data, naive averaging suffers from "client drift" — local updates diverge, slowing or destabilizing convergence.

Reference baselines:
- **FedAvg** — McMahan, Moore, Ramage, Hampson, Agüera y Arcas, AISTATS 2017 ([arXiv:1602.05629](https://arxiv.org/abs/1602.05629)). Server averages client model parameters weighted by `n_k / sum(n_k)` (number of samples per client). No server-side state.
- **FedProx** — Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith, MLSys 2020 ([arXiv:1812.06127](https://arxiv.org/abs/1812.06127)). Same server aggregation as FedAvg, but each client adds a proximal term `(mu/2) * ||w - w_global||^2` to its local objective; default `mu = 0.01`.
- **SCAFFOLD** — Karimireddy, Kale, Mohri, Reddi, Stich, Suresh, ICML 2020 ([arXiv:1910.06378](https://arxiv.org/abs/1910.06378)). Maintains server- and client-side control variates `c, c_i` to correct client drift. Local update: `w <- w - eta * (g_i - c_i + c)`. Server updates `c` after each round from received deltas.

## Implementation Contract
Modify the `Strategy` class in `flower/custom_fl_aggregation.py` (the only editable region). The simulation loop drives it as follows:

```python
class Strategy:
    def __init__(self, global_model, args):
        # Initialize strategy state (momentum buffers, control variates, ...).
        ...

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        # Train one client starting from the global weights.
        # Returns: (state_dict, num_samples, avg_loss).
        # Default: plain SGD for `local_epochs` at `local_lr`. Override to add a
        # proximal term (FedProx), control-variate correction (SCAFFOLD), ...
        ...

    def aggregate(self, global_state_dict, client_updates, round_num):
        # global_state_dict: OrderedDict of current global model parameters
        # client_updates: list of (state_dict, num_samples, avg_loss) tuples
        # round_num: current communication round (0-indexed)
        # Returns: OrderedDict of updated global model parameters.
        ...

    def select_clients(self, num_available, num_to_select, round_num):
        # Returns list of client indices to participate this round.
        ...
```

Every round the harness calls `select_clients`, then `client_local_train` once per selected client, then `aggregate` on the collected updates. The three hooks have working defaults (uniform random selection, plain SGD, sample-weighted averaging), so override only what your method changes; `select_clients` is expected to return `num_to_select` distinct indices. Everything outside the class — models, data loading and partitioning, `_default_client_sgd`, the round loop, and evaluation — is read-only.

## Fixed Pipeline & Evaluation
- **Communication rounds**: 200.
- **Per-round participation (passed to `select_clients` as `num_to_select`)**: 10 of 100 clients.
- **Local training (reference recipe passed to `client_local_train`)**: 5 local epochs per round, SGD with `lr=0.01`.

Benchmarks:
1. **CIFAR-10** with Dirichlet split (`alpha=0.1`) — 100 clients, 10-class image classification.
2. **FEMNIST** (EMNIST ByClass) with Dirichlet split — 100 clients, character recognition.
3. **Shakespeare** (next-character prediction) — naturally non-IID by speaker.

Metric: **test accuracy** after 200 rounds (higher is better).
