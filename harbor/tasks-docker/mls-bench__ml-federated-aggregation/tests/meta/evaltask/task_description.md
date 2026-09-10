# Federated Learning Aggregation Strategy Design

## Research Question
Design a server-side aggregation strategy for federated learning that converges faster and to higher test accuracy under heterogeneous (non-IID) client data. The contribution is the *aggregation rule* (and optionally the client-selection / client-side correction exposed by this interface), not changes to the local optimizer or simulation harness.

## Background
Federated Learning (FL) trains a shared global model across many clients without centralizing data. Under non-IID client data, naive averaging suffers from "client drift" — local updates diverge, slowing or destabilizing convergence.

Reference baselines:
- **FedAvg** — McMahan, Moore, Ramage, Hampson, Agüera y Arcas, AISTATS 2017 ([arXiv:1602.05629](https://arxiv.org/abs/1602.05629)). Server averages client model parameters weighted by `n_k / sum(n_k)` (number of samples per client). No server-side state.
- **FedProx** — Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith, MLSys 2020 ([arXiv:1812.06127](https://arxiv.org/abs/1812.06127)). Same server aggregation as FedAvg, but each client adds a proximal term `(mu/2) * ||w - w_global||^2` to its local objective; default `mu = 0.01`.
- **SCAFFOLD** — Karimireddy, Kale, Mohri, Reddi, Stich, Suresh, ICML 2020 ([arXiv:1910.06378](https://arxiv.org/abs/1910.06378)). Maintains server- and client-side control variates `c, c_i` to correct client drift. Local update: `w <- w - eta * (g_i - c_i + c)`. Server updates `c` after each round from received deltas.

## Implementation Contract
Modify `ServerAggregator` in `flower/custom_fl_aggregation.py`:

```python
class ServerAggregator:
    def __init__(self, global_model, args):
        # Initialize aggregation state (momentum buffers, control variates, ...).
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

## Fixed Pipeline & Evaluation
- **Communication rounds**: 200.
- **Per-round participation**: 10 of 100 clients.
- **Local training**: 5 local epochs per round, SGD with `lr=0.01`.

Benchmarks:
1. **CIFAR-10** with Dirichlet split (`alpha=0.1`) — 100 clients, 10-class image classification.
2. **FEMNIST** (EMNIST ByClass) with Dirichlet split — 100 clients, character recognition.
3. **Shakespeare** (next-character prediction) — naturally non-IID by speaker.

Metric: **test accuracy** after 200 rounds (higher is better).
