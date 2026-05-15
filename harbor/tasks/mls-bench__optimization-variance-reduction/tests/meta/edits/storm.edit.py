"""STORM (STochastic Recursive Momentum) baseline.

Uses momentum-based recursive variance reduction without requiring periodic
full gradient computations. Instead, maintains an exponentially-weighted
running gradient estimate that achieves variance reduction online.

Update:  d_t = (1-a) * grad_i(x_t) + a * (d_{t-1} + grad_i(x_t) - grad_i(x_{t-1}))

Reference: Cutkosky & Orabona, "Momentum-Based Variance Reduction in
Non-Convex SGD", NeurIPS 2019.
"""

_FILE = "opt-vr-bench/custom_vr.py"

_CONTENT = """\
# Design a variance reduction mechanism for stochastic gradient computation.
# You may modify ONLY this section.
#
# Interface contract:
#   - VarianceReductionOptimizer.__init__(model, lr, l2_reg, loss_type, n_train, batch_size, device)
#   - VarianceReductionOptimizer.train_one_epoch(X_train, y_train)
#     -> trains for one epoch, returns dict with 'avg_loss'
#
# Available helper functions (FIXED, defined above):
#   - compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
#     -> returns list of full gradient tensors
#   - compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
#     -> returns list of stochastic gradient tensors on a mini-batch
#   - compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
#     -> returns scalar loss tensor
#
# Constraints:
#   - Must work across all problems with the shared hyperparameter config
#   - May use full gradient computation (compute_full_gradient) at most once
#     per epoch (to maintain sublinear per-epoch cost)
#   - Must respect the provided learning rate and L2 regularization
#   - The model parameters should be updated in-place (via param.data)

class VarianceReductionOptimizer:
    \"\"\"STORM: STochastic Recursive Momentum.

    Maintains a momentum-based gradient estimator that achieves variance
    reduction without requiring periodic full gradient computations (unlike
    SVRG/SARAH).  The key idea is to use an exponential moving average of
    recursively corrected stochastic gradients:

        d_t = (1-a) * g_t + a * (d_{t-1} + g_t - g_{t-1}')

    where g_t = grad_i(x_t), g_{t-1}' = grad_i(x_{t-1}), and a is a
    momentum coefficient.  The first epoch uses a full gradient to warm-start.
    \"\"\"

    def __init__(self, model: nn.Module, lr: float, l2_reg: float,
                 loss_type: str, n_train: int, batch_size: int,
                 device: torch.device):
        self.model = model
        self.lr = lr
        self.l2_reg = l2_reg
        self.loss_type = loss_type
        self.n_train = n_train
        self.batch_size = batch_size
        self.device = device
        self.params = list(model.parameters())
        # Momentum coefficient (STORM paper recommends a = 1 - 1/sqrt(T))
        n_steps_per_epoch = max(1, n_train // batch_size)
        self.momentum = 1.0 - 1.0 / math.sqrt(n_steps_per_epoch)
        # Running gradient estimator
        self.d = None
        # Previous parameters for correction term
        self.prev_params = None
        self.initialized = False

    def _save_params(self):
        return [p.data.clone() for p in self.params]

    def _load_params(self, saved):
        for p, s in zip(self.params, saved):
            p.data.copy_(s)

    def train_one_epoch(self, X_train: torch.Tensor,
                        y_train: torch.Tensor) -> dict:
        self.model.train()
        n = X_train.size(0)
        a = self.momentum
        full_grad_count = 0

        # Initialize with full gradient on first epoch
        if not self.initialized:
            self.d = compute_full_gradient(
                self.model, X_train, y_train, self.loss_type,
                self.l2_reg, self.device
            )
            self.prev_params = self._save_params()
            # First step using full gradient
            with torch.no_grad():
                for p, di in zip(self.params, self.d):
                    p.data.add_(di, alpha=-self.lr)
            self.initialized = True
            full_grad_count = 1

        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idx = indices[start:end]
            Xb = X_train[idx].to(self.device)
            yb = y_train[idx].to(self.device)

            # Current stochastic gradient g_t = grad_i(x_t)
            current_params = self._save_params()
            g_current = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg
            )

            # Previous stochastic gradient g_{t-1}' = grad_i(x_{t-1})
            self._load_params(self.prev_params)
            g_prev = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg
            )
            self._load_params(current_params)

            # STORM update: d_t = (1-a)*g_t + a*(d_{t-1} + g_t - g_{t-1}')
            with torch.no_grad():
                for i, (p, gc, gp, di) in enumerate(zip(
                        self.params, g_current, g_prev, self.d)):
                    self.d[i] = (1 - a) * gc + a * (di + gc - gp)
                    p.data.add_(self.d[i], alpha=-self.lr)

            self.prev_params = self._save_params()

            # Track loss
            with torch.no_grad():
                loss = compute_loss_on_batch(
                    self.model, Xb, yb, self.loss_type, self.l2_reg
                )
                total_loss += loss.item()
            n_batches += 1

        return {"avg_loss": total_loss / max(n_batches, 1),
                "full_grad_count": full_grad_count}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 286,
        "end_line": 370,
        "content": _CONTENT,
    },
]
