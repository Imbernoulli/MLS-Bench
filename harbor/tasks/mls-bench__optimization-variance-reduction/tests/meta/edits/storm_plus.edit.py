"""STORM+ (enhanced STORM with adaptive momentum and per-step adaptive lr).

Like STORM but with adaptive momentum, full gradient warmstart, per-step
adaptive step sizing, and gradient norm clipping.

Reference: Based on Cutkosky & Orabona, NeurIPS 2019, with enhancements.
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
    \"\"\"STORM+ with adaptive momentum and per-step adaptive lr.

    d_t = (1-a_t)*g_t + a_t*(d_{t-1} + g_t - g_{t-1}')
    a_t = min(1 - 1/sqrt(t+1), 0.999)

    Full gradient warmstart on first epoch.
    Per-step lr: min(lr, 0.01 * ||w|| / ||d||).
    Gradient clipping: scale d if ||d|| > 3*||g||.
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
        self.d = None
        self.prev_params = None
        self.initialized = False
        self.global_step = 0

    def _save_params(self):
        return [p.data.clone() for p in self.params]

    def _load_params(self, saved):
        for p, s in zip(self.params, saved):
            p.data.copy_(s)

    def _gnorm(self, grads):
        return math.sqrt(sum(g.pow(2).sum().item() for g in grads))

    def _step_lr(self, direction):
        dnorm = self._gnorm(direction)
        pnorm = math.sqrt(sum(
            p.data.pow(2).sum().item() for p in self.params)) + 1e-8
        return min(self.lr, 0.01 * pnorm / (dnorm + 1e-8))

    def train_one_epoch(self, X_train, y_train):
        self.model.train()
        n = X_train.size(0)
        full_grad_count = 0

        if not self.initialized:
            self.d = compute_full_gradient(
                self.model, X_train, y_train, self.loss_type,
                self.l2_reg, self.device)
            self.prev_params = self._save_params()
            eta = self._step_lr(self.d)
            with torch.no_grad():
                for p, di in zip(self.params, self.d):
                    p.data.add_(di, alpha=-eta)
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

            self.global_step += 1
            a = min(1.0 - 1.0 / math.sqrt(self.global_step + 1), 0.999)

            current_params = self._save_params()
            g_current = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg)

            self._load_params(self.prev_params)
            g_prev = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg)
            self._load_params(current_params)

            with torch.no_grad():
                for i, (gc, gp, di) in enumerate(zip(
                        g_current, g_prev, self.d)):
                    self.d[i] = (1 - a) * gc + a * (di + gc - gp)

                # Clip
                d_norm = self._gnorm(self.d)
                g_norm = self._gnorm(g_current)
                if d_norm > 3.0 * g_norm and g_norm > 1e-8:
                    scale = 3.0 * g_norm / d_norm
                    for di in self.d:
                        di.mul_(scale)

                eta = self._step_lr(self.d)
                for p, di in zip(self.params, self.d):
                    p.data.add_(di, alpha=-eta)

            self.prev_params = self._save_params()

            with torch.no_grad():
                loss = compute_loss_on_batch(
                    self.model, Xb, yb, self.loss_type, self.l2_reg)
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
