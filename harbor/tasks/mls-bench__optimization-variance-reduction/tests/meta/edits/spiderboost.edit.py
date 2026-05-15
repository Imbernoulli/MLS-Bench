"""SpiderBoost (SPIDER with momentum acceleration) baseline.

SPIDER recursive estimator + heavy-ball momentum, with per-step adaptive
step sizing and mini-epoch resets.

Reference: Wang et al., NeurIPS 2019.
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
    \"\"\"SpiderBoost: SPIDER + momentum + per-step adaptive lr.

    v_t = g_t - g_{t-1}' + v_{t-1}  (with periodic resets)
    m_t = beta * m_{t-1} + v_t
    Step with min(lr, 0.01 * ||w|| / ||m||).
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
        self.v = None
        self.momentum_buf = None
        self.prev_params = None
        self.beta = 0.9
        n_steps = max(1, n_train // batch_size)
        self.reset_period = max(1, int(math.sqrt(n_steps)))

    def _save_params(self):
        return [p.data.clone() for p in self.params]

    def _load_params(self, saved):
        for p, s in zip(self.params, saved):
            p.data.copy_(s)

    def _step_lr(self, update_dir):
        gnorm = math.sqrt(sum(g.pow(2).sum().item() for g in update_dir))
        pnorm = math.sqrt(sum(
            p.data.pow(2).sum().item() for p in self.params)) + 1e-8
        return min(self.lr, 0.01 * pnorm / (gnorm + 1e-8))

    def train_one_epoch(self, X_train, y_train):
        self.model.train()
        n = X_train.size(0)

        self.v = compute_full_gradient(
            self.model, X_train, y_train, self.loss_type,
            self.l2_reg, self.device)

        if self.momentum_buf is None:
            self.momentum_buf = [vi.clone() for vi in self.v]
        else:
            with torch.no_grad():
                for i, vi in enumerate(self.v):
                    self.momentum_buf[i] = self.beta * self.momentum_buf[i] + vi

        eta = self._step_lr(self.momentum_buf)
        with torch.no_grad():
            for p, mi in zip(self.params, self.momentum_buf):
                p.data.add_(mi, alpha=-eta)
        self.prev_params = self._save_params()

        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        step_in_epoch = 0

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idx = indices[start:end]
            Xb = X_train[idx].to(self.device)
            yb = y_train[idx].to(self.device)
            step_in_epoch += 1

            current_params = self._save_params()
            grad_current = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg)

            if step_in_epoch % self.reset_period == 0:
                self.v = [g.clone() for g in grad_current]
            else:
                self._load_params(self.prev_params)
                grad_prev = compute_stochastic_gradient(
                    self.model, Xb, yb, self.loss_type, self.l2_reg)
                self._load_params(current_params)

                with torch.no_grad():
                    for i, (gc, gp, vi) in enumerate(zip(
                            grad_current, grad_prev, self.v)):
                        self.v[i] = gc - gp + vi

            with torch.no_grad():
                for i, vi in enumerate(self.v):
                    self.momentum_buf[i] = self.beta * self.momentum_buf[i] + vi

            eta = self._step_lr(self.momentum_buf)
            with torch.no_grad():
                for p, mi in zip(self.params, self.momentum_buf):
                    p.data.add_(mi, alpha=-eta)

            self.prev_params = self._save_params()

            with torch.no_grad():
                loss = compute_loss_on_batch(
                    self.model, Xb, yb, self.loss_type, self.l2_reg)
                total_loss += loss.item()
            n_batches += 1

        return {"avg_loss": total_loss / max(n_batches, 1),
                "full_grad_count": 1}
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
