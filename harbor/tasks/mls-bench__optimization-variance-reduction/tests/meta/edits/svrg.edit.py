"""SVRG (Stochastic Variance Reduced Gradient) baseline.

Periodically computes a full gradient at a snapshot point, then uses it as a
control variate to reduce the variance of stochastic gradient estimates.
Inner loop update:  v_t = grad_i(x_t) - grad_i(x_snap) + full_grad(x_snap)

Reference: Johnson & Zhang, "Accelerating Stochastic Gradient Descent using
Predictive Variance Reduction", NeurIPS 2013.
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
    \"\"\"SVRG with adaptive step sizing and geometric growth cap.

    At the start of each epoch, computes a full gradient at the current
    snapshot point.  Each inner iteration uses the control-variate estimator:
        v_t = grad_i(x_t) - grad_i(x_snap) + mu   (where mu = full_grad(x_snap))

    Step size: eta = min(lr, 0.01 * ||w||/||g||, eta_max).
    eta_max grows geometrically at 1.5x per epoch, allowing the step to
    increase as training progresses (gnorm decreases) while preventing the
    runaway growth that caused divergence in v2.
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
        self.snapshot_params = None
        self.full_grad = None
        self.eta_max = None

    def _save_snapshot(self):
        self.snapshot_params = [p.data.clone() for p in self.params]

    def _load_snapshot(self):
        saved = [p.data.clone() for p in self.params]
        for p, sp in zip(self.params, self.snapshot_params):
            p.data.copy_(sp)
        return saved

    def _restore_params(self, saved):
        for p, s in zip(self.params, saved):
            p.data.copy_(s)

    def train_one_epoch(self, X_train: torch.Tensor,
                        y_train: torch.Tensor) -> dict:
        self.model.train()
        n = X_train.size(0)

        # --- Snapshot ---
        self._save_snapshot()
        self.full_grad = compute_full_gradient(
            self.model, X_train, y_train, self.loss_type,
            self.l2_reg, self.device
        )

        # Standard SVRG: use the provided lr directly. For ill-conditioned
        # MSE problems cap the first-step magnitude by 1/||∇F|| to prevent
        # divergence (previous adaptive 1.5x-geometric schedule blew up to
        # eta≈1e5 and gave final MSE≈1e34).
        if self.loss_type == 'mse':
            gnorm = math.sqrt(sum(
                g.pow(2).sum().item() for g in self.full_grad)) + 1e-8
            effective_lr = min(self.lr, 1.0 / gnorm)
        else:
            effective_lr = self.lr

        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idx = indices[start:end]
            Xb = X_train[idx].to(self.device)
            yb = y_train[idx].to(self.device)

            grad_at_x = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg
            )

            saved = self._load_snapshot()
            grad_at_snap = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg
            )
            self._restore_params(saved)

            # SVRG update: v = grad_i(x_t) - grad_i(x_snap) + mu
            with torch.no_grad():
                for p, gx, gs, mu in zip(self.params, grad_at_x,
                                         grad_at_snap, self.full_grad):
                    vr_grad = gx - gs + mu
                    p.data.add_(vr_grad, alpha=-effective_lr)

            with torch.no_grad():
                loss = compute_loss_on_batch(
                    self.model, Xb, yb, self.loss_type, self.l2_reg
                )
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
