"""SARAH (StochAstic Recursive grAdient algoritHm) baseline.

Uses a recursive gradient estimator that accumulates stochastic differences.
Unlike SVRG, which always references the snapshot gradient, SARAH recursively
updates: v_t = grad_i(x_t) - grad_i(x_{t-1}) + v_{t-1}

Reference: Nguyen, Liu, Scheinberg & Takac, "SARAH: A Novel Method for Machine
Learning Problems Using Stochastic Recursive Gradient", ICML 2017.
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
    \"\"\"SARAH with adaptive step sizing, eta_max cap, and mini-epoch resets.

    At the start of each epoch, computes a full gradient v_0 = full_grad(x_0).
    Each subsequent step uses a recursive estimator:
        v_t = grad_i(x_t) - grad_i(x_{t-1}) + v_{t-1}

    Step size: eta = min(lr, 0.01 * ||w|| / ||full_grad||, eta_max)
    where eta_max grows geometrically (2x per epoch max).  The cap prevents
    when gnorm shrinks faster than pnorm grows during training.

    Mini-epoch resets every sqrt(T) steps + 1.5x gradient norm clipping
    control recursive drift.
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
        self.prev_params = None
        self.eta_max = None  # Set on first epoch

    def _save_params(self):
        return [p.data.clone() for p in self.params]

    def _load_params(self, saved):
        for p, s in zip(self.params, saved):
            p.data.copy_(s)

    def _grad_norm(self, grads):
        return math.sqrt(sum(g.pow(2).sum().item() for g in grads))

    def train_one_epoch(self, X_train: torch.Tensor,
                        y_train: torch.Tensor) -> dict:
        self.model.train()
        n = X_train.size(0)

        # --- Full gradient ---
        self.v = compute_full_gradient(
            self.model, X_train, y_train, self.loss_type,
            self.l2_reg, self.device
        )

        gnorm = self._grad_norm(self.v)
        pnorm = math.sqrt(sum(p.data.pow(2).sum().item()
                               for p in self.params)) + 1e-8

        adaptive_step = 0.01 * pnorm / (gnorm + 1e-8)
        # Only apply eta_max growth cap for regression (MSE) problems.
        # For classification, adaptive_step recovers naturally as gnorm
        # decreases. The eta_max cap would unnecessarily slow MLP convergence.
        if self.loss_type == 'mse':
            if self.eta_max is None:
                self.eta_max = adaptive_step
            else:
                self.eta_max = min(2.0 * self.eta_max, adaptive_step)
            effective_lr = min(self.lr, adaptive_step, self.eta_max)
        else:
            effective_lr = min(self.lr, adaptive_step)

        # Mini-epoch reset interval
        max_inner = max(1, int(math.sqrt(n / self.batch_size)))

        # First step with full gradient
        with torch.no_grad():
            for p, vi in zip(self.params, self.v):
                p.data.add_(vi, alpha=-effective_lr)
        self.prev_params = self._save_params()

        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        inner_step = 0

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idx = indices[start:end]
            Xb = X_train[idx].to(self.device)
            yb = y_train[idx].to(self.device)

            # Mini-epoch reset
            if inner_step > 0 and inner_step % max_inner == 0:
                self.v = compute_stochastic_gradient(
                    self.model, Xb, yb, self.loss_type, self.l2_reg
                )
                self.prev_params = self._save_params()
                with torch.no_grad():
                    for p, vi in zip(self.params, self.v):
                        p.data.add_(vi, alpha=-effective_lr)
                inner_step += 1
                n_batches += 1
                with torch.no_grad():
                    loss = compute_loss_on_batch(
                        self.model, Xb, yb, self.loss_type, self.l2_reg)
                    total_loss += loss.item()
                continue

            current_params = self._save_params()
            grad_at_current = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg
            )

            self._load_params(self.prev_params)
            grad_at_prev = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg
            )
            self._load_params(current_params)

            with torch.no_grad():
                for i, (gc, gp, vi) in enumerate(zip(
                        grad_at_current, grad_at_prev, self.v)):
                    self.v[i] = gc - gp + vi

                # Clip recursive estimate
                vr_norm = self._grad_norm(self.v)
                if vr_norm > 1.0 * gnorm and gnorm > 1e-8:
                    scale = 1.0 * gnorm / vr_norm
                    for vi in self.v:
                        vi.mul_(scale)

                for p, vi in zip(self.params, self.v):
                    p.data.add_(vi, alpha=-effective_lr)

            self.prev_params = self._save_params()
            inner_step += 1

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
