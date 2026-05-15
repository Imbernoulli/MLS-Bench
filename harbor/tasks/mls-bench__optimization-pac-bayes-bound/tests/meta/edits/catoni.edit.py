"""Catoni/Lambda bound (flamb) baseline for opt-pac-bayes-bound.

The PAC-Bayes-lambda bound (Catoni-style):
  B(Q,S) = emp_risk / (1 - lambda/2) + (KL(Q||P) + log(2*sqrt(n)/delta)) / (n*lambda*(1 - lambda/2))

where lambda is a free parameter optimized during training (Thiemann et al., 2017).
This is derived from Catoni's moment-generating function approach.

Reference:
  Catoni, O. (2007). PAC-Bayesian Supervised Classification. Lecture Notes-Monograph Series.
  Thiemann, N., Igel, C., Wintenberger, O., Seldin, Y. (2017).
    "A Strongly Quasiconvex PAC-Bayesian Bound", ALT.
  Implementation: vendor/external_packages/PBB/pbb/bounds.py (flamb branch)
"""

_FILE = "PBB/custom_pac_bayes.py"

_CONTENT = """\
class BoundOptimizer:
    \"\"\"Catoni/Lambda PAC-Bayes bound (flamb).

    Bound: emp_risk / (1 - lam/2) + (KL + log(2*sqrt(n)/delta)) / (n*lam*(1 - lam/2))
    Lambda is a learnable parameter optimized jointly with the posterior.
    Tighter than McAllester when lambda is well-tuned.
    \"\"\"

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-5, initial_lambda=0.5, lambda_lr=0.01):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin
        # Lambda parameter for the Catoni bound (learnable)
        self._lambda_param = torch.tensor(initial_lambda, requires_grad=True)
        self.lambda_lr = lambda_lr
        self._lambda_optimizer = None

    def _get_lambda(self):
        \"\"\"Get clamped lambda value in (0, 2).\"\"\"
        return torch.clamp(self._lambda_param, min=0.01, max=1.99)

    def _ensure_lambda_optimizer(self):
        if self._lambda_optimizer is None:
            self._lambda_optimizer = torch.optim.SGD(
                [self._lambda_param], lr=self.lambda_lr
            )

    def compute_bound(self, empirical_risk, kl, n, delta):
        \"\"\"Catoni/Lambda bound.\"\"\"
        lam = self._get_lambda()
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (
            n * lam * (1.0 - lam / 2.0)
        )
        bound = empirical_risk / (1.0 - lam / 2.0) + kl_term
        return bound

    def train_step(self, model, data, target, device, n_bound, delta):
        \"\"\"Training objective: Catoni/lambda bound with joint lambda optimization.\"\"\"
        # Ensure lambda is on correct device
        if self._lambda_param.device != device:
            self._lambda_param = self._lambda_param.to(device).detach().requires_grad_(True)
            self._lambda_optimizer = None
        self._ensure_lambda_optimizer()

        output = model(data, sample=True)
        log_probs = F.log_softmax(output, dim=1)
        log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
        nll = F.nll_loss(log_probs, target)

        kl = get_total_kl(model)

        # Update lambda on a detached copy — the outer loop's optimizer.step()
        # only knows about posterior params, so lambda would stay frozen at
        # init without this explicit step. Before the fix, lambda=1.0 caused
        # the Catoni bound to double the KL contribution (1-lam/2=0.5), which
        # forced KL to grow to ~10x McAllester's value.
        self._lambda_optimizer.zero_grad()
        lam = self._get_lambda()
        lam_bound = nll.detach() / (1.0 - lam / 2.0) + (
            kl.detach() + math.log(2.0 * math.sqrt(n_bound) / delta)
        ) / (n_bound * lam * (1.0 - lam / 2.0))
        lam_bound.backward()
        self._lambda_optimizer.step()

        bound = self.compute_bound(nll, kl, n_bound, delta)
        return bound

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 mc_samples=1000):
        \"\"\"Evaluate Catoni risk certificate with PAC-Bayes-kl inversion.\"\"\"
        model.eval()
        n_bound = len(bound_loader.dataset)

        # 1. Empirical 0-1 risk via MC sampling
        emp_risk_01 = compute_01_risk(model, bound_loader, device,
                                      mc_samples=mc_samples)

        # 2. NLL-based empirical risk
        total_nll = 0.0
        total_samples = 0
        with torch.no_grad():
            for data, target in bound_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, sample=True)
                log_probs = F.log_softmax(output, dim=1)
                log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
                nll = F.nll_loss(log_probs, target, reduction="sum")
                total_nll += nll.item()
                total_samples += target.size(0)
        emp_nll = total_nll / total_samples

        # 3. KL divergence
        with torch.no_grad():
            dummy_data = next(iter(bound_loader))[0][:1].to(device)
            model(dummy_data, sample=True)
            kl = get_total_kl(model).item()

        # 4. PAC-Bayes-kl inversion for 0-1 loss certificate
        c = (kl + math.log(2.0 * math.sqrt(n_bound) / delta)) / n_bound
        risk_cert_01 = inv_kl(emp_risk_01, c)

        # 5. CE bound using Catoni formula
        emp_nll_t = torch.tensor(emp_nll)
        kl_t = torch.tensor(kl)
        ce_bound = self.compute_bound(emp_nll_t, kl_t, n_bound, delta).item()

        metrics = {
            "empirical_01_risk": emp_risk_01,
            "empirical_nll": emp_nll,
            "kl_divergence": kl,
            "ce_bound": ce_bound,
            "lambda": self._get_lambda().item(),
        }

        return risk_cert_01, metrics
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 460,
        "end_line": 604,
        "content": _CONTENT,
    },
]
