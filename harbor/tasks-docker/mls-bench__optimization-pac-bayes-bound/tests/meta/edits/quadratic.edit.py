"""Quadratic PAC-Bayes bound (fquad) baseline for opt-pac-bayes-bound.

The PAC-Bayes-quadratic bound:
  B(Q,S) = ( sqrt(emp_risk + kl_term) + sqrt(kl_term) )^2
  where kl_term = (KL(Q||P) + log(2*sqrt(n)/delta)) / (2n)

Tighter than McAllester/fclassic when emp_risk is small.

Critical detail replicated from the reference PBB implementation:
  During TRAINING, the surrogate cross-entropy must be rescaled into [0,1] via
  1/log(1/pmin) -- see Perez-Ortiz et al. 2021 JMLR Eq. (11) and
  https://github.com/mperezortiz/PBB/blob/master/pbb/bounds.py
  (`compute_empirical_risk` with `bounded=True`). Without this rescaling the
  training objective and fquad formula are mis-calibrated and the posterior
  drifts far from the prior, inflating KL and loosening the certificate.

Reference:
  Rivasplata, Kuzborskij, Szepesvari, Shawe-Taylor (2019).
    "PAC-Bayes Analysis Beyond the Usual Bounds", NeurIPS 2020.
  Perez-Ortiz, Rivasplata, Shawe-Taylor, Szepesvari (2021).
    "Tighter Risk Certificates for Neural Networks", JMLR 22(227):1-40.
    https://jmlr.csail.mit.edu/papers/volume22/20-879/20-879.pdf
"""

_FILE = "PBB/custom_pac_bayes.py"

_CONTENT = """\
class BoundOptimizer:
    \"\"\"Quadratic PAC-Bayes bound (fquad, Rivasplata 2019 / Perez-Ortiz 2021).

    Bound: (sqrt(emp_risk + kl_term) + sqrt(kl_term))^2
    where kl_term = (KL + log(2*sqrt(n)/delta)) / (2n)

    Tighter than McAllester when empirical risk is low.
    \"\"\"

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin
        # PBB's loss-bounding constant: maps unbounded NLL into [0,1] via
        # ell_tilde = NLL / log(1/pmin).  See Perez-Ortiz 2021 Sec 5.
        self._loss_scale = 1.0 / math.log(1.0 / self.pmin)

    def compute_bound(self, empirical_risk, kl, n, delta):
        \"\"\"Quadratic PAC-Bayes bound (fquad).\"\"\"
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
        # Ensure non-negative under sqrt
        inner = torch.clamp(empirical_risk + kl_term, min=0.0)
        kl_term_clamped = torch.clamp(kl_term, min=0.0)
        bound = (torch.sqrt(inner) + torch.sqrt(kl_term_clamped)) ** 2
        return bound

    def train_step(self, model, data, target, device, n_bound, delta):
        \"\"\"Training objective: bounded NLL passed through the fquad formula.

        The NLL is rescaled by 1/log(1/pmin) so that the surrogate loss lies in
        [0,1], matching the PBB reference implementation. This is essential
        for fquad to actually be tighter than fclassic in practice.
        \"\"\"
        output = model(data, sample=True)
        log_probs = F.log_softmax(output, dim=1)
        log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
        # Bounded NLL surrogate, in [0, 1]
        nll = F.nll_loss(log_probs, target) * self._loss_scale

        kl = get_total_kl(model)
        bound = self.compute_bound(nll, kl, n_bound, delta)
        return bound

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 mc_samples=1000):
        \"\"\"Evaluate quadratic risk certificate with PAC-Bayes-kl inversion.\"\"\"
        model.eval()
        n_bound = len(bound_loader.dataset)

        # 1. Empirical 0-1 risk via MC sampling
        emp_risk_01 = compute_01_risk(model, bound_loader, device,
                                      mc_samples=mc_samples)

        # 2. Bounded NLL empirical risk (same scaling as training)
        total_nll = 0.0
        total_samples = 0
        with torch.no_grad():
            for data, target in bound_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, sample=True)
                log_probs = F.log_softmax(output, dim=1)
                log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
                nll = F.nll_loss(log_probs, target, reduction=\"sum\")
                total_nll += nll.item()
                total_samples += target.size(0)
        emp_nll_bounded = (total_nll / total_samples) * self._loss_scale

        # 3. KL divergence
        with torch.no_grad():
            dummy_data = next(iter(bound_loader))[0][:1].to(device)
            model(dummy_data, sample=True)
            kl = get_total_kl(model).item()

        # 4. PAC-Bayes-kl inversion for 0-1 loss certificate
        c = (kl + math.log(2.0 * math.sqrt(n_bound) / delta)) / n_bound
        risk_cert_01 = inv_kl(emp_risk_01, c)

        # 5. Quadratic bound on bounded CE risk (in [0,1])
        emp_nll_t = torch.tensor(emp_nll_bounded)
        kl_t = torch.tensor(kl)
        ce_bound = self.compute_bound(emp_nll_t, kl_t, n_bound, delta).item()

        metrics = {
            \"empirical_01_risk\": emp_risk_01,
            \"empirical_nll\": emp_nll_bounded,
            \"kl_divergence\": kl,
            \"ce_bound\": ce_bound,
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
