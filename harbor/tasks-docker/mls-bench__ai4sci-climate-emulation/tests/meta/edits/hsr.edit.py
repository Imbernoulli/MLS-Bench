"""Heteroskedastic Regression (HSR) baseline for ai4sci-climate-emulation.

Faithful to the ClimSim reference HSR, Yu et al. NeurIPS 2023 D&B,
`baseline_models/HSR/training/hsr.py` + `hpo.py` (final config):
  - TWO SEPARATE MLPs: a mean network and a log-precision network, each
    hidden_dims=1024, layers=4, dropout=0, block = Linear -> LayerNorm ->
    Dropout -> ReLU, with a linear output layer.
  - Gaussian NLL with precision tau = exp(logprec):
        loss = mean[ tau * (y - mu)^2 - logprec ]
    (i.e. -2x Gaussian log-likelihood up to a constant), clipped to +-1e5.
  - MSE warm-up for the first 1/3 of training epochs (reference: epochs/3),
    then switch to the NLL. Here epochs come from the task budget (NUM_EPOCHS).
  - Inference returns only the mean (ClimSim reports metrics on the mean).

I/O adapted to this task's 556-dim input / 368-dim output. The NLL/warm-up are
injected by overriding the trainer's MSELoss; optimizer/LR/batch are the task's
fixed unified budget (AdamW + cosine).

Reference: Yu et al., "ClimSim..." (NeurIPS 2023 D&B), HSR baseline;
Nix & Weigend (1994), heteroskedastic NN.
"""

_FILE = "ClimSim/custom_emulator.py"

_CONTENT = """\
class _HSRNet(nn.Module):
    \"\"\"One MLP head: `layers` x (Linear -> LayerNorm -> Dropout -> ReLU) + Linear.\"\"\"
    def __init__(self, in_dim, out_dim, hidden=1024, layers=4, dropout=0.0):
        super().__init__()
        blocks = []
        prev = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(prev, hidden), nn.LayerNorm(hidden), nn.Dropout(dropout), nn.ReLU()]
            prev = hidden
        blocks.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class Custom(nn.Module):
    \"\"\"Heteroskedastic regression: separate mean and log-precision networks.\"\"\"

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mean = _HSRNet(input_dim, output_dim, hidden=1024, layers=4, dropout=0.0)
        self.logprec = _HSRNet(input_dim, output_dim, hidden=1024, layers=4, dropout=0.0)
        # Epoch tracking for the MSE -> NLL warm-up (reference: first epochs/3).
        self._epoch = 0
        try:
            # reference switches at epoch < epochs/3 (0-indexed) -> ceil MSE epochs
            self._warmup = max(1, math.ceil(int(os.environ.get('NUM_EPOCHS', 30)) / 3))
        except Exception:
            self._warmup = 1
        self._last_mean = None
        self._last_logprec = None

    def train(self, mode=True):
        # The trainer calls model.train() exactly once at the start of each epoch
        # (model.eval() only runs every EVAL_INTERVAL epochs, so a False->True edge
        # is unreliable). Count every train(True) call as one epoch.
        if mode:
            self._epoch += 1
        return super().train(mode)

    def forward(self, x):
        mu = self.mean(x)
        logprec = torch.clamp(self.logprec(x), min=-10.0, max=10.0)
        self._last_mean = mu
        self._last_logprec = logprec
        return mu  # inference / metrics use the mean

    def hsr_loss(self, mu, logprec, target):
        if self._epoch <= self._warmup:           # MSE warm-up (first 1/3)
            return ((target - mu) ** 2).mean()
        prec = torch.exp(logprec)                 # tau
        nll = (prec * (target - mu) ** 2 - logprec).mean()
        return torch.clamp(nll, min=-1e5, max=1e5)


# --- inject the HSR objective by overriding the trainer's MSELoss ------------
# Subclass the canonical MSELoss (torch.nn.modules.loss), not nn.MSELoss, which
# another baseline's edit may have rebound when all baselines are imported into
# one process (e.g. budget_check.py).
_OrigMSELoss = torch.nn.modules.loss.MSELoss

class _HSRMSELossShim(_OrigMSELoss):
    _active_model = None

    def forward(self, predictions, target):
        m = _HSRMSELossShim._active_model
        if m is not None and getattr(m, '_last_logprec', None) is not None \\
           and m._last_mean is predictions:
            return m.hsr_loss(m._last_mean, m._last_logprec, target)
        return super().forward(predictions, target)

nn.MSELoss = _HSRMSELossShim

_OrigCustomInit = Custom.__init__

def _patched_init(self, input_dim, output_dim):
    _OrigCustomInit(self, input_dim, output_dim)
    _HSRMSELossShim._active_model = self

Custom.__init__ = _patched_init
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 86,
        "end_line": 118,
        "content": _CONTENT,
    },
]
