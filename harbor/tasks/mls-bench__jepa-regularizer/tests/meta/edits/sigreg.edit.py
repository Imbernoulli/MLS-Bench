"""SIGReg/BCS baseline -- rigorous codebase edit ops.

Replaces the placeholder CustomRegularizer with a BCS (Batched Characteristic
Slicing) implementation that uses Epps-Pulley Gaussianity testing on random
projections combined with an invariance loss.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "eb_jepa/custom_regularizer.py"

# ── Replace editable region: CustomRegularizer class + CONFIG_OVERRIDES ─────

_BCS_CLASS = """\
class CustomRegularizer(nn.Module):
    \"\"\"BCS (Batched Characteristic Slicing) regularizer for SIGReg.\"\"\"

    def __init__(self, num_slices=256, lmbd=10.0):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd

    def _epps_pulley(self, x, t_min=-3, t_max=3, n_points=10):
        t = torch.linspace(t_min, t_max, n_points, device=x.device)
        exp_f = torch.exp(-0.5 * t ** 2)
        x_t = x.unsqueeze(2) * t
        ecf = (1j * x_t).exp().mean(0)
        err = exp_f * (ecf - exp_f).abs() ** 2
        T = torch.trapz(err, t, dim=1)
        return T

    def forward(self, z1, z2):
        dev = z1.device
        with torch.no_grad():
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (z1.size(1), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A = A / A.norm(p=2, dim=0)
        view1 = z1 @ A
        view2 = z2 @ A

        self.step += 1
        bcs = (self._epps_pulley(view1).mean() + self._epps_pulley(view2).mean()) / 2
        invariance_loss = F.mse_loss(z1, z2)
        total_loss = invariance_loss + self.lmbd * bcs
        return {
            "loss": total_loss,
            "bcs_loss": bcs,
            "invariance_loss": invariance_loss,
        }


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: proj_output_dim, proj_hidden_dim.
# Paper sigreg.yaml uses 2048->128 — SIGReg's Gaussianity test on random
# projections concentrates better at low output dims (paper rank-1: 91.02%).
CONFIG_OVERRIDES = {"proj_output_dim": 128}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 33,
        "end_line": 58,
        "content": _BCS_CLASS,
    },
]
