"""AWP + TRADES baseline (Wu et al., 2020 + Zhang et al., 2019).

Reference: https://arxiv.org/abs/2004.05884 (AWP paper)
           https://github.com/csdongxian/AWP (official code)
           https://github.com/csdongxian/AWP/blob/main/trades_AWP/utils_awp.py
           https://github.com/csdongxian/AWP/blob/main/trades_AWP/train_trades_cifar.py
           https://arxiv.org/abs/1901.08573 (TRADES)

Key implementation details (follow official trades_AWP):
  * proxy model + proxy SGD optimizer (lr matches model lr, default 0.1)
  * calc_awp minimizes  - (CE(clean) + beta * KL(adv || clean))  on proxy
  * diff = (old_w.norm() / (new_w - old_w).norm()) * (new_w - old_w)
      applied ONLY to params with dim > 1 AND 'weight' in name
  * perturb: model += gamma * diff   (gamma default 0.005)
  * compute TRADES loss under perturbed weights, backward, RESTORE, then step
"""

_FILE = "torchattacks/bench/custom_adv_train.py"

_CONTENT = """\
class AdversarialTrainer:
    \"\"\"AWP + TRADES (Wu et al., 2020 + Zhang et al., 2019).

    Follows official trades_AWP/utils_awp.py:TradesAWP and
    trades_AWP/train_trades_cifar.py (csdongxian/AWP).
    \"\"\"

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        import copy
        from collections import OrderedDict
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes
        self.beta = 6.0        # TRADES regularization weight
        self.gamma = 0.005     # AWP perturbation magnitude (paper default)
        self._EPS_AWP = 1e-20
        # Proxy model + optimizer (lr matches main model's lr=0.1 as in paper).
        # Using a proxy ensures BN running stats of the main model are untouched.
        self.proxy = copy.deepcopy(model)
        # Official uses proxy_optim lr == main lr (0.1). Magnitude is controlled
        # by gamma + weight-norm normalization in _diff_in_weights, not proxy lr.
        self.proxy_optim = torch.optim.SGD(self.proxy.parameters(), lr=0.1)

    def _diff_in_weights(self):
        \"\"\"Return OrderedDict {name: (old.norm()/diff.norm())*diff} for multi-dim weight params.\"\"\"
        from collections import OrderedDict
        diff = OrderedDict()
        model_sd = self.model.state_dict()
        proxy_sd = self.proxy.state_dict()
        for (old_k, old_w), (new_k, new_w) in zip(model_sd.items(), proxy_sd.items()):
            if old_w.dim() <= 1:
                continue
            if 'weight' in old_k:
                diff_w = new_w - old_w
                diff[old_k] = old_w.norm() / (diff_w.norm() + self._EPS_AWP) * diff_w
        return diff

    def _add_into_weights(self, diff, coeff):
        with torch.no_grad():
            names = diff.keys()
            for name, param in self.model.named_parameters():
                if name in names:
                    param.add_(coeff * diff[name])

    def _calc_awp(self, adv_images, clean_images, labels):
        \"\"\"Optimize proxy to INCREASE TRADES loss (=> gradient ASCENT via negated loss).\"\"\"
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        loss_natural = F.cross_entropy(self.proxy(clean_images), labels)
        loss_robust = F.kl_div(
            F.log_softmax(self.proxy(adv_images), dim=1),
            F.softmax(self.proxy(clean_images), dim=1),
            reduction='batchmean',
        )
        loss = -1.0 * (loss_natural + self.beta * loss_robust)
        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()
        return self._diff_in_weights()

    def train_step(self, images, labels, optimizer):
        # Step 1: generate adversarial examples (TRADES-style, maximize KL)
        self.model.eval()
        adv_images = images.detach() + 0.001 * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)
        for _ in range(self.attack_steps):
            adv_images.requires_grad_(True)
            loss_kl = F.kl_div(
                F.log_softmax(self.model(adv_images), dim=1),
                F.softmax(self.model(images), dim=1),
                reduction='sum',
            )
            grad = torch.autograd.grad(loss_kl, adv_images)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign().detach()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        self.model.train()

        # Step 2: AWP — compute weight perturbation via proxy, then apply to model
        diff = self._calc_awp(adv_images, images, labels)
        self._add_into_weights(diff, coeff=1.0 * self.gamma)

        # Step 3: TRADES loss under perturbed weights
        optimizer.zero_grad()
        logits_adv = self.model(adv_images)
        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(self.model(images), dim=1),
            reduction='batchmean',
        )
        logits_clean = self.model(images)
        loss_clean = F.cross_entropy(logits_clean, labels)
        loss = loss_clean + self.beta * loss_robust

        # Step 4: backward, step optimizer, then RESTORE original weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self._add_into_weights(diff, coeff=-1.0 * self.gamma)

        return {'loss': loss.item()}

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 10,
        "end_line": 54,
        "content": _CONTENT,
    }
]
