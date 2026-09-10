"""TRADES baseline (Zhang et al., 2019).

Reference: https://arxiv.org/abs/1901.08573
Decomposes robustness into clean accuracy + boundary regularization:
  loss = CE(clean, y) + beta * KL(clean || adv)
"""

_FILE = "torchattacks/bench/custom_adv_train.py"

_CONTENT = """\
class AdversarialTrainer:
    \"\"\"TRADES (Zhang et al., 2019).\"\"\"

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes
        self.beta = 6.0  # TRADES regularization weight

    def train_step(self, images, labels, optimizer):
        self.model.train()

        # Clean forward pass
        logits_clean = self.model(images)
        loss_clean = F.cross_entropy(logits_clean, labels)

        # Generate adversarial examples by maximizing KL divergence
        self.model.eval()
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

        for _ in range(self.attack_steps):
            adv_images.requires_grad_(True)
            logits_adv = self.model(adv_images)
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean.detach(), dim=1),
                reduction='batchmean',
            )
            grad = torch.autograd.grad(loss_kl, adv_images)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        # TRADES loss: clean CE + beta * KL(clean || adv)
        self.model.train()
        logits_adv = self.model(adv_images)
        loss_kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean.detach(), dim=1),
            reduction='batchmean',
        )
        loss = loss_clean + self.beta * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            'loss': loss.item(),
            'loss_clean': loss_clean.item(),
            'loss_kl': loss_kl.item(),
        }

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
