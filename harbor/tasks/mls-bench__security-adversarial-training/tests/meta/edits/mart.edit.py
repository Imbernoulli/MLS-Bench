"""MART baseline (Wang et al., 2020).

Reference: https://openreview.net/forum?id=rklOg6EFwS
           https://github.com/YisenWang/MART
Misclassification-Aware Regularized Training:
  loss = BCE_boosted(adv, y) + beta * (1 - p(y|x)) * KL(adv || clean)
where BCE_boosted = CE(adv, y) + NLL(log(1 - p_adv), runner_up_class).
"""

_FILE = "torchattacks/bench/custom_adv_train.py"

_CONTENT = """\
class AdversarialTrainer:
    \"\"\"MART (Wang et al., 2020).\"\"\"

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes
        self.beta = 6.0  # MART regularization weight

    def train_step(self, images, labels, optimizer):
        # Generate adversarial examples using PGD (maximize CE loss).
        # Follows official https://github.com/YisenWang/MART/blob/master/mart.py
        self.model.eval()
        adv_images = images.detach() + 0.001 * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

        for _ in range(self.attack_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            grad = torch.autograd.grad(loss, adv_images)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        # MART loss (exactly as in official mart.py)
        self.model.train()
        optimizer.zero_grad()

        logits_clean = self.model(images).detach()  # detach for stable KL target + weighting (matches official mart.py)
        logits_adv = self.model(adv_images)
        adv_probs = F.softmax(logits_adv, dim=1)

        # Boosted CE: standard CE + penalize runner-up class
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(
            tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1],
        )
        loss_adv = F.cross_entropy(logits_adv, labels) + F.nll_loss(
            torch.log(1.0001 - adv_probs + 1e-12), new_y,
        )

        # Misclassification-aware KL regularization
        nat_probs = F.softmax(logits_clean, dim=1)
        true_probs = nat_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        kl_per_sample = F.kl_div(
            torch.log(adv_probs + 1e-12), nat_probs, reduction='none',
        ).sum(dim=1)
        batch_size = images.size(0)
        loss_robust = (1.0 / batch_size) * torch.sum(
            kl_per_sample * (1.0000001 - true_probs)
        )

        loss = loss_adv + self.beta * loss_robust

        loss.backward()
        optimizer.step()

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
