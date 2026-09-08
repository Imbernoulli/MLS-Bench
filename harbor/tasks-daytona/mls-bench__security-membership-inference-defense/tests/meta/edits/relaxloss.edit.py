"""RelaxLoss baseline for security-membership-inference-defense.

Faithful reproduction of Chen et al., ICLR 2022,
"RelaxLoss: Defending Membership Inference Attacks without Losing Utility".

Official reference implementation:
  https://github.com/DingfanChen/RelaxLoss
  (source/cifar/defense/relaxloss.py Trainer.train, lines ~95-130)

Algorithm (per batch):
  Let L_ce = mean cross-entropy over the batch, l_ce_i = per-sample CE.
  Even epochs (gradient ascent towards target level alpha):
      loss = | L_ce - alpha |
  Odd epochs:
      if L_ce > alpha:
          loss = L_ce                      (standard CE descent)
      else:                                (posterior flattening)
          # model's own softmax probability on the true class, clamped to upper
          p_t    = clamp(softmax(logits)[y], max=upper)
          p_else = (1 - p_t) / (K - 1)
          soft_targets = onehot * p_t + (1 - onehot) * p_else   # detached
          loss_i = (1 - correct_i) * CE_soft(logits, soft_targets)_i - l_ce_i
          loss   = mean(loss_i)

Hyperparameters from the released configs that inform this task:
  CIFAR10  resnet20: alpha=1.0,  CIFAR10  vgg11_bn: alpha=1.0
  CIFAR100 resnet20: alpha=3.0,  CIFAR100 vgg11_bn: alpha=0.5
  upper = 1.0 in all configs.

This benchmark selects alpha per dataset using logits.size(1):
  num_classes == 10  -> alpha = 1.0   (CIFAR-10 / FashionMNIST)
  num_classes == 100 -> alpha = 0.5   (CIFAR-100, matches vgg11_bn config)
"""

_FILE = "pytorch-vision/custom_membership_defense.py"

_CONTENT = """\
class MembershipDefense:
    \"\"\"RelaxLoss training rule (Chen et al., ICLR 2022).

    Two-phase alternation per epoch:
      Even epochs: loss = |mean_CE - alpha|   (drives loss toward target level)
      Odd  epochs: if mean_CE > alpha -> CE descent
                   else -> posterior flattening with sign-flipped CE
    See github.com/DingfanChen/RelaxLoss/blob/main/source/cifar/defense/relaxloss.py.
    \"\"\"

    def __init__(self):
        # alpha is the target loss level; chosen per num_classes at call time.
        # upper=1 matches every released config; no clamp effect in practice
        # but kept for faithfulness to the official code.
        self.upper = 1.0

    def compute_loss(self, logits, labels, epoch):
        num_classes = logits.size(1)
        # Released configs use dataset/model-specific alpha values; this task
        # selects alpha by class count for the exposed benchmark cases.
        alpha = 0.5 if num_classes == 100 else 1.0

        loss_ce_full = F.cross_entropy(logits, labels, reduction='none')
        loss_ce = loss_ce_full.mean()

        if epoch % 2 == 0:
            # Gradient ascent / descent toward target level alpha.
            return (loss_ce - alpha).abs()

        # Odd epoch.
        if loss_ce.item() > alpha:
            return loss_ce

        # Posterior flattening.
        probs = torch.softmax(logits, dim=1)
        confidence_target = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        confidence_target = torch.clamp(confidence_target, min=0.0, max=self.upper)
        confidence_else = (1.0 - confidence_target) / (num_classes - 1)

        onehot = F.one_hot(labels, num_classes=num_classes).float()
        soft_targets = (
            onehot * confidence_target.unsqueeze(1)
            + (1.0 - onehot) * confidence_else.unsqueeze(1)
        )
        # Detach targets so the flattening gradient flows only through logits
        # (matches the official implementation: soft_targets is built from a
        # forward pass and used as a target).
        soft_targets = soft_targets.detach()

        log_probs = F.log_softmax(logits, dim=1)
        ce_soft = -(soft_targets * log_probs).sum(dim=1)

        pred = logits.argmax(dim=1)
        correct = pred.eq(labels).float()

        # For correctly-classified samples: -loss_ce_full (gradient ascent on CE).
        # For incorrectly-classified samples: ce_soft - loss_ce_full
        # (pull toward flattened posterior while still descending on CE).
        loss = (1.0 - correct) * ce_soft - loss_ce_full
        return loss.mean()
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 9, "end_line": 29, "content": _CONTENT}
]
