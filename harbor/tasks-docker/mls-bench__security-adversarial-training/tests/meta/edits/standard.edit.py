"""Standard (vanilla) training baseline — no adversarial examples.

This baseline trains on clean images only, providing a reference for
clean accuracy but zero robustness against adversarial attacks.
"""

_FILE = "torchattacks/bench/custom_adv_train.py"

_CONTENT = """\
class AdversarialTrainer:
    \"\"\"Standard training (no adversarial examples).\"\"\"

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes

    def train_step(self, images, labels, optimizer):
        self.model.train()
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
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
