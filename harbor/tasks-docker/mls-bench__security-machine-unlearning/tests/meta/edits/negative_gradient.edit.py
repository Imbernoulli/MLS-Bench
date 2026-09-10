"""Negative-gradient baseline for security-machine-unlearning."""

_FILE = "pytorch-vision/bench/unlearning/custom_unlearning.py"

_CONTENT = """\
class UnlearningMethod:
    \"\"\"Descend retain loss while ascending forget loss.\"\"\"

    def __init__(self):
        self.forget_weight = 0.5

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        retain_x, retain_y = retain_batch
        forget_x, forget_y = forget_batch
        retain_loss = F.cross_entropy(model(retain_x), retain_y)
        forget_loss = F.cross_entropy(model(forget_x), forget_y)
        loss = retain_loss - self.forget_weight * forget_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "retain_loss": retain_loss.item(), "forget_loss": forget_loss.item()}
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 8, "end_line": 22, "content": _CONTENT}
]
