"""Retain-only finetuning baseline for security-machine-unlearning."""

_FILE = "pytorch-vision/bench/unlearning/custom_unlearning.py"

_CONTENT = """\
class UnlearningMethod:
    \"\"\"Continue training on retained data only.\"\"\"

    def __init__(self):
        pass

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        retain_x, retain_y = retain_batch
        logits = model(retain_x)
        loss = F.cross_entropy(logits, retain_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 8, "end_line": 22, "content": _CONTENT}
]
