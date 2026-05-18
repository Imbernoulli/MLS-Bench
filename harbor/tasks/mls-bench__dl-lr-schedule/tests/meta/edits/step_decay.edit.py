"""Step decay learning rate schedule baseline.

Divides the learning rate by 10 at 50% and 75% of total training epochs.
Classic schedule used in the original ResNet paper.

Reference: He et al., "Deep Residual Learning for Image Recognition"
(CVPR 2016)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_schedule.py"

_CONTENT = """\
def get_lr(epoch, total_epochs, base_lr, config):
    \"\"\"Step decay: divide LR by 10 at 50% and 75% of training.

    Milestones at epochs total_epochs*0.5 and total_epochs*0.75.
    \"\"\"
    if epoch >= int(0.75 * total_epochs):
        return base_lr * 0.01
    elif epoch >= int(0.5 * total_epochs):
        return base_lr * 0.1
    return base_lr
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 246,
        "end_line": 269,
        "content": _CONTENT,
    },
]
