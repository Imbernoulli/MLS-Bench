"""Temperature-scaled CE baseline inspired by logit adjustment.

True logit adjustment subtracts log(class_prior) before CE loss. On balanced
CIFAR with a uniform prior this is only a constant shift, so this baseline
uses a small temperature calibration instead.

Reference: Menon et al., "Long-tail learning via logit adjustment"
(ICLR 2021)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_loss.py"

_CONTENT = """\
def compute_loss(logits, targets, config):
    \"\"\"Temperature-scaled CE, motivated by logit adjustment.

    Menon et al.'s logit adjustment is class-frequency dependent. With a
    uniform class prior on balanced CIFAR, it reduces to CE shifted by a
    constant, so this benchmark baseline uses logits / tau as a mild
    calibration-only variant.
    \"\"\"
    C = config['num_classes']
    tau = 1.05
    # Uniform prior adjustment: log(1/C) is constant, so instead we apply
    # a temperature that slightly smooths the logit distribution
    adjusted = logits / tau
    return F.cross_entropy(adjusted, targets)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 246,
        "end_line": 266,
        "content": _CONTENT,
    },
]
