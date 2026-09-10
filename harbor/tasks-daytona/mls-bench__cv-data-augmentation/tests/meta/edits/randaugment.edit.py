"""RandAugment data augmentation baseline.

Applies a sequence of randomly selected augmentation operations with uniform
magnitude, avoiding the expensive search phase of AutoAugment.

Reference: Cubuk et al., "RandAugment: Practical automated data augmentation
with a reduced search space" (CVPR 2020)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_augment.py"

_CONTENT = """\
def build_train_transform(config):
    \"\"\"RandAugment augmentation: automated policy before geometric transforms.

    Pipeline: RandAugment(2, 9) + RandomCrop + HFlip + ToTensor + Normalize.
    \"\"\"
    return transforms.Compose([
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 246,
        "end_line": 275,
        "content": _CONTENT,
    },
]
