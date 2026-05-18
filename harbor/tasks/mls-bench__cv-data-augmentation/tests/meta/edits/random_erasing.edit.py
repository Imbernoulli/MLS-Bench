"""Random Erasing data augmentation baseline.

Randomly selects a rectangle region in an image and erases its pixels with
random values. Applied after ToTensor, probability p=0.5, area ratio 0.02-0.33.

Reference: Zhong et al., "Random Erasing Data Augmentation" (AAAI 2020)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_augment.py"

_CONTENT = """\
def build_train_transform(config):
    \"\"\"Random Erasing augmentation (Zhong et al., AAAI 2020).

    Pipeline: RandomCrop + HFlip + ToTensor + RandomErasing(p=0.5) + Normalize.
    \"\"\"
    return transforms.Compose([
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
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
