"""TrivialAugmentWide data augmentation baseline.

Applies a single randomly selected augmentation with random magnitude per image,
providing strong regularization with zero hyperparameter tuning.

Reference: Mueller & Hutter, "TrivialAugment: Tuning-free Yet State-of-the-Art
Data Augmentation" (ICCV 2021)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_augment.py"

_CONTENT = """\
def build_train_transform(config):
    \"\"\"TrivialAugmentWide: single random op with random magnitude.

    Pipeline: TrivialAugmentWide() + RandomCrop + HFlip + ToTensor + Normalize.
    \"\"\"
    return transforms.Compose([
        transforms.TrivialAugmentWide(),
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
