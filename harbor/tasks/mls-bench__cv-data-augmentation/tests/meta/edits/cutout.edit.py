"""Cutout data augmentation baseline.

Randomly masks out square regions of the input image after converting to tensor,
acting as a regularizer that encourages the network to use broader context.

Reference: DeVries & Taylor, "Improved Regularization of Convolutional Neural
Networks with Cutout" (2017)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_augment.py"

_CONTENT = """\
def build_train_transform(config):
    \"\"\"Cutout augmentation: random square mask after ToTensor.

    Pipeline: RandomCrop + HFlip + ToTensor + Cutout(1, 16) + Normalize.
    \"\"\"
    class Cutout:
        def __init__(self, n_holes=1, length=16):
            self.n_holes = n_holes
            self.length = length

        def __call__(self, img):
            h, w = img.size(1), img.size(2)
            mask = torch.ones_like(img)
            for _ in range(self.n_holes):
                y = torch.randint(0, h, (1,)).item()
                x = torch.randint(0, w, (1,)).item()
                y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
                x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
                mask[:, y1:y2, x1:x2] = 0
            return img * mask

    return transforms.Compose([
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
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
