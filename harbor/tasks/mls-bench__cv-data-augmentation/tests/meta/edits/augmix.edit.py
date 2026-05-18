"""AugMix data augmentation baseline.

Mixes multiple augmented views of an image using random convex combinations,
following AugMix-style transform construction. This task-level baseline
does not add the Jensen-Shannon consistency loss used in the full AugMix
training objective.

Reference: Hendrycks et al., "AugMix: A Simple Data Processing Method to
Improve Robustness and Uncertainty" (ICLR 2020)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_augment.py"

_CONTENT = """\
def build_train_transform(config):
    \"\"\"Transform-only AugMix-style augmentation.

    Mixes multiple augmentation chains with Dirichlet-weighted convex
    combinations plus a skip connection to the clean image.
    Pipeline: AugMix(severity=3, width=3, depth=2) + standard crop/flip.
    \"\"\"
    import random
    from PIL import ImageOps, ImageEnhance

    # Individual augmentation operations (no geometric transforms -- those
    # are handled by the fixed RandomCrop + HFlip).
    def autocontrast(img, _):
        return ImageOps.autocontrast(img)

    def equalize(img, _):
        return ImageOps.equalize(img)

    def posterize(img, v):
        v = max(1, int(v))
        return ImageOps.posterize(img, v)

    def solarize(img, v):
        return ImageOps.solarize(img, int(v))

    def color(img, v):
        return ImageEnhance.Color(img).enhance(v)

    def contrast(img, v):
        return ImageEnhance.Contrast(img).enhance(v)

    def brightness(img, v):
        return ImageEnhance.Brightness(img).enhance(v)

    def sharpness(img, v):
        return ImageEnhance.Sharpness(img).enhance(v)

    aug_list = [
        (autocontrast, 0, 1),
        (equalize, 0, 1),
        (posterize, 4, 8),
        (solarize, 0, 256),
        (color, 0.1, 1.9),
        (contrast, 0.1, 1.9),
        (brightness, 0.1, 1.9),
        (sharpness, 0.1, 1.9),
    ]

    class AugMixTransform:
        def __init__(self, severity=3, width=3, depth=2, alpha=1.0):
            self.severity = severity
            self.width = width
            self.depth = depth
            self.alpha = alpha

        def __call__(self, img):
            import numpy as np
            ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
            m = np.float32(np.random.beta(self.alpha, self.alpha))

            img_np = np.array(img).astype(np.float32)
            mix = np.zeros_like(img_np)

            for i in range(self.width):
                img_aug = img.copy()
                d = self.depth if self.depth > 0 else random.randint(1, 3)
                for _ in range(d):
                    op, lo, hi = random.choice(aug_list)
                    val = lo + (hi - lo) * random.random()
                    img_aug = op(img_aug, val)
                mix += ws[i] * np.array(img_aug).astype(np.float32)

            mixed = m * img_np + (1 - m) * mix
            mixed = np.clip(mixed, 0, 255).astype(np.uint8)
            from PIL import Image
            return Image.fromarray(mixed)

    return transforms.Compose([
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        AugMixTransform(severity=3, width=3, depth=2),
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
