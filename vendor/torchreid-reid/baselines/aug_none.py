"""Weak augmentation baseline: NO augmentation (identical to the eval transform).
Over-fits the small train set. Reference: vendor/torchreid-reid/baselines/aug_none.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_train_transform(img_h, img_w, mean, std):
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize((img_h, img_w)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    tf.name = "no_aug"
    return tf
