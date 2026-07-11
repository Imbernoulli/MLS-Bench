"""Medium augmentation baseline: random horizontal flip + pad-crop (no erasing).
The classic mild geometric augmentation. Reference: vendor/torchreid-reid/baselines/aug_flipcrop.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_train_transform(img_h, img_w, mean, std):
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize((img_h, img_w)),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((img_h, img_w)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    tf.name = "flip_crop"
    return tf
