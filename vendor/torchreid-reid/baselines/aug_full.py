"""SOTA augmentation baseline: flip + pad-crop + RANDOM ERASING (Zhong 2020; Luo 2019).
The canonical strong re-ID augmentation. Reference: vendor/torchreid-reid/baselines/aug_full.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_train_transform(img_h, img_w, mean, std):
    import torchvision.transforms as T
    from torchreid.data.transforms import RandomErasing

    tf = T.Compose([
        T.Resize((img_h, img_w)),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((img_h, img_w)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        RandomErasing(probability=0.5, mean=list(mean)),
    ])
    tf.name = "flip_crop_erase"
    return tf
