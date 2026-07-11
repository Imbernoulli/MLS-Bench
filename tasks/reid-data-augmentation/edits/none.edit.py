"""Weak augmentation baseline: NO augmentation (identical to the eval transform).
Over-fits the small train set. Reference: vendor/torchreid-reid/baselines/aug_none.py
"""
_FILE = "torchreid-reid/solution/augment.py"
_CONTENT = '''def build_train_transform(img_h, img_w, mean, std):
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize((img_h, img_w)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    tf.name = "no_aug"
    return tf'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 21, "content": _CONTENT},
]
