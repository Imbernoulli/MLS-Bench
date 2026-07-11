"""Agent-editable solution surface for person re-identification.

Keep the public callable names and signatures below. Follow the fixed harness
contract for return types, shapes, devices, value ranges, finiteness, and
determinism. The selected implementation is evaluated directly; contract or
runtime failures invalidate the run.
"""
from __future__ import annotations


# EDITABLE REGION
def build_train_transform(img_h, img_w, mean, std):
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize((img_h, img_w)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    tf.name = "no_aug"
    return tf
# END EDITABLE REGION

