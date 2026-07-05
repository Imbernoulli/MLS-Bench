"""Good baseline for cv-count-patch: RANDOM-CROP-and-RESIZE patch augmentation.

Takes a random 3/4-size crop of each training image, keeps the points inside it (with
adjusted coords), then RESIZES the crop back to the full image size (rescaling the point
coordinates accordingly). This is the field-standard crowd-counting augmentation
(MCNN/CSRNet train on random sub-crops): it multiplies effective training data and
exposes the counter to denser, zoomed-in sub-scenes, while keeping the train image size
== the val image size (so there is no train/test resolution mismatch) -> better
generalisation to the shifted val counts -> lower counting MAE.

The GT count of a crop is the number of annotated points inside it, so the
density-integral supervision stays exact per augmented sample.
"""


def crop_train(img, pts):
    import numpy as np

    def resize_bilinear(im, H, W):
        _, h, w = im.shape
        ys = (np.arange(H) + 0.5) * h / H - 0.5
        xs = (np.arange(W) + 0.5) * w / W - 0.5
        y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
        x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        wy = np.clip(ys - y0, 0, 1)[:, None]
        wx = np.clip(xs - x0, 0, 1)[None, :]
        o = np.empty((im.shape[0], H, W), dtype=np.float32)
        for c in range(im.shape[0]):
            cc = im[c]
            top = cc[y0][:, x0] * (1 - wx) + cc[y0][:, x1] * wx
            bot = cc[y1][:, x0] * (1 - wx) + cc[y1][:, x1] * wx
            o[c] = top * (1 - wy) + bot * wy
        return o

    img = np.asarray(img, dtype=np.float32)          # (3,H,W)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    _, H, W = img.shape
    ch = int(H * 0.75); cw = int(W * 0.75)
    if ch < 8 or cw < 8:
        return img, pts
    y0 = np.random.randint(0, H - ch + 1)
    x0 = np.random.randint(0, W - cw + 1)
    crop = img[:, y0:y0 + ch, x0:x0 + cw]
    if len(pts):
        inside = ((pts[:, 0] >= y0) & (pts[:, 0] < y0 + ch) &
                  (pts[:, 1] >= x0) & (pts[:, 1] < x0 + cw))
        kept = pts[inside].copy()
        kept[:, 0] -= y0
        kept[:, 1] -= x0
    else:
        kept = pts
    out = resize_bilinear(crop, H, W)
    if len(kept):
        kept[:, 0] *= H / ch
        kept[:, 1] *= W / cw
    return out, kept
