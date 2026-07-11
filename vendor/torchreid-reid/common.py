"""Fixed person re-identification evaluation harness.

The selected solution surface is loaded through its public contract. Active-surface
load, runtime, shape, type, completeness, or numerical failures invalidate the run.
Other pipeline components remain fixed for a given task.
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import os
import random
import re
import sys
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Fixed paths / determinism
# --------------------------------------------------------------------------- #
EXPECTED_TRAIN_IMAGES = 12_936
EXPECTED_TRAIN_IDS = 751
EXPECTED_QUERY_IMAGES = 3_368
EXPECTED_QUERY_IDS = 750
EXPECTED_GALLERY_IMAGES = 19_732
EXPECTED_TRAIN_MANIFEST_SHA256 = (
    "4f1a5416bad595a67a45652568919252e56a54e99c49fd74f1fd29492123f3d3"
)
EXPECTED_QUERY_MANIFEST_SHA256 = (
    "d34ff6d094521111a10a16f7879f01bb210abdeab24efba4b950fe1f3b9e90f7"
)
EXPECTED_GALLERY_MANIFEST_SHA256 = (
    "7900c8355955f1ca7e2ad5d6844f4be03dddfc3ded1f7a21cf43e55441075c4e"
)
EXPECTED_WEIGHTS_SHA256 = (
    "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
)
REID_PROTOCOL_ID = "market1501-resnet50-60e-v2"
EXPECTED_TOTAL_STEPS = 11_003
EXPECTED_TRAIN_SAMPLES = 704_192
# The measured full P x K run performed 11,003 updates. Pending siblings use a
# deterministic 60-epoch schedule with the same total budget so task-specific
# samplers cannot silently change the amount of optimization work.
EXPECTED_EPOCH_STEPS = (184,) * 23 + (183,) * 37
EXPECTED_TASK_IDS = frozenset({
    "reid-backbone-finetune",
    "reid-batch-mining",
    "reid-data-augmentation",
    "reid-embedding-dim",
    "reid-embedding-head",
    "reid-lr-schedule",
    "reid-metric-loss",
    "reid-optimizer",
    "reid-reranking",
    "reid-spatial-pooling",
})


def data_root() -> Path:
    return Path(os.environ.get("REID_DATA", "/data/torchreid/market1501_full"))


def eval_data_root() -> Path:
    raw = os.environ.get("REID_EVAL_DATA")
    if not raw:
        raise RuntimeError("REID_EVAL_DATA is required for verifier evaluation")
    root = Path(raw)
    if not root.is_absolute():
        raise RuntimeError("REID_EVAL_DATA must be an absolute verifier path")
    return root


def weights_path() -> str:
    """ImageNet-pretrained ResNet-50 weights staged offline."""
    return os.environ.get(
        "REID_WEIGHTS", "/data/torchreid/weights/resnet50_imagenet.pth"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_manifest(split_dir: Path) -> tuple[int, str]:
    """Hash every official JPEG and its filename in a stable manifest format."""
    if not split_dir.is_dir():
        raise FileNotFoundError(f"required Market-1501 split is missing: {split_dir}")
    images = sorted(split_dir.glob("*.jpg"), key=lambda path: path.name)
    manifest = hashlib.sha256()
    for image in images:
        manifest.update(f"{_file_sha256(image)}  ./{image.name}\n".encode("ascii"))
    return len(images), manifest.hexdigest()


def verify_fullscale_inventory() -> dict[str, str | int]:
    """Authenticate the complete official Market-1501 inventory and checkpoint."""
    train_n, train_sha = _image_manifest(data_root() / "train")
    query_n, query_sha = _image_manifest(eval_data_root() / "query")
    gallery_n, gallery_sha = _image_manifest(eval_data_root() / "gallery")
    expected = {
        "train": (EXPECTED_TRAIN_IMAGES, EXPECTED_TRAIN_MANIFEST_SHA256),
        "query": (EXPECTED_QUERY_IMAGES, EXPECTED_QUERY_MANIFEST_SHA256),
        "gallery": (EXPECTED_GALLERY_IMAGES, EXPECTED_GALLERY_MANIFEST_SHA256),
    }
    observed = {
        "train": (train_n, train_sha),
        "query": (query_n, query_sha),
        "gallery": (gallery_n, gallery_sha),
    }
    for name, value in observed.items():
        if value != expected[name]:
            raise RuntimeError(
                f"Market-1501 {name} inventory mismatch: observed={value} "
                f"expected={expected[name]}"
            )

    checkpoint = Path(weights_path())
    if not checkpoint.is_file():
        raise FileNotFoundError(f"required ResNet-50 checkpoint is missing: {checkpoint}")
    weights_sha = _file_sha256(checkpoint)
    if weights_sha != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError(
            f"ResNet-50 checkpoint digest mismatch: {weights_sha}"
        )
    return {
        "train_n": train_n,
        "query_n": query_n,
        "gallery_n": gallery_n,
        "train_sha": train_sha,
        "query_sha": query_sha,
        "gallery_sha": gallery_sha,
        "weights_sha": weights_sha,
    }


def set_seeds(seed: int = 42) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_surface(sol_path: str, attr: str):
    """Import the agent-editable function/callable `attr` from solution/<file>.py."""
    p = Path(sol_path)
    try:
        spec = importlib.util.spec_from_file_location("agent_surface", str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import solution from {p}")
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(p.parent))
        spec.loader.exec_module(mod)  # type: ignore
        if not hasattr(mod, attr):
            raise AttributeError(f"solution must define `{attr}(...)`")
        surface = getattr(mod, attr)
        if not callable(surface):
            raise TypeError(f"solution `{attr}` must be callable")
        return surface
    except Exception as exc:
        print(f"REID_SURFACE_FALLBACK name={attr} reason={exc!r}", flush=True)
        raise RuntimeError(f"failed to load re-ID surface {attr}") from exc


def finite_tensor(value, label: str, *, ndim: int | None = None,
                  batch_size: int | None = None):
    import torch

    if not torch.is_tensor(value):
        print(f"REID_SURFACE_FALLBACK name={label} reason=not_tensor", flush=True)
        raise TypeError(f"{label} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        print(f"REID_SURFACE_FALLBACK name={label} reason=rank_{value.ndim}", flush=True)
        raise ValueError(f"{label} must have rank {ndim}; got shape={tuple(value.shape)}")
    if batch_size is not None and (value.ndim == 0 or value.shape[0] != batch_size):
        print(f"REID_SURFACE_FALLBACK name={label} reason=batch_shape", flush=True)
        raise ValueError(f"{label} has wrong batch shape={tuple(value.shape)}")
    if not torch.isfinite(value).all():
        print(f"REID_NONFINITE stage={label}", flush=True)
        raise RuntimeError(f"{label} contains non-finite values")
    return value


def validate_train_outputs(logits, features, loss, batch_size: int) -> None:
    finite_tensor(logits, "train_logits", ndim=2, batch_size=batch_size)
    finite_tensor(features, "train_features", ndim=2, batch_size=batch_size)
    finite_tensor(loss, "train_loss")
    if loss.numel() != 1:
        print("REID_SURFACE_FALLBACK name=train_loss reason=non_scalar", flush=True)
        raise ValueError(f"training loss must be scalar; got shape={tuple(loss.shape)}")
    if not loss.requires_grad:
        print("REID_SURFACE_FALLBACK name=train_loss reason=no_grad", flush=True)
        raise ValueError("training loss must retain a gradient path")


def backward_and_step(loss, optimizer) -> None:
    import torch

    params = [p for group in optimizer.param_groups for p in group.get("params", [])]
    if not params:
        print("REID_SURFACE_FALLBACK name=optimizer reason=no_parameters", flush=True)
        raise ValueError("optimizer has no parameters")
    optimizer.zero_grad()
    loss.backward()
    if not any(p.grad is not None for p in params):
        print("REID_SURFACE_FALLBACK name=gradient reason=missing", flush=True)
        raise RuntimeError("training produced no optimizer gradients")
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
        print("REID_NONFINITE stage=gradient", flush=True)
        raise RuntimeError("training produced non-finite gradients")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in params):
        print("REID_NONFINITE stage=parameter", flush=True)
        raise RuntimeError("optimizer produced non-finite parameters")


def finite_positive(value, label: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        print(f"REID_SURFACE_FALLBACK name={label} reason=not_scalar", flush=True)
        raise TypeError(f"{label} must be a scalar") from exc
    if not np.isfinite(scalar):
        print(f"REID_NONFINITE stage={label}", flush=True)
        raise ValueError(f"{label} must be finite")
    if scalar <= 0:
        print(f"REID_SURFACE_FALLBACK name={label} reason=non_positive", flush=True)
        raise ValueError(f"{label} must be positive")
    return scalar


# --------------------------------------------------------------------------- #
# Fixed image geometry / normalisation (Market-1501 standard 256x128)
# --------------------------------------------------------------------------- #
IMG_H, IMG_W = 256, 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_transform():
    """FIXED test-time transform: resize -> tensor -> imagenet normalise (no aug)."""
    import torchvision.transforms as T

    return T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def base_train_transform():
    """Community-standard resize/crop/flip transform for Market-1501 training."""
    import torchvision.transforms as T

    return T.Compose([
        T.Resize((288, 144)),
        T.RandomCrop((IMG_H, IMG_W)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Fixed complete Market-1501 dataset, filename-parsed pid/camid
# --------------------------------------------------------------------------- #
_PID_PATTERN = re.compile(r"([-\d]+)_c(\d+)")


def _scan_split(split_dir: Path, *, include_junk: bool = False):
    """Return list of (img_path, pid, camid) for every .jpg in split_dir."""
    items = []
    for img in sorted(glob.glob(str(split_dir / "*.jpg"))):
        m = _PID_PATTERN.search(os.path.basename(img))
        if m is None:
            continue
        pid, camid = int(m.group(1)), int(m.group(2))
        if pid < 0 and not include_junk:
            continue
        items.append((img, pid, camid))
    return items


class ReidImageDataset:
    """A tiny torch Dataset over (path, relabeled_pid, camid). Applies `transform`.

    `pid2label` remaps the arbitrary train pids to a contiguous [0, num_ids) range
    so they can index the classifier. For query/gallery we keep original pids
    (labels are only used for the CMC/mAP matching, not for a classifier).
    """

    def __init__(self, items, transform, pid2label=None):
        from PIL import Image  # noqa: F401 (import check)

        self.items = items
        self.transform = transform
        self.pid2label = pid2label

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from PIL import Image
        import torch

        path, pid, camid = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        if not torch.is_tensor(img) or tuple(img.shape) != (3, IMG_H, IMG_W):
            print("REID_SURFACE_FALLBACK name=image_transform reason=bad_shape", flush=True)
            raise ValueError(f"transform must return [3,{IMG_H},{IMG_W}] tensor")
        if not torch.isfinite(img).all():
            print("REID_NONFINITE stage=image_transform", flush=True)
            raise RuntimeError("image transform returned non-finite values")
        label = self.pid2label[pid] if self.pid2label is not None else pid
        return img, label, camid


def load_train_items():
    """Complete official train split plus contiguous classifier labels."""
    items = _scan_split(data_root() / "train")
    if not items:
        raise SystemExit(f"no train images under {data_root()/'train'}")
    pids = sorted({pid for _, pid, _ in items})
    if len(items) != EXPECTED_TRAIN_IMAGES or len(pids) != EXPECTED_TRAIN_IDS:
        raise RuntimeError(
            f"unexpected training inventory: images={len(items)} ids={len(pids)}"
        )
    pid2label = {pid: i for i, pid in enumerate(pids)}
    return items, pid2label, len(pids)


def load_query_gallery():
    """Complete official query and gallery, including gallery junk distractors."""
    root = eval_data_root()
    q = _scan_split(root / "query")
    g = _scan_split(root / "gallery", include_junk=True)
    if not q or not g:
        raise SystemExit("empty query/gallery split")
    query_ids = {pid for _, pid, _ in q}
    if (len(q), len(query_ids), len(g)) != (
        EXPECTED_QUERY_IMAGES,
        EXPECTED_QUERY_IDS,
        EXPECTED_GALLERY_IMAGES,
    ):
        raise RuntimeError(
            f"unexpected evaluation inventory: query={len(q)} "
            f"query_ids={len(query_ids)} gallery={len(g)}"
        )
    return q, g


# --------------------------------------------------------------------------- #
# Fixed retrieval settings. Every official query is assigned to one difficulty
# bucket by its number of valid cross-camera positives; every bucket is matched
# against the complete official gallery.
# --------------------------------------------------------------------------- #
SETTINGS = ("easy", "medium", "hard")
EXPECTED_QUERY_COUNTS = {"easy": 1122, "medium": 1123, "hard": 1123}


def _split_queries_by_difficulty(q_items, g_items):
    """Partition the complete query inventory by valid-positive count."""
    gallery_by_pid: dict[int, list[int]] = {}
    for _path, pid, camid in g_items:
        if pid >= 0:
            gallery_by_pid.setdefault(pid, []).append(camid)

    ranked = []
    for item in q_items:
        path, pid, camid = item
        n_positive = sum(other_cam != camid for other_cam in gallery_by_pid.get(pid, []))
        if n_positive <= 0:
            raise RuntimeError(f"query has no cross-camera positive: {path}")
        ranked.append((n_positive, path, item))
    ranked.sort(key=lambda row: (row[0], row[1]))

    hard_n = EXPECTED_QUERY_COUNTS["hard"]
    medium_n = EXPECTED_QUERY_COUNTS["medium"]
    groups = {
        "hard": [row[2] for row in ranked[:hard_n]],
        "medium": [row[2] for row in ranked[hard_n: hard_n + medium_n]],
        "easy": [row[2] for row in ranked[hard_n + medium_n:]],
    }
    if {name: len(items) for name, items in groups.items()} != EXPECTED_QUERY_COUNTS:
        raise RuntimeError("query difficulty partition is incomplete")
    return groups


# --------------------------------------------------------------------------- #
# Fixed backbone (ResNet-50 via torchreid, ImageNet-pretrained offline)
# --------------------------------------------------------------------------- #
def build_backbone(num_train_ids: int, loss: str = "triplet"):
    """FIXED backbone: torchreid ResNet-50. `loss='triplet'` makes forward() return
    (logits, features) in train mode and features in eval mode; `loss='softmax'`
    returns logits in train mode. We always build with loss='triplet' so both the
    id-classifier head and the feature embedding are available, and each task's
    surface decides which of {logits, features} to use.

    ImageNet weights are loaded offline from $REID_WEIGHTS.
    """
    import torch
    import torchreid

    model = torchreid.models.build_model(
        name="resnet50",
        num_classes=num_train_ids,
        loss=loss,
        pretrained=False,   # we load weights offline below
        use_gpu=torch.cuda.is_available(),
    )
    _load_imagenet_backbone(model)
    return model


def _load_imagenet_backbone(model):
    """Load torchvision ResNet-50 ImageNet weights into torchreid ResNet-50
    (skips the fc/classifier head, which has the wrong shape)."""
    import torch

    wp = weights_path()
    if not os.path.isfile(wp):
        print(f"REID_WEIGHTS_FALLBACK missing={wp}", flush=True)
        raise FileNotFoundError(f"required ImageNet weights not found: {wp}")
    sd = torch.load(wp, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict) or not sd:
        print("REID_WEIGHTS_FALLBACK reason=invalid_checkpoint", flush=True)
        raise TypeError("ImageNet checkpoint must contain a non-empty state dict")
    model_sd = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if not isinstance(k, str) or not torch.is_tensor(v):
            print("REID_WEIGHTS_FALLBACK reason=invalid_state_entry", flush=True)
            raise TypeError("ImageNet state dict entries must be string/tensor pairs")
        if not torch.isfinite(v).all():
            print(f"REID_NONFINITE stage=checkpoint key={k}", flush=True)
            raise ValueError(f"checkpoint tensor {k} is non-finite")
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
    if loaded < 100:
        print(f"REID_WEIGHTS_FALLBACK reason=insufficient_layers loaded={loaded}", flush=True)
        raise RuntimeError(f"checkpoint matched only {loaded} backbone layers")
    model.load_state_dict(model_sd, strict=True)
    print(f"REID_INIT imagenet_layers_loaded={loaded}", flush=True)


# --------------------------------------------------------------------------- #
# Embedding-head model (for the reid-embedding-head task). The head is the ONLY
# agent-editable piece; backbone pooling, classifier attachment, and the forward
# contract (train -> (logits, head_feat); eval -> head_feat) are fixed here.
# --------------------------------------------------------------------------- #
def head_out_dim(head, feat_dim: int, device: str) -> int:
    """Probe the head's output dimensionality with a dummy input."""
    import torch

    import torch.nn as nn

    if not isinstance(head, nn.Module):
        print("REID_SURFACE_FALLBACK name=head reason=not_module", flush=True)
        raise TypeError("build_head must return nn.Module")
    head.eval()
    with torch.no_grad():
        out = head(torch.zeros(2, feat_dim, device=device))
    finite_tensor(out, "head_probe", ndim=2, batch_size=2)
    if out.shape[1] <= 0:
        print("REID_SURFACE_FALLBACK name=head reason=empty_output", flush=True)
        raise ValueError("embedding head output dimension must be positive")
    return int(out.shape[1])


class HeadModel:
    """Wrap a fixed backbone with an agent-supplied embedding head.

    forward(x): train -> (logits, head_feat); eval -> head_feat.
    The classifier is attached to the head output (fixed). Pooling uses the
    backbone's own featuremaps + global average pool (fixed).
    """

    def __init__(self, backbone, head, out_dim: int, num_ids: int):
        import torch.nn as nn

        # Compose as a proper nn.Module so .to()/.train()/.parameters() work.
        class _M(nn.Module):
            def __init__(m):
                super().__init__()
                m.backbone = backbone
                m.head = head
                m.classifier = nn.Linear(out_dim, num_ids)

            def _pool(m, x):
                f = m.backbone.featuremaps(x)
                v = m.backbone.global_avgpool(f)
                return v.view(v.size(0), -1)

            def forward(m, x):
                v = m._pool(x)
                feat = m.head(v)
                if m.training:
                    return m.classifier(feat), feat
                return feat

        self._module = _M()

    # Delegate the nn.Module surface the harness/eval uses.
    def to(self, *a, **k):
        self._module = self._module.to(*a, **k)
        return self

    def train(self, mode=True):
        self._module.train(mode)
        return self

    def eval(self):
        self._module.eval()
        return self

    def parameters(self):
        return self._module.parameters()

    def state_dict(self, *a, **k):
        return self._module.state_dict(*a, **k)

    def __call__(self, x):
        return self._module(x)


# --------------------------------------------------------------------------- #
# Pooling model (for the reid-pooling task). The spatial-pooling module that
# collapses the backbone's [B, C, H, W] featuremap to a [B, C] vector is the ONLY
# agent-editable piece; the backbone, a fixed BNNeck, and the classifier are
# frozen. forward(x): train -> (logits, bn_feat); eval -> bn_feat.
# --------------------------------------------------------------------------- #
class PoolModel:
    """Wrap a fixed backbone with an agent-supplied spatial pooling module + a
    fixed BNNeck. The pooling module maps [B, C, H, W] -> [B, C]."""

    def __init__(self, backbone, pool, feat_dim: int, num_ids: int):
        import torch.nn as nn

        class _M(nn.Module):
            def __init__(m):
                super().__init__()
                m.backbone = backbone
                m.pool = pool
                m.bnneck = nn.BatchNorm1d(feat_dim)
                m.bnneck.bias.requires_grad_(False)
                m.classifier = nn.Linear(feat_dim, num_ids, bias=False)

            def forward(m, x):
                fmap = m.backbone.featuremaps(x)          # [B, C, H, W]
                v = m.pool(fmap)                           # [B, C, 1, 1] or [B, C]
                v = v.view(v.size(0), -1)
                bn = m.bnneck(v)
                if m.training:
                    return m.classifier(bn), v            # id on bn, triplet on v
                return bn

        self._module = _M()

    def to(self, *a, **k):
        self._module = self._module.to(*a, **k)
        return self

    def train(self, mode=True):
        self._module.train(mode)
        return self

    def eval(self):
        self._module.eval()
        return self

    def parameters(self):
        return self._module.parameters()

    def state_dict(self, *a, **k):
        return self._module.state_dict(*a, **k)

    def __call__(self, x):
        return self._module(x)


# --------------------------------------------------------------------------- #
# Fixed optimiser / budget
# --------------------------------------------------------------------------- #
def build_optimizer(model, lr: float = 3e-4, weight_decay: float = 5e-4):
    import torch

    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _validate_task_id(task_id: str) -> str:
    if task_id not in EXPECTED_TASK_IDS:
        raise ValueError(f"unknown re-ID task identity: {task_id!r}")
    return task_id


def emit_fullscale_protocol(*, task_id: str, seed: int, epochs: int,
                            batch_size: int,
                            num_instances: int = 4) -> dict[str, str | int]:
    """Validate the immutable budget/assets and emit its unique proof line."""
    if (seed, epochs, batch_size, num_instances) != (42, 60, 64, 4):
        raise ValueError("fixed protocol requires seed=42 epochs=60 batch=64 instances=4")
    task_id = _validate_task_id(task_id)
    inventory = verify_fullscale_inventory()
    print(
        f"REID_PROTOCOL schema=2 task={task_id} protocol={REID_PROTOCOL_ID} "
        "seed=42 model=resnet50 epochs=60 batch=64 instances=4 "
        f"train_images={inventory['train_n']} query_images={inventory['query_n']} "
        f"gallery_images={inventory['gallery_n']} train_ids={EXPECTED_TRAIN_IDS} "
        f"query_ids={EXPECTED_QUERY_IDS} train_sha={inventory['train_sha']} "
        f"query_sha={inventory['query_sha']} gallery_sha={inventory['gallery_sha']} "
        f"weights_sha={inventory['weights_sha']}",
        flush=True,
    )
    return inventory


def fullscale_loader(dataset, sampler, *, batch_size: int = 64):
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    if len(loader) < 150:
        raise RuntimeError(f"full training loader is unexpectedly short: {len(loader)}")
    return loader


def default_epoch_scheduler(optimizer):
    import torch

    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 50], gamma=0.1
    )


def run_fullscale_training(model, loader, optimizer, loss_step, *, epochs: int = 60,
                           epoch_scheduler=None, lr_at_step=None) -> tuple[int, int]:
    """Execute every batch of all 60 epochs and emit auditable counters."""
    import torch

    if epochs != 60:
        raise ValueError("full-scale training requires exactly 60 epochs")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("full-scale re-ID verification requires one CUDA GPU")

    total_steps = 0
    train_samples = 0
    if len(EXPECTED_EPOCH_STEPS) != epochs or sum(EXPECTED_EPOCH_STEPS) != EXPECTED_TOTAL_STEPS:
        raise RuntimeError("invalid fixed epoch-step schedule")
    for epoch, expected_epoch_steps in enumerate(EXPECTED_EPOCH_STEPS):
        model.train()
        running = 0.0
        epoch_steps = 0
        iterator = iter(loader)
        while epoch_steps < expected_epoch_steps:
            try:
                imgs, labels, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                try:
                    imgs, labels, _ = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError("full training loader produced no batches") from exc
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if lr_at_step is not None:
                lr = finite_positive(lr_at_step(total_steps), "learning_rate")
                for group in optimizer.param_groups:
                    group["lr"] = lr

            logits, features, loss = loss_step(imgs, labels)
            validate_train_outputs(logits, features, loss, imgs.shape[0])
            backward_and_step(loss, optimizer)
            running += float(loss.item())
            epoch_steps += 1
            total_steps += 1
            train_samples += int(imgs.shape[0])

        if epoch_steps != expected_epoch_steps:
            raise RuntimeError(
                f"epoch {epoch} was incomplete: {epoch_steps}/{expected_epoch_steps} steps"
            )
        if epoch_scheduler is not None:
            epoch_scheduler.step()
        lr = finite_positive(optimizer.param_groups[0]["lr"], "learning_rate")
        print(
            f"REID_EPOCH epoch={epoch} steps={epoch_steps} total_steps={total_steps} "
            f"loss={running / epoch_steps:.6f} lr={lr:.8g}",
            flush=True,
        )

    if (total_steps, train_samples) != (
        EXPECTED_TOTAL_STEPS,
        EXPECTED_TRAIN_SAMPLES,
    ):
        print(
            f"REID_PROTOCOL_ERROR stage=train_budget total_steps={total_steps} "
            f"train_samples={train_samples}",
            flush=True,
        )
        raise RuntimeError(
            "training budget mismatch: "
            f"observed={(total_steps, train_samples)} "
            f"expected={(EXPECTED_TOTAL_STEPS, EXPECTED_TRAIN_SAMPLES)}"
        )
    print(
        f"REID_TRAIN_COMPLETE epochs={epochs} total_steps={total_steps} "
        f"train_samples={train_samples}",
        flush=True,
    )
    return total_steps, train_samples


def finish_fullscale_evaluation(model, started_at: float, *, task_id: str,
                                total_steps: int, train_samples: int,
                                rerank_fn=None) -> dict:
    import time

    task_id = _validate_task_id(task_id)
    if (total_steps, train_samples) != (
        EXPECTED_TOTAL_STEPS,
        EXPECTED_TRAIN_SAMPLES,
    ):
        raise RuntimeError("evaluation cannot follow an incomplete training budget")
    if rerank_fn is None:
        results = evaluate_reid_multi(model)
    else:
        results = evaluate_reid_multi_rerank(model, rerank_fn=rerank_fn)
    elapsed = time.perf_counter() - started_at
    emit_multi_metrics(results, elapsed)
    print(
        f"REID_EVAL_COMPLETE schema=2 task={task_id} protocol={REID_PROTOCOL_ID} "
        "settings=easy,medium,hard query_total=3368 gallery=19732 "
        f"total_steps={total_steps} train_samples={train_samples} status=ok",
        flush=True,
    )
    return results


# --------------------------------------------------------------------------- #
# Fixed feature extraction + re-ID scoring (mAP + CMC). NOT agent-editable.
# --------------------------------------------------------------------------- #
def extract_features(model, items, batch_size: int = 64):
    """Extract L2-NORMALISED features for `items`. Uses the FIXED eval transform.
    Returns (feats [N, D] float tensor on cpu, pids [N], camids [N])."""
    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not items:
        raise ValueError("feature extraction received no items")
    ds = ReidImageDataset(items, eval_transform(), pid2label=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    feats, pids, camids = [], [], []
    with torch.no_grad():
        for imgs, labels, cams in loader:
            imgs = imgs.to(device)
            out = model(imgs)                 # eval mode -> features (v)
            finite_tensor(out, "eval_features_raw", ndim=2, batch_size=imgs.shape[0])
            norms = torch.linalg.vector_norm(out, dim=1)
            finite_tensor(norms, "eval_feature_norms", ndim=1, batch_size=imgs.shape[0])
            if torch.any(norms <= 0):
                print("REID_SURFACE_FALLBACK name=eval_features reason=zero_norm", flush=True)
                raise ValueError("evaluation features must have positive norm")
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            finite_tensor(out, "eval_features_normalized", ndim=2, batch_size=imgs.shape[0])
            feats.append(out.cpu())
            pids.append(labels)
            camids.append(cams)
    feats = torch.cat(feats, 0)
    pids = torch.cat(pids, 0).numpy()
    camids = torch.cat(camids, 0).numpy()
    if feats.shape[0] != len(items) or len(pids) != len(items) or len(camids) != len(items):
        raise RuntimeError(f"incomplete feature extraction: {feats.shape[0]}/{len(items)}")
    return feats, pids, camids


def evaluate_reid(model):
    """Single-setting compatibility wrapper for the full fixed gallery."""
    return evaluate_reid_multi(model)["hard"]


def evaluate_reid_multi(model):
    """Score the same trained model under all fixed gallery settings."""
    return evaluate_reid_multi_rerank(model, rerank_fn=None)


def evaluate_reid_multi_rerank(model, rerank_fn=None):
    """Like evaluate_reid_multi, but optionally applies an agent-supplied
    `rerank_fn(distmat, qf, gf) -> distmat` to the query-gallery distance matrix
    of EACH difficulty setting before CMC/mAP. `qf`/`gf` are L2-normalised
    [Nq, D] / [Ng, D] numpy arrays; `distmat` is the [Nq, Ng] cosine distance.
    """
    import torch

    q_items, g_items = load_query_gallery()
    qf_all, q_pids_all, q_cams_all = extract_features(model, q_items)
    gf, g_pids, g_cams = extract_features(model, g_items)

    query_groups = _split_queries_by_difficulty(q_items, g_items)
    path_to_query_row = {item[0]: index for index, item in enumerate(q_items)}
    results = {}
    for name in SETTINGS:
        sub = query_groups[name]
        rows = [path_to_query_row[item[0]] for item in sub]
        qf = qf_all[rows]
        q_pids = q_pids_all[rows]
        q_cams = q_cams_all[rows]
        distmat = 1.0 - torch.mm(qf, gf.t()).numpy()
        expected_shape = (len(sub), len(g_items))
        if distmat.shape != expected_shape or not np.all(np.isfinite(distmat)):
            print(f"REID_NONFINITE stage=distance setting={name}", flush=True)
            raise RuntimeError(f"invalid distance matrix shape/values for {name}")
        if rerank_fn is not None:
            distmat = rerank_fn(distmat, qf.numpy(), gf.numpy())
            if not isinstance(distmat, np.ndarray) or distmat.shape != expected_shape:
                print(f"REID_SURFACE_FALLBACK name=rerank reason=bad_shape setting={name}", flush=True)
                raise TypeError(f"rerank must return numpy array with shape {expected_shape}")
            if not np.all(np.isfinite(distmat)):
                print(f"REID_NONFINITE stage=rerank setting={name}", flush=True)
                raise RuntimeError("rerank returned non-finite distances")
        res = _eval_market(distmat, q_pids, g_pids, q_cams, g_cams)
        if res["num_query"] != EXPECTED_QUERY_COUNTS[name]:
            raise RuntimeError(
                f"incomplete {name} query evaluation: {res['num_query']}"
            )
        res["num_gallery"] = int(len(g_items))
        results[name] = res
    return results


def _eval_market(distmat, q_pids, g_pids, q_cams, g_cams, max_rank: int = 50):
    """Deterministic mAP + CMC (standard cross-camera re-ID protocol)."""
    if not isinstance(distmat, np.ndarray) or distmat.ndim != 2:
        raise TypeError("distmat must be a rank-2 numpy array")
    num_q, num_g = distmat.shape
    if num_q <= 0 or num_g <= 0:
        raise ValueError("distmat must contain queries and gallery items")
    if any(len(x) != expected for x, expected in (
            (q_pids, num_q), (q_cams, num_q), (g_pids, num_g), (g_cams, num_g))):
        raise ValueError("distance matrix and identity metadata lengths disagree")
    if not np.all(np.isfinite(distmat)):
        print("REID_NONFINITE stage=market_distance", flush=True)
        raise ValueError("distmat contains non-finite values")
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)               # ascending distance
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_ap, num_valid_q = [], [], 0.0
    for qi in range(num_q):
        order = indices[qi]
        # remove gallery samples with same pid AND same camid as the query
        remove = (g_pids[order] == q_pids[qi]) & (g_cams[order] == q_cams[qi])
        keep = np.invert(remove)

        raw_cmc = matches[qi][keep]
        if not np.any(raw_cmc):
            continue  # this query has no ground-truth match in gallery -> skip
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        all_ap.append(tmp_cmc.sum() / num_rel)

    if num_valid_q == 0:
        raise SystemExit("no valid query (no gallery match after camera filtering)")

    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    mAP = float(np.mean(all_ap))
    result = {
        "mAP": mAP,
        "rank1": float(all_cmc[0]),
        "rank5": float(all_cmc[4]) if len(all_cmc) >= 5 else float(all_cmc[-1]),
        "num_query": int(num_valid_q),
    }
    if not all(np.isfinite(result[k]) and 0.0 <= result[k] <= 1.0
               for k in ("mAP", "rank1", "rank5")):
        print("REID_NONFINITE stage=market_metrics", flush=True)
        raise RuntimeError(f"invalid re-ID metrics {result}")
    return result


# --------------------------------------------------------------------------- #
# Metric emission (one REID_METRICS line per difficulty setting). The parser
# reads `map_<setting>` / `rank1_<setting>` from these lines.
# --------------------------------------------------------------------------- #
def emit_multi_metrics(results: dict, elapsed: float) -> None:
    """Print one metrics line per fixed setting."""
    if not isinstance(results, dict) or set(results) != set(SETTINGS):
        raise ValueError(f"results must contain exactly {SETTINGS}")
    elapsed = float(elapsed)
    if not np.isfinite(elapsed) or elapsed <= 0:
        print("REID_NONFINITE stage=elapsed", flush=True)
        raise ValueError("elapsed time must be finite and non-negative")
    for name in SETTINGS:
        r = results[name]
        if not isinstance(r, dict):
            raise TypeError(f"result for {name} must be a dict")
        for key in ("mAP", "rank1", "rank5"):
            value = float(r[key])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                print(f"REID_NONFINITE stage=emit_metrics setting={name}", flush=True)
                raise ValueError(f"invalid {key}={value} for {name}")
        if int(r.get("num_gallery", 0)) <= 0 or int(r.get("num_query", 0)) <= 0:
            raise ValueError(f"incomplete evaluation counts for {name}")
        print(
            f"REID_METRICS setting={name} map={r['mAP']:.6f} "
            f"rank1={r['rank1']:.6f} rank5={r['rank5']:.6f} "
            f"num_query={r.get('num_query', 0)} "
            f"num_gallery={r.get('num_gallery', 0)} elapsed={elapsed:.1f}",
            flush=True,
        )
