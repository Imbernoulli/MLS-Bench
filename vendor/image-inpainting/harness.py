#!/usr/bin/env python3
"""Fail-closed full-resolution image-inpainting verifier.

The verifier trains exactly one selected solution surface. A selected surface is
never replaced after a load, contract, numerical, training, or evaluation failure.
The data, optimization budget, fixed components, masks, and proof format are part
of the pinned protocol below.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

PROTOCOL_ID = "places365-val256-fullres-v1"
SCHEMA_VERSION = 1
TORCH_VERSION = "2.4.1"
IMAGE_SIZE = 256
BASE_CH = 64
TRAIN_COUNT = 32_000
VAL_COUNT = 4_500
TRAIN_STEPS = 100_000
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
PROGRESS_EVERY = 10_000
SPLIT_SEED = 20_260_705
MASK_SEED = 999
OPTIMIZER_LR = 1.0e-4
MAX_PARAMETERS = 120_000_000

PLACES_URL = "https://data.csail.mit.edu/places/places365/val_256.tar"
PLACES_TAR_BYTES = 525_158_400
PLACES_TAR_MD5 = "e27b17d8d44f4af9a78502beb927f808"
PLACES_TAR_SHA256 = "24b4e639ef12a0012af525bc4cb443e4ab4aaea8369a1fb009b70e4a4aad5d48"

SETTINGS = ("small", "large", "strokes")
MASK_RANGES = {
    "small": (0.06, 0.12),
    "large": (0.22, 0.38),
    "strokes": (0.14, 0.28),
}
SETTING_OFFSETS = {"small": 101, "large": 202, "strokes": 303}

SURFACE_ATTRS = {
    "activation": "make_activation",
    "arch": "build_net",
    "attention": "build_bottleneck",
    "dilation": "build_dilation",
    "fusion": "fuse",
    "gate": "apply_gate",
    "loss": "compute_loss",
    "masking": "make_holes",
    "norm": "make_norm",
    "upsample": "build_upsample",
}


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validate_protocol_manifest(root: Path) -> tuple[dict[str, object], bytes]:
    path = root / "protocol_manifest.json"
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or raw != _canonical_json_bytes(manifest):
        raise RuntimeError("protocol manifest is missing or not canonical JSON")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_ID,
        "dataset": "Places365-Standard",
        "source_split": "val_256",
        "source_url": PLACES_URL,
        "archive_bytes": PLACES_TAR_BYTES,
        "archive_md5": PLACES_TAR_MD5,
        "archive_sha256": PLACES_TAR_SHA256,
        "archive_image_count": TRAIN_COUNT + VAL_COUNT,
        "split_seed": SPLIT_SEED,
        "mask_seed": MASK_SEED,
        "image_size": IMAGE_SIZE,
        "train_count": TRAIN_COUNT,
        "val_count": VAL_COUNT,
        "train_steps": TRAIN_STEPS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
    }
    if set(manifest) != set(fixed) | {
        "train_manifest_sha256", "val_manifest_sha256"
    }:
        raise RuntimeError("protocol manifest has an unexpected schema")
    for key, expected in fixed.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"protocol manifest mismatch for {key}")
    for key in ("train_manifest_sha256", "val_manifest_sha256"):
        if not _is_sha256(manifest.get(key)):
            raise RuntimeError(f"protocol manifest has an invalid {key}")
    return manifest, raw


def _validate_split(
    root: Path,
    split: str,
    expected_count: int,
    expected_manifest_sha256: str,
) -> tuple[list[dict[str, object]], bytes]:
    manifest_path = root / split / "manifest.json"
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise RuntimeError(f"{split} manifest digest mismatch")
    items = json.loads(raw)
    if raw != _canonical_json_bytes(items):
        raise RuntimeError(f"{split} manifest is not canonical JSON")
    if not isinstance(items, list) or len(items) != expected_count:
        raise RuntimeError(f"{split} manifest count mismatch")

    seen_sources: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != {
            "index", "path", "source", "bytes", "sha256"
        }:
            raise RuntimeError(f"malformed {split} row {index}")
        if item["index"] != index or not isinstance(item["source"], str):
            raise RuntimeError(f"out-of-order {split} row {index}")
        if item["source"] in seen_sources:
            raise RuntimeError(f"duplicate {split} source {item['source']!r}")
        seen_sources.add(item["source"])
        if not isinstance(item["bytes"], int) or item["bytes"] <= 0:
            raise RuntimeError(f"invalid byte count in {split} row {index}")
        if not _is_sha256(item["sha256"]):
            raise RuntimeError(f"invalid digest in {split} row {index}")
        relpath = PurePosixPath(item["path"])
        if (
            relpath.is_absolute()
            or ".." in relpath.parts
            or len(relpath.parts) != 3
            or relpath.parts[:2] != (split, "images")
        ):
            raise RuntimeError(f"unsafe path in {split} row {index}")
        path = root.joinpath(*relpath.parts)
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"missing or truncated file in {split} row {index}")
        if _sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"file digest mismatch in {split} row {index}")
    return items, raw


def load_and_validate_data(
    root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    protocol, protocol_raw = _validate_protocol_manifest(root)
    train, train_raw = _validate_split(
        root, "train", TRAIN_COUNT, protocol["train_manifest_sha256"]
    )
    val, val_raw = _validate_split(
        root, "val", VAL_COUNT, protocol["val_manifest_sha256"]
    )
    if {item["source"] for item in train} & {item["source"] for item in val}:
        raise RuntimeError("train and validation sources overlap")
    digest = hashlib.sha256(protocol_raw + train_raw + val_raw).hexdigest()
    return train, val, digest


class PlacesDataset(Dataset):
    def __init__(self, root: Path, items: list[dict[str, object]]) -> None:
        self.root = root
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        item = self.items[index]
        path = self.root.joinpath(*PurePosixPath(item["path"]).parts)
        with Image.open(path) as image:
            image.load()
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"decoded image {index} has wrong dimensions")
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise RuntimeError(f"decoded image {index} has wrong shape")
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)
        return tensor, index, item["sha256"]


def _square_mask(rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    for _ in range(400):
        target = float(rng.uniform(lo, hi))
        side = int(round(math.sqrt(target) * IMAGE_SIZE))
        side = min(max(side, 8), IMAGE_SIZE - 2)
        top = int(rng.integers(0, IMAGE_SIZE - side + 1))
        left = int(rng.integers(0, IMAGE_SIZE - side + 1))
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        mask[top:top + side, left:left + side] = 1.0
        if lo <= float(mask.mean()) <= hi:
            return mask
    raise RuntimeError("could not construct square mask within the pinned range")


def _stroke_mask(rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    for _ in range(400):
        canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
        draw = ImageDraw.Draw(canvas)
        for _stroke in range(int(rng.integers(3, 9))):
            x = int(rng.integers(0, IMAGE_SIZE))
            y = int(rng.integers(0, IMAGE_SIZE))
            points = [(x, y)]
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            for _vertex in range(int(rng.integers(4, 13))):
                angle += float(rng.uniform(-1.2, 1.2))
                length = int(rng.integers(16, 65))
                x = min(max(int(round(x + math.cos(angle) * length)), 0), IMAGE_SIZE - 1)
                y = min(max(int(round(y + math.sin(angle) * length)), 0), IMAGE_SIZE - 1)
                points.append((x, y))
            draw.line(points, fill=255, width=int(rng.integers(8, 25)), joint="curve")
        mask = (np.asarray(canvas, dtype=np.uint8) > 0).astype(np.float32)
        if lo <= float(mask.mean()) <= hi:
            return mask
    raise RuntimeError("could not construct stroke mask within the pinned range")


def _mask_for_setting(rng: np.random.Generator, setting: str) -> np.ndarray:
    lo, hi = MASK_RANGES[setting]
    if setting == "strokes":
        return _stroke_mask(rng, lo, hi)
    return _square_mask(rng, lo, hi)


def validation_mask(index: int, setting: str) -> torch.Tensor:
    seed = MASK_SEED + SETTING_OFFSETS[setting] + index * 1_000_003
    rng = np.random.default_rng(seed)
    return torch.from_numpy(_mask_for_setting(rng, setting)).unsqueeze(0)


def default_make_holes(gt: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    masks = []
    for _ in range(gt.shape[0]):
        setting = SETTINGS[int(rng.integers(0, len(SETTINGS)))]
        masks.append(_mask_for_setting(rng, setting))
    return torch.from_numpy(np.stack(masks)).unsqueeze(1).to(gt.device)


def default_compute_loss(out: torch.Tensor, gt: torch.Tensor, hole: torch.Tensor) -> torch.Tensor:
    valid = 1.0 - hole
    channels = out.shape[1]
    hole_l1 = (torch.abs(out - gt) * hole).sum() / (hole.sum() * channels + 1.0e-8)
    valid_l1 = (torch.abs(out - gt) * valid).sum() / (valid.sum() * channels + 1.0e-8)
    dy_out = out[:, :, 1:] - out[:, :, :-1]
    dy_gt = gt[:, :, 1:] - gt[:, :, :-1]
    dx_out = out[:, :, :, 1:] - out[:, :, :, :-1]
    dx_gt = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    gradient = torch.mean(torch.abs(dy_out - dy_gt)) + torch.mean(torch.abs(dx_out - dx_gt))
    return 6.0 * hole_l1 + valid_l1 + 0.1 * gradient


def load_surface(solution_path: Path, attr: str):
    if not solution_path.is_file():
        raise RuntimeError(f"solution file does not exist: {solution_path}")
    spec = importlib.util.spec_from_file_location("selected_inpaint_surface", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create solution import spec")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(solution_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    value = getattr(module, attr)
    if not callable(value):
        raise TypeError(f"{attr} must be callable")
    return value


class Identity(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class FixedDilatedStack(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
            for dilation in (1, 2, 4, 8)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = F.elu(block(value), inplace=True)
        return value


def _group_norm(channels: int) -> nn.Module:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConfigUNet(nn.Module):
    """Full-resolution, four-level encoder-decoder used outside the arch task."""

    def __init__(self, hooks: dict[str, object] | None = None) -> None:
        super().__init__()
        hooks = hooks or {}
        c = BASE_CH
        bottleneck_channels = 8 * c
        self.activation = hooks.get("activation", nn.ELU(inplace=True))
        if not isinstance(self.activation, nn.Module):
            raise TypeError("activation hook must be an nn.Module")

        self.e1 = nn.Conv2d(4, c, 5, 1, 2)
        self.e2 = nn.Conv2d(c, 2 * c, 4, 2, 1)
        self.e3 = nn.Conv2d(2 * c, 4 * c, 4, 2, 1)
        self.e4 = nn.Conv2d(4 * c, bottleneck_channels, 4, 2, 1)
        self.mid = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, 1, 1)
        self.d3 = nn.Conv2d(bottleneck_channels + 4 * c, 4 * c, 3, 1, 1)
        self.d2 = nn.Conv2d(4 * c + 2 * c, 2 * c, 3, 1, 1)
        self.d1 = nn.Conv2d(2 * c + c, c, 3, 1, 1)
        self.out = nn.Conv2d(c, 3, 3, 1, 1)

        norm_factory = hooks.get("norm")
        self.norms = nn.ModuleList()
        for channels in (c, 2 * c, 4 * c, bottleneck_channels, bottleneck_channels,
                         4 * c, 2 * c, c):
            norm = _group_norm(channels) if norm_factory is None else norm_factory(channels)
            if not isinstance(norm, nn.Module):
                raise TypeError("make_norm must return nn.Module for every channel count")
            self.norms.append(norm)

        attention = hooks.get("attention", Identity())
        dilation = hooks.get("dilation", FixedDilatedStack(bottleneck_channels))
        if not isinstance(attention, nn.Module) or not isinstance(dilation, nn.Module):
            raise TypeError("bottleneck hooks must be nn.Module instances")
        self.attention = attention
        self.dilation = dilation

        upsample_factory = hooks.get("upsample")
        self.up3 = self._make_upsample(upsample_factory, bottleneck_channels)
        self.up2 = self._make_upsample(upsample_factory, 4 * c)
        self.up1 = self._make_upsample(upsample_factory, 2 * c)
        self.fuse = hooks.get("fusion")
        self.gate = hooks.get("gate")
        self.g1 = nn.Conv2d(c, c, 1)
        self.g2 = nn.Conv2d(2 * c, 2 * c, 1)
        self.g3 = nn.Conv2d(4 * c, 4 * c, 1)
        self.g4 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 1)

    @staticmethod
    def _make_upsample(factory, channels: int) -> nn.Module:
        module = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if factory is None
            else factory(channels)
        )
        if not isinstance(module, nn.Module):
            raise TypeError("build_upsample must return nn.Module")
        return module

    def _apply_gate(
        self,
        feature: torch.Tensor,
        logits: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        learned = torch.sigmoid(logits)
        output = feature * learned if self.gate is None else self.gate(feature, learned, valid)
        if not torch.is_tensor(output) or output.shape != feature.shape:
            raise TypeError("apply_gate must return a tensor with the feature shape")
        return output

    def _fuse(self, decoded: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        output = (
            torch.cat([decoded, skip], dim=1)
            if self.fuse is None
            else self.fuse(decoded, skip)
        )
        expected = (decoded.shape[0], decoded.shape[1] + skip.shape[1],
                    decoded.shape[2], decoded.shape[3])
        if not torch.is_tensor(output) or tuple(output.shape) != expected:
            raise TypeError(f"fuse must return shape {expected}")
        return output

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        valid = value[:, 3:4]
        e1 = self.activation(self.norms[0](self.e1(value)))
        e1 = self._apply_gate(e1, self.g1(e1), F.interpolate(valid, size=e1.shape[-2:]))
        e2 = self.activation(self.norms[1](self.e2(e1)))
        e2 = self._apply_gate(e2, self.g2(e2), F.interpolate(valid, size=e2.shape[-2:]))
        e3 = self.activation(self.norms[2](self.e3(e2)))
        e3 = self._apply_gate(e3, self.g3(e3), F.interpolate(valid, size=e3.shape[-2:]))
        e4 = self.activation(self.norms[3](self.e4(e3)))
        e4 = self._apply_gate(e4, self.g4(e4), F.interpolate(valid, size=e4.shape[-2:]))
        middle = self.activation(self.norms[4](self.mid(e4)))
        middle = self.dilation(self.attention(middle))
        d3 = self.activation(self.norms[5](self.d3(self._fuse(self.up3(middle), e3))))
        d2 = self.activation(self.norms[6](self.d2(self._fuse(self.up2(d3), e2))))
        d1 = self.activation(self.norms[7](self.d1(self._fuse(self.up1(d2), e1))))
        return torch.sigmoid(self.out(d1))


def _probe_module(module: nn.Module, channels: int) -> None:
    module.eval()
    with torch.no_grad():
        probe = module(torch.zeros(1, channels, 32, 32))
    if not torch.is_tensor(probe) or probe.shape != (1, channels, 32, 32):
        raise TypeError("bottleneck module changed the pinned tensor shape")
    if not torch.isfinite(probe).all():
        raise ValueError("bottleneck module probe is non-finite")


def build_selected_pipeline(surface: str, solution_path: Path):
    selected = load_surface(solution_path, SURFACE_ATTRS[surface])
    loss_fn = default_compute_loss
    holes_fn = default_make_holes

    if surface == "arch":
        model = selected(4)
        if not isinstance(model, nn.Module):
            raise TypeError("build_net must return nn.Module")
    elif surface == "loss":
        loss_fn = selected
        model = ConfigUNet()
    elif surface == "masking":
        holes_fn = selected
        model = ConfigUNet()
    else:
        hooks: dict[str, object] = {}
        if surface == "activation":
            module = selected()
            if not isinstance(module, nn.Module):
                raise TypeError("make_activation must return nn.Module")
            hooks["activation"] = module
        elif surface == "attention":
            module = selected(8 * BASE_CH)
            if not isinstance(module, nn.Module):
                raise TypeError("build_bottleneck must return nn.Module")
            _probe_module(module, 8 * BASE_CH)
            hooks["attention"] = module
        elif surface == "dilation":
            module = selected(8 * BASE_CH)
            if not isinstance(module, nn.Module):
                raise TypeError("build_dilation must return nn.Module")
            _probe_module(module, 8 * BASE_CH)
            hooks["dilation"] = module
        elif surface == "fusion":
            hooks["fusion"] = selected
        elif surface == "gate":
            hooks["gate"] = selected
        elif surface == "norm":
            hooks["norm"] = selected
        elif surface == "upsample":
            hooks["upsample"] = selected
        model = ConfigUNet(hooks)

    if not isinstance(model, nn.Module):
        raise TypeError("selected pipeline did not produce an nn.Module")
    model.eval()
    with torch.no_grad():
        probe = model(torch.zeros(1, 4, IMAGE_SIZE, IMAGE_SIZE))
    if not torch.is_tensor(probe) or probe.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise TypeError("model probe returned the wrong shape")
    if not torch.isfinite(probe).all() or torch.any(probe < 0) or torch.any(probe > 1):
        raise ValueError("model probe must be finite and within [0, 1]")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable <= 0 or parameters > MAX_PARAMETERS:
        raise ValueError(
            f"model parameter contract failed: trainable={trainable} total={parameters}"
        )
    return model, loss_fn, holes_fn, parameters


def _validate_holes(holes: object, gt: torch.Tensor) -> torch.Tensor:
    expected = (gt.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE)
    if not torch.is_tensor(holes) or tuple(holes.shape) != expected:
        raise TypeError(f"make_holes must return tensor shape {expected}")
    if holes.device != gt.device or not holes.dtype.is_floating_point:
        raise TypeError("make_holes must preserve device and return floating-point data")
    if not torch.isfinite(holes).all() or torch.any(holes < 0) or torch.any(holes > 1):
        raise ValueError("make_holes returned values outside [0, 1]")
    fractions = holes.mean(dim=(1, 2, 3))
    if torch.any(fractions < 0.05) or torch.any(fractions > 0.45):
        raise ValueError("every training mask must cover between 5% and 45% of the image")
    return holes


def _validate_output(output: object, gt: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(output) or output.shape != gt.shape or output.device != gt.device:
        raise TypeError("network output must match the target tensor shape and device")
    if not output.dtype.is_floating_point or not torch.isfinite(output).all():
        raise ValueError("network output must be finite floating-point data")
    if torch.any(output < 0) or torch.any(output > 1):
        raise ValueError("network output must remain within [0, 1]")
    return output


def _validate_loss(loss: object) -> torch.Tensor:
    if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss):
        raise TypeError("compute_loss must return one finite scalar tensor")
    if loss.item() < 0 or not loss.requires_grad:
        raise ValueError("compute_loss must be non-negative and retain a gradient path")
    return loss


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=workers > 0,
        generator=generator,
    )


def train_and_evaluate(
    surface: str,
    solution_path: Path,
    data_root: Path,
    seed: int,
) -> None:
    if seed != 42:
        raise ValueError("the pinned protocol requires seed 42")
    if torch.__version__.split("+")[0] != TORCH_VERSION:
        raise RuntimeError(f"expected torch {TORCH_VERSION}, got {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("the full-resolution protocol requires a CUDA worker")
    set_all_seeds(seed)

    train_items, val_items, data_digest = load_and_validate_data(data_root)
    solution_digest = _sha256_file(solution_path)
    model, loss_fn, holes_fn, parameters = build_selected_pipeline(surface, solution_path)
    device = torch.device("cuda")
    model = model.to(device).train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=OPTIMIZER_LR, betas=(0.5, 0.999)
    )
    train_loader = _loader(
        PlacesDataset(data_root, train_items), TRAIN_BATCH_SIZE, True, workers=4
    )
    train_iterator = iter(train_loader)
    mask_rng = np.random.default_rng(seed + MASK_SEED)

    print(
        "INPAINT_PROTOCOL "
        f"schema={SCHEMA_VERSION} protocol={PROTOCOL_ID} "
        f"dataset_sha256={PLACES_TAR_SHA256} image_size={IMAGE_SIZE} "
        f"train_count={TRAIN_COUNT} val_count={VAL_COUNT} "
        f"train_steps={TRAIN_STEPS} batch_size={TRAIN_BATCH_SIZE} seed={seed} "
        f"surface={surface} settings=small,large,strokes "
        f"data_manifest_sha256={data_digest} "
        f"solution_sha256={solution_digest} parameters={parameters}",
        flush=True,
    )

    loss_window: list[float] = []
    for step in range(1, TRAIN_STEPS + 1):
        try:
            gt, _indices, _digests = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            gt, _indices, _digests = next(train_iterator)
        gt = gt.to(device, non_blocking=True)
        holes = _validate_holes(holes_fn(gt, mask_rng), gt)
        valid = 1.0 - holes
        inputs = torch.cat([gt * valid, valid], dim=1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = _validate_output(model(inputs), gt)
            loss = _validate_loss(loss_fn(output, gt, holes))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=10.0, error_if_nonfinite=True
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError("training produced a non-finite gradient norm")
        optimizer.step()
        loss_window.append(float(loss.detach()))
        if step % PROGRESS_EVERY == 0:
            mean_loss = math.fsum(loss_window) / len(loss_window)
            if not math.isfinite(mean_loss):
                raise RuntimeError("training progress loss is non-finite")
            print(
                f"INPAINT_PROGRESS step={step} train_loss={mean_loss:.9f}",
                flush=True,
            )
            loss_window.clear()

    model.eval()
    val_loader = _loader(
        PlacesDataset(data_root, val_items), EVAL_BATCH_SIZE, False, workers=4
    )
    for setting in SETTINGS:
        print(f"INPAINT_SETTING setting={setting} count={VAL_COUNT}", flush=True)
        item_rows: list[tuple[int, int, float, float, float]] = []
        emitted = 0
        with torch.no_grad():
            for gt, indices, source_digests in val_loader:
                gt = gt.to(device, non_blocking=True)
                masks = torch.stack(
                    [validation_mask(int(index), setting) for index in indices], dim=0
                ).to(device, non_blocking=True)
                valid = 1.0 - masks
                inputs = torch.cat([gt * valid, valid], dim=1)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = _validate_output(model(inputs), gt)
                completed = masks * output.float() + valid * gt
                difference = torch.abs(completed - gt)
                squared = (completed - gt).square()
                hole_pixels = masks.sum(dim=(1, 2, 3)).to(torch.int64).cpu().tolist()
                hole_abs = (difference * masks).sum(dim=(1, 2, 3)).double().cpu().tolist()
                hole_sq = (squared * masks).sum(dim=(1, 2, 3)).double().cpu().tolist()
                full_abs = difference.sum(dim=(1, 2, 3)).double().cpu().tolist()
                for batch_index, source_digest in enumerate(source_digests):
                    index = int(indices[batch_index])
                    if index != emitted:
                        raise RuntimeError("validation loader produced an out-of-order item")
                    values = (
                        int(hole_pixels[batch_index]),
                        float(f"{hole_abs[batch_index]:.9f}"),
                        float(f"{hole_sq[batch_index]:.9f}"),
                        float(f"{full_abs[batch_index]:.9f}"),
                    )
                    item_rows.append((index, *values))
                    print(
                        f"INPAINT_ITEM setting={setting} index={index} "
                        f"source_sha256={source_digest} hole_pixels={values[0]} "
                        f"hole_abs_sum={values[1]:.9f} hole_sq_sum={values[2]:.9f} "
                        f"full_abs_sum={values[3]:.9f}",
                        flush=True,
                    )
                    emitted += 1

        if emitted != VAL_COUNT:
            raise RuntimeError(f"evaluation emitted {emitted} items; expected {VAL_COUNT}")
        total_hole_pixels = sum(row[1] for row in item_rows)
        total_hole_abs = math.fsum(row[2] for row in item_rows)
        total_hole_sq = math.fsum(row[3] for row in item_rows)
        total_full_abs = math.fsum(row[4] for row in item_rows)
        denominator = total_hole_pixels * 3
        hole_l1 = total_hole_abs / denominator
        hole_mse = total_hole_sq / denominator
        hole_psnr = 99.0 if hole_mse < 1.0e-12 else 10.0 * math.log10(1.0 / hole_mse)
        full_l1 = total_full_abs / (VAL_COUNT * 3 * IMAGE_SIZE * IMAGE_SIZE)
        hole_frac = total_hole_pixels / (VAL_COUNT * IMAGE_SIZE * IMAGE_SIZE)
        metrics = (hole_l1, hole_psnr, full_l1, hole_frac)
        if not all(math.isfinite(value) for value in metrics):
            raise RuntimeError("terminal metrics are non-finite")
        print(
            f"INPAINT_METRICS surface={surface} setting={setting} count={VAL_COUNT} "
            f"hole_l1={hole_l1:.9f} hole_psnr={hole_psnr:.9f} "
            f"full_l1={full_l1:.9f} hole_frac={hole_frac:.9f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--solution", required=True, type=Path)
    parser.add_argument("--surface", required=True, choices=sorted(SURFACE_ATTRS))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    train_and_evaluate(
        args.surface,
        args.solution,
        args.data_root,
        args.seed,
    )


if __name__ == "__main__":
    main()
