#!/usr/bin/env python3
"""Full Adobe Composition-1K harness for the ten cv-matting-* siblings.

One model is trained for 100,000 optimizer steps on all 43,100 canonical training
composites, then evaluated on all 1,000 canonical test composites under three
trimap widths. Editable hooks fail closed; no hook error is replaced by a default
implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


PROTOCOL = "composition1k-full-v1"
SOURCE_REVISION = "adobe-composition-1k-licensed-full-v1"
# Must match the licensed source manifest pinned in prepare_data.py. Leaving this
# empty makes verification fail closed until the operator backfills the real digest.
EXPECTED_SOURCE_MANIFEST_SHA256 = ""
TRAIN_COUNT = 43_100
TEST_COUNT = 1_000
TRAIN_CROP = 320
TRAIN_ITERS = 100_000
TRAIN_TRIMAP_WIDTH = 6
TRIMAP_WIDTHS = {"medium": 6, "wide": 9, "xwide": 12}
SETTINGS = tuple(TRIMAP_WIDTHS)
BASE = 32
TASK_SURFACES = {
    "cv-matting-arch": "arch",
    "cv-matting-attention": "attention",
    "cv-matting-decoder-design": "decoder",
    "cv-matting-dilation": "dilation",
    "cv-matting-loss-design": "loss",
    "cv-matting-norm": "norm",
    "cv-matting-refinement": "refine",
    "cv-matting-skip": "skip",
    "cv-matting-trimap-encoding": "trimap",
    "cv-matting-upsampling": "upsampling",
}
SURFACES = set(TASK_SURFACES.values())
SHA256 = re.compile(r"[0-9a-f]{64}")
_VERIFIED_MEMBERS: set[tuple[str, str]] = set()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(split_dir: Path, value: object, expected_sha256: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest paths must be non-empty strings")
    if not isinstance(expected_sha256, str) or not SHA256.fullmatch(expected_sha256):
        raise ValueError(f"manifest digest is missing or malformed for {value}")
    path = (split_dir / value).resolve()
    path.relative_to(split_dir.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    cache_key = (str(path), expected_sha256)
    if cache_key not in _VERIFIED_MEMBERS:
        observed = _sha256(path)
        if observed != expected_sha256:
            raise ValueError(
                f"manifest member digest mismatch for {value}: "
                f"expected {expected_sha256}, got {observed}"
            )
        _VERIFIED_MEMBERS.add(cache_key)
    return path


def derive_trimap(alpha: torch.Tensor, width: int) -> torch.Tensor:
    """Erode certain foreground/background to create a standard unknown band."""
    if alpha.ndim != 2 or not torch.isfinite(alpha).all():
        raise ValueError("alpha matte must be one finite HxW tensor")
    kernel = 2 * int(width) + 1
    fg = (alpha >= 0.999).float()[None, None]
    bg = (alpha <= 0.001).float()[None, None]
    eroded_fg = F.max_pool2d(1.0 - fg, kernel, stride=1, padding=width).eq(0)[0, 0]
    eroded_bg = F.max_pool2d(1.0 - bg, kernel, stride=1, padding=width).eq(0)[0, 0]
    trimap = torch.full_like(alpha, 0.5)
    trimap[eroded_bg] = 0.0
    trimap[eroded_fg] = 1.0
    if int((trimap == 0.5).sum()) < 50:
        raise ValueError("alpha matte does not produce a non-trivial trimap")
    return trimap


def _pil_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _pil_alpha(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").copy()


def _resize(images: list[Image.Image], size: tuple[int, int]) -> list[Image.Image]:
    return [image.resize(size, Image.Resampling.BILINEAR) for image in images]


def _to_rgb_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def _to_alpha_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).clamp_(0.0, 1.0)


class Composition1K:
    def __init__(self, root: str, split: str, trimap_width: int, training: bool):
        self.root = Path(root).resolve()
        self.split = split
        self.split_dir = self.root / split
        self.trimap_width = int(trimap_width)
        self.training = training

        top_path = self.root / "manifest.json"
        split_path = self.split_dir / "manifest.json"
        if not top_path.is_file() or not split_path.is_file():
            raise FileNotFoundError("Composition-1K manifests are missing")
        if not SHA256.fullmatch(EXPECTED_SOURCE_MANIFEST_SHA256):
            raise ValueError("Composition-1K source manifest digest is not operator-pinned")
        observed_source_sha256 = _sha256(top_path)
        if observed_source_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256:
            raise ValueError("Composition-1K source manifest digest mismatch")
        top = json.loads(top_path.read_text(encoding="utf-8"))
        if (
            top.get("protocol") != PROTOCOL
            or top.get("revision") != SOURCE_REVISION
            or int(top.get("crop_size", -1)) != TRAIN_CROP
        ):
            raise ValueError("unexpected Composition-1K protocol manifest")
        expected = TRAIN_COUNT if split == "train" else TEST_COUNT
        proof = top.get("splits", {}).get(split, {})
        if int(proof.get("count", -1)) != expected:
            raise ValueError(f"unexpected {split} count in protocol manifest")
        if proof.get("manifest_sha256") != _sha256(split_path):
            raise ValueError(f"{split} manifest hash mismatch")
        self.data_manifest_sha256 = observed_source_sha256

        self.rows = json.loads(split_path.read_text(encoding="utf-8"))
        if not isinstance(self.rows, list) or len(self.rows) != expected:
            raise ValueError(f"expected {expected} {split} records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        if not isinstance(row, dict):
            raise ValueError(f"malformed {self.split} manifest row {index}")
        image = _pil_rgb(
            _safe_member(self.split_dir, row.get("image"), row.get("image_sha256"))
        )
        alpha = _pil_alpha(
            _safe_member(self.split_dir, row.get("alpha"), row.get("alpha_sha256"))
        )
        if image.size != alpha.size:
            raise ValueError(f"image/alpha size mismatch at {self.split}/{index}")

        if self.training:
            foreground = _pil_rgb(
                _safe_member(
                    self.split_dir,
                    row.get("foreground"),
                    row.get("foreground_sha256"),
                )
            )
            background = _pil_rgb(
                _safe_member(
                    self.split_dir,
                    row.get("background"),
                    row.get("background_sha256"),
                )
            )
            if foreground.size != image.size or background.size != image.size:
                raise ValueError(f"composition layer size mismatch at train/{index}")
            images = [image, alpha, foreground, background]
            width, height = image.size
            scale = max(TRAIN_CROP / width, TRAIN_CROP / height, 1.0)
            if scale > 1.0:
                images = _resize(images, (math.ceil(width * scale), math.ceil(height * scale)))
                width, height = images[0].size
            left = random.randint(0, width - TRAIN_CROP)
            top = random.randint(0, height - TRAIN_CROP)
            box = (left, top, left + TRAIN_CROP, top + TRAIN_CROP)
            images = [value.crop(box) for value in images]
            if random.random() < 0.5:
                images = [value.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for value in images]
            image, alpha, foreground, background = images
        else:
            foreground = Image.new("RGB", image.size)
            background = Image.new("RGB", image.size)

        alpha_tensor = _to_alpha_tensor(alpha)
        trimap = derive_trimap(alpha_tensor, self.trimap_width)
        return {
            "alpha": alpha_tensor,
            "background": _to_rgb_tensor(background),
            "foreground": _to_rgb_tensor(foreground),
            "image": _to_rgb_tensor(image),
            "trimap": trimap,
            "unknown": trimap.eq(0.5),
        }


def load_surface(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("agent_surface", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _grad_mag(alpha: torch.Tensor) -> torch.Tensor:
    dx = F.pad(alpha[:, 1:] - alpha[:, :-1], (0, 1, 0, 0))
    dy = F.pad(alpha[1:, :] - alpha[:-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-8)


def eval_metrics(pred: torch.Tensor, gt: torch.Tensor, trimap: torch.Tensor) -> dict[str, float]:
    if pred.shape != gt.shape or not torch.isfinite(pred).all():
        raise ValueError("prediction is wrong-shape or non-finite")
    if float(pred.min()) < 0.0 or float(pred.max()) > 1.0:
        raise ValueError("prediction must stay in [0,1]")
    pred = torch.where(trimap.eq(0.0), torch.zeros_like(pred), pred)
    pred = torch.where(trimap.eq(1.0), torch.ones_like(pred), pred)
    diff = pred - gt
    return {
        "grad": float((_grad_mag(pred) - _grad_mag(gt)).abs().sum()) / 1000.0,
        "mse": float(diff.square().mean()) * 1000.0,
        "sad": float(diff.abs().sum()) / 1000.0,
        "unk_frac": float(trimap.eq(0.5).float().mean()),
    }


def _norm(factory, channels: int) -> nn.Module:
    if factory is None:
        return nn.BatchNorm2d(channels)
    module = factory(channels)
    if not isinstance(module, nn.Module):
        raise TypeError("make_norm() must return torch.nn.Module")
    return module


def _block(cin: int, cout: int, norm_factory=None) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        _norm(norm_factory, cout),
        nn.ReLU(True),
        nn.Conv2d(cout, cout, 3, padding=1),
        _norm(norm_factory, cout),
        nn.ReLU(True),
    )


class Encoder(nn.Module):
    channels = [32, 64, 96, 128]

    def __init__(self, cin: int, norm_factory=None):
        super().__init__()
        self.e0 = _block(cin, 32, norm_factory)
        self.e1 = _block(32, 64, norm_factory)
        self.e2 = _block(64, 96, norm_factory)
        self.e3 = _block(96, 128, norm_factory)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        e0 = self.e0(x)
        e1 = self.e1(self.pool(e0))
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        return [e0, e1, e2, e3]


class DefaultUpsampler(nn.Module):
    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class AgentUpsampler(nn.Module):
    def __init__(self, factory):
        super().__init__()
        self.modules_by_channel = nn.ModuleDict()
        for channels in (64, 96, 128):
            module = factory(channels)
            if not isinstance(module, nn.Module):
                raise TypeError("build_upsampler() must return torch.nn.Module")
            self.modules_by_channel[str(channels)] = module

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        out = self.modules_by_channel[str(x.shape[1])](x)
        expected = (x.shape[0], x.shape[1], ref.shape[-2], ref.shape[-1])
        if not torch.is_tensor(out) or tuple(out.shape) != expected:
            raise ValueError(f"upsampler output must have exact shape {expected}")
        if not torch.isfinite(out).all():
            raise ValueError("upsampler output is non-finite")
        return out


class ConfigMattingNet(nn.Module):
    def __init__(self, cin: int, *, attention=None, dilation=None, fuse=None,
                 norm_factory=None, upsampler=None):
        super().__init__()
        self.encoder = Encoder(cin, norm_factory)
        self.attention = attention
        self.dilation = dilation if dilation is not None else nn.Identity()
        if self.attention is not None and not isinstance(self.attention, nn.Module):
            raise TypeError("attention factory must return torch.nn.Module")
        if not isinstance(self.dilation, nn.Module):
            raise TypeError("dilation factory must return torch.nn.Module")
        self.fuse = fuse
        self.upsampler = upsampler if upsampler is not None else DefaultUpsampler()
        self.up3 = _block(128 + 96, 96, norm_factory)
        self.up2 = _block(96 + 64, 64, norm_factory)
        self.up1 = _block(64 + 32, 32, norm_factory)
        self.head = nn.Conv2d(32, 1, 1)

    def _fuse(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = torch.cat([decoder, skip], dim=1) if self.fuse is None else self.fuse(decoder, skip)
        expected = (decoder.shape[0], decoder.shape[1] + skip.shape[1], *skip.shape[-2:])
        if not torch.is_tensor(out) or tuple(out.shape) != expected or not torch.isfinite(out).all():
            raise ValueError(f"fuse() must return finite shape {expected}")
        return out

    def forward(self, x: torch.Tensor, image=None, trimap=None) -> torch.Tensor:
        e0, e1, e2, e3 = self.encoder(x)
        bottleneck = e3
        if self.attention is not None:
            gate = self.attention(e3)
            expected_gate = (e3.shape[0], e3.shape[1], 1, 1)
            if not torch.is_tensor(gate) or tuple(gate.shape) != expected_gate:
                raise ValueError(f"attention gate must have exact shape {expected_gate}")
            if (
                not torch.isfinite(gate).all()
                or float(gate.min()) < 0.0
                or float(gate.max()) > 1.0
            ):
                raise ValueError("attention gate must be finite and in [0,1]")
            bottleneck = e3 * gate
        bottleneck = self.dilation(bottleneck)
        if tuple(bottleneck.shape) != tuple(e3.shape) or not torch.isfinite(bottleneck).all():
            raise ValueError("dilation output must be finite and same-shape")
        d2 = self.up3(self._fuse(self.upsampler(bottleneck, e2), e2))
        d1 = self.up2(self._fuse(self.upsampler(d2, e1), e1))
        d0 = self.up1(self._fuse(self.upsampler(d1, e0), e0))
        return torch.sigmoid(self.head(d0)).squeeze(1)


class DecoderMattingNet(nn.Module):
    def __init__(self, cin: int, decoder: nn.Module):
        super().__init__()
        self.encoder = Encoder(cin)
        self.decoder = decoder

    def forward(self, x: torch.Tensor, image=None, trimap=None) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class RefinementMattingNet(nn.Module):
    def __init__(self, refine_fn):
        super().__init__()
        self.coarse = ConfigMattingNet(4)
        self.refine_fn = refine_fn
        self._gradient_contract_checked = False

    def forward(self, x: torch.Tensor, image=None, trimap=None) -> torch.Tensor:
        if image is None or trimap is None:
            raise ValueError("refinement requires image and trimap tensors")
        coarse = self.coarse(x, image=image, trimap=trimap)
        out = self.refine_fn(coarse, image, trimap)
        if not torch.is_tensor(out) or tuple(out.shape) != tuple(coarse.shape):
            raise ValueError("refine() must preserve the alpha shape")
        if out.device != coarse.device or out.dtype != coarse.dtype:
            raise ValueError("refine() must preserve alpha device and dtype")
        if not torch.isfinite(out).all() or float(out.min()) < 0.0 or float(out.max()) > 1.0:
            raise ValueError("refine() must return finite alpha in [0,1]")
        if self.training and not self._gradient_contract_checked:
            if not coarse.requires_grad or not out.requires_grad:
                raise ValueError("refine() must preserve a gradient path to coarse_alpha")
            coarse_grad = torch.autograd.grad(
                out.sum(), coarse, retain_graph=True, allow_unused=True
            )[0]
            if coarse_grad is None or not torch.isfinite(coarse_grad).all():
                raise ValueError("refine() has no finite gradient path to coarse_alpha")
            self._gradient_contract_checked = True
        return out


class TrimapEncodingMattingNet(nn.Module):
    def __init__(self, encode_fn, encoded_channels: int):
        super().__init__()
        self.encode_fn = encode_fn
        self.net = ConfigMattingNet(3 + encoded_channels)

    def _encode(self, trimap: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_fn(trimap)
        if not torch.is_tensor(encoded) or encoded.ndim != 4:
            raise ValueError("encode_trimap() must return a BxKxHxW tensor")
        if (
            encoded.shape[0] != trimap.shape[0]
            or encoded.shape[-2:] != trimap.shape[-2:]
            or not 1 <= encoded.shape[1] <= 8
        ):
            raise ValueError("encode_trimap() returned invalid batch/channel/spatial shape")
        if encoded.device != trimap.device or not encoded.dtype.is_floating_point:
            raise ValueError("encode_trimap() must preserve device and return floating point")
        if not torch.isfinite(encoded).all():
            raise ValueError("encode_trimap() returned non-finite features")
        return encoded

    def forward(self, x: torch.Tensor, image=None, trimap=None) -> torch.Tensor:
        if image is None or trimap is None:
            raise ValueError("trimap encoding requires image and trimap tensors")
        encoded = self._encode(trimap)
        return self.net(torch.cat([image, encoded], dim=1), image=image, trimap=trimap)


def default_loss(pred, gt, image, foreground, background, trimap, unknown):
    mask = unknown.float()
    denom = mask.sum(dim=(-2, -1)).clamp(min=1.0)
    alpha_loss = ((pred - gt).abs() * mask).sum(dim=(-2, -1)) / denom
    composite = pred.unsqueeze(1) * foreground + (1.0 - pred.unsqueeze(1)) * background
    composition_loss = ((composite - image).abs().mean(1) * mask).sum(dim=(-2, -1)) / denom
    return (alpha_loss + 0.5 * composition_loss).mean()


def _build_model(surface: str, module, cin: int) -> nn.Module:
    if surface == "arch":
        factory = getattr(module, "build_net", None)
        if not callable(factory):
            raise TypeError("solution must define callable build_net()")
        model = factory(cin)
        if not isinstance(model, nn.Module):
            raise TypeError("build_net() must return torch.nn.Module")
        return model
    if surface == "decoder":
        factory = getattr(module, "build_decoder", None)
        if not callable(factory):
            raise TypeError("solution must define callable build_decoder()")
        decoder = factory(list(Encoder.channels))
        if not isinstance(decoder, nn.Module):
            raise TypeError("build_decoder() must return torch.nn.Module")
        return DecoderMattingNet(cin, decoder)
    if surface == "refine":
        refine_fn = getattr(module, "refine", None)
        if not callable(refine_fn):
            raise TypeError("solution must define callable refine()")
        return RefinementMattingNet(refine_fn)
    if surface == "trimap":
        encode_fn = getattr(module, "encode_trimap", None)
        if not callable(encode_fn):
            raise TypeError("solution must define callable encode_trimap()")
        probe = torch.zeros(2, 16, 16)
        encoded = encode_fn(probe)
        if (
            not torch.is_tensor(encoded)
            or encoded.ndim != 4
            or encoded.shape[0] != 2
            or encoded.shape[-2:] != (16, 16)
            or not 1 <= encoded.shape[1] <= 8
            or not encoded.dtype.is_floating_point
            or not torch.isfinite(encoded).all()
        ):
            raise ValueError("encode_trimap() failed its static shape/finiteness probe")
        return TrimapEncodingMattingNet(encode_fn, int(encoded.shape[1]))

    kwargs = {}
    if surface == "attention":
        factory = getattr(module, "build_attention", None)
        if not callable(factory):
            raise TypeError("solution must define callable build_attention()")
        active = factory(128)
        if not isinstance(active, nn.Module):
            raise TypeError("build_attention() must return torch.nn.Module")
        kwargs["attention"] = active
    elif surface == "dilation":
        factory = getattr(module, "build_dilation", None)
        if not callable(factory):
            raise TypeError("solution must define callable build_dilation()")
        active = factory(128)
        if not isinstance(active, nn.Module):
            raise TypeError("build_dilation() must return torch.nn.Module")
        kwargs["dilation"] = active
    elif surface == "skip":
        fuse = getattr(module, "fuse", None)
        if not callable(fuse):
            raise TypeError("solution must define callable fuse()")
        kwargs["fuse"] = fuse
    elif surface == "norm":
        factory = getattr(module, "make_norm", None)
        if not callable(factory):
            raise TypeError("solution must define callable make_norm()")
        kwargs["norm_factory"] = factory
    elif surface == "upsampling":
        factory = getattr(module, "build_upsampler", None)
        if not callable(factory):
            raise TypeError("solution must define callable build_upsampler()")
        kwargs["upsampler"] = AgentUpsampler(factory)
    return ConfigMattingNet(cin, **kwargs)


def _batch(dataset: Composition1K, indexes: list[int], device: torch.device) -> dict:
    records = [dataset[index] for index in indexes]
    return {
        key: torch.stack([record[key] for record in records]).to(device)
        for key in ("alpha", "background", "foreground", "image", "trimap", "unknown")
    }


def _forward(model: nn.Module, batch: dict) -> torch.Tensor:
    height, width = batch["alpha"].shape[-2:]
    pad_height = (-height) % 8
    pad_width = (-width) % 8
    padding = (0, pad_width, 0, pad_height)
    image = F.pad(batch["image"], padding, mode="replicate") if any(padding) else batch["image"]
    trimap = F.pad(batch["trimap"], padding, mode="replicate") if any(padding) else batch["trimap"]
    inputs = torch.cat([image, trimap.unsqueeze(1)], dim=1)
    pred = model(inputs, image=image, trimap=trimap)
    expected = (batch["alpha"].shape[0], height + pad_height, width + pad_width)
    if not torch.is_tensor(pred) or tuple(pred.shape) != expected:
        raise ValueError("matting network must return alpha with shape BxHxW")
    if not torch.isfinite(pred).all() or float(pred.min()) < 0.0 or float(pred.max()) > 1.0:
        raise ValueError("matting network output must be finite and in [0,1]")
    return pred[..., :height, :width]


def _evaluate(model: nn.Module, dataset: Composition1K, device: torch.device) -> dict[str, float]:
    values = []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = _batch(dataset, [index], device)
            pred = _forward(model, batch)[0]
            values.append(eval_metrics(pred, batch["alpha"][0], batch["trimap"][0]))
    if len(values) != TEST_COUNT:
        raise RuntimeError("full test evaluation did not complete")
    return {key: float(np.mean([value[key] for value in values])) for key in values[0]}


def train_and_evaluate(surface: str, module, train: Composition1K,
                       tests: dict[str, Composition1K], device: torch.device,
                       iters: int, seed: int, batch_size: int = 8) -> dict[str, dict[str, float]]:
    set_all_seeds(seed)
    loss_fn = default_loss
    if surface == "loss":
        factory = getattr(module, "get_matting_loss", None)
        if not callable(factory):
            raise TypeError("solution must define callable get_matting_loss()")
        loss_fn = factory()
        if not callable(loss_fn):
            raise TypeError("get_matting_loss() must return a callable")

    model = _build_model(surface, module, cin=4).to(device).train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("matting model must have trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters)
    order = list(range(len(train)))

    for step in range(iters):
        if step % math.ceil(len(train) / batch_size) == 0:
            random.shuffle(order)
        indexes = [order[(step * batch_size + offset) % len(train)] for offset in range(batch_size)]
        batch = _batch(train, indexes, device)
        pred = _forward(model, batch)
        loss = torch.as_tensor(
            loss_fn(
                pred,
                batch["alpha"],
                batch["image"],
                batch["foreground"],
                batch["background"],
                batch["trimap"],
                batch["unknown"],
            ),
            device=device,
            dtype=torch.float32,
        )
        if loss.numel() != 1 or not torch.isfinite(loss).all() or float(loss) < 0.0:
            raise ValueError("matting loss must be one finite non-negative scalar")
        if step == 0:
            if not loss.requires_grad:
                raise ValueError("matting loss must have a gradient path to pred")
            pred_grad = torch.autograd.grad(
                loss, pred, retain_graph=True, allow_unused=True
            )[0]
            if pred_grad is None or not torch.isfinite(pred_grad).all():
                raise ValueError("matting loss has no finite gradient path to pred")
        optimizer.zero_grad(set_to_none=True)
        loss.reshape(()).backward()
        active_grads = [parameter.grad for parameter in trainable if parameter.grad is not None]
        if not active_grads:
            raise ValueError("matting loss produced no model-parameter gradients")
        if not all(torch.isfinite(gradient).all() for gradient in active_grads):
            raise ValueError("matting loss produced non-finite model-parameter gradients")
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        scheduler.step()
        if step % 20_000 == 0 or step == iters - 1:
            print(f"MATTING_TRAIN step={step} loss={float(loss):.6f}", flush=True)

    print(f"MATTING_TRAIN_COMPLETE steps={iters} batch={batch_size}", flush=True)
    return {setting: _evaluate(model, dataset, device) for setting, dataset in tests.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--task-id", required=True, choices=sorted(TASK_SURFACES))
    parser.add_argument("--surface", required=True, choices=sorted(SURFACES))
    parser.add_argument("--iters", type=int, default=TRAIN_ITERS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if TASK_SURFACES[args.task_id] != args.surface:
        raise SystemExit("task-id/surface mismatch")
    if args.iters != TRAIN_ITERS or args.seed != 42:
        raise SystemExit(f"full protocol requires iters={TRAIN_ITERS}, seed=42")
    if not torch.cuda.is_available():
        raise SystemExit("image-matting verification requires CUDA")

    set_all_seeds(args.seed)
    device = torch.device("cuda")
    print(f"DEVICE cuda torch={torch.__version__}", flush=True)
    train = Composition1K(args.data_root, "train", TRAIN_TRIMAP_WIDTH, training=True)
    tests = {
        setting: Composition1K(args.data_root, "test", width, training=False)
        for setting, width in TRIMAP_WIDTHS.items()
    }
    hashes = {train.data_manifest_sha256} | {
        dataset.data_manifest_sha256 for dataset in tests.values()
    }
    if len(hashes) != 1:
        raise RuntimeError("dataset manifest changed between settings")
    manifest_sha256 = hashes.pop()
    print(
        f"MATTING_DATA protocol={PROTOCOL} task={args.task_id} surface={args.surface} "
        f"train={len(train)} test={len(tests['medium'])} "
        f"crop={TRAIN_CROP} train_trimap_width={TRAIN_TRIMAP_WIDTH} "
        f"manifest_sha256={manifest_sha256}",
        flush=True,
    )

    module = load_surface(Path(args.solution))
    results = train_and_evaluate(
        args.surface, module, train, tests, device, args.iters, args.seed
    )
    for setting in SETTINGS:
        metrics = results[setting]
        print(
            f"MATTING_RESULT protocol={PROTOCOL} task={args.task_id} "
            f"surface={args.surface} setting={setting} "
            f"trimap_width={TRIMAP_WIDTHS[setting]} "
            f"sad={metrics['sad']:.6f} mse={metrics['mse']:.6f} "
            f"grad={metrics['grad']:.6f} unk_frac={metrics['unk_frac']:.6f} "
            f"train={len(train)} test={len(tests[setting])} iters={args.iters} seed={args.seed}",
            flush=True,
        )
    print(
        f"MATTING_COMPLETE protocol={PROTOCOL} task={args.task_id} "
        f"surface={args.surface} settings=3 "
        f"manifest_sha256={manifest_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
