#!/usr/bin/env python3
"""Train the frozen CIFAR-10 classifier with the public OpenOOD baseline recipe."""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment(value: torch.Tensor) -> torch.Tensor:
    """OpenOOD CIFAR augmentation: zero-pad 4, per-image random crop, flip."""
    batch, channels, _, _ = value.shape
    padded = F.pad(value, (4, 4, 4, 4), mode="constant", value=0.0)
    offset_y = torch.randint(0, 9, (batch,), device=value.device)
    offset_x = torch.randint(0, 9, (batch,), device=value.device)
    row = offset_y[:, None] + torch.arange(32, device=value.device)[None, :]
    col = offset_x[:, None] + torch.arange(32, device=value.device)[None, :]
    cropped = padded.gather(2, row[:, None, :, None].expand(-1, channels, -1, 40))
    cropped = cropped.gather(3, col[:, None, None, :].expand(-1, channels, 32, -1))
    flip = torch.rand(batch, device=value.device) < 0.5
    cropped[flip] = cropped[flip].flip(3)
    return cropped


@torch.no_grad()
def accuracy(
    model,
    images: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    *,
    normalized_input: bool,
) -> float:
    model.eval()
    correct = 0
    for start in range(0, images.shape[0], 512):
        value = torch.from_numpy(images[start:start + 512]).to(device).float()
        if not normalized_input:
            value.div_(255.0)
            value = (value - MEAN.to(device)) / STD.to(device)
        prediction = model(value).argmax(1).cpu().numpy()
        correct += int(np.count_nonzero(prediction == labels[start:start + 512]))
    return correct / images.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--legacy-normalized-data",
        action="store_true",
        help="read Xtr_all/Xte_id from the earlier float32 archive for staging only",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("classifier staging requires exactly one visible CUDA GPU")

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "ood-detection-lab"))
    from openood_resnet18 import ResNet18_32x32

    set_seeds(args.seed)
    device = torch.device("cuda")
    with np.load(args.data, allow_pickle=False) as archive:
        if args.legacy_normalized_data:
            train_images = archive["Xtr_all"]
            train_labels = archive["ytr_all"]
            id_images = archive["Xte_id"]
            id_labels = archive["yte_id"]
        else:
            train_images = archive["train_images"]
            train_labels = archive["train_labels"]
            id_images = archive["id_images"]
            id_labels = archive["id_labels"]
    expected_id = 5_000 if args.legacy_normalized_data else 10_000
    if train_images.shape != (50_000, 3, 32, 32) or id_images.shape != (
        expected_id, 3, 32, 32,
    ):
        raise RuntimeError("full CIFAR-10 train inventory and the expected ID split are required")
    expected_image_dtype = np.float32 if args.legacy_normalized_data else np.uint8
    if train_images.dtype != expected_image_dtype or id_images.dtype != expected_image_dtype:
        raise RuntimeError(
            f"classifier input dtype is {train_images.dtype}/{id_images.dtype}, "
            f"expected {expected_image_dtype}"
        )

    model = ResNet18_32x32().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    steps_per_epoch = math.ceil(train_images.shape[0] / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1e-5 + (1.0 - 1e-5) * 0.5 * (
            1.0 + math.cos(step / total_steps * math.pi)
        ),
    )
    mean = MEAN.to(device)
    std = STD.to(device)
    generator = torch.Generator().manual_seed(args.seed)
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(train_images.shape[0], generator=generator).numpy()
        loss_sum = 0.0
        seen = 0
        for start in range(0, permutation.size, args.batch_size):
            index = permutation[start:start + args.batch_size]
            value = torch.from_numpy(train_images[index]).to(device).float()
            target = torch.from_numpy(train_labels[index]).to(device)
            if args.legacy_normalized_data:
                value = value * std + mean
            else:
                value.div_(255.0)
            value = augment(value)
            value = (value - mean) / std
            optimizer.zero_grad(set_to_none=True)
            logits = model(value)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"non-finite classifier loss at step {global_step}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * index.size
            seen += index.size
            global_step += 1
        print(
            f"OPENOOD_TRAIN epoch={epoch + 1}/{args.epochs} steps={global_step} "
            f"loss={loss_sum / seen:.6f} lr={scheduler.get_last_lr()[0]:.8f}",
            flush=True,
        )

    test_accuracy = accuracy(
        model,
        id_images,
        id_labels,
        device,
        normalized_input=args.legacy_normalized_data,
    )
    if not 0.90 <= test_accuracy <= 1.0:
        raise RuntimeError(f"trained classifier accuracy is invalid: {test_accuracy:.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, args.output)
    print(
        f"OPENOOD_TRAIN_COMPLETE protocol=openood_cifar10_resnet18_full_v1 "
        f"epochs={args.epochs} steps={global_step} n_train=50000 n_id={expected_id} "
        f"id_acc={test_accuracy:.6f} checkpoint={args.output} status=ok",
        flush=True,
    )


if __name__ == "__main__":
    main()
