"""Evaluation harness for white-box Linf adversarial attack task."""

import argparse
import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from custom_attack import run_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--eps", type=float, default=2.0 / 255.0)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(name: str, data_dir: str) -> tuple[torch.utils.data.Dataset, int]:
    transform = transforms.ToTensor()
    if name == "cifar10":
        return datasets.CIFAR10(data_dir, train=False, transform=transform, download=False), 10
    return datasets.CIFAR100(data_dir, train=False, transform=transform, download=False), 100


def load_model(dataset: str, arch: str, device: torch.device) -> torch.nn.Module:
    entry = f"{dataset}_{arch}"
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        entry,
        pretrained=True,
        trust_repo=True,
    )
    return model.to(device).eval()


def collect_correct_subset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    selected_images: list[torch.Tensor] = []
    selected_labels: list[torch.Tensor] = []
    collected = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            mask = preds.eq(labels)
            if mask.any():
                selected_images.append(images[mask].detach().cpu())
                selected_labels.append(labels[mask].detach().cpu())
                collected += int(mask.sum().item())
            if collected >= n_samples:
                break

    if collected == 0:
        raise RuntimeError("No correctly classified samples found on clean inputs.")

    images_all = torch.cat(selected_images, dim=0)[:n_samples]
    labels_all = torch.cat(selected_labels, dim=0)[:n_samples]
    return images_all, labels_all


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, n_classes = load_dataset(args.dataset, args.data_dir)
    # score_model is the trusted reference — never passed to user code.
    score_model = load_model(args.dataset, args.arch, device)

    clean_images_cpu, clean_labels_cpu = collect_correct_subset(
        model=score_model,
        dataset=dataset,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device,
    )
    n_eval = clean_images_cpu.shape[0]

    # Low-severity: warn when fewer correctly-classified samples than requested.
    if n_eval < args.n_samples:
        print(
            f"[Eval] WARNING: requested {args.n_samples} samples but only {n_eval} "
            "correctly-classified samples found; evaluation proceeds with reduced count."
        )

    eval_loader = DataLoader(
        TensorDataset(clean_images_cpu, clean_labels_cpu),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    robust_correct = 0
    all_linf_ok = True
    all_range_ok = True
    per_sample_max_delta_all: list[torch.Tensor] = []
    valid_mask_all: list[torch.Tensor] = []

    for images_cpu, labels_cpu in eval_loader:
        images = images_cpu.to(device)
        labels = labels_cpu.to(device)
        # High-severity fix: pass a fresh deep copy so user code cannot tamper with
        # score_model's weights, BatchNorm statistics, or forward hooks.
        attack_model = copy.deepcopy(score_model)
        adv_images = run_attack(attack_model, images, labels, args.eps, device, n_classes)
        del attack_model

        if adv_images.shape != images.shape:
            raise RuntimeError(
                f"run_attack returned wrong shape: got {tuple(adv_images.shape)}, "
                f"expected {tuple(images.shape)}"
            )

        # Medium-severity fix: reject samples whose pixel values leave [0, 1].
        in_range_mask = (
            torch.isfinite(adv_images).all(dim=(1, 2, 3))
            & (adv_images.flatten(1).min(dim=1).values >= -1e-6)
            & (adv_images.flatten(1).max(dim=1).values <= 1.0 + 1e-6)
        )

        # Per-sample Linf: max over (C, H, W)
        per_sample_max_delta = (adv_images - images).abs().flatten(1).max(dim=1).values
        linf_mask = per_sample_max_delta <= (args.eps + 1e-6)
        valid_mask = in_range_mask & linf_mask
        per_sample_max_delta_all.append(per_sample_max_delta.detach().cpu())
        valid_mask_all.append(valid_mask.detach().cpu())
        # Track the two constraints independently for accurate reporting.
        all_linf_ok = all_linf_ok and bool(linf_mask.all().item())
        all_range_ok = all_range_ok and bool(in_range_mask.all().item())

        with torch.no_grad():
            adv_preds = score_model(adv_images).argmax(dim=1)
            # Invalid samples (budget or range violation) are treated as attack failures.
            robust_mask = (~valid_mask) | adv_preds.eq(labels)
            robust_correct += int(robust_mask.sum().item())

    clean_acc = 1.0
    robust_acc = robust_correct / n_eval
    asr = 1.0 - robust_acc
    if per_sample_max_delta_all:
        per_sample_max_delta_cat = torch.cat(per_sample_max_delta_all, dim=0)
        valid_mask_cat = torch.cat(valid_mask_all, dim=0)
        # valid_rate reflects both [0,1]-range and Linf budget, matching the scoring logic.
        valid_rate = float(valid_mask_cat.float().mean().item())
        mean_max_delta = float(per_sample_max_delta_cat.mean().item())
        max_delta = float(per_sample_max_delta_cat.max().item())
    else:
        valid_rate = 0.0
        mean_max_delta = 0.0
        max_delta = 0.0

    print(f"[Eval] Samples evaluated: {n_eval}")
    print(f"[Eval] Linf constraint satisfied: {all_linf_ok} (eps={args.eps:.6f}, max_delta={max_delta:.6f})")
    print(f"[Eval] Pixel range [0,1] satisfied: {all_range_ok}")
    print(
        f"[Eval] valid_rate={valid_rate:.6f} "
        f"mean_max_delta={mean_max_delta:.6f} "
        f"max_delta={max_delta:.6f}"
    )
    print(f"ATTACK_METRICS asr={asr:.6f} clean_acc={clean_acc:.6f} robust_acc={robust_acc:.6f}")


if __name__ == "__main__":
    main()
