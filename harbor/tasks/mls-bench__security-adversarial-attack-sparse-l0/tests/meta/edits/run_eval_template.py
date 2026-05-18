"""Evaluation harness for sparse L0 adversarial attack task."""

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
    # RobustBench L2-robust CIFAR-10 model-zoo key (Croce et al., AAAI 2022
    # canonical sparse-L0 threat model: k=24, untargeted, robust models).
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="/data/robustbench_models")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--pixels", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(data_dir: str) -> tuple[torch.utils.data.Dataset, int]:
    # CIFAR-10 only: the Sparse-RS paper never evaluates L0 on CIFAR-100.
    transform = transforms.ToTensor()
    return datasets.CIFAR10(data_dir, train=False, transform=transform, download=False), 10


def load_model(model_name: str, model_dir: str, device: torch.device) -> torch.nn.Module:
    # Adversarially-robust L2 CIFAR-10 target from the RobustBench model zoo,
    # matching the Sparse-RS paper's L0 evaluation (App. A.5). Checkpoints are
    # pre-fetched by the data-prep step into model_dir (compute nodes are
    # offline); robustbench.load_model reads them without network access.
    from robustbench.utils import load_model as rb_load_model

    model = rb_load_model(
        model_name,
        model_dir=model_dir,
        dataset="cifar10",
        threat_model="L2",
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

    dataset, n_classes = load_dataset(args.data_dir)
    # score_model is the trusted reference — never passed to user code.
    score_model = load_model(args.model_name, args.model_dir, device)

    clean_images_cpu, clean_labels_cpu = collect_correct_subset(
        model=score_model,
        dataset=dataset,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device,
    )
    n_eval = clean_images_cpu.shape[0]

    eval_loader = DataLoader(
        TensorDataset(clean_images_cpu, clean_labels_cpu),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    robust_correct = 0
    all_l0_ok = True
    all_range_ok = True
    changed_pixels_all: list[torch.Tensor] = []
    valid_mask_all: list[torch.Tensor] = []

    for images_cpu, labels_cpu in eval_loader:
        images = images_cpu.to(device)
        labels = labels_cpu.to(device)

        # High-severity fix: pass a fresh deep copy so user code cannot tamper with
        # score_model's weights, BatchNorm statistics, or forward hooks.
        attack_model = copy.deepcopy(score_model)
        adv_images = run_attack(attack_model, images, labels, args.pixels, device, n_classes)
        del attack_model

        if adv_images.shape != images.shape:
            raise RuntimeError(
                f"run_attack returned wrong shape: got {tuple(adv_images.shape)}, "
                f"expected {tuple(images.shape)}"
            )

        # Reject samples whose pixel values leave [0, 1].
        in_range_mask = (
            torch.isfinite(adv_images).all(dim=(1, 2, 3))
            & (adv_images.flatten(1).min(dim=1).values >= -1e-6)
            & (adv_images.flatten(1).max(dim=1).values <= 1.0 + 1e-6)
        )

        changed_pixels = ((adv_images - images).abs() > 1e-5).any(dim=1).sum(dim=(1, 2))
        l0_mask = changed_pixels <= args.pixels
        valid_mask = in_range_mask & l0_mask
        changed_pixels_all.append(changed_pixels.detach().cpu())
        valid_mask_all.append(valid_mask.detach().cpu())
        all_l0_ok = all_l0_ok and bool(l0_mask.all().item())
        all_range_ok = all_range_ok and bool(in_range_mask.all().item())

        with torch.no_grad():
            adv_preds = score_model(adv_images).argmax(dim=1)
            # Invalid samples (budget or range violation) are treated as attack failures.
            robust_mask = (~valid_mask) | adv_preds.eq(labels)
            robust_correct += int(robust_mask.sum().item())

    clean_acc = 1.0
    robust_acc = robust_correct / n_eval
    asr = 1.0 - robust_acc
    changed_pixels_cat = torch.cat(changed_pixels_all, dim=0)
    valid_mask_cat = torch.cat(valid_mask_all, dim=0)
    valid_rate = float(valid_mask_cat.float().mean().item())
    mean_changed_pixels = float(changed_pixels_cat.float().mean().item())
    max_changed_pixels = int(changed_pixels_cat.max().item())

    print(f"[Eval] Samples evaluated: {n_eval}")
    print(
        f"[Eval] L0 constraint satisfied: {all_l0_ok} "
        f"(budget={args.pixels}, max_changed_pixels={max_changed_pixels})"
    )
    print(f"[Eval] Pixel range [0,1] satisfied: {all_range_ok}")
    print(
        f"[Eval] valid_rate={valid_rate:.6f} "
        f"mean_changed_pixels={mean_changed_pixels:.4f} "
        f"max_changed_pixels={max_changed_pixels}"
    )
    print(f"ATTACK_METRICS asr={asr:.6f} clean_acc={clean_acc:.6f} robust_acc={robust_acc:.6f}")


if __name__ == "__main__":
    main()
