"""Trusted evaluation harness for score-based query black-box attack task."""

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from custom_attack import run_attack


class QueryLimitedBlackBox(torch.nn.Module):
    """Query-limited wrapper with no gradient path and budget tracking."""

    def __init__(self, model: torch.nn.Module, n_classes: int, max_queries: int):
        super().__init__()
        self.model = model
        self.n_classes = n_classes
        self.max_queries = int(max_queries)
        self.queries_used = 0
        self.budget_exhausted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = int(x.shape[0])
        self.queries_used += batch

        if self.queries_used > self.max_queries:
            self.budget_exhausted = True
            return torch.zeros(
                (batch, self.n_classes),
                device=x.device,
                dtype=x.dtype,
            )

        with torch.no_grad():
            return self.model(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
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

    clean_model = load_model(args.dataset, args.arch, device)
    dataset, n_classes = load_dataset(args.dataset, args.data_dir)
    clean_images_cpu, clean_labels_cpu = collect_correct_subset(
        model=clean_model,
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
    total_queries = 0
    exhausted_batches = 0
    all_linf_ok = True
    all_range_ok = True
    per_sample_max_delta_all: list[torch.Tensor] = []
    valid_mask_all: list[torch.Tensor] = []

    for images_cpu, labels_cpu in eval_loader:
        images = images_cpu.to(device)
        labels = labels_cpu.to(device)
        batch_size = int(images.shape[0])

        query_model = QueryLimitedBlackBox(
            model=clean_model,
            n_classes=n_classes,
            max_queries=batch_size * int(args.n_queries),
        )

        adv_images = run_attack(
            query_model,
            images,
            labels,
            args.eps,
            args.n_queries,
            device,
            n_classes,
        )

        total_queries += min(query_model.queries_used, query_model.max_queries)

        if adv_images.shape != images.shape:
            raise RuntimeError(
                f"run_attack returned wrong shape: got {tuple(adv_images.shape)}, "
                f"expected {tuple(images.shape)}"
            )

        # Query budget exhaustion — entire batch treated as failure.
        if query_model.queries_used > query_model.max_queries:
            exhausted_batches += 1
            robust_correct += batch_size
            continue

        # Per-sample [0,1] range check.
        in_range_mask = (
            torch.isfinite(adv_images).all(dim=(1, 2, 3))
            & (adv_images.flatten(1).min(dim=1).values >= -1e-6)
            & (adv_images.flatten(1).max(dim=1).values <= 1.0 + 1e-6)
        )

        # Per-sample Linf check.
        per_sample_max_delta = (adv_images - images).abs().flatten(1).max(dim=1).values
        linf_mask = per_sample_max_delta <= (args.eps + 1e-6)
        valid_mask = in_range_mask & linf_mask
        per_sample_max_delta_all.append(per_sample_max_delta.detach().cpu())
        valid_mask_all.append(valid_mask.detach().cpu())
        all_linf_ok = all_linf_ok and bool(linf_mask.all().item())
        all_range_ok = all_range_ok and bool(in_range_mask.all().item())

        with torch.no_grad():
            adv_preds = clean_model(adv_images).argmax(dim=1)
            # Invalid samples (budget or range violation) are treated as attack failures.
            robust_mask = (~valid_mask) | adv_preds.eq(labels)
            robust_correct += int(robust_mask.sum().item())

    clean_acc = 1.0
    robust_acc = robust_correct / n_eval
    asr = 1.0 - robust_acc
    avg_queries = total_queries / n_eval
    if per_sample_max_delta_all:
        per_sample_max_delta_cat = torch.cat(per_sample_max_delta_all, dim=0)
        valid_mask_cat = torch.cat(valid_mask_all, dim=0)
        valid_rate = float(valid_mask_cat.float().mean().item())
        max_delta = float(per_sample_max_delta_cat.max().item())
    else:
        valid_rate = 0.0
        max_delta = 0.0

    print(f"[Eval] Samples evaluated: {n_eval}")
    print(f"[Eval] Per-sample query budget: {args.n_queries}")
    print(f"[Eval] Budget exhausted batches: {exhausted_batches}")
    print(f"[Eval] Linf constraint satisfied: {all_linf_ok} (max_delta={max_delta:.6f})")
    print(f"[Eval] Pixel range [0,1] satisfied: {all_range_ok}")
    print(f"[Eval] valid_rate={valid_rate:.6f}")
    print(
        "ATTACK_METRICS "
        f"asr={asr:.6f} clean_acc={clean_acc:.6f} "
        f"robust_acc={robust_acc:.6f} avg_queries={avg_queries:.2f}"
    )


if __name__ == "__main__":
    main()
