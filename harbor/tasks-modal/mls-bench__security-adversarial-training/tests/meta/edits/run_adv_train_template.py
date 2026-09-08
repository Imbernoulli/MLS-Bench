"""Training and evaluation harness for adversarial training task."""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from custom_adv_train import AdversarialTrainer
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, required=True,
                        choices=["smallcnn", "preact_resnet18", "vgg11_bn"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--attack-steps", type=int, default=10)
    parser.add_argument("--eval-attack-steps", type=int, default=50)
    parser.add_argument("--eval-alpha", type=float, default=None,
                        help="Step size for eval PGD. Defaults to eps/4 if not set.")
    parser.add_argument("--eval-restarts", type=int, default=1,
                        help="Number of random restarts for eval PGD.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_datasets(dataset_name, data_dir):
    """Load train and test datasets."""
    if dataset_name == "mnist":
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(data_dir, train=True, transform=transform_train, download=False)
        test_set = datasets.MNIST(data_dir, train=False, transform=transform_test, download=False)
        num_classes = 10
    elif dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(data_dir, train=True, transform=transform_train, download=False)
        test_set = datasets.CIFAR10(data_dir, train=False, transform=transform_test, download=False)
        num_classes = 10
    elif dataset_name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR100(data_dir, train=True, transform=transform_train, download=False)
        test_set = datasets.CIFAR100(data_dir, train=False, transform=transform_test, download=False)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_set, test_set, num_classes


def fgsm_attack(model, images, labels, eps):
    """FGSM attack for evaluation."""
    model.eval()
    images_cp = images.clone().detach().requires_grad_(True)
    outputs = model(images_cp)
    loss = F.cross_entropy(outputs, labels)
    grad = torch.autograd.grad(loss, images_cp)[0]
    adv_images = images + eps * grad.sign()
    return torch.clamp(adv_images, 0.0, 1.0).detach()


def pgd_attack(model, images, labels, eps, alpha, steps, restarts=1):
    """PGD attack for evaluation with multi-restart support."""
    model.eval()
    best_adv = images.clone().detach()
    best_loss = torch.full((images.size(0),), -float('inf'), device=images.device)

    for _ in range(restarts):
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

        for _ in range(steps):
            adv_images.requires_grad_(True)
            outputs = model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            grad = torch.autograd.grad(loss, adv_images)[0]
            adv_images = adv_images.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        # Keep best adversarial examples (highest per-sample loss)
        with torch.no_grad():
            outputs = model(adv_images)
            loss_per_sample = F.cross_entropy(outputs, labels, reduction='none')
            improved = loss_per_sample > best_loss
            best_adv[improved] = adv_images[improved]
            best_loss[improved] = loss_per_sample[improved]

    return best_adv


def evaluate_clean(model, test_loader, device):
    """Evaluate clean accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


def evaluate_robust(model, test_loader, eps, eval_alpha, steps, device, attack_type="pgd", restarts=1):
    """Evaluate robust accuracy under adversarial attack."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        if attack_type == "pgd":
            adv_images = pgd_attack(model, images, labels, eps, eval_alpha, steps, restarts)
        elif attack_type == "fgsm":
            adv_images = fgsm_attack(model, images, labels, eps)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Data
    train_set, test_set, num_classes = get_datasets(args.dataset, args.data_dir)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Model
    model = get_model(args.arch, num_classes).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Adversarial trainer
    trainer = AdversarialTrainer(
        model=model,
        eps=args.eps,
        alpha=args.alpha,
        attack_steps=args.attack_steps,
        num_classes=num_classes,
    )

    # Training loop
    print(
        f"[Train] arch={args.arch} dataset={args.dataset} epochs={args.epochs} "
        f"eps={args.eps:.6f} alpha={args.alpha:.6f} attack_steps={args.attack_steps}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            result = trainer.train_step(images, labels, optimizer)
            total_loss += result['loss']
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"TRAIN_METRICS epoch={epoch+1} loss={avg_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}",
                flush=True,
            )

    # Evaluation
    print("[Eval] Evaluating clean accuracy...", flush=True)
    clean_acc = evaluate_clean(model, test_loader, device)
    print(f"[Eval] clean_acc={clean_acc:.4f}", flush=True)

    print("[Eval] Evaluating FGSM robustness...", flush=True)
    robust_acc_fgsm = evaluate_robust(
        model, test_loader, args.eps, args.eps, 1, device, "fgsm",
    )
    print(f"[Eval] robust_acc_fgsm={robust_acc_fgsm:.4f}", flush=True)

    eval_alpha = args.eval_alpha if args.eval_alpha is not None else args.eps / 4.0
    print(
        f"[Eval] Evaluating PGD-{args.eval_attack_steps} robustness "
        f"(alpha={eval_alpha:.6f}, restarts={args.eval_restarts})...",
        flush=True,
    )
    robust_acc_pgd = evaluate_robust(
        model, test_loader, args.eps, eval_alpha, args.eval_attack_steps, device,
        "pgd", restarts=args.eval_restarts,
    )
    print(f"[Eval] robust_acc_pgd={robust_acc_pgd:.4f}", flush=True)

    print(
        f"TEST_METRICS clean_acc={clean_acc:.4f} "
        f"robust_acc_fgsm={robust_acc_fgsm:.4f} "
        f"robust_acc_pgd={robust_acc_pgd:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
