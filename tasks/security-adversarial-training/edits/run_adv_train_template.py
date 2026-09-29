"""Training and evaluation harness for adversarial training task.

Training and evaluation run in two different interpreters:

* The training worker (this file, re-started with ``--train-worker-ckpt``) is
  the only process that imports ``custom_adv_train``.  It trains the model and
  writes ``model.state_dict()`` to a checkpoint path chosen by the evaluator.
* The evaluator (this process) never imports the submitted code.  After the
  worker has exited it builds a fresh model of the fixed architecture from
  ``models.py``, loads only the checkpoint's tensors into it (strict key and
  shape match), and attacks that model in eval mode.  Hooks, patched
  ``forward`` methods, wrapper modules or patched library functions set up by
  the training code therefore do not exist at evaluation time: only the trained
  weights are evaluated.  The worker's output is relayed with the metric tag
  defused, so the only ``TEST_METRICS`` line is the one printed here.

Each attack (FGSM and PGD) is run twice, once on the cross-entropy loss and
once on the Carlini-Wagner margin loss ``max_{j != y} z_j - z_y`` (Carlini &
Wagner, 2017), whose gradient sign does not depend on the logit scale and does
not vanish when the softmax saturates.  A test image counts as robust only if
it is still classified correctly under both.
"""

import argparse
import importlib.util
import os
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_METRIC_TAG = b"TEST_METRICS"
_ATTACK_LOSSES = ("ce", "margin")
_UNPRIVILEGED_UID = 65534  # nobody


def _load_get_model():
    """Load the FIXED architecture module (models.py) by its file path.

    The evaluator is started with ``python -I`` so that a stdlib-shadowing
    file the agent creates in the harness directory (e.g. ``bench/argparse.py``)
    is never imported in this trusted process.  Under ``-I`` the harness
    directory is not on ``sys.path``, so ``models.py`` is loaded explicitly by
    path instead.  ``models.py`` is a non-editable file whose hash the edit
    guard verifies before any eval runs, so the copy loaded here is the
    pristine one."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.py")
    spec = importlib.util.spec_from_file_location("_mlsb_fixed_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_model


get_model = _load_get_model()


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
    # Internal: set only by the evaluator when it starts the training worker.
    parser.add_argument("--train-worker-ckpt", type=str, default=None,
                        help=argparse.SUPPRESS)
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


def _attack_loss(outputs, labels, loss_type):
    """Per-sample attack objective (higher = closer to / further past a misclassification)."""
    if loss_type == "ce":
        return F.cross_entropy(outputs, labels, reduction='none')
    if loss_type == "margin":
        # Carlini-Wagner margin: best wrong logit minus true logit.
        true_logit = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
        is_true = F.one_hot(labels, outputs.size(1)).bool()
        other_logit = outputs.masked_fill(is_true, float('-inf')).amax(dim=1)
        return other_logit - true_logit
    raise ValueError(f"Unknown attack loss: {loss_type}")


def _batch_attack_loss(outputs, labels, loss_type):
    if loss_type == "ce":
        return F.cross_entropy(outputs, labels)
    return _attack_loss(outputs, labels, loss_type).sum()


def _input_grad(model, adv_images, labels, loss_type):
    adv_images = adv_images.clone().detach().requires_grad_(True)
    outputs = model(adv_images)
    loss = _batch_attack_loss(outputs, labels, loss_type)
    grad = torch.autograd.grad(loss, adv_images)[0]
    return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


def _correct(outputs, labels):
    """Correct = finite logits and arg-max on the true label."""
    return torch.isfinite(outputs).all(dim=1) & outputs.argmax(dim=1).eq(labels)


def fgsm_attack(model, images, labels, eps, loss_type="ce"):
    """FGSM attack for evaluation."""
    model.eval()
    grad = _input_grad(model, images, labels, loss_type)
    adv_images = images + eps * grad.sign()
    return torch.clamp(adv_images, 0.0, 1.0).detach()


def pgd_attack(model, images, labels, eps, alpha, steps, restarts=1, loss_type="ce"):
    """PGD attack for evaluation with multi-restart support."""
    model.eval()
    best_adv = images.clone().detach()
    best_loss = torch.full((images.size(0),), -float('inf'), device=images.device)

    for _ in range(restarts):
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

        for _ in range(steps):
            grad = _input_grad(model, adv_images, labels, loss_type)
            adv_images = adv_images.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        # Keep best adversarial examples (highest per-sample loss)
        with torch.no_grad():
            outputs = model(adv_images)
            loss_per_sample = _attack_loss(outputs, labels, loss_type)
            # A restart that makes the logits non-finite counts as a success.
            loss_per_sample = torch.nan_to_num(loss_per_sample, nan=float('inf'))
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
            total += labels.size(0)
            correct += _correct(outputs, labels).sum().item()
    return correct / total


def evaluate_robust(model, test_loader, eps, eval_alpha, steps, device, attack_type="pgd", restarts=1):
    """Evaluate robust accuracy under adversarial attack.

    Returns a dict with the accuracy under each attack loss and ``"worst"``, the
    accuracy on images that survive every attack loss (the reported metric).
    """
    model.eval()
    correct = {name: 0 for name in _ATTACK_LOSSES + ("worst",)}
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        survived = torch.ones_like(labels, dtype=torch.bool)
        for loss_type in _ATTACK_LOSSES:
            if attack_type == "pgd":
                adv_images = pgd_attack(model, images, labels, eps, eval_alpha, steps,
                                        restarts, loss_type=loss_type)
            elif attack_type == "fgsm":
                adv_images = fgsm_attack(model, images, labels, eps, loss_type=loss_type)
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

            with torch.no_grad():
                ok = _correct(model(adv_images), labels)
            correct[loss_type] += ok.sum().item()
            survived &= ok
        correct["worst"] += survived.sum().item()
        total += labels.size(0)

    return {name: n / total for name, n in correct.items()}


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_worker(args):
    """Training process: the only one that imports the submitted trainer."""
    from custom_adv_train import AdversarialTrainer

    set_seed(args.seed)
    device = _pick_device()

    # Data
    train_set, _, num_classes = get_datasets(args.dataset, args.data_dir)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
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

    torch.save(model.state_dict(), args.train_worker_ckpt)


def _worker_preexec():
    """Isolate + de-privilege the training worker before it execs.

    New session (so the evaluator can reap the whole tree) and, when running
    as root, drop to ``nobody`` so the agent's trainer cannot write any path a
    trusted process reads or imports (the harness directory, ``site-packages``,
    a ``sitecustomize``/``.pth`` a later ``-I`` process still runs) — only the
    dedicated checkpoint directory below, which the evaluator owns."""
    def _fn():
        os.setsid()
        if os.geteuid() == 0:
            os.setgroups([])
            os.setgid(_UNPRIVILEGED_UID)
            os.setuid(_UNPRIVILEGED_UID)
    return _fn


def run_train_worker(ckpt_path):
    """Start the training worker and relay its output with the metric tag defused.

    The worker is the only process that imports the submitted trainer.  It is
    NOT started with ``-I`` (it needs the harness directory on ``sys.path`` to
    import ``custom_adv_train``), but it is dropped to an unprivileged uid so
    it can write nothing the evaluator later reads except its checkpoint."""
    here = os.path.dirname(os.path.abspath(__file__))
    # Give the unprivileged worker its own scratch HOME/TMPDIR (torch extension
    # / kernel caches want a writable home) and make sure it can read the
    # (root-owned, at verify time) submitted trainer and fixed modules.
    home = tempfile.mkdtemp(prefix="mlsb_advtrain_")
    os.chmod(home, 0o700)
    if os.geteuid() == 0:
        os.chown(home, _UNPRIVILEGED_UID, _UNPRIVILEGED_UID)
        # The checkpoint the worker must write; the evaluator (root) reads it.
        os.chown(os.path.dirname(ckpt_path), _UNPRIVILEGED_UID, _UNPRIVILEGED_UID)
        for name in ("custom_adv_train.py", "models.py"):
            p = os.path.join(here, name)
            try:
                os.chmod(p, os.stat(p).st_mode | 0o444)
            except OSError:
                pass
    # Inherit the (trusted, harness-set) environment so SEED, CUDA and thread
    # budgets reach training unchanged; only redirect HOME/TMPDIR to a scratch
    # dir the unprivileged worker owns.
    env = os.environ.copy()
    env["HOME"] = env["TMPDIR"] = home

    cmd = [sys.executable, "-B", "-u", os.path.abspath(__file__)]
    cmd += sys.argv[1:] + ["--train-worker-ckpt", ckpt_path]
    sys.stdout.flush()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, preexec_fn=_worker_preexec(), close_fds=True,
        )
        out = sys.stdout.buffer
        for line in iter(proc.stdout.readline, b""):
            out.write(line.replace(_METRIC_TAG, b"test_metrics(train-worker)"))
            out.flush()
        proc.stdout.close()
        return proc.wait()
    finally:
        subprocess.run(["rm", "-rf", home], check=False)


def load_trained_model(ckpt_path, arch, num_classes, device):
    """Fresh model of the fixed architecture holding only the trained tensors."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()
    ):
        raise RuntimeError("training checkpoint is not a state_dict of tensors")
    model = get_model(arch, num_classes)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    args = parse_args()
    if args.train_worker_ckpt is not None:
        train_worker(args)
        return

    ckpt_dir = tempfile.mkdtemp(prefix="adv_train_ckpt_")
    try:
        ckpt_path = os.path.join(ckpt_dir, "model.pt")
        rc = run_train_worker(ckpt_path)
        if rc != 0:
            raise SystemExit(f"[Eval] training worker failed with exit code {rc}")
        if not os.path.isfile(ckpt_path):
            raise SystemExit("[Eval] training worker wrote no checkpoint")

        set_seed(args.seed)
        device = _pick_device()
        _, test_set, num_classes = get_datasets(args.dataset, args.data_dir)
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        model = load_trained_model(ckpt_path, args.arch, num_classes, device)
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    # Evaluation
    print("[Eval] Evaluating clean accuracy...", flush=True)
    clean_acc = evaluate_clean(model, test_loader, device)
    print(f"[Eval] clean_acc={clean_acc:.4f}", flush=True)

    print("[Eval] Evaluating FGSM robustness (CE and CW-margin losses)...", flush=True)
    fgsm = evaluate_robust(
        model, test_loader, args.eps, args.eps, 1, device, "fgsm",
    )
    robust_acc_fgsm = fgsm["worst"]
    print(
        f"[Eval] robust_acc_fgsm_ce={fgsm['ce']:.4f} "
        f"robust_acc_fgsm_margin={fgsm['margin']:.4f} "
        f"robust_acc_fgsm={robust_acc_fgsm:.4f}",
        flush=True,
    )

    eval_alpha = args.eval_alpha if args.eval_alpha is not None else args.eps / 4.0
    print(
        f"[Eval] Evaluating PGD-{args.eval_attack_steps} robustness "
        f"(alpha={eval_alpha:.6f}, restarts={args.eval_restarts}, "
        f"CE and CW-margin losses)...",
        flush=True,
    )
    pgd = evaluate_robust(
        model, test_loader, args.eps, eval_alpha, args.eval_attack_steps, device,
        "pgd", restarts=args.eval_restarts,
    )
    robust_acc_pgd = pgd["worst"]
    print(
        f"[Eval] robust_acc_pgd_ce={pgd['ce']:.4f} "
        f"robust_acc_pgd_margin={pgd['margin']:.4f} "
        f"robust_acc_pgd={robust_acc_pgd:.4f}",
        flush=True,
    )

    print(
        f"TEST_METRICS clean_acc={clean_acc:.4f} "
        f"robust_acc_fgsm={robust_acc_fgsm:.4f} "
        f"robust_acc_pgd={robust_acc_pgd:.4f}",
        flush=True,
    )

if __name__ == "__main__":
    main()
