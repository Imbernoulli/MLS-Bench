"""Optimization-parity scaffold for MLS-Bench.

The fixed evaluation samples hidden sparse parity functions and asks the agent
to control only:
  1. model initialization
  2. training-data generation
  3. AdamW hyperparameters
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn


# =====================================================================
# FIXED: Benchmark configuration
# =====================================================================
@dataclass(frozen=True)
class TaskConfig:
    n_features: int = 32
    secret_size: int = 8
    hidden_width: int = 512
    batch_size: int = 128
    max_steps: int = 30_000
    max_train_examples: int = 12_800_000
    num_hidden_secrets: int = 5
    num_orderings: int = 3
    test_set_size: int = 16_384
    log_interval: int = 250
    min_steps_before_stop: int = 1_000
    early_stop_acc: float = 0.999
    early_stop_windows: int = 4


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    wd: float
    beta1: float
    beta2: float


@dataclass(frozen=True)
class RunResult:
    secret_index: int
    order_index: int
    steps: int
    test_accuracy: float


DEFAULT_TASK = TaskConfig()


def build_model(config: TaskConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.n_features, config.hidden_width),
        nn.ReLU(),
        nn.Linear(config.hidden_width, 1),
        nn.Sigmoid(),
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_hidden_secrets(config: TaskConfig, seed: int) -> list[tuple[int, ...]]:
    max_unique = math.comb(config.n_features, config.secret_size)
    if config.num_hidden_secrets > max_unique:
        raise ValueError("Requested more hidden secrets than unique subsets.")

    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    secrets: list[tuple[int, ...]] = []
    while len(secrets) < config.num_hidden_secrets:
        secret = tuple(sorted(rng.sample(range(config.n_features), config.secret_size)))
        if secret not in seen:
            seen.add(secret)
            secrets.append(secret)
    return secrets


def parity_labels(x: torch.Tensor, secret: tuple[int, ...]) -> torch.Tensor:
    secret_index = torch.tensor(secret, dtype=torch.long)
    return (x.index_select(dim=1, index=secret_index).sum(dim=1).remainder(2)).to(
        torch.float32
    )


def make_test_set(
    secret: tuple[int, ...],
    config: TaskConfig,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(
        low=0,
        high=2,
        size=(config.test_set_size, config.n_features),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float32)
    y = parity_labels(x, secret)
    return x, y


def normalize_dataset(
    dataset: object,
    config: TaskConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(dataset, dict):
        if "x" not in dataset or "y" not in dataset:
            raise ValueError("Dataset dict must contain 'x' and 'y'.")
        x, y = dataset["x"], dataset["y"]
    elif isinstance(dataset, (tuple, list)) and len(dataset) == 2:
        x, y = dataset
    else:
        raise TypeError("Dataset must be a (x, y) pair or a dict with keys 'x' and 'y'.")

    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32).view(-1)

    if x.ndim != 2:
        raise ValueError(f"Expected x to have shape [num_examples, n_features], got {tuple(x.shape)}.")
    if x.shape[1] != config.n_features:
        raise ValueError(
            f"Expected x.shape[1] == {config.n_features}, got {x.shape[1]}."
        )
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of examples.")
    if x.shape[0] == 0:
        raise ValueError("Training dataset must contain at least one example.")
    if x.shape[0] > config.max_train_examples:
        raise ValueError(
            f"Training dataset size {x.shape[0]} exceeds limit {config.max_train_examples}."
        )
    if not torch.all((x == 0) | (x == 1)):
        raise ValueError("Training inputs must stay in {0, 1}.")
    if not torch.all((y == 0) | (y == 1)):
        raise ValueError("Training labels must stay in {0, 1}.")
    return x.contiguous(), y.contiguous()


def normalize_optimizer_config(config_dict: dict[str, float]) -> OptimizerConfig:
    required = {"lr", "wd", "beta1", "beta2"}
    missing = required - set(config_dict)
    if missing:
        raise ValueError(f"Missing optimizer hyperparameters: {sorted(missing)}")

    config = OptimizerConfig(
        lr=float(config_dict["lr"]),
        wd=float(config_dict["wd"]),
        beta1=float(config_dict["beta1"]),
        beta2=float(config_dict["beta2"]),
    )
    if not config.lr > 0.0:
        raise ValueError("AdamW learning rate must be positive.")
    if not config.wd >= 0.0:
        raise ValueError("AdamW weight decay must be non-negative.")
    if not 0.0 < config.beta1 < 1.0:
        raise ValueError("AdamW beta1 must satisfy 0 < beta1 < 1.")
    if not 0.0 < config.beta2 < 1.0:
        raise ValueError("AdamW beta2 must satisfy 0 < beta2 < 1.")
    return config


def evaluate_accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = start + batch_size
            batch_x = x[start:end].to(device)
            batch_y = y[start:end].to(device)
            preds = model(batch_x).view(-1)
            correct += ((preds >= 0.5) == (batch_y >= 0.5)).sum().item()
            total += batch_y.numel()
    return correct / max(total, 1)


def maybe_log_final_window(
    secret_index: int,
    order_index: int,
    steps: int,
    window_loss: float,
    window_acc: float,
    window_count: int,
) -> None:
    if window_count == 0:
        return
    print(
        "TRAIN_METRICS "
        f"secret={secret_index} order={order_index} step={steps} "
        f"loss={window_loss / window_count:.6f} acc={window_acc / window_count:.6f}",
        flush=True,
    )


# =====================================================================
# EDITABLE: init_model, make_dataset, get_optimizer_config
# =====================================================================
def init_model(model: nn.Sequential, config: TaskConfig) -> None:
    """Initialize the fixed two-layer MLP without using the hidden secret."""
    for layer in model:
        if isinstance(layer, nn.Linear):
            gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)


def make_dataset(
    secret: tuple[int, ...],
    config: TaskConfig,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a reproducible training dataset for one hidden secret."""
    generator = torch.Generator().manual_seed(seed)
    num_examples = 4_096
    x = torch.randint(
        low=0,
        high=2,
        size=(num_examples, config.n_features),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float32)
    y = parity_labels(x, secret)
    return x, y


def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
    """Return AdamW hyperparameters for the fixed training loop."""
    return {
        "lr": 1e-3,
        "wd": 1e-2,
        "beta1": 0.9,
        "beta2": 0.999,
    }


# =====================================================================
# FIXED: training and evaluation driver
# =====================================================================
def train_one_run(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    config: TaskConfig,
    device: torch.device,
    run_seed: int,
    order_seed: int,
    secret_index: int,
    order_index: int,
) -> RunResult:
    set_global_seed(run_seed)

    model = build_model(config).to(device)
    init_model(model, config)
    optimizer_config = normalize_optimizer_config(get_optimizer_config(config))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.lr,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.wd,
    )
    criterion = nn.BCELoss()

    steps = 0
    stable_windows = 0
    window_loss = 0.0
    window_acc = 0.0
    window_count = 0
    last_logged_step = 0
    permutation_generator = torch.Generator().manual_seed(order_seed)

    while steps < config.max_steps:
        permutation = torch.randperm(train_x.shape[0], generator=permutation_generator)
        for start in range(0, train_x.shape[0], config.batch_size):
            batch_indices = permutation[start : start + config.batch_size]
            batch_x = train_x.index_select(0, batch_indices).to(device)
            batch_y = train_y.index_select(0, batch_indices).to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_x).view(-1)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            batch_acc = ((preds >= 0.5) == (batch_y >= 0.5)).float().mean().item()
            window_loss += loss.item()
            window_acc += batch_acc
            window_count += 1
            steps += 1

            should_log = steps == 1 or steps % config.log_interval == 0 or steps == config.max_steps
            if should_log:
                avg_loss = window_loss / window_count
                avg_acc = window_acc / window_count
                print(
                    "TRAIN_METRICS "
                    f"secret={secret_index} order={order_index} step={steps} "
                    f"loss={avg_loss:.6f} acc={avg_acc:.6f}",
                    flush=True,
                )
                last_logged_step = steps
                if steps >= config.min_steps_before_stop and avg_acc >= config.early_stop_acc:
                    stable_windows += 1
                else:
                    stable_windows = 0
                window_loss = 0.0
                window_acc = 0.0
                window_count = 0
                if stable_windows >= config.early_stop_windows:
                    break

            if steps >= config.max_steps:
                break
        if stable_windows >= config.early_stop_windows or steps >= config.max_steps:
            break

    if last_logged_step != steps:
        maybe_log_final_window(
            secret_index=secret_index,
            order_index=order_index,
            steps=steps,
            window_loss=window_loss,
            window_acc=window_acc,
            window_count=window_count,
        )

    test_accuracy = evaluate_accuracy(model, test_x, test_y, device)
    print(
        "RUN_METRICS "
        f"secret={secret_index} order={order_index} steps={steps} "
        f"test_accuracy={test_accuracy:.6f}",
        flush=True,
    )
    return RunResult(
        secret_index=secret_index,
        order_index=order_index,
        steps=steps,
        test_accuracy=test_accuracy,
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_apply_smoke_mode(config: TaskConfig, enabled: bool) -> TaskConfig:
    if not enabled:
        return config
    return replace(
        config,
        num_hidden_secrets=2,
        num_orderings=2,
        test_set_size=2_048,
        max_steps=4_000,
        log_interval=100,
        min_steps_before_stop=400,
        early_stop_windows=3,
    )


def run_benchmark(
    config: TaskConfig,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    print(
        "TASK_CONFIG "
        + " ".join(
            [
                f"N={config.n_features}",
                f"K={config.secret_size}",
                f"W={config.hidden_width}",
                f"num_hidden_secrets={config.num_hidden_secrets}",
                f"num_orderings={config.num_orderings}",
                f"test_set_size={config.test_set_size}",
                f"batch_size={config.batch_size}",
                f"max_steps={config.max_steps}",
            ]
        ),
        flush=True,
    )

    secrets = sample_hidden_secrets(config, seed + 17)
    results: list[RunResult] = []

    for secret_index, secret in enumerate(secrets):
        train_dataset_seed = seed * 10_000 + secret_index
        train_x, train_y = normalize_dataset(
            make_dataset(secret, config, train_dataset_seed),
            config,
        )
        test_x, test_y = make_test_set(
            secret=secret,
            config=config,
            seed=seed * 20_000 + secret_index,
        )
        positive_rate = float(train_y.mean().item())
        print(
            "DATASET_METRICS "
            f"secret={secret_index} num_examples={train_x.shape[0]} "
            f"positive_rate={positive_rate:.6f}",
            flush=True,
        )

        for order_index in range(config.num_orderings):
            run_seed = seed * 1_000_000 + secret_index * 1_000 + order_index
            order_seed = seed * 2_000_000 + secret_index * 1_000 + order_index
            results.append(
                train_one_run(
                    train_x=train_x,
                    train_y=train_y,
                    test_x=test_x,
                    test_y=test_y,
                    config=config,
                    device=device,
                    run_seed=run_seed,
                    order_seed=order_seed,
                    secret_index=secret_index,
                    order_index=order_index,
                )
            )

    accuracy_tensor = torch.tensor([result.test_accuracy for result in results], dtype=torch.float64)
    step_tensor = torch.tensor([result.steps for result in results], dtype=torch.float64)
    final_metrics = {
        "test_accuracy": float(accuracy_tensor.mean().item()),
        "score": float(accuracy_tensor.mean().item()),
        "test_accuracy_std": float(accuracy_tensor.std(unbiased=False).item()),
        "mean_steps": float(step_tensor.mean().item()),
        "num_runs": int(len(results)),
    }
    print(
        "FINAL_METRICS "
        + " ".join(
            f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in final_metrics.items()
        ),
        flush=True,
    )
    # Also print TEST_METRICS for framework compatibility
    print(
        f"TEST_METRICS test_accuracy={final_metrics['test_accuracy']:.6f} "
        f"score={final_metrics['score']:.6f}",
        flush=True,
    )
    return {
        "config": asdict(config),
        "metrics": final_metrics,
        "results": [asdict(result) for result in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MLS-Bench optimization-parity task.")
    parser.add_argument("--seed", type=int, default=42, help="Top-level benchmark seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for a JSON summary.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="eval",
        help="Optional label stored in the JSON summary.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a smaller local sanity check without changing the benchmark defaults in code.",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=None,
        help="Override n_features in TaskConfig.",
    )
    parser.add_argument(
        "--secret-size",
        type=int,
        default=None,
        help="Override secret_size in TaskConfig.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = maybe_apply_smoke_mode(DEFAULT_TASK, args.smoke)
    if args.n_features is not None:
        config = replace(config, n_features=args.n_features)
    if args.secret_size is not None:
        config = replace(config, secret_size=args.secret_size)
    device = resolve_device(args.device)
    summary = run_benchmark(config=config, seed=args.seed, device=device)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"{args.label}_seed{args.seed}.json"
        output_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
