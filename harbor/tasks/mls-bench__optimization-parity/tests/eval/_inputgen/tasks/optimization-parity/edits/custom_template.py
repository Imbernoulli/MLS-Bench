"""Optimization-parity scaffold for MLS-Bench.

The fixed evaluation trains a fixed two-layer MLP to learn hidden sparse parity
functions and asks the agent to control only:
  1. model initialization
  2. training-data construction (selecting/transforming the provided pool)
  3. AdamW hyperparameters

NOTE: The hidden parity secret S and the held-out test labels are NOT part of
this program. The harness pre-generates the (unlabeled) training inputs and
their labels for you and scores your predictions against held-out truth in a
separate host-side process. Your editable hooks only ever see binary inputs and
their labels — never the secret subset S, and never the test labels. The runner
trains your model and emits its predictions on a held-out test set; the host
regenerates the test labels and computes test accuracy.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
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


def predict_on(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Return raw model outputs (sigmoid probabilities) for every row of x."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = start + batch_size
            batch_x = x[start:end].to(device)
            preds = model(batch_x).view(-1)
            outputs.append(preds.detach().cpu())
    return torch.cat(outputs) if outputs else torch.empty(0)


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
# FIXED: held-out input loading (the harness pre-generates these; the
# secret and the test labels are never present in this process)
# =====================================================================
def _inputs_dir() -> str:
    """Directory holding the pre-generated parity inputs for this task."""
    env = os.environ.get("PARITY_INPUTS_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_parity_inputs")


def _config_tag(config: TaskConfig) -> str:
    return f"n{config.n_features}_k{config.secret_size}"


def gen_train_pool_x(config: TaskConfig, train_dataset_seed: int) -> torch.Tensor:
    """Regenerate the full unlabeled training x-pool (no secret involved)."""
    generator = torch.Generator().manual_seed(train_dataset_seed)
    return torch.randint(
        low=0,
        high=2,
        size=(config.max_train_examples, config.n_features),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float32)


def gen_test_x(config: TaskConfig, test_seed: int) -> torch.Tensor:
    """Regenerate the held-out test inputs (no secret; labels are withheld)."""
    generator = torch.Generator().manual_seed(test_seed)
    return torch.randint(
        low=0,
        high=2,
        size=(config.test_set_size, config.n_features),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float32)


def load_train_labels(config: TaskConfig, seed: int, secret_index: int) -> torch.Tensor:
    """Load the bit-packed training-pool labels for one hidden secret.

    Only the labels are provided (the secret that produced them is held out).
    The labels are bit-packed for one row per training example over the full
    ``max_train_examples`` pool; unpack to a float tensor in {0, 1}.
    """
    import numpy as np

    tag = _config_tag(config)
    path = os.path.join(_inputs_dir(), f"{tag}_seed{seed}_s{secret_index}.labels.b64")
    with open(path, "r") as f:
        packed = np.frombuffer(base64.b64decode(f.read()), dtype=np.uint8)
    bits = np.unpackbits(packed)[: config.max_train_examples]
    return torch.from_numpy(bits.astype("float32"))


# =====================================================================
# EDITABLE: init_model, make_dataset, get_optimizer_config
# =====================================================================
def init_model(model: nn.Sequential, config: TaskConfig) -> None:
    """Initialize the fixed two-layer MLP."""
    for layer in model:
        if isinstance(layer, nn.Linear):
            gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)


def make_dataset(
    x_pool: torch.Tensor,
    y_pool: torch.Tensor,
    config: TaskConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the training dataset from the provided labeled pool.

    The harness supplies a large pool of labeled binary examples (x_pool, y_pool)
    drawn from the same distribution as the held-out test set. Select and/or
    transform it however you like; the result must be a binary (x, y) pair (or a
    dict with keys 'x' and 'y'). The hidden parity secret is never exposed.
    """
    num_examples = 4_096
    return x_pool[:num_examples], y_pool[:num_examples]


def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
    """Return AdamW hyperparameters for the fixed training loop."""
    return {
        "lr": 1e-3,
        "wd": 1e-2,
        "beta1": 0.9,
        "beta2": 0.999,
    }


# =====================================================================
# FIXED: training and prediction driver
# =====================================================================
def train_one_run(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    config: TaskConfig,
    device: torch.device,
    run_seed: int,
    order_seed: int,
    secret_index: int,
    order_index: int,
) -> tuple[RunResult, torch.Tensor]:
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

    test_preds = predict_on(model, test_x, device)
    print(
        "RUN_METRICS "
        f"secret={secret_index} order={order_index} steps={steps}",
        flush=True,
    )
    return (
        RunResult(
            secret_index=secret_index,
            order_index=order_index,
            steps=steps,
        ),
        test_preds,
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


def _emit_pred(
    config: TaskConfig,
    seed: int,
    secret_index: int,
    order_index: int,
    test_preds: torch.Tensor,
) -> None:
    """Emit the model's held-out predictions for the host-side scorer.

    Predictions are thresholded at 0.5 (the same threshold the metric uses) and
    bit-packed. We do NOT have the test labels, so we cannot (and do not) compute
    the metric here.
    """
    import numpy as np

    pred_bits = (test_preds.numpy() >= 0.5).astype(np.uint8)
    payload = base64.b64encode(np.packbits(pred_bits).tobytes()).decode("ascii")
    print(
        "PARITY_PRED "
        f"config={_config_tag(config)} seed={seed} secret={secret_index} "
        f"order={order_index} n={int(test_preds.numel())} preds={payload}",
        flush=True,
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

    results: list[RunResult] = []

    for secret_index in range(config.num_hidden_secrets):
        train_dataset_seed = seed * 10_000 + secret_index
        x_pool = gen_train_pool_x(config, train_dataset_seed)
        y_pool = load_train_labels(config, seed, secret_index)
        train_x, train_y = normalize_dataset(
            make_dataset(x_pool, y_pool, config),
            config,
        )
        test_x = gen_test_x(config, seed * 20_000 + secret_index)
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
            result, test_preds = train_one_run(
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                config=config,
                device=device,
                run_seed=run_seed,
                order_seed=order_seed,
                secret_index=secret_index,
                order_index=order_index,
            )
            results.append(result)
            _emit_pred(config, seed, secret_index, order_index, test_preds)

    step_tensor = torch.tensor([result.steps for result in results], dtype=torch.float64)
    print(
        "BENCH_DONE "
        f"num_runs={len(results)} mean_steps={float(step_tensor.mean().item()):.6f}",
        flush=True,
    )
    return {
        "config": asdict(config),
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
