# Custom federated learning aggregation strategy for MLS-Bench
#
# EDITABLE section: ServerAggregator class (aggregate method + helpers).
# FIXED sections: everything else (config, data partitioning, client training,
#                 FL simulation loop, evaluation).
import argparse
import copy
import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset


# =====================================================================
# FIXED: Configuration
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "femnist", "shakespeare"])
    parser.add_argument("--data-dir", type=str, default="/data")
    parser.add_argument("--num-clients", type=int, default=100,
                        help="Total number of clients")
    parser.add_argument("--clients-per-round", type=int, default=10,
                        help="Number of clients sampled per round")
    parser.add_argument("--num-rounds", type=int, default=200,
                        help="Number of communication rounds")
    parser.add_argument("--local-epochs", type=int, default=5,
                        help="Number of local training epochs per round")
    parser.add_argument("--local-lr", type=float, default=0.01,
                        help="Local SGD learning rate")
    parser.add_argument("--local-batch-size", type=int, default=64,
                        help="Local training batch size")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.1,
                        help="Dirichlet concentration for non-IID split (CIFAR-10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Evaluate global model every N rounds")
    return parser.parse_args()


# =====================================================================
# FIXED: Models
# =====================================================================
class CifarCNN(nn.Module):
    """Simple CNN for CIFAR-10 (used in FedAvg/FedProx literature)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class FemnistCNN(nn.Module):
    """CNN for FEMNIST (62 classes: digits + upper + lower)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CharLSTM(nn.Module):
    """Character-level LSTM for Shakespeare next-char prediction."""

    def __init__(self, vocab_size=80, embed_dim=8, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        # x: (batch, seq_len) of char indices
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        logits = self.fc(out)  # (batch, seq_len, vocab_size)
        return logits


# =====================================================================
# FIXED: Dataset loading and non-IID partitioning
# =====================================================================
class ShakespeareCharDataset(Dataset):
    """Character-level Shakespeare dataset for next-char prediction."""

    def __init__(self, text, seq_len=80, char2idx=None):
        self.seq_len = seq_len
        if char2idx is not None:
            self.char2idx = char2idx
        else:
            chars = sorted(set(text))
            self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        self.data = [self.char2idx[c] for c in text if c in self.char2idx]

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def load_cifar10(data_dir):
    """Load CIFAR-10 using torchvision."""
    import torchvision
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465),
                                                      (0.2023, 0.1994, 0.2010))])
    train_set = torchvision.datasets.CIFAR10(
        os.path.join(data_dir, "cifar10"), train=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(
        os.path.join(data_dir, "cifar10"), train=False, transform=transform)
    return train_set, test_set


def load_femnist(data_dir):
    """Load EMNIST ByClass split (simulates FEMNIST)."""
    import torchvision
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.EMNIST(
        os.path.join(data_dir, "emnist"), split="byclass",
        train=True, transform=transform)
    test_set = torchvision.datasets.EMNIST(
        os.path.join(data_dir, "emnist"), split="byclass",
        train=False, transform=transform)
    return train_set, test_set


def load_shakespeare(data_dir):
    """Load Shakespeare text and split by character/speaker."""
    text_path = os.path.join(data_dir, "shakespeare", "input.txt")
    with open(text_path, "r") as f:
        text = f.read()
    return text


def dirichlet_partition(targets, num_clients, alpha, seed=42):
    """Partition dataset indices using Dirichlet distribution for non-IID split."""
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)
        splits = np.split(class_indices, split_points[:-1])
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    # Shuffle each client's data
    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def shakespeare_partition(text, num_clients, seed=42):
    """Partition Shakespeare text by speaker (naturally non-IID).

    Falls back to chunk-based partitioning if parsing fails.
    """
    rng = np.random.default_rng(seed)
    # Split by speaker blocks (lines starting with all-caps name followed by colon)
    import re
    blocks = re.split(r'\n(?=[A-Z][A-Z ]+:)', text)
    blocks = [b for b in blocks if len(b.strip()) > 100]

    if len(blocks) < num_clients:
        # Fallback: chunk-based
        chunk_size = len(text) // num_clients
        return [text[i * chunk_size:(i + 1) * chunk_size] for i in range(num_clients)]

    rng.shuffle(blocks)
    client_texts = [""] * num_clients
    for i, block in enumerate(blocks):
        client_texts[i % num_clients] += block

    return client_texts


def prepare_data(args):
    """Prepare dataset, partition among clients, return client datasets + test set."""
    if args.dataset == "cifar10":
        train_set, test_set = load_cifar10(args.data_dir)
        targets = train_set.targets
        client_indices = dirichlet_partition(
            targets, args.num_clients, args.dirichlet_alpha, args.seed)
        client_datasets = [Subset(train_set, idx) for idx in client_indices]
        model_fn = CifarCNN
        loss_fn = nn.CrossEntropyLoss()
        return client_datasets, test_set, model_fn, loss_fn

    elif args.dataset == "femnist":
        train_set, test_set = load_femnist(args.data_dir)
        targets = train_set.targets.numpy()
        client_indices = dirichlet_partition(
            targets, args.num_clients, args.dirichlet_alpha, args.seed)
        client_datasets = [Subset(train_set, idx) for idx in client_indices]
        model_fn = FemnistCNN
        loss_fn = nn.CrossEntropyLoss()
        return client_datasets, test_set, model_fn, loss_fn

    elif args.dataset == "shakespeare":
        text = load_shakespeare(args.data_dir)
        client_texts = shakespeare_partition(text, args.num_clients, args.seed)
        # Create per-client datasets
        full_ds = ShakespeareCharDataset(text)
        vocab_size = full_ds.vocab_size
        char2idx = full_ds.char2idx
        client_datasets = [ShakespeareCharDataset(t, char2idx=char2idx)
                           for t in client_texts if len(t) > 100]
        # Pad to num_clients if needed
        while len(client_datasets) < args.num_clients:
            client_datasets.append(client_datasets[-1])
        # Test set: last 10% of full text
        split_pt = int(len(text) * 0.9)
        test_ds = ShakespeareCharDataset(text[split_pt:], char2idx=char2idx)
        model_fn = lambda: CharLSTM(vocab_size=vocab_size)
        loss_fn = nn.CrossEntropyLoss()
        return client_datasets, test_ds, model_fn, loss_fn

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


# =====================================================================
# FIXED: default helpers shared by every FL Strategy
# =====================================================================
def _default_client_sgd(model, loader, loss_fn, local_epochs, local_lr, device,
                        loss_aug=None):
    """Plain SGD loop used by the default Strategy. ``loss_aug`` optionally
    adds a term to each mini-batch loss (e.g. FedProx's proximal term)."""
    optimizer = optim.SGD(model.parameters(), lr=local_lr)
    total_loss, total_samples = 0.0, 0
    for _ in range(local_epochs):
        for batch_data in loader:
            if len(batch_data) != 2:
                continue
            inputs, targets = batch_data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.dim() == 3:
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            loss = loss_fn(outputs, targets)
            if loss_aug is not None:
                loss = loss + loss_aug(model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / max(total_samples, 1), total_samples


# =====================================================================
# FIXED: Evaluation
# =====================================================================
@torch.no_grad()
def evaluate_global_model(model, test_set, loss_fn, device, batch_size=256):
    """Evaluate the global model on the test set; returns (loss, accuracy)."""
    model.eval()
    model.to(device)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_data in loader:
        inputs, targets = batch_data
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        if outputs.dim() == 3:
            # Shakespeare: flatten for loss and accuracy
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)
            loss = loss_fn(outputs_flat, targets_flat)
            preds = outputs_flat.argmax(dim=-1)
            total_correct += (preds == targets_flat).sum().item()
            total_samples += targets_flat.numel()
        else:
            loss = loss_fn(outputs, targets)
            preds = outputs.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

        total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    model.cpu()
    return avg_loss, accuracy


# =====================================================================
# EDITABLE: FL Strategy — owns BOTH client-side and server-side logic
# =====================================================================
class Strategy:
    """End-to-end FL strategy.

    Unlike a pure ServerAggregator, this class is responsible for the FULL
    FL recipe: how each client trains locally AND how the server aggregates.
    This matches the scope of real FL methods (FedProx / SCAFFOLD / FedDyn /
    MOON / ...) whose innovations live inside the local loop — a server-only
    API can only approximate them.

    The run_fl_simulation loop calls:
        strategy = Strategy(global_model, args)
        selected = strategy.select_clients(N, K, round_num)
        # for each client i:
        state_i, n_i, loss_i = strategy.client_local_train(
            global_state, client_dataset, model_fn, loss_fn,
            local_epochs, local_lr, local_batch_size, device, client_idx)
        global_state = strategy.aggregate(global_state, [(state_i, n_i, loss_i)], round_num)

    You SHOULD override at least ``client_local_train`` and/or ``aggregate``
    to implement your method. Sensible defaults (plain SGD + weighted
    average) are provided so trivial subclasses still run.

    Innovation space:
        * client-side: proximal regularization (FedProx), per-step
          control-variate correction (SCAFFOLD), dynamic regularizer
          (FedDyn), contrastive loss (MOON), adaptive local LR, ...
        * server-side: sample-weighted / robust / learning-rate-adapted
          aggregation, server momentum, drift estimation, ...
    """

    def __init__(self, global_model, args):
        """Initialize strategy state.

        Args:
            global_model: the initial global nn.Module.
            args: parsed CLI args (num_clients, clients_per_round, ...).
        """
        self.args = args

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        """Train one client locally.

        Must return ``(state_dict, num_samples, avg_loss)``. The default
        implementation does plain SGD (no momentum). Override to add
        proximal / correction / regularizer terms to the local objective
        or to modify the optimizer / gradient flow.
        """
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)
        avg_loss, _ = _default_client_sgd(
            model, loader, loss_fn, local_epochs, local_lr, device)
        return model.cpu().state_dict(), len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
        """Aggregate client updates into a new global state_dict.

        Default is sample-count-weighted average (FedAvg). Override for
        server-side innovations (momentum, robust median, control-variate
        server update, ...).
        """
        total_samples = sum(max(upd[1], 1) for upd in client_updates)
        new_state = OrderedDict()
        for key, ref in global_state_dict.items():
            if not ref.is_floating_point():
                new_state[key] = client_updates[0][0][key].detach().clone()
                continue
            acc = torch.zeros_like(ref, device="cpu", dtype=torch.float32)
            for st, n, _ in client_updates:
                acc += st[key].detach().cpu().float() * (max(n, 1) / total_samples)
            new_state[key] = acc.to(ref.dtype)
        return new_state

    def select_clients(self, num_available, num_to_select, round_num):
        """Pick client indices for this round. Default: uniform random."""
        return random.sample(range(num_available), min(num_to_select, num_available))


# =====================================================================
# FIXED: FL Simulation Loop
# =====================================================================
def run_fl_simulation(args):
    """Main federated learning simulation."""
    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Prepare data
    print(f"Loading dataset: {args.dataset}", flush=True)
    client_datasets, test_set, model_fn, loss_fn = prepare_data(args)
    print(f"Number of clients: {len(client_datasets)}", flush=True)

    # Initialize global model
    global_model = model_fn()
    global_state = copy.deepcopy(global_model.state_dict())

    # Initialize FL strategy (owns both client-side and server-side logic)
    strategy = Strategy(global_model, args)

    best_accuracy = 0.0
    start_time = time.time()

    for round_num in range(args.num_rounds):
        round_start = time.time()

        # Client selection
        selected = strategy.select_clients(
            len(client_datasets), args.clients_per_round, round_num)

        # Local training (simulated sequentially)
        client_updates = []
        round_loss = 0.0
        for client_idx in selected:
            updated_state, n_samples, avg_loss = strategy.client_local_train(
                global_state, client_datasets[client_idx],
                model_fn, loss_fn,
                args.local_epochs, args.local_lr,
                args.local_batch_size, device,
                client_idx)
            client_updates.append((updated_state, n_samples, avg_loss))
            round_loss += avg_loss

        avg_round_loss = round_loss / len(selected)

        # Server aggregation
        global_state = strategy.aggregate(global_state, client_updates, round_num)

        # Log training metrics
        round_time = time.time() - round_start
        if (round_num + 1) % 5 == 0 or round_num == 0:
            print(f"TRAIN_METRICS round={round_num+1} avg_loss={avg_round_loss:.4f} "
                  f"round_time={round_time:.1f}s", flush=True)

        # Periodic evaluation
        if (round_num + 1) % args.eval_every == 0 or round_num == args.num_rounds - 1:
            global_model.load_state_dict(global_state)
            test_loss, test_acc = evaluate_global_model(
                global_model, test_set, loss_fn, device)
            elapsed = time.time() - start_time
            print(f"EVAL round={round_num+1} test_loss={test_loss:.4f} "
                  f"test_accuracy={test_acc:.4f} elapsed={elapsed:.0f}s", flush=True)
            if test_acc > best_accuracy:
                best_accuracy = test_acc

    # Final evaluation
    global_model.load_state_dict(global_state)
    test_loss, test_acc = evaluate_global_model(global_model, test_set, loss_fn, device)
    print(f"TEST_METRICS test_accuracy={test_acc:.4f} test_loss={test_loss:.4f} "
          f"best_accuracy={best_accuracy:.4f}", flush=True)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "dataset": args.dataset,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "best_accuracy": best_accuracy,
        "num_rounds": args.num_rounds,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return test_acc, best_accuracy


if __name__ == "__main__":
    args = parse_args()
    run_fl_simulation(args)
