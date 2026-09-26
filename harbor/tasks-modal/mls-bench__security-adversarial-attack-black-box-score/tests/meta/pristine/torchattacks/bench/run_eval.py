"""Trusted evaluation harness for score-based query black-box attack task.

The submitted attack never runs in this process.  ``run_attack`` is imported
and called only in a separate worker interpreter (this same file, started with
``--attack-worker-fd``), and that worker holds no classifier.  The ``model``
it hands to ``run_attack`` is a stub whose every call ships the query batch to
this process over a socket and gets back a fresh copy of the logits.  The
classifier, the query counter and the metric line therefore live only here:
there is no underlying module to reach through the wrapper, no gradient path,
and no counter the attack could reset.  Nothing the worker sends is unpickled
(queries and adversarial images arrive as raw float32 buffers), and the
worker's output is relayed with the metric tag defused, so the only
``ATTACK_METRICS`` line is the one printed below.

Process isolation (why the worker cannot forge the score):

* This trusted process is started by the eval script with ``python -I``
  (isolated mode).  Under ``-I`` the interpreter puts neither the script's
  directory (``bench/``, which is agent-writable) nor the user site nor any
  ``PYTHON*`` environment on ``sys.path``, so a stdlib-shadowing file the
  agent creates next to this script (e.g. ``bench/argparse.py``) is never
  imported here and cannot run in the trusted process.
* The worker is spawned with ``preexec_fn`` dropping it to an unprivileged
  uid (``nobody``) when this process runs as root, so agent code cannot
  write ``bench/`` (a shadow module a later trusted process would import),
  ``site-packages`` (a ``.pth``/``sitecustomize`` a later ``-I`` process
  still runs) or the victim checkpoint cache.
* Before the worker starts, the victim-weight cache (``TORCH_HOME``) is made
  unreadable to any other uid, so the unprivileged worker cannot reload the
  served checkpoints from disk and mount a white-box attack.  (The victims
  are still public checkpoints; see the task description.)
"""

import argparse
import atexit
import os
import random
import signal
import socket
import struct
import subprocess
import sys
import threading
import tempfile
from multiprocessing.connection import Connection

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

_WORKER_FLAG = "--attack-worker-fd"
_METRIC_TAG = b"ATTACK_METRICS"
_UNPRIVILEGED_UID = 65534  # nobody


def _lock_victim_cache() -> None:
    """Make the torch.hub victim-weight cache unreadable to other uids.

    The trusted process has already loaded the victims; from here on only it
    (root) needs the files.  Removing group/other permissions stops the
    unprivileged worker from reloading the exact served checkpoints and
    running a white-box attack.  Best-effort: a failure never aborts the run,
    it only leaves the pre-existing (already public) exposure in place."""
    roots = []
    th = os.environ.get("TORCH_HOME")
    if th:
        roots.append(th)
    try:
        roots.append(torch.hub.get_dir())
    except Exception:
        pass
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for name in [dirpath] + [os.path.join(dirpath, f) for f in filenames]:
                try:
                    mode = os.stat(name).st_mode
                    os.chmod(name, mode & ~0o077)
                except OSError:
                    pass


def _worker_preexec():
    """Isolate + de-privilege the attack worker before it execs.

    New session (so the trusted process can reap the whole tree) and, when
    running as root, drop to ``nobody`` so agent code cannot write any path a
    trusted process reads or imports, nor the victim cache."""
    def _fn():
        os.setsid()
        if os.geteuid() == 0:
            os.setgroups([])
            os.setgid(_UNPRIVILEGED_UID)
            os.setuid(_UNPRIVILEGED_UID)
    return _fn


def _encode_array(kind: bytes, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    head = struct.pack("<cB", kind, arr.ndim)
    head += struct.pack(f"<{arr.ndim}q", *arr.shape)
    return head + arr.tobytes()


def _decode_array(buf: bytes) -> tuple[bytes, np.ndarray]:
    """Parse a worker message without unpickling anything the worker wrote."""
    if len(buf) < 2:
        raise RuntimeError("malformed message from attack worker")
    kind, ndim = struct.unpack_from("<cB", buf, 0)
    offset = 2 + 8 * ndim
    if ndim > 8 or len(buf) < offset:
        raise RuntimeError("malformed message from attack worker")
    shape = struct.unpack_from(f"<{ndim}q", buf, 2)
    if any(d < 0 for d in shape) or len(buf) - offset != 4 * int(np.prod(shape, dtype=np.int64)):
        raise RuntimeError("malformed message from attack worker")
    arr = np.frombuffer(buf, dtype=np.float32, offset=offset).reshape(shape).copy()
    return kind, arr


class QueryLimitedBlackBox(torch.nn.Module):
    """What ``run_attack`` receives: ``model(x)`` returns logits, nothing else.

    Lives in the worker process, which has no classifier.  Each call sends
    ``x`` to the evaluation process, which counts ``x.shape[0]`` queries
    against the batch budget and answers with the logits (all zeros once the
    budget is exceeded, which also fails the whole batch).  The result is a
    new tensor with no autograd history.
    """

    def __init__(self, conn: Connection, device: torch.device):
        super().__init__()
        # Zero-size, frozen: only so tools that read the device from
        # next(model.parameters()) (e.g. torchattacks.Attack) keep working.
        self._device_anchor = torch.nn.Parameter(
            torch.zeros(0, device=device), requires_grad=False
        )
        self._conn = conn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x) or x.dim() != 4:
            raise ValueError("the black-box model takes a 4-D tensor of images")
        self._conn.send_bytes(
            _encode_array(b"Q", x.detach().to("cpu", torch.float32).numpy())
        )
        kind, payload = self._conn.recv()
        if kind != "L":
            raise RuntimeError(payload)
        return torch.from_numpy(payload).to(x.device)


def _attack_worker(fd: int) -> None:
    conn = Connection(fd)
    cfg = conn.recv()
    from custom_attack import run_attack

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    while True:
        msg = conn.recv()
        if msg[0] == "done":
            return
        _, images_np, labels_np = msg
        images = torch.from_numpy(images_np).to(device)
        labels = torch.from_numpy(labels_np).to(device)
        query_model = QueryLimitedBlackBox(conn, device)
        adv_images = run_attack(
            query_model,
            images,
            labels,
            cfg["eps"],
            cfg["n_queries"],
            device,
            cfg["n_classes"],
        )
        if not torch.is_tensor(adv_images):
            raise TypeError("run_attack must return a tensor")
        conn.send_bytes(
            _encode_array(b"R", adv_images.detach().to("cpu", torch.float32).numpy())
        )


def _relay(stream) -> None:
    out = sys.stdout.buffer
    for line in iter(stream.readline, b""):
        out.write(line.replace(_METRIC_TAG, b"attack_metrics(worker)"))
        out.flush()


def _start_worker(cfg: dict) -> tuple[subprocess.Popen, Connection, threading.Thread]:
    # The worker imports the agent's custom_attack.py from bench/ (its script
    # dir, which stays on sys.path because the worker is NOT started with -I);
    # make sure the unprivileged worker can read it and its own scratch HOME.
    worker_module = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "custom_attack.py")
    home = tempfile.mkdtemp(prefix="mlsb_attack_")
    os.chmod(home, 0o700)
    if os.geteuid() == 0:
        os.chown(home, _UNPRIVILEGED_UID, _UNPRIVILEGED_UID)
        try:
            os.chmod(worker_module, os.stat(worker_module).st_mode | 0o444)
        except OSError:
            pass
    # Inherit the (trusted, harness-set) environment so SEED, CUDA and thread
    # budgets reach the attack unchanged; only redirect HOME/TMPDIR to a
    # scratch dir the unprivileged worker owns.
    env = os.environ.copy()
    env["HOME"] = env["TMPDIR"] = home

    parent_sock, child_sock = socket.socketpair()
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.abspath(__file__), _WORKER_FLAG, str(child_sock.fileno())],
        pass_fds=(child_sock.fileno(),),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=_worker_preexec(),
        close_fds=True,
    )
    child_sock.close()
    atexit.register(_kill_group, proc.pid)
    atexit.register(lambda: subprocess.run(["rm", "-rf", home], check=False))
    relay = threading.Thread(target=_relay, args=(proc.stdout,), daemon=True)
    relay.start()
    conn = Connection(parent_sock.detach())
    conn.send(cfg)
    return proc, conn, relay


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass


def _stop_worker(proc: subprocess.Popen, conn: Connection, relay: threading.Thread) -> None:
    try:
        conn.send(("done",))
    except OSError:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass
    _kill_group(proc.pid)
    proc.wait()
    relay.join(timeout=10)
    conn.close()


def _serve_batch(
    conn: Connection,
    model: torch.nn.Module,
    images_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    n_classes: int,
    max_queries: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Answer the worker's queries for one batch; return (adv_images, queries_used)."""
    conn.send(("batch", images_cpu.numpy(), labels_cpu.numpy()))
    queries_used = 0
    while True:
        try:
            buf = conn.recv_bytes()
        except (EOFError, OSError):
            raise RuntimeError(
                "attack worker exited before returning adversarial images (see its output above)"
            ) from None
        kind, arr = _decode_array(buf)
        if kind == b"R":
            return torch.from_numpy(arr).to(device), queries_used
        if kind != b"Q":
            raise RuntimeError("malformed message from attack worker")
        batch = int(arr.shape[0]) if arr.ndim else 0
        queries_used += batch
        if queries_used > max_queries:
            conn.send(("L", np.zeros((batch, n_classes), dtype=np.float32)))
            continue
        try:
            with torch.no_grad():
                logits = model(torch.from_numpy(arr).to(device))
            conn.send(("L", logits.detach().float().cpu().numpy()))
        except Exception as exc:  # bad query shape etc.: raised inside the attack
            conn.send(("X", f"black-box query failed: {type(exc).__name__}: {exc}"))


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

    # Victims are loaded; deny the unprivileged worker any on-disk path to the
    # exact served checkpoints before it starts.
    _lock_victim_cache()

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

    worker, conn, relay = _start_worker(
        {
            "seed": int(args.seed),
            "device": str(device),
            "eps": float(args.eps),
            "n_queries": int(args.n_queries),
            "n_classes": int(n_classes),
        }
    )

    for images_cpu, labels_cpu in eval_loader:
        images = images_cpu.to(device)
        labels = labels_cpu.to(device)
        batch_size = int(images.shape[0])
        max_queries = batch_size * int(args.n_queries)

        adv_images, queries_used = _serve_batch(
            conn,
            clean_model,
            images_cpu,
            labels_cpu,
            n_classes,
            max_queries,
            device,
        )

        total_queries += min(queries_used, max_queries)

        if adv_images.shape != images.shape:
            raise RuntimeError(
                f"run_attack returned wrong shape: got {tuple(adv_images.shape)}, "
                f"expected {tuple(images.shape)}"
            )

        # Query budget exhaustion — entire batch treated as failure.
        if queries_used > max_queries:
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

    _stop_worker(worker, conn, relay)

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
    if len(sys.argv) == 3 and sys.argv[1] == _WORKER_FLAG:
        _attack_worker(int(sys.argv[2]))
    else:
        main()
