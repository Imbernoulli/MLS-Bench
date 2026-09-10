#!/usr/bin/env python3
"""Lightweight GPU-aware task scheduler for MLS-Bench.

Manages a queue of mlsbench commands (agent/baseline), assigns GPUs based on
compute requirements, and limits concurrency to available GPU resources.

Usage:
    # Schedule tasks interactively
    python scripts/scheduler.py add agent rl-offline-cql --model claude-sonnet-4-6
    python scripts/scheduler.py add baseline rl-offline-continuous --name default
    python scripts/scheduler.py list
    python scripts/scheduler.py start

    # Batch-add all tasks for a domain
    python scripts/scheduler.py batch agent pde --model claude-sonnet-4-6

    # Specify GPUs and config
    python scripts/scheduler.py start --gpus 0,1,2,3 --config configs/config.yaml

    # Monitor
    python scripts/scheduler.py status
    python scripts/scheduler.py cancel <job_id>
"""

import argparse
from contextlib import contextmanager
import fcntl
from functools import lru_cache
import json
import math
import os
import signal
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCHED_DIR_OVERRIDE = os.environ.get("MLSBENCH_SCHEDULER_DIR", "").strip()
if _SCHED_DIR_OVERRIDE:
    SCHEDULER_DIR = Path(_SCHED_DIR_OVERRIDE).expanduser().resolve()
else:
    SCHEDULER_DIR = PROJECT_ROOT / ".scheduler"
QUEUE_FILE = SCHEDULER_DIR / "queue.json"
STATUS_FILE = SCHEDULER_DIR / "status.json"
PID_FILE = SCHEDULER_DIR / "scheduler.pid"
LOG_DIR = SCHEDULER_DIR / "logs"
LOCK_FILE = SCHEDULER_DIR / "queue.lock"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id: int
    command: str          # "agent" or "baseline"
    task: str
    args: list[str]       # extra CLI args (--model, --name, etc.)
    config: str
    gpus_needed: int      # number of GPUs needed (default 1)
    min_gpus_needed: int = 1  # minimum GPUs required to make forward progress
    state: str = "queued" # queued, running, completed, failed, cancelled
    gpus: str = ""        # assigned GPU indices, e.g. "0,1"
    pid: int = 0
    started_at: str = ""
    finished_at: str = ""
    exit_code: int = -1
    log_file: str = ""
    timeout_secs: int = 0  # max wall-time in seconds; 0 = no limit

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Queue persistence
# ---------------------------------------------------------------------------

def _ensure_dirs():
    SCHEDULER_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)


def _load_queue_unlocked() -> list[Job]:
    if not QUEUE_FILE.exists():
        return []
    with open(QUEUE_FILE) as f:
        return [Job.from_dict(d) for d in json.load(f)]


def _save_queue_unlocked(jobs: list[Job]):
    tmp = QUEUE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump([j.to_dict() for j in jobs], f, indent=2)
    tmp.rename(QUEUE_FILE)


@contextmanager
def _queue_transaction():
    """Lock the queue file, load jobs, and persist changes on exit."""
    _ensure_dirs()
    with open(LOCK_FILE, "a+") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        jobs = _load_queue_unlocked()
        try:
            yield jobs
        finally:
            _save_queue_unlocked(jobs)
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _load_queue() -> list[Job]:
    with _queue_transaction() as jobs:
        return [Job.from_dict(j.to_dict()) for j in jobs]


def _save_queue(jobs: list[Job]):
    with _queue_transaction() as locked_jobs:
        locked_jobs[:] = jobs


def _requeue_stale_running_jobs(
    jobs: list[Job],
    tracked_job_ids: set[int] | None = None,
) -> list[int]:
    """Convert dead running jobs back to queued so the scheduler can recover."""
    tracked_job_ids = tracked_job_ids or set()
    requeued: list[int] = []
    for job in jobs:
        if job.state != "running":
            continue
        if job.job_id in tracked_job_ids:
            continue
        if job.pid and _job_alive(job.pid):
            continue
        job.state = "queued"
        job.gpus = ""
        job.pid = 0
        job.started_at = ""
        job.finished_at = ""
        job.exit_code = -1
        requeued.append(job.job_id)
    return requeued


def _next_id(jobs: list[Job]) -> int:
    return max((j.job_id for j in jobs), default=0) + 1


def _read_scheduler_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_exists(pid: int) -> bool:
    """Best-effort liveness check that tolerates restricted signal permissions."""
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        probe = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return probe.returncode == 0 and bool(probe.stdout.strip())
    except OSError:
        return False


def _process_group_alive(pgid: int) -> bool:
    """Check if any process in the process group *pgid* is still alive.

    Only meaningful when the job was started with ``start_new_session=True``
    (so pid == pgid).  For older jobs where the pid is NOT a group leader,
    ``os.killpg`` raises ESRCH and we return False — a safe fallback.
    """
    try:
        os.killpg(pgid, 0)
        return True
    except PermissionError:
        return True          # group exists but we lack permission
    except OSError:          # ESRCH — no such process group
        return False


def _job_alive(pid: int) -> bool:
    """Check whether a scheduler-launched job (or any of its descendants) is alive."""
    if _pid_exists(pid):
        return True
    # The direct leader may have exited but children in its process group
    # can still be running (only valid for start_new_session jobs).
    return _process_group_alive(pid)


def _collect_descendant_pids(root_pid: int) -> set[int]:
    """Return descendants of *root_pid* using the host process table."""
    try:
        probe = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return set()
    if probe.returncode != 0:
        return set()

    children: dict[int, list[int]] = {}
    for line in probe.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)

    descendants: set[int] = set()
    stack = list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        stack.extend(children.get(pid, []))
    return descendants


def _signal_pids(pids: set[int], sig: int):
    """Best-effort signal delivery to individual PIDs."""
    for pid in sorted(pids, reverse=True):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
        except OSError:
            pass


def _wait_for_pids_to_exit(pids: set[int], timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(_pid_exists(pid) for pid in pids):
            return True
        time.sleep(0.25)
    return not any(_pid_exists(pid) for pid in pids)


def _kill_process_tree(root_pid: int, timeout: float = 5.0):
    """Terminate a process and descendants, including children in new PGIDs."""
    pids = _collect_descendant_pids(root_pid)
    if _pid_exists(root_pid):
        pids.add(root_pid)
    if not pids:
        return

    _signal_pids(pids, signal.SIGTERM)
    if _wait_for_pids_to_exit(pids, timeout):
        return

    # Refresh once in case children forked while TERM was being handled.
    pids.update(_collect_descendant_pids(root_pid))
    _signal_pids(pids, signal.SIGKILL)


def _kill_process_group(pgid: int, timeout: float = 5.0):
    """SIGTERM the process group, escalate to SIGKILL if it doesn't die."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return                # already gone

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _process_group_alive(pgid):
            return
        time.sleep(0.5)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


def _docker_container_names_from_script(script_path: str) -> list[str]:
    """Extract MLS-Bench Docker container names referenced by a run script."""
    path = Path(script_path)
    if not path.exists() or not path.is_file():
        return []
    try:
        text = path.read_text()
    except OSError:
        return []

    names: set[str] = set()
    for line in text.splitlines():
        if "docker" not in line or "run" not in line or "--name" not in line:
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        for idx, token in enumerate(tokens):
            if token.lstrip("(") != "docker":
                continue
            if idx + 1 >= len(tokens) or tokens[idx + 1] != "run":
                continue
            docker_tokens = ["docker", *tokens[idx + 1:]]
            for tok_idx, docker_token in enumerate(docker_tokens):
                name: str | None = None
                if docker_token == "--name" and tok_idx + 1 < len(docker_tokens):
                    name = docker_tokens[tok_idx + 1]
                elif docker_token.startswith("--name="):
                    name = docker_token.split("=", 1)[1]
                if name:
                    name = name.strip().strip("()")
                    if name.startswith("mlsbench-"):
                        names.add(name)
    return sorted(names)


def _cleanup_docker_containers_for_script(script_path: str):
    """Force-remove Docker containers launched by a generated scheduler script."""
    for name in _docker_container_names_from_script(script_path):
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except Exception:
            pass


def _kill_job_processes(job: Job, pid: int | None = None, timeout: float = 5.0):
    """Best-effort cleanup for a scheduler job.

    Docker containers launched by rootless Docker can outlive the docker client
    process and move outside the scheduler process group, so process-group
    cleanup alone is not enough.
    """
    root_pid = pid or job.pid
    if job.command == "script":
        _cleanup_docker_containers_for_script(job.task)
    if root_pid:
        # Collect the process tree before the group leader can exit and
        # reparent descendants into a different subtree.
        _kill_process_tree(root_pid, timeout=timeout)
        _kill_process_group(root_pid, timeout=timeout)
    if job.command == "script":
        _cleanup_docker_containers_for_script(job.task)


def _scheduler_pid_is_valid(pid: int) -> bool:
    """Return whether *pid* appears to be an mlsbench scheduler process."""
    if not _pid_exists(pid):
        return False

    argv: list[str] = []
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        argv = [part.decode("utf-8", errors="ignore") for part in raw.split(b"\x00") if part]
    except OSError:
        probe = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        argv = probe.stdout.strip().split()

    if not argv:
        return False

    exe = Path(argv[0]).name.lower()
    if "python" not in exe:
        return False
    return "-m" in argv and "mlsbench.scheduler" in argv and "start" in argv


# ---------------------------------------------------------------------------
# GPU tracking
# ---------------------------------------------------------------------------

def _get_available_gpu_count() -> int:
    """Detect number of GPUs via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return len(out.stdout.strip().splitlines())
    except Exception:
        return 0


def _get_free_gpus(all_gpus: list[int], running_jobs: list[Job]) -> list[int]:
    """Return GPU indices not currently assigned to running jobs."""
    used = set()
    for j in running_jobs:
        if j.state == "running" and j.gpus and j.pid and _job_alive(j.pid):
            for g in j.gpus.split(","):
                used.add(int(g))
    return [g for g in all_gpus if g not in used]


def _query_gpu_memory_used() -> dict[int, int]:
    """Return current GPU memory usage in MiB keyed by GPU index."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return {}
    if out.returncode != 0:
        return {}

    usage: dict[int, int] = {}
    for line in out.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            usage[int(parts[0])] = int(parts[1])
        except ValueError:
            continue
    return usage


def _filter_externally_busy_gpus(
    candidate_gpus: list[int],
    running_jobs: list[Job],
    busy_memory_threshold_mb: int = 4096,
) -> list[int]:
    """Drop free GPUs that already have substantial external memory usage."""
    if not candidate_gpus:
        return []

    scheduler_owned = set()
    for j in running_jobs:
        if j.state == "running" and j.gpus and j.pid and _job_alive(j.pid):
            scheduler_owned.update(int(g) for g in j.gpus.split(","))

    memory_used = _query_gpu_memory_used()
    filtered: list[int] = []
    for gpu in candidate_gpus:
        if gpu in scheduler_owned:
            filtered.append(gpu)
            continue
        if memory_used.get(gpu, 0) >= busy_memory_threshold_mb:
            continue
        filtered.append(gpu)
    return filtered


def _load_task_config(task_name: str) -> dict:
    config_path = PROJECT_ROOT / "tasks" / task_name / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")
    return json.loads(config_path.read_text())


def _load_run_config(config_path: str | None) -> dict:
    """Load a scheduler/runtime config file."""
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


@lru_cache(maxsize=None)
def _load_package_config(package_name: str) -> dict:
    config_path = PROJECT_ROOT / "vendor" / "pkg_configs" / package_name / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def _test_cmd_uses_gpu(entry: dict, task_config: dict | None = None) -> bool:
    """Return whether a test command should consume GPU capacity.

    Resolution order is task config first, then package config, then a
    conservative fallback of assuming the job needs a GPU.
    """
    for key in ("use_cuda", "requires_gpu"):
        if key in entry:
            return bool(entry[key])

    task_config = task_config or {}
    for key in ("use_cuda", "requires_gpu"):
        if key in task_config:
            return bool(task_config[key])

    package_name = entry.get("package")
    if package_name:
        package_config = _load_package_config(package_name)
        for key in ("use_cuda", "requires_gpu"):
            if key in package_config:
                return bool(package_config[key])

    return True


_DEFAULT_CPU_ONLY_CONCURRENCY = 4


def _resolve_cpu_only_concurrency(config_path: str | None, cli_value: int | None) -> int:
    """Return the max number of concurrent CPU-only jobs.

    Resolution order: CLI flag → config file → sensible default.
    Never returns None — unlimited CPU-only concurrency is too dangerous
    on shared machines.
    """
    if cli_value is not None:
        return max(1, cli_value)

    run_config = _load_run_config(config_path)
    value = run_config.get("cpu_only_concurrency")
    if value is not None:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            pass

    return _DEFAULT_CPU_ONLY_CONCURRENCY


def _job_uses_gpu(job: Job) -> bool:
    """Return whether a queued/running job should count against GPU capacity."""
    return bool(job.gpus) or max(job.gpus_needed, job.min_gpus_needed) > 0


def _filter_test_cmds(
    test_cmds: list[dict],
    groups: list[int] | None = None,
    labels: list[str] | None = None,
) -> list[dict]:
    filtered = list(test_cmds)
    if groups:
        allowed_groups = set(groups)
        filtered = [entry for entry in filtered if entry.get("group") in allowed_groups]
    if labels:
        allowed_labels = set(labels)
        filtered = [entry for entry in filtered if entry.get("label") in allowed_labels]
    return filtered


def infer_gpus_needed(
    task_name: str,
    groups: list[int] | None = None,
    labels: list[str] | None = None,
) -> int:
    """Infer the peak concurrent GPU count for a task after group/label filters."""
    task_config = _load_task_config(task_name)
    test_cmds = _filter_test_cmds(task_config.get("test_cmds", []), groups=groups, labels=labels)
    if not test_cmds:
        return 1
    if not any(_test_cmd_uses_gpu(entry, task_config) for entry in test_cmds):
        return 0

    grouped: dict[int, list[dict]] = {}
    auto_group = 10000
    for entry in test_cmds:
        group = entry.get("group")
        if group is None:
            group = auto_group
            auto_group += 1
        grouped.setdefault(group, []).append(entry)

    peak_gpus = 0
    for entries in grouped.values():
        whole_gpu_jobs = 0
        fractional = 0.0
        for entry in entries:
            if not _test_cmd_uses_gpu(entry, task_config):
                continue
            compute = float(entry.get("compute", 1) or 1)
            if compute >= 1.0:
                whole_gpu_jobs += max(1, math.ceil(compute))
            else:
                fractional += compute
        peak_gpus = max(peak_gpus, whole_gpu_jobs + max(0, math.ceil(fractional)))

    return max(1, peak_gpus)


def infer_min_gpus_needed(
    task_name: str,
    groups: list[int] | None = None,
    labels: list[str] | None = None,
) -> int:
    """Infer the minimum GPU count required to make progress on a task.

    This is lower than ``infer_gpus_needed`` when the peak demand comes from
    parallel commands in the same group rather than a single multi-GPU command.
    """
    task_config = _load_task_config(task_name)
    test_cmds = _filter_test_cmds(task_config.get("test_cmds", []), groups=groups, labels=labels)
    if not test_cmds:
        return 1
    if not any(_test_cmd_uses_gpu(entry, task_config) for entry in test_cmds):
        return 0

    min_gpus = 0
    for entry in test_cmds:
        if not _test_cmd_uses_gpu(entry, task_config):
            continue
        compute = float(entry.get("compute", 1) or 1)
        min_gpus = max(min_gpus, max(1, math.ceil(compute)))
    return max(1, min_gpus)


def _build_extra_args(
    args,
    subcmd: str,
    *,
    include_seed: bool = True,
    include_labels: bool = True,
) -> list[str]:
    extra_args: list[str] = []
    if subcmd == "agent":
        extra_args.extend(["--model", args.model])
    elif subcmd == "baseline":
        if args.name:
            extra_args.extend(["--name", args.name])

    if include_seed and getattr(args, "seed", None) is not None:
        extra_args.extend(["--seed", str(args.seed)])
    for group in getattr(args, "group", []) or []:
        extra_args.extend(["--group", str(group)])
    for label in (getattr(args, "label", []) or []) if include_labels else []:
        extra_args.extend(["--label", label])
    return extra_args


def _can_expand_baseline_jobs(test_cmds: list[dict]) -> bool:
    """Return whether a baseline can be split into single-GPU jobs safely."""
    if not test_cmds:
        return False
    return all(float(entry.get("compute", 1) or 1) <= 1.0 for entry in test_cmds)


def _baseline_names(task_config: dict, requested_name: str | None) -> list[str | None]:
    """Return the baseline names that should become scheduler jobs."""
    if requested_name:
        return [requested_name]
    baselines = task_config.get("baselines", {}) or {}
    if baselines:
        return list(baselines.keys())
    return [None]


def _build_job_specs(args) -> list[dict]:
    """Expand one scheduler add request into one or more concrete jobs."""
    if args.subcmd != "baseline":
        extra_args = _build_extra_args(args, args.subcmd)
        gpus_needed = args.gpus_needed
        if gpus_needed is None:
            gpus_needed = infer_gpus_needed(args.task, groups=args.group, labels=args.label)
        min_gpus_needed = infer_min_gpus_needed(args.task, groups=args.group, labels=args.label)
        return [{
            "task": args.task,
            "args": extra_args,
            "gpus_needed": gpus_needed,
            "min_gpus_needed": min_gpus_needed,
        }]

    task_config = _load_task_config(args.task)
    filtered_cmds = _filter_test_cmds(task_config.get("test_cmds", []), groups=args.group, labels=args.label)
    baselines = task_config.get("baselines", {}) or {}
    should_expand = _can_expand_baseline_jobs(filtered_cmds) and (
        args.task.startswith("pde-") or bool(baselines)
    )
    if not should_expand:
        extra_args = _build_extra_args(args, args.subcmd)
        gpus_needed = args.gpus_needed
        if gpus_needed is None:
            gpus_needed = infer_gpus_needed(args.task, groups=args.group, labels=args.label)
        min_gpus_needed = infer_min_gpus_needed(args.task, groups=args.group, labels=args.label)
        return [{
            "task": args.task,
            "args": extra_args,
            "gpus_needed": gpus_needed,
            "min_gpus_needed": min_gpus_needed,
        }]

    run_config = _load_run_config(args.config)
    task_seeds = task_config.get("seeds")
    if getattr(args, "seed", None) is not None:
        seeds = [args.seed]
    elif task_seeds:
        seeds = task_seeds
    else:
        seeds = run_config.get("seeds", [42])
    labels = [entry["label"] for entry in filtered_cmds if entry.get("label")]
    if not labels:
        labels = [None]
    baseline_names = _baseline_names(task_config, getattr(args, "name", None))
    specs: list[dict] = []
    # Iterate label-first so the queue's first wave spans as many distinct
    # baseline/seed groups as possible. This avoids front-loading multiple
    # jobs that all wait on the same shared training checkpoint.
    for label in labels:
        for seed in seeds:
            for baseline_name in baseline_names:
                extra_args = _build_extra_args(
                    args,
                    args.subcmd,
                    include_seed=False,
                    include_labels=False,
                )
                if baseline_name and "--name" not in extra_args:
                    extra_args.extend(["--name", baseline_name])
                if "--seed" not in extra_args:
                    extra_args.extend(["--seed", str(seed)])
                if label and "--label" not in extra_args:
                    extra_args.extend(["--label", label])
                label_filter = [label] if label else None
                gpus_needed = args.gpus_needed
                if gpus_needed is None:
                    gpus_needed = infer_gpus_needed(args.task, groups=args.group, labels=label_filter)
                min_gpus_needed = infer_min_gpus_needed(args.task, groups=args.group, labels=label_filter)
                specs.append({
                    "task": args.task,
                    "args": extra_args,
                    "gpus_needed": gpus_needed,
                    "min_gpus_needed": min_gpus_needed,
                })
    return specs


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_add(args):
    """Add an agent or baseline job to the queue."""
    _ensure_dirs()
    job_specs = _build_job_specs(args)
    with _queue_transaction() as jobs:
        created: list[Job] = []
        for spec in job_specs:
            job = Job(
                job_id=_next_id(jobs),
                command=args.subcmd,
                task=spec["task"],
                args=spec["args"],
                config=args.config,
                gpus_needed=spec["gpus_needed"],
                min_gpus_needed=spec["min_gpus_needed"],
            )
            jobs.append(job)
            created.append(job)
    for job in created:
        print(
            f"Added job #{job.job_id}: {job.command} {job.task} {' '.join(job.args)} "
            f"[gpus={job.min_gpus_needed}-{job.gpus_needed}]"
        )


def cmd_batch(args):
    """Batch-add all tasks matching a domain prefix."""
    _ensure_dirs()
    tasks_dir = PROJECT_ROOT / "tasks"

    matching = []
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "config.json").exists():
            continue
        if args.domain == "all" or d.name.startswith(args.domain + "-") or d.name == args.domain:
            if args.exclude and d.name in args.exclude.split(","):
                continue
            matching.append(d.name)

    if not matching:
        print(f"No tasks found for domain '{args.domain}'")
        return

    created_count = 0
    with _queue_transaction() as jobs:
        for task in matching:
            task_args = argparse.Namespace(**vars(args))
            task_args.task = task
            job_specs = _build_job_specs(task_args)
            for spec in job_specs:
                job = Job(
                    job_id=_next_id(jobs),
                    command=args.subcmd,
                    task=task,
                    args=spec["args"],
                    config=args.config,
                    gpus_needed=spec["gpus_needed"],
                    min_gpus_needed=spec["min_gpus_needed"],
                )
                jobs.append(job)
                created_count += 1
                print(
                    f"  Added job #{job.job_id}: {job.command} {task} "
                    f"{' '.join(job.args)} [gpus={job.min_gpus_needed}-{job.gpus_needed}]"
                )

    print(f"\nAdded {created_count} jobs. Run 'python -m mlsbench.scheduler start' to begin.")


def cmd_list(args):
    """List all jobs in the queue."""
    jobs = _load_queue()
    if not jobs:
        print("Queue is empty.")
        return

    print(f"{'ID':>4}  {'State':<10}  {'GPUs':<6}  {'Command':<10}  {'Task':<40}  {'Extra'}")
    print("-" * 100)
    for j in jobs:
        extra = " ".join(j.args)
        if j.gpus:
            gpus = j.gpus
        elif max(j.gpus_needed, j.min_gpus_needed) == 0:
            gpus = "CPU"
        else:
            gpus = f"need:{j.min_gpus_needed}-{j.gpus_needed}"
        print(f"{j.job_id:>4}  {j.state:<10}  {gpus:<6}  {j.command:<10}  {j.task:<40}  {extra}")


def cmd_status(args):
    """Show scheduler status and running jobs."""
    pid = _read_scheduler_pid()
    if pid is not None:
        if _scheduler_pid_is_valid(pid):
            print(f"Scheduler running (PID {pid})")
        else:
            print("Scheduler not running (stale PID file)")
    else:
        print("Scheduler not running")

    jobs = _load_queue()
    running = [j for j in jobs if j.state == "running" and j.pid and _job_alive(j.pid)]
    stale = [j for j in jobs if j.state == "running" and (not j.pid or not _job_alive(j.pid))]
    queued = [j for j in jobs if j.state == "queued"]
    completed = [j for j in jobs if j.state == "completed"]
    failed = [j for j in jobs if j.state == "failed"]

    print(f"\nQueued: {len(queued)}  Running: {len(running)}  "
          f"Completed: {len(completed)}  Failed: {len(failed)}  Stale: {len(stale)}")

    if running:
        print("\nRunning:")
        for j in running:
            gpu_label = f"GPU {j.gpus}" if j.gpus else "CPU-only"
            print(f"  #{j.job_id} {j.command} {j.task} [{gpu_label}] "
                  f"PID={j.pid} started={j.started_at}")
    if stale:
        print("\nStale:")
        for j in stale:
            print(f"  #{j.job_id} {j.command} {j.task} [last GPU {j.gpus or '-'}] "
                  f"dead_pid={j.pid or '-'}")
    if queued:
        print("\nNext in queue:")
        for j in queued[:5]:
            if max(j.gpus_needed, j.min_gpus_needed) == 0:
                res_label = "CPU-only"
            else:
                res_label = f"needs {j.min_gpus_needed}-{j.gpus_needed} GPU"
            print(f"  #{j.job_id} {j.command} {j.task} ({res_label})")


def cmd_cancel(args):
    """Cancel a job (kill if running, including all descendant processes)."""
    with _queue_transaction() as jobs:
        for j in jobs:
            if j.job_id == args.job_id:
                if j.state == "running" and j.pid:
                    print(f"Killing job #{j.job_id} process tree from PID {j.pid} ...")
                    _kill_job_processes(j, timeout=5.0)
                    print(f"Job #{j.job_id} processes terminated.")
                j.state = "cancelled"
                j.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Job #{j.job_id} cancelled.")
                return
    print(f"Job #{args.job_id} not found.")


def cmd_clear(args):
    """Remove completed/failed/cancelled jobs from the queue."""
    with _queue_transaction() as jobs:
        kept = [j for j in jobs if j.state in ("queued", "running")]
        removed = len(jobs) - len(kept)
        jobs[:] = kept
    print(f"Cleared {removed} finished jobs, {len(kept)} remaining.")


def cmd_start(args):
    """Start the scheduler loop."""
    _ensure_dirs()

    existing_pid = _read_scheduler_pid()
    if existing_pid is not None and existing_pid != os.getpid() and _scheduler_pid_is_valid(existing_pid):
        print(f"Scheduler already running (PID {existing_pid})", file=sys.stderr)
        sys.exit(1)

    # Parse GPU list
    if args.gpus:
        all_gpus = [int(g) for g in args.gpus.split(",")]
    else:
        n = _get_available_gpu_count()
        all_gpus = list(range(n))

    if not all_gpus:
        print("ERROR: No GPUs detected. Use --gpus to specify manually.", file=sys.stderr)
        sys.exit(1)

    print(f"Scheduler starting with GPUs: {all_gpus}")

    # Resolve CPU-only concurrency: CLI flag → scheduler --config → first job's config → default
    _start_config = args.config
    if not _start_config:
        with _queue_transaction() as _q:
            for _j in _q:
                if _j.config:
                    _start_config = _j.config
                    break
    cpu_only_limit = _resolve_cpu_only_concurrency(_start_config, getattr(args, "max_cpu_only", None))
    print(f"CPU-only concurrency limit: {cpu_only_limit}")

    # Write PID
    PID_FILE.write_text(str(os.getpid()))

    # Handle signals
    stop = False
    def _handle_signal(sig, frame):
        nonlocal stop
        print(f"\nReceived signal {sig}, stopping after current jobs finish...")
        stop = True
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    processes: dict[int, subprocess.Popen] = {}  # job_id -> Popen

    with _queue_transaction() as jobs:
        requeued = _requeue_stale_running_jobs(jobs)
        for job_id in requeued:
            print(f"Recovered stale job #{job_id}: requeued")

    try:
        while not stop:
            with _queue_transaction() as jobs:
                requeued = _requeue_stale_running_jobs(jobs, tracked_job_ids=set(processes))
                for job_id in requeued:
                    print(f"Recovered stale job #{job_id}: requeued")

                # Check running jobs for completion or timeout
                for j in jobs:
                    if j.state == "running" and j.job_id in processes:
                        proc = processes[j.job_id]
                        ret = proc.poll()
                        if ret is not None:
                            # Main process exited — kill any orphaned descendants
                            # that may still be holding GPUs.
                            _kill_job_processes(j, proc.pid, timeout=5.0)
                            j.exit_code = ret
                            j.state = "completed" if ret == 0 else "failed"
                            j.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            del processes[j.job_id]
                            status = "completed" if ret == 0 else f"failed (exit={ret})"
                            print(f"[{j.finished_at}] Job #{j.job_id} {j.task} {status}")
                        elif j.timeout_secs > 0 and j.started_at:
                            # Enforce wall-time limit
                            started = datetime.strptime(j.started_at, "%Y-%m-%d %H:%M:%S")
                            elapsed = (datetime.now() - started).total_seconds()
                            if elapsed > j.timeout_secs:
                                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Job #{j.job_id} "
                                      f"TIMEOUT after {elapsed:.0f}s (limit {j.timeout_secs}s)")
                                _kill_job_processes(j, proc.pid, timeout=5.0)
                                j.exit_code = -9
                                j.state = "failed"
                                j.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                del processes[j.job_id]

                # Find free GPUs
                free_gpus = _get_free_gpus(all_gpus, jobs)
                if not getattr(args, "ignore_busy", False):
                    free_gpus = _filter_externally_busy_gpus(free_gpus, jobs)
                running_cpu_only = sum(
                    1
                    for j in jobs
                    if j.state == "running" and j.pid and _job_alive(j.pid) and not _job_uses_gpu(j)
                )

                # Launch queued jobs if GPUs available
                for j in jobs:
                    if j.state != "queued":
                        continue
                    if _job_uses_gpu(j):
                        if j.min_gpus_needed > len(free_gpus):
                            continue
                        # Assign GPUs
                        assign_count = min(j.gpus_needed, len(free_gpus))
                        assigned = free_gpus[:assign_count]
                        free_gpus = free_gpus[assign_count:]
                        j.gpus = ",".join(str(g) for g in assigned)
                    else:
                        if running_cpu_only >= cpu_only_limit:
                            continue
                        j.gpus = ""
                        running_cpu_only += 1

                    # Build command
                    config = args.config or j.config
                    if j.command == "script":
                        # Raw script execution: task field holds the script path
                        cmd = ["bash", j.task]
                    else:
                        cmd = [
                            sys.executable, "-m", "mlsbench",
                            j.command, j.task,
                            "--config", config,
                        ] + j.args

                    # Log file
                    task_slug = Path(j.task).stem if j.command == "script" else j.task
                    log_name = f"job{j.job_id}_{j.command}_{task_slug}_{datetime.now():%Y%m%d_%H%M%S}.log"
                    log_path = LOG_DIR / log_name
                    j.log_file = str(log_path)

                    # Set CUDA_VISIBLE_DEVICES
                    env = os.environ.copy()
                    env["MLSBENCH_SCHEDULER_MANAGED"] = "1"
                    env["CUDA_VISIBLE_DEVICES"] = j.gpus
                    env["NVIDIA_VISIBLE_DEVICES"] = j.gpus
                    src_path = str(PROJECT_ROOT / "src")
                    existing_pythonpath = env.get("PYTHONPATH", "")
                    env["PYTHONPATH"] = (
                        src_path if not existing_pythonpath
                        else f"{src_path}{os.pathsep}{existing_pythonpath}"
                    )

                    # Launch — each job gets its own session so we can
                    # track and kill the entire process tree via PGID.
                    log_fh = open(log_path, "w")
                    proc = subprocess.Popen(
                        cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                        env=env, cwd=str(PROJECT_ROOT),
                        start_new_session=True,
                    )
                    log_fh.close()
                    processes[j.job_id] = proc
                    j.pid = proc.pid
                    j.state = "running"
                    j.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    gpu_label = f"GPU {j.gpus}" if j.gpus else "CPU-only"
                    print(f"[{j.started_at}] Started job #{j.job_id}: {j.command} {j.task} "
                          f"[{gpu_label}] PID={j.pid}")

            # If nothing queued and nothing running, exit (unless --daemon)
            if not getattr(args, "daemon", False):
                if not any(j.state in ("queued", "running") for j in jobs):
                    print("All jobs finished.")
                    break

            time.sleep(5)

    finally:
        current_pid = _read_scheduler_pid()
        if current_pid == os.getpid():
            PID_FILE.unlink(missing_ok=True)
        # Kill any still-running process *groups* on exit
        for job_id, proc in processes.items():
            if proc.poll() is None or _process_group_alive(proc.pid):
                print(f"Terminating job #{job_id} (process tree from PID {proc.pid})")
                cleanup_job = next((j for j in _load_queue() if j.job_id == job_id), None)
                if cleanup_job is not None:
                    _kill_job_processes(cleanup_job, proc.pid, timeout=5.0)
                else:
                    _kill_process_group(proc.pid, timeout=5.0)
                    _kill_process_tree(proc.pid, timeout=5.0)


# ---------------------------------------------------------------------------
# Programmatic API for submitting script jobs (used by LocalSchedulerExecutor)
# ---------------------------------------------------------------------------


def submit_script_job(
    script_path: str,
    gpus_needed: int = 1,
    job_name: str = "",
    timeout_secs: int = 0,
) -> int:
    """Add a 'script' job to the queue. Returns the job_id.

    Args:
        timeout_secs: Wall-time limit in seconds. 0 = no limit.
    """
    _ensure_dirs()
    with _queue_transaction() as jobs:
        job = Job(
            job_id=_next_id(jobs),
            command="script",
            task=script_path,
            args=[],
            config="",
            gpus_needed=gpus_needed,
            min_gpus_needed=gpus_needed,
            timeout_secs=timeout_secs,
        )
        jobs.append(job)
    return job.job_id


def poll_job(job_id: int) -> tuple[str, int, str]:
    """Check a job's current state. Returns (state, exit_code, log_file)."""
    jobs = _load_queue()
    for j in jobs:
        if j.job_id == job_id:
            return j.state, j.exit_code, j.log_file
    return "unknown", -1, ""


def wait_for_job(job_id: int, poll_interval: float = 3.0, timeout: float = 7200) -> str:
    """Block until a job reaches a terminal state. Returns the final state."""
    terminal = {"completed", "failed", "cancelled"}
    start = time.time()
    while True:
        state, _, _ = poll_job(job_id)
        if state in terminal:
            return state
        if time.time() - start > timeout:
            return "timeout"
        time.sleep(poll_interval)


def is_scheduler_running() -> bool:
    """Check if the scheduler daemon is alive."""
    pid = _read_scheduler_pid()
    return pid is not None and _scheduler_pid_is_valid(pid)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lightweight GPU-aware task scheduler for MLS-Bench",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # add
    p_add = sub.add_parser("add", help="Add a single job")
    p_add.add_argument("subcmd", choices=["agent", "baseline"], help="mlsbench subcommand")
    p_add.add_argument("task", help="Task name")
    p_add.add_argument("--model", default="claude-sonnet-4-6")
    p_add.add_argument("--name", default=None, help="Baseline name")
    p_add.add_argument("--seed", type=int, default=None, help="Optional seed override passed through to mlsbench")
    p_add.add_argument("--group", type=int, action="append", default=None, help="Optional test group filter")
    p_add.add_argument("--label", action="append", default=None, help="Optional test label filter")
    p_add.add_argument("--config", default="configs/config.yaml")
    p_add.add_argument("--gpus-needed", type=int, default=None, help="GPUs per job (default: infer from task config)")
    p_add.set_defaults(func=cmd_add)

    # batch
    p_batch = sub.add_parser("batch", help="Batch-add tasks by domain")
    p_batch.add_argument("subcmd", choices=["agent", "baseline"])
    p_batch.add_argument("domain", help="Domain prefix or 'all'")
    p_batch.add_argument("--model", default="claude-sonnet-4-6")
    p_batch.add_argument("--name", default=None, help="Baseline name (for baseline)")
    p_batch.add_argument("--seed", type=int, default=None, help="Optional seed override passed through to mlsbench")
    p_batch.add_argument("--group", type=int, action="append", default=None, help="Optional test group filter")
    p_batch.add_argument("--label", action="append", default=None, help="Optional test label filter")
    p_batch.add_argument("--config", default="configs/config.yaml")
    p_batch.add_argument("--gpus-needed", type=int, default=None)
    p_batch.add_argument("--exclude", default="", help="Comma-separated tasks to exclude")
    p_batch.set_defaults(func=cmd_batch)

    # list
    p_list = sub.add_parser("list", help="List all jobs")
    p_list.set_defaults(func=cmd_list)

    # status
    p_status = sub.add_parser("status", help="Show scheduler status")
    p_status.set_defaults(func=cmd_status)

    # start
    p_start = sub.add_parser("start", help="Start the scheduler")
    p_start.add_argument("--gpus", default="", help="Comma-separated GPU indices (default: auto-detect)")
    p_start.add_argument("--config", default=None, help="Override config for all jobs")
    p_start.add_argument("--ignore-busy", action="store_true",
                         help="Skip external GPU memory check (use when sharing GPUs with other workloads)")
    p_start.add_argument("--max-cpu-only", type=int, default=None,
                         help="Max concurrent CPU-only jobs (default: read cpu_only_concurrency from config)")
    p_start.add_argument("--daemon", action="store_true",
                         help="Keep running even when queue is empty (for dynamic job submission from agents)")
    p_start.set_defaults(func=cmd_start)

    # cancel
    p_cancel = sub.add_parser("cancel", help="Cancel a job")
    p_cancel.add_argument("job_id", type=int)
    p_cancel.set_defaults(func=cmd_cancel)

    # clear
    p_clear = sub.add_parser("clear", help="Remove finished jobs from queue")
    p_clear.set_defaults(func=cmd_clear)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
