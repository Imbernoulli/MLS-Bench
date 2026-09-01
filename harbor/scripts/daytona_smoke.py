#!/usr/bin/env python3
"""Run isolated MLS-Bench Daytona environment smoke tests.

Harbor can schedule several trials concurrently, but the question here is
whether each independently-built environment works on Daytona.  This helper
launches a fresh ``harbor run`` subprocess for every selected task, batching
those subprocesses with ``--concurrency``.  A subprocess also gives each run a
fresh Daytona client singleton and an unambiguous job directory.

By default one representative task is selected for each package environment
(the task requesting the most GPUs, then the lexicographically first task).
Use ``--scope task`` to exercise all non-API tasks instead.  The two benchmark
tasks that call model APIs are excluded unless ``--include-api`` is supplied.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10 (supported by the adapter)
    import tomli as tomllib  # type: ignore[no-redef]


# These are the only MLS-Bench tasks whose verifier scripts intentionally call
# a model provider.  A Daytona key authenticates the sandbox, not these APIs.
EXTERNAL_API_TASKS = frozenset({"agent-tool-reasoning", "mas-topology"})


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    package: str
    gpus: int
    path: Path
    external_api: bool


def discover_tasks(tasks_dir: Path) -> list[TaskRecord]:
    """Read task metadata and return records sorted by task id."""

    records: list[TaskRecord] = []
    for task_toml in sorted(tasks_dir.glob("*/task.toml")):
        data = tomllib.loads(task_toml.read_text())
        metadata = data.get("metadata") or {}
        environment = data.get("environment") or {}
        task_id = str(metadata.get("mls_bench_task_id") or "")
        package = str(metadata.get("mls_bench_package") or "")
        if not task_id or not package:
            raise ValueError(f"{task_toml} lacks metadata task id/package")
        try:
            gpus = int(environment.get("gpus", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid GPU count in {task_toml}") from exc
        if gpus < 0:
            raise ValueError(f"Negative GPU count in {task_toml}")
        records.append(
            TaskRecord(
                task_id=task_id,
                package=package,
                gpus=gpus,
                path=task_toml.parent.resolve(),
                external_api=task_id in EXTERNAL_API_TASKS,
            )
        )
    if not records:
        raise ValueError(f"No task.toml files found under {tasks_dir}")
    return records


def select_tasks(
    records: Iterable[TaskRecord], *, scope: str, include_api: bool
) -> list[TaskRecord]:
    """Filter API tasks and select either every task or one per package."""

    eligible = [r for r in records if include_api or not r.external_api]
    if scope == "task":
        return sorted(eligible, key=lambda r: r.task_id)
    if scope != "environment":
        raise ValueError(f"Unknown scope: {scope}")

    by_package: dict[str, list[TaskRecord]] = {}
    for record in eligible:
        by_package.setdefault(record.package, []).append(record)
    # Prefer a GPU task when a package has one: this exercises the Daytona GPU
    # strategy and still covers CPU-only packages normally.
    return sorted(
        (
            max(tasks, key=lambda r: (r.gpus, r.task_id))
            for tasks in by_package.values()
        ),
        key=lambda r: r.package,
    )


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return result or "task"


def build_command(
    record: TaskRecord,
    *,
    harbor_cmd: str,
    import_path: str,
    jobs_dir: Path,
    agent: str,
    verify: bool,
    force_build: bool,
    delete: bool,
    overrides: dict[str, int | None],
) -> list[str]:
    """Build an argv list for one isolated Harbor invocation."""

    job_name = f"daytona-smoke-{_slug(record.task_id)}"
    command = [
        harbor_cmd,
        "run",
        "--path",
        str(record.path),
        "--job-name",
        job_name,
        "--jobs-dir",
        str(jobs_dir),
        "--n-concurrent",
        "1",
        "--agent",
        agent,
        "--environment-import-path",
        import_path,
        "--yes",
    ]
    if not verify:
        command.append("--disable-verification")
    if force_build:
        command.append("--force-build")
    else:
        command.append("--no-force-build")
    command.append("--delete" if delete else "--no-delete")
    for option, value in overrides.items():
        # A GPU override is only meaningful for tasks that declare GPUs. Do
        # not turn CPU-only tasks into GPU sandboxes when capping a matrix run
        # to an organization's available GPU quota.
        if option == "--override-gpus" and record.gpus <= 0:
            continue
        # Daytona's CPU sandbox class has a much smaller per-sandbox disk
        # ceiling than GPU sandboxes.  Large GPU base images may need a
        # storage override (for example, the CFGpp image is >100 GB), but
        # forwarding that same value to CPU tasks would make their create
        # request fail before the environment is exercised.
        if option == "--override-storage-mb" and record.gpus <= 0:
            # Daytona's CPU sandbox class is capped at 10 GB.  MLS-Bench's
            # task files historically request 60 GB for local Docker, so
            # provide a safe CPU default when no explicit value was supplied.
            if value is None or value > 10 * 1024:
                value = 10 * 1024
        if value is not None:
            command.extend([option, str(value)])
    return command


def _result_path(jobs_dir: Path, task_id: str) -> Path:
    expected = jobs_dir / f"daytona-smoke-{_slug(task_id)}" / "result.json"
    if expected.is_file():
        return expected
    # Harbor may append a suffix when a job-name already exists.  Pick the
    # newest matching result rather than reporting a false missing result.
    candidates = sorted(
        jobs_dir.glob(f"daytona-smoke-{_slug(task_id)}*/result.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else expected


def summarize_result(path: Path, process_returncode: int) -> tuple[str, str]:
    """Return ``(status, exceptions)`` from Harbor's result JSON."""

    if not path.is_file():
        return ("process-error" if process_returncode else "missing-result", "")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ("invalid-result", "")
    stats = data.get("stats") or {}
    errors: list[str] = []
    for evaluation in (stats.get("evals") or {}).values():
        errors.extend((evaluation.get("exception_stats") or {}).keys())
    # Provider interruption/cancellation is not a stable environment failure;
    # mark it non-terminal so ``--resume`` retries the environment.
    if "CancelledError" in errors:
        return ("incomplete", "cancelled")
    if errors:
        return ("error", ";".join(sorted(set(errors))))
    if stats.get("n_errored_trials", 0):
        return ("error", "trial-error")
    if stats.get("n_running_trials", 0) or stats.get("n_pending_trials", 0):
        return ("incomplete", "running-trials")
    if process_returncode:
        return ("process-error", "")
    return ("passed", "")


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "task_id",
                "package",
                "gpus",
                "external_api",
                "status",
                "exceptions",
                "result_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


async def _daytona_sandboxes(
    *, request_delete: bool = False, protected_ids: set[str] | None = None
) -> list[str]:
    """Return active Daytona sandbox IDs, optionally requesting deletion.

    Harbor normally deletes a sandbox in its strategy ``stop`` hook.  During
    an image build, however, Daytona can leave a short-lived BUILDING_SNAPSHOT
    or BUILD_FAILED record behind after the client process exits.  Those
    records still count against the organisation's one-GPU quota.  If explicit
    cleanup is requested, only IDs not in ``protected_ids`` are deleted.  This
    is necessary because ``client.list()`` is organization-wide and must not
    delete another user's sandbox.
    """

    from daytona import AsyncDaytona, DaytonaConfig

    client = AsyncDaytona(DaytonaConfig(api_key=None))
    try:
        ids: list[str] = []
        async for sandbox in client.list():
            ids.append(str(sandbox.id))
            # Never perform organization-wide deletion.  A caller must pass a
            # snapshot of protected IDs; ``None`` disables deletion entirely.
            if request_delete and protected_ids is not None and str(sandbox.id) not in protected_ids:
                try:
                    await sandbox.delete()
                except BaseException:
                    # The sandbox may disappear between list and delete; the
                    # next poll will verify whether it still occupies quota.
                    pass
        return ids
    finally:
        # The SDK's websocket disconnect can raise CancelledError during
        # interpreter shutdown; this cleanup is best-effort and must not mask
        # the actual Harbor result.
        try:
            await client.close()
        except BaseException:
            pass


def wait_for_daytona_quiescence(
    timeout_sec: int, *, protected_ids: set[str] | None = None
) -> list[str]:
    """Wait until the previous sandbox deletion releases account resources."""

    if timeout_sec <= 0:
        return []
    try:
        import asyncio

        deadline = time.monotonic() + timeout_sec
        last_ids: list[str] = []
        while True:
            all_ids = asyncio.run(_daytona_sandboxes(request_delete=False))
            removable = [
                sandbox_id
                for sandbox_id in all_ids
                if protected_ids is not None and sandbox_id not in protected_ids
            ]
            if removable:
                asyncio.run(
                    _daytona_sandboxes(
                        request_delete=True, protected_ids=protected_ids
                    )
                )
            last_ids = asyncio.run(_daytona_sandboxes(request_delete=False))
            remaining = [
                sandbox_id
                for sandbox_id in last_ids
                if protected_ids is not None and sandbox_id not in protected_ids
            ]
            if not remaining:
                return []
            if time.monotonic() >= deadline:
                return last_ids
            print(
                f"    waiting for Daytona cleanup ({len(remaining)} sandbox(es))...",
                flush=True,
            )
            time.sleep(5)
    except BaseException as exc:
        print(f"    warning: cannot poll Daytona cleanup: {exc}", file=sys.stderr)
        return []


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    harbor_dir = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, default=harbor_dir / "tasks")
    parser.add_argument(
        "--scope",
        choices=("environment", "task"),
        default="environment",
        help="one representative per package (default) or every task",
    )
    parser.add_argument(
        "--resource",
        choices=("all", "gpu", "cpu"),
        default="all",
        help="optionally run only GPU or CPU records (useful for parallel runs)",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="restrict the run to one or more task IDs (repeat this option)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing result.json files in jobs-dir and run only missing records",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="with --resume, retry terminal error results instead of reusing them",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="maximum number of Harbor subprocesses in one batch",
    )
    parser.add_argument(
        "--gpu-limit",
        type=int,
        default=None,
        help="aggregate GPU quota for a batch; tasks are queued by declared GPU count",
    )
    parser.add_argument("--include-api", action="store_true")
    parser.add_argument("--harbor-cmd", default="harbor")
    parser.add_argument(
        "--import-path", default="harbor_env:DaytonaEnvironment", dest="import_path"
    )
    parser.add_argument(
        "--jobs-dir", type=Path, default=harbor_dir / "jobs-daytona-smoke"
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--agent", default="nop")
    parser.add_argument("--verify", action="store_true", help="run verifiers")
    parser.add_argument("--no-force-build", action="store_true")
    parser.add_argument("--no-delete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--settle-timeout-sec",
        type=int,
        default=300,
        help=(
            "with --cleanup-orphans, wait this long for sandbox deletion "
            "before the next batch"
        ),
    )
    parser.add_argument(
        "--cleanup-orphans",
        action="store_true",
        help=(
            "delete Daytona sandboxes created after this runner starts; unsafe "
            "when another Daytona runner is active, so disabled by default"
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--override-cpus", type=int, default=None)
    parser.add_argument("--override-memory-mb", type=int, default=None)
    parser.add_argument("--override-storage-mb", type=int, default=None)
    parser.add_argument("--override-gpus", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks_dir = args.tasks_dir.expanduser().resolve()
    jobs_dir = args.jobs_dir.expanduser().resolve()
    report_path = (args.report or jobs_dir / "report.csv").expanduser().resolve()
    records = select_tasks(
        discover_tasks(tasks_dir), scope=args.scope, include_api=args.include_api
    )
    if args.resource == "gpu":
        records = [record for record in records if record.gpus > 0]
    elif args.resource == "cpu":
        records = [record for record in records if record.gpus <= 0]
    if args.task_ids:
        wanted = set(args.task_ids)
        records = [record for record in records if record.task_id in wanted]
    if args.limit is not None:
        records = records[: max(0, args.limit)]
    if args.concurrency < 1:
        print("--concurrency must be at least 1", file=sys.stderr)
        return 2
    if not records:
        print("No eligible tasks selected.", file=sys.stderr)
        return 2

    print(
        f"Selected {len(records)} {args.scope}(s) from {tasks_dir} "
        f"({'including' if args.include_api else 'excluding'} external-API tasks)."
    )
    if args.dry_run:
        for record in records:
            command = build_command(
                record,
                harbor_cmd=args.harbor_cmd,
                import_path=args.import_path,
                jobs_dir=jobs_dir,
                agent=args.agent,
                verify=args.verify,
                force_build=not args.no_force_build,
                delete=not args.no_delete,
                overrides={
                    "--override-cpus": args.override_cpus,
                    "--override-memory-mb": args.override_memory_mb,
                    "--override-storage-mb": args.override_storage_mb,
                    "--override-gpus": args.override_gpus,
                },
            )
            print(shlex.join(command))
        return 0

    jobs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    script_dir = Path(__file__).resolve().parent
    harbor_dir = script_dir.parent
    child_env = os.environ.copy()
    pythonpath = [str(harbor_dir)]
    if child_env.get("PYTHONPATH"):
        pythonpath.append(child_env["PYTHONPATH"])
    child_env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    def run_record(index: int, record: TaskRecord) -> dict[str, str] | None:
        existing_result = _result_path(jobs_dir, record.task_id)
        if args.resume and existing_result.is_file():
            status, exceptions = summarize_result(existing_result, 0)
            # A result left with running/pending trials is a cancelled or
            # interrupted job, not a terminal measurement.  Re-run it on
            # resume so stale partial records (common after provider build
            # failures) do not permanently hide an environment.
            if status == "passed" or (status == "error" and not args.retry_errors):
                print(f"[{index}/{len(records)}] {record.task_id} -> reused {status}")
                return {
                    "task_id": record.task_id,
                    "package": record.package,
                    "gpus": str(record.gpus),
                    "external_api": str(record.external_api).lower(),
                    "status": status,
                    "exceptions": exceptions,
                    "result_json": str(existing_result),
                }
            print(f"[{index}/{len(records)}] {record.task_id} -> rerunning {status}")
        command = build_command(
            record,
            harbor_cmd=args.harbor_cmd,
            import_path=args.import_path,
            jobs_dir=jobs_dir,
            agent=args.agent,
            verify=args.verify,
            force_build=not args.no_force_build,
            delete=not args.no_delete,
            overrides={
                "--override-cpus": args.override_cpus,
                "--override-memory-mb": args.override_memory_mb,
                "--override-storage-mb": args.override_storage_mb,
                "--override-gpus": args.override_gpus,
            },
        )
        print(f"[{index}/{len(records)}] {record.task_id} ({record.package}, gpus={record.gpus})")
        completed = subprocess.run(command, cwd=harbor_dir, env=child_env, check=False)
        result_json = _result_path(jobs_dir, record.task_id)
        status, exceptions = summarize_result(result_json, completed.returncode)
        print(f"    -> {status}{f' ({exceptions})' if exceptions else ''}")
        return {
            "task_id": record.task_id,
            "package": record.package,
            "gpus": str(record.gpus),
            "external_api": str(record.external_api).lower(),
            "status": status,
            "exceptions": exceptions,
            "result_json": str(result_json),
        }

    # Pack batches by aggregate declared GPU count when requested.  This keeps
    # each task's resource contract intact: a task declaring 8 GPUs receives 8
    # GPUs and waits for capacity instead of being silently reduced to 1.
    batches: list[list[tuple[int, TaskRecord]]] = []
    if args.resource == "gpu" and args.gpu_limit is not None:
        if args.gpu_limit < 1:
            print("--gpu-limit must be at least 1", file=sys.stderr)
            return 2
        current: list[tuple[int, TaskRecord]] = []
        used_gpus = 0
        for index, record in enumerate(records, start=1):
            requested = int(record.gpus)
            if requested > args.gpu_limit:
                print(
                    f"{record.task_id} requests {requested} GPUs, exceeding "
                    f"--gpu-limit={args.gpu_limit}; running it alone",
                    file=sys.stderr,
                )
            if current and (
                len(current) >= args.concurrency
                or used_gpus + requested > args.gpu_limit
            ):
                batches.append(current)
                current = []
                used_gpus = 0
            current.append((index, record))
            used_gpus += requested
        if current:
            batches.append(current)
    else:
        batches = [
            list(enumerate(records[offset : offset + args.concurrency], start=offset + 1))
            for offset in range(0, len(records), args.concurrency)
        ]

    # Harbor deletes the sandbox owned by each trial.  Organization-wide
    # incremental cleanup is deliberately opt-in: two concurrent runners can
    # both start from the same baseline, and either runner would otherwise
    # delete the other's live sandboxes when its own batch finishes.
    protected_ids: set[str] | None = None
    if args.cleanup_orphans:
        try:
            protected_ids = set(
                asyncio.run(_daytona_sandboxes(request_delete=False))
            )
        except BaseException:
            # If the baseline cannot be read, disable deletion for safety.
            protected_ids = None

    # Run records in batches.  A batch avoids racing the cleanup poller while
    # still allowing independent sandboxes to use available provider quota.
    for batch in batches:
        if len(batch) == 1:
            batch_rows = [run_record(*batch[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                batch_rows = list(pool.map(lambda item: run_record(*item), batch))
        rows.extend(row for row in batch_rows if row is not None)
        write_report(report_path, rows)
        if args.cleanup_orphans:
            remaining = wait_for_daytona_quiescence(
                args.settle_timeout_sec, protected_ids=protected_ids
            )
            if remaining:
                print(
                    "    warning: Daytona still reports sandbox(es) after "
                    f"{args.settle_timeout_sec}s: {', '.join(remaining)}",
                    file=sys.stderr,
                )
    write_report(report_path, rows)
    failed = sum(row["status"] != "passed" for row in rows)
    print(f"Report: {report_path} ({len(rows) - failed} passed, {failed} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # ``... --dry-run | head`` is a convenient way to inspect a large
        # selection; do not turn the closed output pipe into a traceback.
        pass
