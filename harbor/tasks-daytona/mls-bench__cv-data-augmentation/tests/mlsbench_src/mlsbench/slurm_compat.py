from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlsbench import scheduler  # noqa: E402


STATE_MAP = {
    "queued": "PD",
    "running": "R",
    "completed": "CD",
    "failed": "F",
    "cancelled": "CA",
}


def _effective_state(job: scheduler.Job) -> str:
    if job.state == "running" and job.pid and not scheduler._pid_exists(job.pid):
        return "ST"
    return STATE_MAP.get(job.state, job.state.upper())


def _selected_jobs(job_ids: set[int] | None = None) -> list[scheduler.Job]:
    jobs = scheduler._load_queue()
    if not job_ids:
        return jobs
    return [job for job in jobs if job.job_id in job_ids]


def _format_job_name(job: scheduler.Job) -> str:
    if job.command == "script":
        # Extract meaningful name from script path:
        # .../logs/<task>/<model>/<timestamp>/group_<g>/run.sh → <task> <model> g<g>
        from pathlib import Path
        p = Path(job.task)
        parts_list = p.parts
        try:
            logs_idx = parts_list.index("logs")
            task = parts_list[logs_idx + 1] if len(parts_list) > logs_idx + 1 else "?"
            model = parts_list[logs_idx + 2] if len(parts_list) > logs_idx + 2 else ""
            group = p.parent.name  # e.g. "group_1_0"
            return f"{task} ({model}) {group}"
        except (ValueError, IndexError):
            return f"script {p.name}"
    parts: list[str] = [job.command, job.task]
    if job.args:
        parts.append(" ".join(job.args))
    return " ".join(parts)


def _parse_job_ids(raw_values: list[str] | None) -> set[int] | None:
    if not raw_values:
        return None

    parsed: set[int] = set()
    for value in raw_values:
        for piece in value.split(","):
            piece = piece.strip()
            if not piece:
                continue
            parsed.add(int(piece))
    return parsed


def cmd_squeue(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mlsbench-squeue", description="Show queued and running MLS-Bench jobs")
    parser.add_argument("-a", "--all", action="store_true", help="Show finished jobs too")
    parser.add_argument("-j", "--jobs", action="append", help="Comma-separated job ids to include")
    parser.add_argument("--noheader", action="store_true", help="Suppress the header row")
    args = parser.parse_args(argv)

    jobs = _selected_jobs(_parse_job_ids(args.jobs))
    if not args.all:
        jobs = [job for job in jobs if _effective_state(job) in {"PD", "R", "ST"}]

    if not args.noheader:
        print(
            f"{'JOBID':>5} {'ST':<3} {'GPUS':<7} {'PID':<8} "
            f"{'TASK':<32} {'DETAILS'}"
        )
    for job in jobs:
        # For script jobs, show a short task slug instead of the full path
        if job.command == "script":
            from pathlib import Path
            p = Path(job.task)
            parts_list = p.parts
            try:
                logs_idx = parts_list.index("logs")
                task_display = parts_list[logs_idx + 1] if len(parts_list) > logs_idx + 1 else p.stem
            except ValueError:
                task_display = p.stem
        else:
            task_display = job.task
        print(
            f"{job.job_id:>5} {_effective_state(job):<3} "
            f"{(job.gpus or f'need:{job.min_gpus_needed}-{job.gpus_needed}'):<7} "
            f"{(str(job.pid) if job.pid else '-'): <8} "
            f"{task_display:<32} {_format_job_name(job)}"
        )
    return 0


def cmd_sacct(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mlsbench-sacct", description="Show MLS-Bench job history")
    parser.add_argument("-j", "--jobs", action="append", help="Comma-separated job ids to include")
    parser.add_argument("--noheader", action="store_true", help="Suppress the header row")
    args = parser.parse_args(argv)

    jobs = _selected_jobs(_parse_job_ids(args.jobs))
    if not args.noheader:
        print(
            f"{'JOBID':>5} {'STATE':<4} {'EXIT':<5} {'GPUS':<7} "
            f"{'STARTED':<19} {'FINISHED':<19} {'TASK'}"
        )
    for job in jobs:
        print(
            f"{job.job_id:>5} {_effective_state(job):<4} {job.exit_code:<5} "
            f"{(job.gpus or '-'): <7} {(job.started_at or '-'): <19} "
            f"{(job.finished_at or '-'): <19} {job.task}"
        )
    return 0


def cmd_scancel(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mlsbench-scancel", description="Cancel MLS-Bench scheduler jobs")
    parser.add_argument("job_ids", nargs="+", type=int, help="Job ids to cancel")
    args = parser.parse_args(argv)

    exit_code = 0
    for job_id in args.job_ids:
        before = {job.job_id: job.state for job in scheduler._load_queue()}
        scheduler.cmd_cancel(argparse.Namespace(job_id=job_id))
        after = {job.job_id: job.state for job in scheduler._load_queue()}
        if before.get(job_id) == after.get(job_id):
            exit_code = 1
    return exit_code


def cmd_sbatch(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="mlsbench-sbatch",
        description="Queue an MLS-Bench scheduler job. Usage mirrors 'python scripts/scheduler.py add'.",
    )
    parser.add_argument("subcmd", choices=["agent", "baseline"])
    parser.add_argument("task")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--group", type=int, action="append", default=None)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--gpus-needed", type=int, default=None)
    args = parser.parse_args(argv)

    scheduler.cmd_add(args)
    return 0


_COMMAND_ALIASES = {
    "squeue": "squeue", "mlsbench-squeue": "squeue",
    "sacct": "sacct", "mlsbench-sacct": "sacct",
    "scancel": "scancel", "mlsbench-scancel": "scancel",
    "sbatch": "sbatch", "mlsbench-sbatch": "sbatch",
}

_DISPATCH = {
    "squeue": cmd_squeue,
    "sacct": cmd_sacct,
    "scancel": cmd_scancel,
    "sbatch": cmd_sbatch,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if argv and argv[0] in _COMMAND_ALIASES:
        command = _COMMAND_ALIASES[argv.pop(0)]
    else:
        command = _COMMAND_ALIASES.get(Path(sys.argv[0]).name, Path(sys.argv[0]).name)

    handler = _DISPATCH.get(command)
    if handler:
        return handler(argv)

    print(f"Unknown command shim: {command}", file=sys.stderr)
    return 2


def squeue_main() -> None:
    raise SystemExit(main(["squeue", *sys.argv[1:]]))


def sacct_main() -> None:
    raise SystemExit(main(["sacct", *sys.argv[1:]]))


def scancel_main() -> None:
    raise SystemExit(main(["scancel", *sys.argv[1:]]))


def sbatch_main() -> None:
    raise SystemExit(main(["sbatch", *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
