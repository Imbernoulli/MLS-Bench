"""MLS-Bench score normalization system.

Provides unified [0,1] scoring across tasks with different metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register_score_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``score`` subcommand with the main CLI parser."""
    p_score = subparsers.add_parser(
        "score",
        help="Compute normalized scores from leaderboard results",
    )

    p_score.add_argument(
        "task", nargs="?", default=None,
        help="Task name to score. Omit with --all to score everything.",
    )
    p_score.add_argument(
        "--all", action="store_true", dest="score_all",
        help="Score all tasks that have a score_spec.py",
    )
    p_score.add_argument(
        "--model", default=None,
        help="Only score a specific model (default: all models in leaderboard)",
    )
    p_score.add_argument(
        "--format", choices=["table", "json", "csv"], default="table",
        dest="out_format",
        help="Output format (default: table)",
    )
    p_score.add_argument(
        "--summary", action="store_true",
        help="Only show task-level scores (with --all)",
    )
    p_score.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-term details",
    )
    p_score.add_argument(
        "--check", action="store_true",
        help="Validate score_spec files without scoring",
    )
    p_score.add_argument(
        "--anchors", action="store_true",
        help="Show baseline anchors for a task",
    )
    p_score.add_argument(
        "--autogen", action="store_true",
        help="Auto-generate score_spec.py templates from leaderboard data",
    )
    p_score.add_argument(
        "--force", action="store_true",
        help="Overwrite existing score_spec.py (with --autogen)",
    )
    p_score.add_argument(
        "--dry-run", action="store_true",
        help="Print generated spec without writing (with --autogen)",
    )

    p_score.set_defaults(func=cmd_score)


def cmd_score(args: argparse.Namespace) -> None:
    """Handler for the ``mlsbench score`` subcommand."""
    from mlsbench import PROJECT_ROOT
    tasks_dir = PROJECT_ROOT / "tasks"

    if args.autogen:
        _cmd_autogen(args, tasks_dir)
        return

    if args.anchors:
        _cmd_anchors(args, tasks_dir)
        return

    if args.check:
        _cmd_check(args, tasks_dir)
        return

    if args.score_all:
        _cmd_score_all(args, tasks_dir)
    elif args.task:
        _cmd_score_task(args, tasks_dir)
    else:
        print("Error: specify a task name or use --all", file=sys.stderr)
        sys.exit(1)


def _cmd_score_task(args: argparse.Namespace, tasks_dir: Path) -> None:
    from mlsbench.scoring.evaluate import evaluate_task
    from mlsbench.scoring.report import format_task_detail

    results = evaluate_task(args.task, model=args.model, tasks_dir=tasks_dir)
    if not results:
        print(f"No results for task '{args.task}'. Check score_spec.py and leaderboard.csv exist.")
        return

    if args.out_format == "json":
        from mlsbench.scoring.report import results_to_json
        print(results_to_json({args.task: results}))
    elif args.out_format == "csv":
        from mlsbench.scoring.report import results_to_csv
        print(results_to_csv({args.task: results}))
    else:
        print(format_task_detail(results, verbose=args.verbose))


def _cmd_score_all(args: argparse.Namespace, tasks_dir: Path) -> None:
    from mlsbench.scoring.evaluate import evaluate_all
    from mlsbench.scoring.report import format_summary_table, format_task_detail

    models = [args.model] if args.model else None
    all_results = evaluate_all(tasks_dir=tasks_dir, models=models)

    if not all_results:
        print("No scored tasks found. Ensure tasks have score_spec.py files.")
        return

    if args.out_format == "json":
        from mlsbench.scoring.report import results_to_json
        print(results_to_json(all_results))
    elif args.out_format == "csv":
        from mlsbench.scoring.report import results_to_csv
        print(results_to_csv(all_results))
    else:
        if args.summary:
            print(format_summary_table(all_results, summary_only=True))
        else:
            # Print summary table first
            print(format_summary_table(all_results))
            print()
            # Then per-task details if verbose
            if args.verbose:
                for task_name in sorted(all_results.keys()):
                    print(format_task_detail(all_results[task_name], verbose=True))
                    print()


def _cmd_check(args: argparse.Namespace, tasks_dir: Path) -> None:
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import load_expanded_spec
    from mlsbench.scoring.spec import leaderboard_declared_metrics, validate_score_spec

    targets: list[Path] = []
    if args.task:
        targets.append(tasks_dir / args.task)
    else:
        for d in sorted(tasks_dir.iterdir()):
            if d.is_dir() and (d / "score_spec.py").exists():
                targets.append(d)

    if not targets:
        print("No score_spec.py files found.")
        return

    total_warns = 0
    for task_dir in targets:
        task_name = task_dir.name
        anchors = BaselineAnchors(task_dir)
        spec = load_expanded_spec(task_dir, anchors)
        if spec is None:
            print(f"  {task_name}: no score_spec.py")
            continue

        if (task_dir / "leaderboard.csv").exists():
            available_metrics = sorted(set(anchors.metric_columns()) | set(leaderboard_declared_metrics(task_dir)))
        else:
            available_metrics = [ts.metric for ts in spec.terms.values()]
        warns = validate_score_spec(spec, available_metrics)
        if not (task_dir / "leaderboard.csv").exists():
            warns.append("leaderboard.csv not found; baseline anchors unavailable")
        else:
            missing_anchor_metrics: list[str] = []
            for tspec in spec.terms.values():
                if tspec.role == "drop":
                    continue
                if tspec.metric not in available_metrics:
                    continue
                if anchors.worst_for(tspec.metric, tspec.direction) is None:
                    missing_anchor_metrics.append(tspec.metric)
            if missing_anchor_metrics:
                metrics = sorted(set(missing_anchor_metrics))
                preview = ", ".join(metrics[:8])
                if len(metrics) > 8:
                    preview += f", ... (+{len(metrics) - 8} more)"
                warns.append(
                    f"{len(metrics)} metric(s) have no current baseline anchor: {preview}"
                )

        status = "OK" if not warns else f"{len(warns)} warning(s)"
        print(f"  {task_name}: {len(spec.terms)} terms, {len(spec.settings)} settings — {status}")
        for w in warns:
            print(f"    WARNING: {w}")
            total_warns += 1

    print(f"\nChecked {len(targets)} task(s), {total_warns} warning(s) total.")


def _cmd_anchors(args: argparse.Namespace, tasks_dir: Path) -> None:
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import load_expanded_spec

    if not args.task:
        print("Error: --anchors requires a task name", file=sys.stderr)
        sys.exit(1)

    task_dir = tasks_dir / args.task
    anchors = BaselineAnchors(task_dir)
    spec = load_expanded_spec(task_dir, anchors)
    directions: dict[str, str] = {}
    if spec:
        for term in spec.terms.values():
            previous = directions.get(term.metric)
            if previous is None:
                directions[term.metric] = term.direction
            elif previous != term.direction:
                directions[term.metric] = "mixed"

    bl_names = anchors.baseline_names()
    if not bl_names:
        print(f"No baselines found for task '{args.task}'.")
        return

    print(f"Task: {args.task}")
    print(f"Baselines: {', '.join(bl_names)}")
    print()

    if spec and spec.terms:
        cols = []
        seen = set()
        for term in spec.terms.values():
            if term.metric not in seen:
                cols.append(term.metric)
                seen.add(term.metric)
    else:
        cols = anchors.metric_columns()
    if not cols:
        print("No metric columns found.")
        return

    max_col = max(len(c) for c in cols)
    print(f"  {'Metric':<{max_col}}  {'Direction':>9}  {'Worst':>12}  {'Best':>12}")
    print(f"  {'-' * max_col}  {'-' * 9}  {'-' * 12}  {'-' * 12}")
    for c in cols:
        a = anchors.get(c)
        if a:
            direction = directions.get(c, "raw")
            if direction in {"higher", "lower"}:
                worst = anchors.worst_for(c, direction)
                best = anchors.best_for(c, direction)
            else:
                worst = a.worst
                best = a.best
            print(f"  {c:<{max_col}}  {direction:>9}  {worst:>12.4f}  {best:>12.4f}")
        else:
            direction = directions.get(c, "raw")
            print(f"  {c:<{max_col}}  {direction:>9}  {'missing':>12}  {'missing':>12}")


def _cmd_autogen(args: argparse.Namespace, tasks_dir: Path) -> None:
    from mlsbench.scoring.autogen import autogen_score_spec

    targets: list[str] = []
    if args.task:
        targets.append(args.task)
    else:
        # All tasks with leaderboard but no score_spec (unless --force)
        for d in sorted(tasks_dir.iterdir()):
            if not d.is_dir():
                continue
            if not (d / "leaderboard.csv").exists():
                continue
            if (d / "score_spec.py").exists() and not args.force:
                continue
            targets.append(d.name)

    if not targets:
        print("No tasks to generate score_spec for.")
        return

    generated = 0
    skipped = 0
    for task_name in targets:
        task_dir = tasks_dir / task_name
        if (task_dir / "score_spec.py").exists() and not args.force:
            skipped += 1
            continue

        source = autogen_score_spec(task_name, tasks_dir=tasks_dir, dry_run=args.dry_run)

        if args.dry_run:
            print(f"=== {task_name} ===")
            print(source)
            print()
        else:
            print(f"  {task_name}: generated")
        generated += 1

    action = "would generate" if args.dry_run else "generated"
    print(f"\n{action} {generated} score_spec(s), skipped {skipped} existing.")
