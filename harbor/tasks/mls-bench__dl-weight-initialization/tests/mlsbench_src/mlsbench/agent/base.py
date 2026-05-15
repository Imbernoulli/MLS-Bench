"""BaseAgent: abstract base class for MLS-Bench agents."""

import copy
import importlib.util
import json
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from mlsbench import PROJECT_ROOT
from mlsbench.agent.logger import RunLogger


class BaseAgent(ABC):
    """Abstract agent that runs the modify→test loop against a task workspace."""

    # Subclasses override this to tag leaderboard rows with the agent type
    # (e.g. "openevolve", "discover"). Empty string = no prefix, which keeps
    # historical InteractiveAgent rows backward-compatible.
    agent_label: str = ""

    def __init__(self, task_name: str, global_config: dict, workspace_root: str | Path | None = None):
        self.task_name = task_name
        self.global_config = global_config
        self.project_root = PROJECT_ROOT
        self.workspace_root = (
            Path(workspace_root) if workspace_root else PROJECT_ROOT / "vendor" / "workspace"
        )

        # Load per-task config
        task_dir = self.project_root / "tasks" / task_name

        config_path = task_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        self.config_task: dict = config
        self.config_edit: list[dict] = config.get("files", [])

        from mlsbench.agent.tools import load_pre_edit_ops, load_mid_edit_ops
        pkg_configs_dir = self.project_root / "vendor" / "pkg_configs"
        self.pre_edit_ops: list[dict] = load_pre_edit_ops(config, pkg_configs_dir)
        self.mid_edit_ops: list[dict] = load_mid_edit_ops(task_name, self.project_root / "tasks")

        # Leaderboard (per-task CSV)
        from mlsbench.agent.leaderboard import Leaderboard
        self.leaderboard = Leaderboard(task_dir / "leaderboard.csv")

        # Instantiate tools
        from mlsbench.agent.tools import WorkspaceTools
        use_cuda_override = global_config.get("use_cuda")
        if use_cuda_override is not None:
            use_cuda_override = bool(use_cuda_override)
        self.tools = WorkspaceTools(
            task_name=task_name,
            config_task=self.config_task,
            config_edit=self.config_edit,
            workspace_root=self.workspace_root,
            project_root=self.project_root,
            max_tests=global_config.get("max_tests", 5),
            model_name=(
                (f"{self.agent_label}:" if self.agent_label else "")
                + global_config.get("model", "")
            ),
            leaderboard=self.leaderboard,
            save_path=global_config.get("save_path", ""),
            seeds=self.config_task.get("seeds") or global_config.get("seeds"),
            slurm_config=global_config.get("slurm"),
            container_runtime=global_config.get("container_runtime", "apptainer"),
            use_cuda=use_cuda_override,
            platform=global_config.get("platform", ""),
            gpu_devices=global_config.get("gpu_devices", ""),
            global_config=global_config,
            allow_web_search=global_config.get("allow_web_search", False),
            tavily_api_key=(global_config.get("providers", {}).get("tavily", {}) or {}).get("api_key", ""),
            max_web_credits=global_config.get("max_web_credits", 20),
            hide_hidden=global_config.get("hide_hidden", False),
        )

        # Resolve --extra-context. If requested, the matching context file MUST
        # exist for this task — otherwise abort, since silently no-op-ing would
        # let the agent run without the prompt the user explicitly asked for.
        self._extra_context_request: str | None = global_config.get("extra_context") or None
        self._extra_context_text: str = ""
        if self._extra_context_request in ("baseline", "theory"):
            suffix = ("baseline_derivation_context.md"
                      if self._extra_context_request == "baseline"
                      else "deep_theory_context.md")
            ctx_path = (
                self.project_root
                / "context_packs" / "science_priors_10tasks_v3" / "contexts"
                / f"{task_name}__{suffix}"
            )
            if not ctx_path.exists():
                raise FileNotFoundError(
                    f"--extra-context {self._extra_context_request} requested but "
                    f"context file does not exist for task {task_name!r}: {ctx_path}"
                )
            self._extra_context_text = ctx_path.read_text()
            self.tools.extra_context = self._extra_context_request

        self.max_steps: int = global_config.get("max_steps", 20)
        self.verbose: bool = global_config.get("verbose", False)

        # Agent conversation & file logger
        exp_name = global_config.get("model", "unknown")
        # Allow concurrent runs of the same model to log to distinct dirs by
        # setting MLSBENCH_LOG_LABEL=<suffix> (appended to exp_name).
        log_label = os.environ.get("MLSBENCH_LOG_LABEL", "").strip()
        if log_label:
            exp_name = f"{exp_name}__{log_label}"
        logs_dir = self.project_root / "logs" / task_name / exp_name / "agent"
        self.logger = RunLogger(logs_dir)
        # Surface the log dir to WorkspaceTools so sidecar artifacts
        # (e.g. param_counts.jsonl from budget_check logging) land next
        # to messages.jsonl.
        try:
            self.tools.agent_log_dir = logs_dir
        except Exception:
            pass

        # Running token totals, summed over every LLM call the agent makes.
        # Populated from the ``usage`` key that ModelClient.call now returns.
        self._token_totals: dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "calls": 0,
        }

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------
    def _accumulate_tokens(self, usage: dict) -> None:
        """Sum a per-call usage dict into self._token_totals."""
        for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                  "cached_tokens", "cache_creation_tokens"):
            v = usage.get(k) or 0
            if isinstance(v, (int, float)):
                self._token_totals[k] += int(v)
        self._token_totals["calls"] += 1

    # ------------------------------------------------------------------
    # Workspace setup
    # ------------------------------------------------------------------

    def setup_workspace(self) -> None:
        """Copy external packages to the workspace and apply pre_edit operations."""
        workspace_task_dir = self.tools.workspace_task_dir
        workspace_task_dir.mkdir(parents=True, exist_ok=True)
        ext_dir = self.project_root / "vendor" / "external_packages"

        # Collect all packages from all test_cmds (deduplicated)
        all_packages: list[str] = []
        seen_norm: set[str] = set()
        for entry in self.tools.test_cmd_entries:
            pkg = entry.get("package")
            if pkg:
                norm = pkg.lower().replace("-", "").replace("_", "")
                if norm not in seen_norm:
                    seen_norm.add(norm)
                    all_packages.append(pkg)

        any_copied = False
        for pkg in all_packages:
            dst = workspace_task_dir / pkg
            if dst.exists():
                print(f"[workspace] Already exists, skipping copy: {dst}")
                continue

            # Find the source package (case-insensitive search)
            src = None
            norm = pkg.lower().replace("-", "").replace("_", "")
            for d in ext_dir.iterdir():
                if d.is_dir() and d.name.lower().replace("-", "").replace("_", "") == norm:
                    src = d
                    break
            if src is None:
                raise FileNotFoundError(
                    f"External package '{pkg}' not found in {ext_dir}"
                )

            print(f"[workspace] Copying {src} -> {dst}")
            shutil.copytree(src, dst, symlinks=True)
            print(f"[workspace] Copy complete: {dst}")
            any_copied = True

        # Apply pre_edit ops (package-level patches).
        # - 'create' ops always run (reset editable files to template state).
        # - Mutation ops (insert/replace/delete) only run on fresh copies to
        #   avoid double-applying (e.g. re-injecting TRAIN_METRICS lines).
        if self.pre_edit_ops:
            print(f"[workspace] Applying pre_edit ({len(self.pre_edit_ops)} op(s), fresh_copy={any_copied})")
            self.tools.apply_pre_edit(self.pre_edit_ops, mutations_only_if_fresh=not any_copied)
            print("[workspace] Pre-edit applied")

        # Apply mid_edit ops (task-specific workspace setup).
        if self.mid_edit_ops:
            print(f"[workspace] Applying mid_edit ({len(self.mid_edit_ops)} op(s), fresh_copy={any_copied})")
            self.tools.apply_pre_edit(self.mid_edit_ops, mutations_only_if_fresh=not any_copied)
            print("[workspace] Mid-edit applied")

    # ------------------------------------------------------------------
    # Initial prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _adjusted_edit_ranges(
        edit_ranges: list[dict],
        edit_file: Path,
        target_file: str | None = None,
    ) -> list[dict]:
        """Compute edit ranges adjusted for line-count changes from baseline ops.

        Ops are assumed to be ordered bottom-to-top (codebase convention) so
        that each op's ``start_line``/``end_line`` refers to the file state
        *after* all preceding (= lower-in-file) ops have been applied.
        """
        spec = importlib.util.spec_from_file_location("_bl_adj", edit_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ops = getattr(module, "OPS", [])
        if target_file:
            ops = [op for op in ops if op.get("file", target_file) == target_file]
        if not ops:
            return list(edit_ranges)

        # Mutable 1-based [start, end] inclusive boundaries.
        bounds: list[list[int] | None] = []
        for r in edit_ranges:
            if r["start"] == -1:
                bounds.append(None)
            else:
                bounds.append([r["start"], r["end"]])

        for op in ops:
            op_type = op.get("op")
            if op_type == "replace":
                op_s = op["start_line"]
                op_e = op["end_line"]
                content = op.get("content", "")
                if not content.endswith("\n"):
                    content += "\n"
                delta = len(content.splitlines()) - (op_e - op_s + 1)
            elif op_type == "insert":
                op_s = op["after_line"]
                op_e = op_s  # insertion point
                content = op.get("content", "")
                if not content.endswith("\n"):
                    content += "\n"
                delta = len(content.splitlines())
            elif op_type == "delete":
                op_s = op["start_line"]
                op_e = op.get("end_line", op_s)
                delta = -(op_e - op_s + 1)
            else:
                continue

            for b in bounds:
                if b is None:
                    continue
                if op_type == "replace":
                    if op_s >= b[0] and op_e <= b[1]:
                        # Op is within this range → extend/shrink end.
                        b[1] += delta
                    elif op_e < b[0]:
                        # Op is entirely before this range → shift both.
                        b[0] += delta
                        b[1] += delta
                elif op_type == "insert":
                    if op_s >= b[0] and op_s <= b[1]:
                        b[1] += delta
                    elif op_s < b[0]:
                        b[0] += delta
                        b[1] += delta
                elif op_type == "delete":
                    if op_s >= b[0] and op_e <= b[1]:
                        b[1] += delta
                    elif op_e < b[0]:
                        b[0] += delta
                        b[1] += delta

        return [
            {"start": -1, "end": -1} if b is None else {"start": b[0], "end": b[1]}
            for b in bounds
        ]

    @staticmethod
    def _apply_edit_ops(
        template_content: str,
        edit_file: Path,
        target_file: str | None = None,
    ) -> str:
        """Apply edit operations from a baseline edit file to template content.

        If target_file is given, only ops whose "file" field matches it are applied.
        Returns the modified content as a string.
        """
        spec = importlib.util.spec_from_file_location("baseline_edit", edit_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ops = getattr(module, "OPS", [])
        if target_file:
            ops = [op for op in ops if op.get("file", target_file) == target_file]

        lines = template_content.splitlines(keepends=True)
        for op in ops:
            op_type = op["op"]
            if op_type == "replace":
                start = op["start_line"]
                end = op["end_line"]
                content = op["content"]
                if not content.endswith("\n"):
                    content += "\n"
                content_lines = content.splitlines(keepends=True)
                lines[start - 1 : end] = content_lines
            elif op_type == "insert":
                after_line = op["after_line"]
                content = op["content"]
                if not content.endswith("\n"):
                    content += "\n"
                content_lines = content.splitlines(keepends=True)
                for i, cl in enumerate(content_lines):
                    lines.insert(after_line + i, cl)
            elif op_type == "delete":
                start = op["start_line"]
                end = op.get("end_line", start)
                del lines[start - 1 : end]

        return "".join(lines)

    def build_initial_prompt(self) -> str:
        """Build the initial user prompt with task description and annotated code."""
        task_dir = self.project_root / "tasks" / self.task_name
        rigorous = self.config_task.get("rigorous_codebase", False)
        baselines = self.config_task.get("baselines", {})

        # Task description
        task_desc_path = task_dir / "task_description.md"
        task_desc = (
            task_desc_path.read_text()
            if task_desc_path.exists()
            else "(no task description provided)"
        )

        sections = []
        if self._extra_context_text:
            kind_label = (
                "Baseline derivations"
                if self._extra_context_request == "baseline"
                else "Deep theoretical context"
            )
            sections.append(
                f"# {kind_label} (reference material)\n\n"
                f"{self._extra_context_text.rstrip()}\n"
            )
        sections.append(f"# Task: {self.task_name}\n\n{task_desc}")

        # Track first editable filename for the read-only file skip-logic below.
        editable_filename = next(
            (e["filename"] for e in self.config_edit if "edit" in e),
            None,
        )

        # Code sections for each config_edit entry that has read ranges.
        # Each file header includes an editability annotation so the model knows
        # exactly which files (and lines) it may change.
        for entry in self.config_edit:
            filename = entry["filename"]
            read_ranges = entry.get("read", [])
            if not read_ranges:
                continue

            editable = "edit" in entry

            # In rigorous mode, skip read-only files — replaced by baseline variants below.
            if rigorous and baselines and not editable:
                continue

            edit_ranges = entry.get("edit", [])
            if not editable:
                edit_note = "READ-ONLY — do not edit"
            elif not edit_ranges:
                edit_note = "READ-ONLY — do not edit"
            else:
                range_strs = [
                    "entire file" if r["start"] == -1 else f"lines {r['start']}–{r['end']}"
                    for r in edit_ranges
                ]
                edit_note = f"EDITABLE — {', '.join(range_strs)} only"

            try:
                path = self.tools._resolve_workspace_path(filename)
                all_lines = path.read_text().splitlines()
            except Exception as exc:
                sections.append(f"\n## {filename}  [{edit_note}]\n(could not read: {exc})")
                continue

            file_sections = []
            for rng in read_ranges:
                start = rng["start"]
                end = rng["end"]
                if start == -1 and end == -1:
                    numbered = "\n".join(
                        f"{i + 1:6d}: {line}" for i, line in enumerate(all_lines)
                    )
                    file_sections.append(numbered)
                else:
                    slice_lines = all_lines[start - 1 : end]
                    numbered = "\n".join(
                        f"{start + i:6d}: {line}" for i, line in enumerate(slice_lines)
                    )
                    file_sections.append(f"Lines {start}-{end}:\n{numbered}")

            if file_sections:
                content = "\n\n".join(file_sections)
                lang = "bash" if filename.endswith(".sh") else "python"
                header = (
                    f"\n## {filename}\n"
                    if not edit_note
                    else f"\n## {filename}  [{edit_note}]\n"
                )
                sections.append(f"{header}```{lang}\n{content}\n```")

        # In rigorous mode, generate baseline variants for each editable file.
        # Only shows the EDITABLE region of each baseline (FIXED parts are identical
        # to the template already shown above, so repeating them wastes context).
        if rigorous and baselines:
            for entry in self.config_edit:
                if "edit" not in entry:
                    continue
                bl_filename = entry["filename"]
                edit_ranges = entry.get("edit", [])
                try:
                    bl_path = self.tools._resolve_workspace_path(bl_filename)
                    bl_template = bl_path.read_text()
                except Exception:
                    continue

                for bl_name, bl_config in baselines.items():
                    edit_ops_rel = bl_config.get("edit_ops")
                    if not edit_ops_rel:
                        continue
                    edit_file = task_dir / edit_ops_rel
                    if not edit_file.exists():
                        continue

                    try:
                        generated = self._apply_edit_ops(
                            bl_template, edit_file, target_file=bl_filename
                        )
                        # Skip if this baseline has no ops targeting this file
                        if generated.rstrip() == bl_template.rstrip():
                            continue
                        all_lines = generated.splitlines()

                        # Adjust edit ranges to account for line-count changes
                        # from baseline ops (e.g. a 60-line template region
                        # replaced by 100 lines of baseline code).
                        adj_ranges = self._adjusted_edit_ranges(
                            edit_ranges, edit_file, target_file=bl_filename
                        )

                        # Only show EDITABLE regions (with a few lines of context)
                        context_lines = 3
                        bl_parts: list[str] = []
                        for rng in adj_ranges:
                            start = rng["start"]
                            end = rng["end"]
                            if start == -1:
                                # Entire file is editable — show all
                                numbered = "\n".join(
                                    f"{i + 1:6d}: {line}" for i, line in enumerate(all_lines)
                                )
                                bl_parts.append(numbered)
                            else:
                                ctx_start = max(0, start - 1 - context_lines)
                                ctx_end = min(len(all_lines), end + context_lines)
                                slice_lines = all_lines[ctx_start:ctx_end]
                                numbered = "\n".join(
                                    f"{ctx_start + i + 1:6d}: {line}"
                                    for i, line in enumerate(slice_lines)
                                )
                                bl_parts.append(f"Lines {start}–{end}:\n{numbered}")

                        if bl_parts:
                            content = "\n\n".join(bl_parts)
                            sections.append(
                                f"\n## {bl_name} baseline — editable region  "
                                "[READ-ONLY — reference implementation]\n"
                                f"```python\n{content}\n```"
                            )
                    except Exception as exc:
                        sections.append(
                            f"\n## {bl_name} baseline  [READ-ONLY]\n"
                            f"(could not generate: {exc})"
                        )

        # Evaluation commands + compute budget
        test_cmds = self.config_task.get("test_cmds", [])
        hide_hidden = bool(getattr(self.tools, "hide_hidden", False))
        if hide_hidden:
            test_cmds = [e for e in test_cmds if not e.get("hidden")]
        if test_cmds:
            cmd_lines = [
                f"  - `{e['cmd']}` → label: `{e['label']}`"
                for e in test_cmds
                if e.get("cmd") and e.get("label")
            ]
            if cmd_lines:
                sections.append(
                    "\n## Evaluation Commands\nYour algorithm is evaluated by running:\n"
                    + "\n".join(cmd_lines)
                )

            # Compute budget table
            budget_rows = []
            for e in test_cmds:
                if not e.get("cmd") or not e.get("label"):
                    continue
                compute = float(e.get("compute", 1) or 1)
                time_str = e.get("time", "1:00:00")
                # Format compute as GPU description
                if compute >= 1.0:
                    gpu_desc = f"{int(compute)} GPU(s)" if compute == int(compute) else f"{compute:.1f} GPU(s)"
                else:
                    frac = f"1/{int(round(1/compute))}" if compute > 0 else "0"
                    gpu_desc = f"{frac} GPU"
                budget_rows.append(f"| `{e['label']}` | {gpu_desc} | {time_str} |")
            if budget_rows:
                sections.append(
                    "\n## Compute Budget\n"
                    "All evaluation runs on **NVIDIA H100 80GB** GPU(s). "
                    "Your algorithm must complete within the time limits below. "
                    "If a command exceeds its time limit, the run is killed and the result is "
                    "**invalid** (it will not count as a valid test result). "
                    "Design your model to be efficient enough to train and evaluate within these constraints.\n\n"
                    "| Command | GPUs | Time Limit |\n"
                    "| --- | --- | --- |\n"
                    + "\n".join(budget_rows)
                )

        # Baseline results (loaded from per-task leaderboard)
        labels = [e["label"] for e in test_cmds if e.get("label")]
        baseline_names = set((self.config_task.get("baselines") or {}).keys())
        baseline_rows: list[tuple[str, dict]] = []
        for r in self.leaderboard.all_records():
            model = str(r.get("model", ""))
            if model.startswith("baseline:"):
                name = model[len("baseline:"):]
            elif model in baseline_names:
                # Older task rows sometimes used the bare baseline name.
                name = model
            else:
                continue
            if baseline_names and name not in baseline_names:
                continue
            baseline_rows.append((name, r))

        def _row_metric_count(record: dict) -> int:
            count = 0
            for key, value in record.items():
                if key in {"timestamp", "model", "is_final", "seed"}:
                    continue
                if str(key).startswith("elapsed_") or str(key).endswith("_std"):
                    continue
                if value in ("", None):
                    continue
                count += 1
            return count

        def _is_final_baseline_row(record: dict) -> bool:
            return str(record.get("is_final", "")).lower() == "true"

        def _baseline_row_priority(record: dict) -> tuple[int, bool, bool, str]:
            return (
                _row_metric_count(record),
                record.get("seed") == "mean",
                _is_final_baseline_row(record),
                str(record.get("timestamp", "")),
            )

        grouped_baselines: dict[str, list[dict]] = {}
        for name, record in baseline_rows:
            grouped_baselines.setdefault(name, []).append(record)

        baseline_records = []
        for name, records in grouped_baselines.items():
            records_with_metrics = [r for r in records if _row_metric_count(r) > 0]
            if not records_with_metrics:
                continue
            chosen = max(records_with_metrics, key=_baseline_row_priority)
            baseline_records.append((name, chosen))
        if baseline_records:
            header = ["baseline"] + labels
            def _find_metric(record: dict, label: str):
                """Find metric value in record by label, handling key normalization."""
                # Direct match first
                if label in record:
                    return record[label]
                # Try normalized: replace hyphens with underscores, match suffix
                norm = label.replace("-", "_")
                for k, v in record.items():
                    if k == norm or k.endswith("_" + norm):
                        return v
                return None

            rows = []
            for name, r in sorted(baseline_records):
                cells = [name]
                for l in labels:
                    val = _find_metric(r, l)
                    if isinstance(val, (int, float)):
                        cells.append(f"{val:.2f}")
                    else:
                        cells.append(str(val) if val is not None else "N/A")
                rows.append(cells)
            md = (
                "| " + " | ".join(header) + " |\n"
                "| " + " | ".join(["---"] * len(header)) + " |\n"
                + "\n".join("| " + " | ".join(row) + " |" for row in rows)
            )
            sections.append(
                "\n## Baseline Results\n"
                "Beat these with your algorithm:\n\n"
                + md
            )

        # Budget summary: tell the model exactly how many steps/tests it has
        max_tests = self.tools.max_tests
        budget_lines = [
            f"- **Action budget**: {self.max_steps} total tool calls "
            "(every edit / test / undo / web_search / web_extract counts; submit does not)",
            f"- **Test invocations**: at most {max_tests} "
            "(each test() call also consumes one action from the budget above)",
        ]
        if max_tests >= 2:
            budget_lines.append(
                "- You **must** iterate at least once "
                "(edit → test → review → edit → test) before submitting."
            )
        else:
            budget_lines.append(
                "- ⚠️ **CRITICAL — single-test mode (max_tests=1)**: "
                "your one and only `test()` call is automatically the FINAL "
                "submission. Whatever metrics it returns are written to the "
                "leaderboard; whatever bugs it hits are recorded as a failed "
                "submission with empty metrics — there is **no second chance**."
            )
            budget_lines.append(
                "- Before you call `test()`, run through this checklist mentally: "
                "(1) tensor shapes match between your model layers and the "
                "expected input/output, (2) dtypes / device are consistent, "
                "(3) any new module is actually used in `forward()`, "
                "(4) loss is finite on a tiny dummy input, "
                "(5) you handled the corner cases the task description warns about."
            )
            budget_lines.append(
                "- If you are unsure, spend remaining edit budget tightening the "
                "code rather than rushing to test. **A crashed test is a wasted "
                "submission.** Only call `test()` when you can defend each line."
            )
        sections.append(
            "\n## Your Budget\n" + "\n".join(budget_lines)
        )

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Abstract: get next action
    # ------------------------------------------------------------------

    @abstractmethod
    def get_action(self, messages: list) -> dict | None:
        """Return a tool_use dict {'name': str, 'input': dict, ...}, or None to stop."""

    # ------------------------------------------------------------------
    # Pretty-print model actions (git-style colored diff)
    # ------------------------------------------------------------------

    # ANSI color codes
    _RED = "\033[31m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _CYAN = "\033[36m"
    _BOLD = "\033[1m"
    _DIM = "\033[2m"
    _RESET = "\033[0m"

    _MAGENTA = "\033[35m"

    def _log_thinking(self, thinking: str):
        """Pretty-print model thinking/reasoning."""
        D, M, RST = self._DIM, self._MAGENTA, self._RESET
        lines = thinking.strip().splitlines()
        limit = None if self.verbose else 20
        print(f"{M}{'─' * 60}{RST}")
        print(f"{M}Thinking:{RST}")
        show = lines if limit is None else lines[:limit]
        for line in show:
            print(f"{D}  {line}{RST}")
        if limit is not None and len(lines) > limit:
            print(f"{D}  ... ({len(lines) - limit} more lines, use -v to show all){RST}")
        print(f"{M}{'─' * 60}{RST}")

    def _log_result(self, result: str):
        """Pretty-print a tool result with color coding."""
        R, G, D, RST = self._RED, self._GREEN, self._DIM, self._RESET
        result_str = str(result)
        is_error = result_str.startswith("ERROR")

        # First line = status
        first_line = result_str.split("\n", 1)[0]
        color = R if is_error else G
        print(f"\n{color}{'▶' if not is_error else '✘'} {first_line}{RST}")

        # Remaining lines (file snapshot, etc.)
        rest = result_str.split("\n", 1)[1] if "\n" in result_str else ""
        if rest:
            rest_lines = rest.splitlines()
            limit = None if self.verbose else 60
            show = rest_lines if limit is None else rest_lines[:limit]
            for line in show:
                print(f"{D}  {line}{RST}")
            if limit is not None and len(rest_lines) > limit:
                print(f"{D}  ... ({len(rest_lines) - limit} more lines, use -v to show all){RST}")
        print()

    def _log_action(self, step: int, tool_name: str, tool_input: dict):
        """Pretty-print a model action in git-diff style with ANSI colors."""
        B, R, G, Y, C, D, RST = (
            self._BOLD, self._RED, self._GREEN,
            self._YELLOW, self._CYAN, self._DIM, self._RESET,
        )

        print(f"{Y}{'─' * 60}{RST}")
        print(f"{B}{Y}Step {step}{RST}  {B}{tool_name}{RST}")
        print(f"{Y}{'─' * 60}{RST}")

        if tool_name == "edit":
            self._log_edit_diff(tool_input, B, R, G, Y, C, D, RST)
        elif tool_name == "test":
            will_be_final = self.tools.test_count + 1 >= self.tools.max_tests
            if will_be_final:
                icon = f"{B}{Y}FINAL (max_tests reached){RST}"
            else:
                test_num = self.tools.test_count + 1
                icon = f"{C}test #{test_num}{RST}"
            print(f"  Running tests ({icon})")
        elif tool_name == "submit":
            n = tool_input.get("n", -1)
            print(f"  {B}{Y}Submitting test #{n} as FINAL{RST}")
        elif tool_name == "undo":
            n = tool_input.get("n", 1)
            print(f"  {Y}Reverting last {n} edit(s){RST}")
        elif tool_name == "web_search":
            q = tool_input.get("query", "")
            n = tool_input.get("max_results", 5)
            print(f"  {C}web_search{RST} {q!r}  (max_results={n})")
        elif tool_name == "web_extract":
            urls = tool_input.get("urls", [])
            q = tool_input.get("query", "")
            cps = tool_input.get("chunks_per_source", 3)
            url_preview = urls if isinstance(urls, list) else [urls]
            print(f"  {C}web_extract{RST} {len(url_preview)} url(s)  query={q!r}  chunks={cps}")
            for u in url_preview[:3]:
                print(f"    - {u}")
        else:
            import json as _json
            print(f"  {_json.dumps(tool_input, indent=2)}")

        print(f"{Y}{'─' * 60}{RST}")

    def _print_lines(self, lines: list[str], fmt: str, start: int, limit: int | None, D, RST):
        """Helper: print lines with optional truncation. fmt has {num} and {line} placeholders."""
        show = lines if limit is None else lines[:limit]
        for i, line in enumerate(show):
            print(fmt.format(num=start + i, line=line))
        if limit is not None and len(lines) > limit:
            print(f"{D}  ... ({len(lines) - limit} more lines, use -v to show all){RST}")

    def _log_edit_diff(self, tool_input: dict, B, R, G, Y, C, D, RST):
        """Show edit operations as a git-style diff."""
        op = tool_input.get("op", "?")
        fname = tool_input.get("filename", "?")
        content = tool_input.get("content", "")
        new_lines = content.splitlines()
        limit = None if self.verbose else 40

        if op == "replace":
            s = tool_input.get("start_line", 0)
            e = tool_input.get("end_line", 0)
            # Read old lines from file before edit
            old_lines: list[str] = []
            try:
                path = self.tools._resolve_workspace_path(fname)
                if path.exists():
                    all_lines = path.read_text().splitlines()
                    old_lines = all_lines[s - 1 : e]
            except Exception:
                pass

            print(f"{B}diff --agent a/{fname} b/{fname}{RST}")
            print(f"{C}@@ -{s},{e - s + 1} +{s},{len(new_lines)} @@{RST}")
            self._print_lines(old_lines, f"{R}-{{num:4d}} | {{line}}{RST}", s, limit, D, RST)
            self._print_lines(new_lines, f"{G}+{{num:4d}} | {{line}}{RST}", s, limit, D, RST)

        elif op == "insert":
            after = tool_input.get("after_line", 0)
            print(f"{B}diff --agent a/{fname} b/{fname}{RST}")
            print(f"{C}@@ insert after line {after}: +{len(new_lines)} lines @@{RST}")
            self._print_lines(new_lines, f"{G}+{{num:4d}} | {{line}}{RST}", after + 1, limit, D, RST)

        elif op == "create":
            print(f"{B}new file: {fname}{RST}")
            print(f"{C}@@ +1,{len(new_lines)} @@{RST}")
            self._print_lines(new_lines, f"{G}+{{num:4d}} | {{line}}{RST}", 1, limit, D, RST)

        else:
            print(f"  {Y}op={op}{RST}  file={fname}")

    # ------------------------------------------------------------------
    # Resume from existing log
    # ------------------------------------------------------------------

    def resume_from_log(self) -> tuple[list[dict], bool]:
        """Rebuild conversation and restore tool state from existing messages.jsonl.

        Returns (messages, success). On success, tools.step_count, test_count,
        _test_history, and workspace files are restored.
        """
        if not self.logger.has_messages():
            return [], False

        records = self.logger.read_messages()
        if not records:
            return [], False

        # 1) Rebuild the messages list (Anthropic-format conversation)
        messages: list[dict] = []
        step_count = 0
        test_count = 0
        test_history: list[dict] = []
        done = False
        last_tool_name = None
        _cancelled_test_pairs: list[tuple[int, int]] = []

        # Track which original records to keep on disk after resume.  Start
        # by keeping everything, then mark drops as we detect orphan /
        # cancelled entries so we can rewrite messages.jsonl cleanly at the
        # end and prevent duplicate accumulation across resume cycles.
        rec_keep: list[bool] = [True] * len(records)
        # msg_to_rec[i] = index into `records` that produced messages[i].
        # -1 means "not tied to a specific record" (e.g., generated later).
        msg_to_rec: list[int] = []

        # Track the tool_name associated with the pending (unpaired) assistant msg
        pending_tool_name = None
        pending_rec_idx = -1

        for rec_idx, rec in enumerate(records):
            role = rec.get("role")

            if role == "_meta":
                # Note: do NOT restore exp_name here. setup_workspace() has
                # already created a fresh workspace with a new exp_name, and
                # overriding it would point tools at a stale/non-existent
                # directory.  The saved exp_name is preserved in the log for
                # reference but no longer applied.
                saved_exp = rec.get("exp_name")
                if saved_exp:
                    print(f"[resume] Original exp_name: {saved_exp} "
                          f"(keeping current: {getattr(self.tools, 'exp_name', '?')})")
                continue

            if role == "user" and rec.get("step") == 0:
                # Initial prompt
                messages.append({"role": "user", "content": rec["content"]})
                msg_to_rec.append(rec_idx)

            elif role == "assistant":
                tool_name = rec.get("tool_name", "")
                tool_input = rec.get("tool_input", {})

                # If previous message was also assistant (no tool_result in between),
                # drop the previous one — it was an incomplete action (crash, parse error).
                if messages and messages[-1].get("role") == "assistant":
                    messages.pop()
                    dropped_rec_idx = msg_to_rec.pop()
                    if 0 <= dropped_rec_idx < len(rec_keep):
                        rec_keep[dropped_rec_idx] = False
                    if pending_tool_name and pending_tool_name != "submit":
                        step_count = max(0, step_count - 1)
                    if pending_tool_name == "test":
                        test_count = max(0, test_count - 1)

                last_tool_name = tool_name
                pending_tool_name = tool_name
                pending_rec_idx = rec_idx

                # Build assistant message (tool_use block)
                tool_id = f"tool_{step_count + 1}"
                content_blocks = []

                # Note: we intentionally do NOT preserve thinking blocks
                # from previous turns during resume. Thinking blocks are
                # large and some proxy/API combinations (e.g., LiteLLM →
                # Bedrock) mishandle them, breaking tool_use/tool_result
                # pairing. The model can still reason from the tool results.

                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                })
                messages.append({"role": "assistant", "content": content_blocks})
                msg_to_rec.append(rec_idx)

                # Update counters (mirrors dispatch logic)
                if tool_name != "submit":
                    step_count += 1
                if tool_name == "test":
                    test_count += 1

            elif role == "tool_result":
                result = rec.get("result", "")
                meta = rec.get("meta") or {}
                pending_tool_name = None
                pending_rec_idx = -1
                # Find the tool_use_id from the last assistant message
                tool_id = f"tool_{step_count}"
                if messages and messages[-1].get("role") == "assistant":
                    for block in messages[-1].get("content", []):
                        if block.get("type") == "tool_use":
                            tool_id = block.get("id", tool_id)
                            break

                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": str(result),
                    }],
                })
                msg_to_rec.append(rec_idx)

                # Reconstruct _test_history from test results.
                # Only count tests that actually produced metrics — cancelled/
                # empty SLURM tests should not consume the test budget.
                _has_real_metrics = (
                    last_tool_name == "test" and result and (
                        "TEST_METRICS" in result
                        or "Final metrics" in result
                        or "[Leaderboard]" in result
                    )
                )
                if _has_real_metrics:
                    restored = meta.get("test_history_entry")
                    if isinstance(restored, dict):
                        test_history.append({
                            "feedback": restored.get("feedback", result),
                            "seed_metrics": [dict(m) for m in restored.get("seed_metrics", [])],
                            "seeds": list(restored.get("seeds", [])),
                            "had_failures": bool(restored.get("had_failures", False)),
                        })
                    else:
                        test_history.append({
                            "feedback": result,
                            "seed_metrics": [],  # legacy logs do not carry structured metrics
                            "seeds": [],
                            "had_failures": False,
                        })
                elif last_tool_name == "test" and result and not _has_real_metrics:
                    # Cancelled/empty test — mark for removal in post-processing
                    # pass (after the full messages list is built) to avoid
                    # breaking conversation structure during iteration.
                    # Do NOT adjust step_count here — it must stay monotonic
                    # to keep tool_use IDs unique across the iteration.
                    if len(messages) >= 2:
                        _cancelled_test_pairs.append((len(messages) - 2, len(messages) - 1))
                    print(f"[resume] Marked cancelled test (no metrics) for removal")

                # Check if agent was done — only if submit actually succeeded
                if last_tool_name == "submit":
                    result_str = str(result)[:200]
                    if "ERROR" not in result_str and "No valid metrics" not in result_str:
                        done = True

        # Post-process: remove cancelled test round-trips AND any submit
        # calls that reference those cancelled tests (which also lack metrics).
        if _cancelled_test_pairs:
            # Collect all indices to remove
            _remove_indices: set[int] = set()
            for use_idx, result_idx in _cancelled_test_pairs:
                _remove_indices.add(use_idx)
                _remove_indices.add(result_idx)

            # Also mark submit pairs that immediately follow a cancelled test
            # result and whose own result lacks valid metrics. Pattern:
            #   [cancelled_test_result_idx] -> [submit_use_idx] -> [submit_result_idx]
            for _, result_idx in _cancelled_test_pairs:
                scan = result_idx + 1
                while scan + 1 < len(messages):
                    msg_a = messages[scan]
                    msg_b = messages[scan + 1]
                    # Check if msg_a is assistant/submit and msg_b is its tool_result
                    is_submit = (
                        msg_a.get("role") == "assistant"
                        and any(
                            b.get("name") == "submit"
                            for b in msg_a.get("content", [])
                            if isinstance(b, dict) and b.get("type") == "tool_use"
                        )
                    )
                    is_result = (
                        msg_b.get("role") == "user"
                        and any(
                            b.get("type") == "tool_result"
                            for b in msg_b.get("content", [])
                            if isinstance(b, dict)
                        )
                    )
                    if is_submit and is_result:
                        # Check if this submit result also lacks real metrics
                        sub_content = ""
                        for b in msg_b.get("content", []):
                            if isinstance(b, dict) and b.get("type") == "tool_result":
                                sub_content += str(b.get("content", ""))
                        has_metrics = (
                            "Final metrics" in sub_content
                            or "[Leaderboard]" in sub_content
                            or "TEST_METRICS" in sub_content
                        )
                        if not has_metrics:
                            _remove_indices.add(scan)
                            _remove_indices.add(scan + 1)
                            scan += 2
                            continue
                    break

            # Mark the underlying records for removal from disk too.
            for i in _remove_indices:
                if 0 <= i < len(msg_to_rec):
                    r_idx = msg_to_rec[i]
                    if 0 <= r_idx < len(rec_keep):
                        rec_keep[r_idx] = False

            # Filter messages, keeping only non-removed ones
            messages = [m for i, m in enumerate(messages) if i not in _remove_indices]
            msg_to_rec = [r for i, r in enumerate(msg_to_rec) if i not in _remove_indices]
            n_removed_tests = len(_cancelled_test_pairs)
            n_removed_total = len(_remove_indices)
            step_count = max(0, step_count - n_removed_tests)
            test_count = max(0, test_count - n_removed_tests)
            print(f"[resume] Removed {n_removed_tests} cancelled test "
                  f"round-trip(s) and {(n_removed_total - 2 * n_removed_tests) // 2} "
                  f"related submit pair(s) from conversation")

            # Recalculate done: only True if a valid submit with real metrics
            # still exists in the remaining messages.
            done = False
            for i in range(len(messages) - 1):
                msg_a = messages[i]
                msg_b = messages[i + 1] if i + 1 < len(messages) else None
                if (msg_a.get("role") == "assistant" and msg_b
                        and msg_b.get("role") == "user"):
                    is_submit = any(
                        b.get("name") == "submit"
                        for b in msg_a.get("content", [])
                        if isinstance(b, dict) and b.get("type") == "tool_use"
                    )
                    if is_submit:
                        sub_text = ""
                        for b in msg_b.get("content", []):
                            if isinstance(b, dict) and b.get("type") == "tool_result":
                                sub_text += str(b.get("content", ""))
                        if ("ERROR" not in sub_text
                                and "No valid metrics" not in sub_text
                                and ("Final metrics" in sub_text
                                     or "[Leaderboard]" in sub_text
                                     or "TEST_METRICS" in sub_text)):
                            done = True

            # Re-number tool IDs sequentially to ensure API compatibility.
            # Each assistant tool_use block gets a new sequential ID, and the
            # corresponding user tool_result block is updated to match.
            _new_id_counter = 0
            for i, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    for block in msg.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            _new_id_counter += 1
                            old_id = block.get("id")
                            new_id = f"tool_{_new_id_counter}"
                            block["id"] = new_id
                            # Find the matching tool_result in the next message
                            if i + 1 < len(messages):
                                next_msg = messages[i + 1]
                                if next_msg.get("role") == "user":
                                    for rb in next_msg.get("content", []):
                                        if (isinstance(rb, dict)
                                                and rb.get("type") == "tool_result"
                                                and rb.get("tool_use_id") == old_id):
                                            rb["tool_use_id"] = new_id

        # 2) Restore workspace files from latest snapshots
        snapshots = self.logger.get_latest_snapshots()
        restored_files = 0
        for _sname, snap_path in snapshots.items():
            content = snap_path.read_text()
            # Try to find the original filename from the sanitized name
            # The snapshot content IS the file — write it to the workspace
            # We need to find which workspace file this maps to
            for entry in self.config_edit:
                if "edit" not in entry:
                    continue
                fn = entry["filename"]
                sanitized = self.logger._sanitize(fn)
                if _sname == sanitized:
                    try:
                        path = self.tools._resolve_workspace_path(fn)
                        path.write_text(content)
                        restored_files += 1
                        print(f"[resume] Restored {fn} from snapshot")
                    except Exception as exc:
                        print(f"[resume] Warning: could not restore {fn}: {exc}")
                    break

        # 3) Restore tool counters
        # If the last message is an assistant (no tool_result), the agent crashed
        # mid-execution.
        self._resume_has_pending_test = False
        if messages and messages[-1].get("role") == "assistant":
            if last_tool_name == "test" and hasattr(self, 'tools') and self.tools.slurm_executor:
                # Pending SLURM test — keep counters but mark for recovery.
                # run() will execute the test by recovering the orphaned SLURM job.
                messages.pop()
                dropped_rec_idx = msg_to_rec.pop() if msg_to_rec else -1
                if 0 <= dropped_rec_idx < len(rec_keep):
                    rec_keep[dropped_rec_idx] = False
                step_count = max(0, step_count - 1)
                test_count = max(0, test_count - 1)
                self.tools._recover_pending_slurm = True
                self._resume_has_pending_test = True
                print(f"[resume] Pending SLURM test detected — will recover orphaned job")
            else:
                # Non-SLURM or non-test action — roll back so it can be re-attempted.
                messages.pop()
                dropped_rec_idx = msg_to_rec.pop() if msg_to_rec else -1
                if 0 <= dropped_rec_idx < len(rec_keep):
                    rec_keep[dropped_rec_idx] = False
                step_count = max(0, step_count - 1)
                if last_tool_name == "test":
                    test_count = max(0, test_count - 1)
                print(f"[resume] Last action ({last_tool_name}) had no result — rolled back")

        self.tools.step_count = step_count
        # Derive test_count from actual successful test runs (test_history),
        # not from counting test() tool calls in the log.  Error responses
        # like "budget exhausted" don't produce history entries, so repeated
        # resume sessions no longer inflate the counter.
        effective_test_count = len(test_history)
        if self._resume_has_pending_test:
            effective_test_count += 1  # pending SLURM test will be recovered
        if effective_test_count != test_count:
            print(f"[resume] test_count adjusted: {test_count} (from log) → "
                  f"{effective_test_count} (from test_history)")
        self.tools.test_count = effective_test_count
        self.tools._test_history = test_history

        # If agent still has test budget remaining but was marked done
        # (from a previous submit), reopen the session so it can continue
        # iterating.  Strip the trailing submit round-trip and inject a
        # nudge telling the agent how many tests remain.
        if done and effective_test_count < self.tools.max_tests:
            # Remove only the trailing submit round-trip (assistant submit +
            # its tool_result). We must NOT strip non-submit tool_results
            # that happen to be at the end — those belong to test() calls.
            _stripped_submit = False
            while len(messages) >= 2:
                last = messages[-1]
                second_last = messages[-2]
                # Check: second-to-last = assistant/submit, last = its tool_result
                is_submit_call = (
                    second_last.get("role") == "assistant"
                    and isinstance(second_last.get("content"), list)
                    and any(
                        b.get("type") == "tool_use" and b.get("name") == "submit"
                        for b in second_last["content"]
                    )
                )
                is_its_result = (
                    last.get("role") == "user"
                    and isinstance(last.get("content"), list)
                    and any(b.get("type") == "tool_result" for b in last["content"])
                )
                if is_submit_call and is_its_result:
                    messages.pop()   # remove tool_result
                    r1 = msg_to_rec.pop() if msg_to_rec else -1
                    if 0 <= r1 < len(rec_keep):
                        rec_keep[r1] = False
                    messages.pop()   # remove assistant/submit
                    r2 = msg_to_rec.pop() if msg_to_rec else -1
                    if 0 <= r2 < len(rec_keep):
                        rec_keep[r2] = False
                    _stripped_submit = True
                else:
                    break
            done = False
            remaining = self.tools.max_tests - effective_test_count
            nudge = (
                f"[SYSTEM] Your previous run was interrupted by infrastructure issues. "
                f"Some of your earlier test attempts were lost to job cancellations. "
                f"You still have {remaining} test(s) remaining out of {self.tools.max_tests}. "
                f"You have {effective_test_count} valid result(s) so far. "
                f"Please continue improving your solution and use your remaining test budget."
            )
            messages.append({"role": "user", "content": nudge})
            print(f"[resume] Reopened session: {remaining} tests remaining, "
                  f"stripped submit, injected nudge")

        self.tools.done = done

        # Rewrite messages.jsonl to drop orphan/cancelled records from disk.
        # Prevents duplicate entries from accumulating across resume cycles.
        n_dropped = sum(1 for k in rec_keep if not k)
        if n_dropped > 0:
            clean_records = [r for r, keep in zip(records, rec_keep) if keep]
            self.logger.rewrite_messages(clean_records)
            print(f"[resume] Cleaned {n_dropped} orphan/cancelled record(s) from messages.jsonl")

        # 4) Recompute protected ranges from the restored file state
        # (The ranges may have shifted due to edits — re-derive from current file sizes
        # and the original config. For resumed runs with replace ops that change line count,
        # we scan the workspace file and update ranges.)
        for entry in self.config_edit:
            if "edit" not in entry:
                continue
            fn = entry["filename"]
            try:
                path = self.tools._resolve_workspace_path(fn)
                if path.exists():
                    current_lines = len(path.read_text().splitlines())
                    # Re-derive: we can't perfectly reconstruct, but the ranges
                    # logged in the last tool_result snapshot give us the info.
                    # Parse the last edit result for this file to get current range.
                    last_range = self._extract_range_from_results(records, fn)
                    if last_range:
                        self.tools.live_protected_ranges[fn] = self.tools._allowed_to_protected(
                            [last_range]
                        )
            except Exception:
                pass

        print(f"[resume] Restored state: {step_count} steps, {test_count} tests, "
              f"{restored_files} files, done={done}")
        return messages, True

    @staticmethod
    def _extract_range_from_results(records: list[dict], filename: str) -> list[int] | None:
        """Extract the latest editable range for a file from tool_result messages.

        Looks for patterns like 'editable: 214–413' in results.
        """
        import re
        last_range = None
        for rec in records:
            if rec.get("role") != "tool_result":
                continue
            result = str(rec.get("result", ""))
            if filename not in result:
                continue
            # Match "editable: N–M" or "editable: N-M"
            m = re.search(r"editable:\s*(\d+)[–\-](\d+)", result)
            if m:
                last_range = [int(m.group(1)), int(m.group(2))]
        return last_range

    # ------------------------------------------------------------------
    # Per-run structured summary (raw metrics for downstream analysis)
    # ------------------------------------------------------------------

    def _write_run_summary(self, *, extra: dict | None = None,
                           error: str | None = None) -> None:
        """Persist <log_dir>/summary.json with the run's raw metrics.

        Idempotent — overwrites atomically. Captures everything needed to
        reconstruct a (task, agent, run) entry for parity analysis without
        having to re-parse messages.jsonl: agent_label, task, model, the
        env-derived log_label, max_steps/max_tests, seeds, token totals, and
        the full _test_history (per-seed metrics + had_failures flag).
        Subclasses pass agent-specific fields through ``extra``.
        """
        from datetime import datetime, timezone

        try:
            log_dir = self.logger.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            test_history = []
            for entry in (getattr(self.tools, "_test_history", None) or []):
                test_history.append({
                    "feedback": entry.get("feedback", ""),
                    "seed_metrics": entry.get("seed_metrics", []),
                    "seeds": list(entry.get("seeds", []) or []),
                    "had_failures": bool(entry.get("had_failures", False)),
                })

            summary = {
                "schema_version": 1,
                "agent_label": self.agent_label or "interactive",
                "agent_class": type(self).__name__,
                "task": self.task_name,
                "model": self.global_config.get("model"),
                "log_label": os.environ.get("MLSBENCH_LOG_LABEL", "").strip(),
                "log_dir": str(log_dir),
                "exp_name": log_dir.parent.name,
                "wrote_at": datetime.now(tz=timezone.utc).isoformat(),
                "max_steps": self.max_steps,
                "max_tests": getattr(self.tools, "max_tests", None),
                "seeds": list(getattr(self.tools, "seeds", []) or []),
                "steps": getattr(self.tools, "step_count", 0),
                "tests": getattr(self.tools, "test_count", 0),
                "done": bool(getattr(self.tools, "done", False)),
                "tokens": dict(self._token_totals),
                "test_history": test_history,
            }
            if error:
                summary["error"] = error
            if extra:
                summary["extra"] = extra

            out_path = log_dir / "summary.json"
            tmp_path = out_path.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            os.replace(tmp_path, out_path)
        except Exception as exc:  # don't break the agent on summary failure
            print(f"[agent] _write_run_summary failed: {exc}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, resume: bool = False) -> dict:
        """Run the agent loop: setup → prompt → modify→test loop → summary."""
        self.setup_workspace()

        if resume and not self.logger.has_messages():
            print("[agent] --resume specified but no messages.jsonl found, starting fresh")
            resume = False

        if resume:
            messages, ok = self.resume_from_log()
            if ok and not self.tools.done:
                print(f"[agent] Resuming from step {self.tools.step_count}, "
                      f"test {self.tools.test_count}/{self.tools.max_tests}")
                # Recover pending SLURM test before entering the main loop
                if self._resume_has_pending_test:
                    print("[resume] Recovering pending test...")
                    step_num = self.tools.step_count + 1
                    self.logger.log_assistant(step_num, {"name": "test", "input": {}})
                    result = self.tools.dispatch("test", {})
                    self._log_result(result)
                    meta = None
                    entry = self.tools.latest_test_history_entry()
                    if entry is not None:
                        meta = {"test_history_entry": copy.deepcopy(entry)}
                    self.logger.log_tool_result(step_num, str(result), meta=meta)
                    # Append assistant + tool_result to messages
                    tool_id = f"tool_{self.tools.step_count}"
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": tool_id, "name": "test", "input": {}}],
                    })
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": str(result)}],
                    })
                    print(f"[resume] Test recovery complete — step {self.tools.step_count}, "
                          f"test {self.tools.test_count}/{self.tools.max_tests}")
            elif ok and self.tools.done:
                # Check if final results are actually complete in the leaderboard
                # (covers: submit with empty seed_metrics, single-seed submit, etc.)
                expected_seeds = set(str(s) for s in self.tools.seeds)
                final_seeds = set(
                    str(r.get("seed"))
                    for r in self.leaderboard.all_records()
                    if r.get("model") == self.tools.model_name
                    and str(r.get("is_final", "")) == "true"
                    and str(r.get("seed")) != "mean"
                    and self.leaderboard.has_real_metrics(r)
                )
                if expected_seeds <= final_seeds:
                    print("[agent] Previous run already completed with full results. Nothing to resume.")
                    self._write_run_summary()
                    return {
                        "steps": self.tools.step_count,
                        "tests": self.tools.test_count,
                        "done": True,
                    }
                missing = expected_seeds - final_seeds
                print(f"[resume] Previous run marked done but final results incomplete "
                      f"(missing seeds: {missing}). Re-running missing seeds...")
                # Run test only for missing seeds using the same submitted code
                missing_seed_list = sorted(int(s) for s in missing)
                print(f"[resume] Running test for missing seeds only: {missing_seed_list}")
                original_seeds = self.tools.seeds
                self.tools.seeds = missing_seed_list
                self.tools.done = False
                self.tools.test_count = self.tools.max_tests - 1  # so next test() is final
                result = self.tools.test()
                self.tools.seeds = original_seeds  # restore full seed list
                print(f"[resume] Missing-seed test result: {result[:200]}...")
                # test() no longer auto-submits, so explicitly submit the latest result
                if not self.tools.done:
                    self.tools.submit(n=len(self.tools._test_history), _force=True)
                self._write_run_summary()
                return {
                    "steps": self.tools.step_count,
                    "tests": self.tools.test_count,
                    "done": self.tools.done,
                }
            else:
                print("[agent] Resume failed, starting fresh")
                resume = False

        if not resume:
            self.logger.reset()
            initial_prompt = self.build_initial_prompt()
            messages = [{"role": "user", "content": initial_prompt}]
            self.logger.log_initial_prompt(initial_prompt)
            # Save exp_name so resume can restore the same workspace/log paths
            self.logger._append({"role": "_meta", "exp_name": self.tools.exp_name})

            # Log the initial prompt sent to the model
            C, B, D, RST = self._CYAN, self._BOLD, self._DIM, self._RESET
            print(f"{C}{'═' * 60}{RST}")
            print(f"{B}{C}Initial prompt → model{RST}")
            print(f"{C}{'═' * 60}{RST}")
            prompt_lines = initial_prompt.splitlines()
            limit = None if self.verbose else 80
            show = prompt_lines if limit is None else prompt_lines[:limit]
            for line in show:
                print(f"{D}  {line}{RST}")
            if limit is not None and len(prompt_lines) > limit:
                print(f"{D}  ... ({len(prompt_lines) - limit} more lines, use -v to show all){RST}")
            print(f"{C}{'═' * 60}{RST}")
            print(f"{D}  (total {len(prompt_lines)} lines, {len(initial_prompt)} chars){RST}\n")

        consecutive_failed_submits = 0
        while True:
            # Check step limit
            if self.tools.step_count >= self.max_steps:
                print(f"[agent] Max steps ({self.max_steps}) reached, stopping")
                break

            # Get next action from the agent (retry up to 3 times on None).
            # Even when get_action returns None, the underlying LLM call still
            # consumed tokens; capture those from client._last_usage so our
            # accounting stays honest across retries.
            tool_use = None
            for _no_action_attempt in range(3):
                tool_use = self.get_action(messages)
                if tool_use is None:
                    usage = getattr(getattr(self, "client", None), "_last_usage", None)
                    if usage:
                        self._accumulate_tokens(usage)
                        self.logger.log_tokens(self.tools.step_count, usage)
                if tool_use is not None:
                    break
                print(f"[agent] No action returned (attempt {_no_action_attempt + 1}/3), nudging")
                # Append a nudge so the model returns a tool call next time
                messages.append({
                    "role": "user",
                    "content": "Please use one of the available tools (edit, test, or undo) to continue working on the task.",
                })
            if tool_use is None:
                print("[agent] No action returned after 3 attempts, stopping")
                break

            tool_name = tool_use["name"]
            tool_input = tool_use.get("input", {})

            # Log thinking if present
            thinking = tool_use.get("thinking")
            if thinking:
                self._log_thinking(thinking)

            step_num = self.tools.step_count + 1
            self._log_action(step_num, tool_name, tool_input)
            self.logger.log_assistant(step_num, tool_use)

            # Token accounting: ModelClient.call returns a "usage" dict.
            usage = tool_use.get("usage")
            if usage:
                self._accumulate_tokens(usage)
                self.logger.log_tokens(step_num, usage)

            # Execute the tool (also increments step_count)
            result = self.tools.dispatch(tool_name, tool_input)

            # Log the result
            self._log_result(result)
            meta = None
            if tool_name == "test":
                entry = self.tools.latest_test_history_entry()
                if entry is not None:
                    meta = {"test_history_entry": copy.deepcopy(entry)}
            self.logger.log_tool_result(step_num, str(result), meta=meta)

            # Snapshot file after edit
            if tool_name == "edit":
                fname = tool_input.get("filename", "")
                try:
                    fpath = self.tools._resolve_workspace_path(fname)
                    if fpath.exists():
                        self.logger.log_file_snapshot(step_num, fname, fpath.read_text())
                except Exception:
                    pass

            # Build a unique ID for this tool call
            tool_id = f"tool_{self.tools.step_count}"

            # Use assistant_message from the call() result if available (preserves thinking blocks),
            # otherwise construct a minimal one from tool_name/input.
            if "assistant_message" in tool_use:
                assistant_message = tool_use["assistant_message"]
            else:
                assistant_message = {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": tool_input,
                        }
                    ],
                }

            # Ensure the tool_use block has the right id for the tool_result below
            # Find the tool_use block in the assistant_message content and use its id
            actual_tool_id = tool_id
            for block in assistant_message.get("content", []):
                if block.get("type") == "tool_use":
                    actual_tool_id = block.get("id", tool_id)
                    break

            messages.append(assistant_message)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": actual_tool_id,
                        "content": str(result),
                    }
                ],
            })

            # Reminder: if action budget is almost exhausted and no test()
            # has run yet, nudge the model to call test() before it loses
            # the chance to record any result. Fires at most once per run.
            remaining_steps = self.max_steps - self.tools.step_count
            if (self.tools.test_count == 0
                    and remaining_steps <= 2
                    and not getattr(self, "_test_reminder_sent", False)):
                reminder = (
                    f"[REMINDER] You have {remaining_steps} action(s) left out of "
                    f"{self.max_steps} and have not called test() yet. Every tool "
                    f"call (edit / test / undo / web_*) consumes one action. If you "
                    f"do not call test() before the budget runs out, your run will "
                    f"end with no recorded result. Call test() now — it both "
                    f"evaluates your current code and (when it is the final test) "
                    f"records the result to the leaderboard."
                )
                messages.append({"role": "user", "content": reminder})
                self.logger.log_user_message(reminder)
                self._test_reminder_sent = True
                print(f"[agent] Injected test() reminder ({remaining_steps} actions left, 0 tests)")

            # Check if we're done (final test was called)
            if self.tools.done:
                print("[agent] Done (final test reached)")
                break

            # Guard against infinite submit loops: submit doesn't increment
            # step_count, so the step limit never fires.  If the model keeps
            # calling submit and it keeps failing, bail out.
            if tool_name == "submit" and not self.tools.done:
                consecutive_failed_submits += 1
                if consecutive_failed_submits >= 5:
                    print(f"[agent] {consecutive_failed_submits} consecutive failed submits, stopping")
                    break
            else:
                consecutive_failed_submits = 0

        # Record empty finals for seeds missing valid final results
        # (e.g. all tests crashed with code bugs)
        self.tools.record_zero_if_no_finals()

        if self._token_totals["calls"] > 0:
            print(f"[agent] token totals: {self._token_totals}")

        self._write_run_summary()

        return {
            "steps": self.tools.step_count,
            "tests": self.tools.test_count,
            "done": self.tools.done,
            "tokens": dict(self._token_totals),
        }
