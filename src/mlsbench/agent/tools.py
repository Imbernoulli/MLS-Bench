"""WorkspaceTools: file editing and test execution tools for the MLS-Bench agent."""

import copy
import fcntl
import fnmatch
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time as _time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Tool schemas (Anthropic-style, compatible with OpenAI function calling)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "edit",
        "description": (
            "Edit files in the workspace. Three operations are supported:\n"
            "  create: Create a new file with the given content. Only available if allow_create=true.\n"
            "  insert: Insert one or more lines immediately after `after_line` (1-indexed).\n"
            "  replace: Replace lines `start_line`..`end_line` (inclusive, 1-indexed) with `content`.\n"
            "File paths are relative to the package root (e.g. 'LLaMA-Factory/src/...').\n"
            "Lines within protected ranges must NOT be modified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["create", "insert", "replace"],
                    "description": "The edit operation to perform.",
                },
                "filename": {
                    "type": "string",
                    "description": "Package-relative path to the file (e.g. 'LLaMA-Factory/src/llamafactory/train/dpo/trainer.py').",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for create/replace) or insert.",
                },
                "after_line": {
                    "type": "integer",
                    "description": "Line number after which to insert (required for op='insert').",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to replace, 1-indexed (required for op='replace').",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to replace, 1-indexed inclusive (required for op='replace').",
                },
            },
            "required": ["op", "filename", "content"],
        },
    },
    {
        "name": "test",
        "description": (
            "Run a new experiment. Executes training and evaluation, then returns metrics. "
            "Each run is numbered #1, #2, etc. All runs use all configured seeds. "
            "You have a limited test budget."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "submit",
        "description": (
            "Submit a previous test result as your final answer. This does NOT re-run "
            "anything — it selects a result you already obtained. You must have run "
            "test() at least once before calling submit()."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "The test number to submit (1-indexed). e.g. n=1 submits the result from test #1.",
                },
            },
            "required": ["n"],
        },
    },
    {
        "name": "undo",
        "description": "Revert the last n file modification actions (create/insert/replace) by restoring pre-edit snapshots. Does not undo test calls.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of edit actions to undo (default: 1).",
                },
            },
        },
    },
]


# Opt-in tools — appended to TOOL_SCHEMAS only when --allow-web-search is set.
# Kept separate so they never reach the model unless the flag is on.
WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web (Tavily) and return up to `max_results` hits, each with title, "
        "URL, relevance score, and a content snippet (~few hundred chars). When "
        "`include_answer=true`, also returns a synthesized one-paragraph answer. "
        "`include_raw_content` is best-effort — Tavily often returns null for it, and it "
        "is not query-focused. For ACTUAL page content (paper derivations, code "
        "discussion), call web_extract on the URLs you want to read."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "max_results": {
                "type": "integer",
                "description": "Max results to return (default 5, capped at 10).",
            },
            "include_answer": {
                "type": "boolean",
                "description": "If true, prepend a synthesized answer paragraph to the results.",
            },
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "description": (
                    "'basic' (default, 1 credit/call) for quick lookups; 'advanced' "
                    "(2 credits/call) for higher-quality results on technical or "
                    "ambiguous queries. Use 'advanced' when 'basic' returned weak hits."
                ),
            },
            "time_range": {
                "type": "string",
                "enum": ["day", "week", "month", "year"],
                "description": "Restrict results to the last day/week/month/year. Useful for recent papers.",
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Whitelist of domains to include, e.g. ['arxiv.org', "
                    "'semanticscholar.org']. Empty = no whitelist."
                ),
            },
            "exclude_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Blacklist of domains to skip.",
            },
            "include_raw_content": {
                "type": "string",
                "enum": ["text", "markdown"],
                "description": (
                    "Best-effort full-page text/markdown alongside each result. Often "
                    "null — when you need reliable, focused content, prefer web_extract."
                ),
            },
        },
        "required": ["query"],
    },
}


WEB_EXTRACT_SCHEMA = {
    "name": "web_extract",
    "description": (
        "Fetch and extract content from specific URLs (typically picked from a "
        "previous web_search). When `query` is set together with `chunks_per_source`, "
        "returns ONLY the chunks of each page most relevant to that query — much more "
        "token-efficient than full-page extraction. This is the right tool when you "
        "need to actually read what a paper or doc says about a specific concern "
        "(derivation, code, API, hyperparameter discussion). Each call counts against "
        "your step budget AND your web_search budget."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-5 URLs to extract from.",
            },
            "query": {
                "type": "string",
                "description": (
                    "If provided, returns only chunks most relevant to this query "
                    "(strongly recommended — without it you get the whole page)."
                ),
            },
            "chunks_per_source": {
                "type": "integer",
                "description": "Relevant chunks per URL (1-5, default 3). Effective only when query is set.",
            },
            "extract_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "description": (
                    "'basic' (1 credit/URL) or 'advanced' (2 credits/URL). Use "
                    "'advanced' for PDFs and complex pages — strongly recommended for arxiv PDFs."
                ),
            },
            "format": {
                "type": "string",
                "enum": ["text", "markdown"],
                "description": "Output format (default 'text').",
            },
        },
        "required": ["urls"],
    },
}


# ---------------------------------------------------------------------------
# compute_scale
# ---------------------------------------------------------------------------

def scale_test_cmd_entries(
    entries: list[dict], scale: float, task_name: str = ""
) -> list[dict]:
    """Apply the host GPU `compute_scale` to a list of test_cmd entries.

    `scale` is the perf factor vs the H100 baseline the tasks are tuned on
    (1.0 = H100; 0.5 = H200 ~= 2x H100). Returns deep-copied entries with the
    `h200` helper key stripped, so the result is always safe to hand to the
    container builders / GPU packers. Per entry:

    - `h200` override block present (llm-pretrain / llm-rl): apply its `cmd` and
      `compute`, and merge its `env` over the entry's (the task's H200 config).
      The fewer-GPUs change is paired with a larger per-GPU batch / TP=1 so the
      macro-batch — and thus the result — stays constant.
    - no override, `compute <= 1` (fractional / single-GPU): multiply `compute`
      by `scale` for denser packing (the original behaviour; never changes a
      single job's result). Offered for all such tasks but only really needed
      for the two families above.
    - no override, `compute > 1`: left untouched and warned. Cutting a multi-GPU
      data-parallel job's GPU count alone would change its global batch / result,
      so we don't do it silently.

    The standalone scheduler (scheduler.py) mirrors the compute-number transform
    independently to keep its GPU inference dependency-free.
    """
    scale = float(scale or 1.0)
    scaled: list[dict] = []
    for entry in entries:
        entry = copy.deepcopy(entry)
        override = entry.pop("h200", None)
        if scale == 1.0:
            scaled.append(entry)
            continue
        if override:
            if "cmd" in override:
                entry["cmd"] = override["cmd"]
            entry["compute"] = override.get(
                "compute", float(entry.get("compute", 1) or 1) * scale
            )
            if override.get("env"):
                merged = dict(entry.get("env", {}))
                merged.update(override["env"])
                entry["env"] = merged
        else:
            # Sub-1 (fractional / single-GPU) jobs scale for denser packing —
            # the original compute_scale behaviour, which never changes a single
            # job's result. A multi-GPU data-parallel job (compute > 1) without
            # an h200 block is left untouched + warned, since cutting its GPU
            # count alone would change its global batch / result.
            compute = float(entry.get("compute", 1) or 1)
            if compute <= 1.0:
                # Write even when `compute` was implicit (default 1.0) so this
                # matches scheduler._scaled_compute, which scales the default.
                entry["compute"] = compute * scale
            else:
                print(
                    f"[compute_scale] WARNING: {task_name or 'task'} test_cmd "
                    f"'{entry.get('label', entry.get('cmd', ''))}' has compute="
                    f"{compute} but no 'h200' override; left at {compute} GPUs "
                    f"(not scaled by {scale}). Cutting a data-parallel job's GPU "
                    f"count alone changes its global batch/result; compute_scale "
                    f"is only really needed for llm-pretrain / llm-rl. Add an "
                    f"'h200' block to adapt it.",
                    file=sys.stderr,
                )
        scaled.append(entry)
    return scaled


# ---------------------------------------------------------------------------
# WorkspaceTools
# ---------------------------------------------------------------------------

class WorkspaceTools:
    """Tools for editing workspace files and running experiments."""

    # Class-level lock for image building — shared across all instances so that
    # parallel baseline threads (each with its own WorkspaceTools) don't race
    # to build the same container image simultaneously.
    _class_build_lock = threading.Lock()

    def __init__(
        self,
        task_name,
        config_task,
        config_edit,
        workspace_root,
        project_root,
        max_tests,
        model_name: str = "",
        parser=None,
        leaderboard=None,
        save_path: str = "",
        seeds: list[int] | None = None,
        slurm_config: dict | None = None,
        exp_name: str = "",
        container_runtime: str = "apptainer",
        use_cuda: bool | None = None,
        platform: str = "",
        gpu_devices: str = "",
        compute_scale: float = 1.0,
        global_config: dict | None = None,
        allow_web_search: bool = False,
        tavily_api_key: str = "",
        max_web_credits: int = 20,
        extra_context: str | None = None,
        hide_hidden: bool = False,
        extra_env: dict | None = None,
    ):
        self.task_name = task_name
        self.config_task = config_task
        self.config_edit = config_edit
        self.workspace_root = Path(workspace_root)
        self.project_root = Path(project_root)
        self.max_tests = max_tests
        self.model_name = model_name
        self.leaderboard = leaderboard
        self.save_path = save_path
        self.seeds = seeds or [42]
        self.container_runtime = container_runtime
        self._use_cuda_override = use_cuda   # None = defer to pkg config
        self._platform = platform             # e.g. "linux/amd64" for Rosetta
        self.gpu_devices = gpu_devices
        # GPU perf/packing scaler vs the H100 baseline the tasks are tuned on.
        # <1 means each GPU is faster/bigger (e.g. 0.5 for H200 ~= 2x H100).
        # Only retunes llm-pretrain / llm-rl tasks (which declare an `h200`
        # block); see scale_test_cmd_entries.
        self.compute_scale = float(compute_scale or 1.0)
        self.global_config = dict(global_config or {})
        self.allow_web_search = bool(allow_web_search)
        self.tavily_api_key = tavily_api_key or ""
        # 0 = unlimited; otherwise hard cap on Tavily credits per run.
        # Pricing: basic search = 1, advanced search = 2,
        #          basic extract = 1/URL, advanced extract = 2/URL.
        self.max_web_credits = int(max_web_credits or 0)
        self.web_credits_used = 0
        # Set by BaseAgent only when the corresponding context file exists for this task,
        # so leaderboard rows only record extra_context when it actually went into the prompt.
        self.extra_context: str | None = extra_context if extra_context else None
        # Param-count log written by _log_budget_output. Populated lazily.
        # BaseAgent overrides this via `tools.agent_log_dir = ...` once the
        # logger is constructed, so the file lives next to messages.jsonl.
        self.agent_log_dir: Path | None = None
        self._param_log_path: Path | None = None
        # Most recent agent param count parsed from budget_check.py stdout,
        # keyed by env label. Used to attach an `agent_params` column on
        # leaderboard rows when budget_check.py emits param counts.
        self._last_agent_params: dict[str, int] = {}
        # Per-baseline / per-run env vars injected into every test command's
        # container environment (after pkg_config["env"], before SAVE_PATH/SEED/ENV).
        self.extra_env: dict = dict(extra_env or {})

        # Experiment name for OUTPUT_DIR: baseline passes explicit name,
        # agent auto-generates from model_name + timestamp
        if exp_name:
            self.exp_name = exp_name
        else:
            sanitized = re.sub(r'[/: ]', '_', model_name)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_name = f"{sanitized}_{ts}" if sanitized else ts

        # Job scheduler executor (None = direct execution)
        self._recover_pending_slurm = False  # set by resume to recover orphaned SLURM jobs
        self.slurm_executor = None
        if slurm_config:
            from mlsbench.agent.slurm import SlurmExecutor
            self.slurm_executor = SlurmExecutor(slurm_config, self.project_root)
        elif not os.environ.get("MLSBENCH_SCHEDULER_MANAGED"):
            # Auto-detect local GPU scheduler for test-time GPU allocation.
            # Skip if we're already inside a scheduler-managed job (baseline via scheduler).
            try:
                from mlsbench.scheduler import is_scheduler_running
                if is_scheduler_running():
                    from mlsbench.agent.local_executor import LocalSchedulerExecutor
                    self.slurm_executor = LocalSchedulerExecutor(self.project_root, self.global_config)
                    print("[info] Local GPU scheduler detected — test jobs will be submitted for GPU allocation")
            except ImportError:
                pass

        # Build the list of test_cmd entries (supports old single-string format too)
        self.test_cmd_entries: list[dict] = self._build_test_cmd_entries()

        # Union of all packages across all test_cmds (for edit validation)
        seen = {}
        for entry in self.test_cmd_entries:
            pkg = entry.get("package")
            if pkg:
                seen[self._normalize_pkg_name(pkg)] = pkg
        self.all_external_packages: list[str] = list(seen.values())

        # Load parser (task-specific if available)
        if parser is not None:
            self.parser = parser
        else:
            from mlsbench.agent.parsers import load_parser
            self.parser = load_parser(task_name, self.project_root)

        # Live protected ranges: filename -> list of [start, end] (non-editable zones)
        # Computed as the complement of the allowed edit ranges from config.
        self.live_protected_ranges: dict[str, list[list[int]]] = {}
        for entry in config_edit:
            if "edit" in entry:
                fn = entry["filename"]
                allowed = [[r["start"], r["end"]] for r in entry["edit"]]
                # Empty edit list → entire file protected (read-only).
                # _allowed_to_protected([]) already returns [[-1,-1]].
                self.live_protected_ranges[fn] = self._allowed_to_protected(allowed)

        # Snapshot history for undo: each entry is {filename, path, content, ranges}
        self._history: list[dict] = []

        # Instance-level lock kept for backward compat (single-instance parallel cmds)
        self._build_lock = self._class_build_lock
        self._docker_rootless_cache: bool | None = None

        # Counters
        self.step_count = 0   # incremented on every dispatch
        self.test_count = 0   # incremented on every test call
        self.done = False

        # History of all test results (1-indexed for submit)
        self._test_history: list[dict] = []  # each: {feedback, seed_metrics, seeds}
        self._current_test_had_failures = False
        self._last_test_had_failures = False

        # Per-metric glob patterns that are withheld from the agent during
        # intermediate (non-final) tests. Metrics still land in the leaderboard
        # CSV; only the agent-visible feedback text has matching ``key=value``
        # segments stripped. Mirrors the semantics of script-level ``hidden``.
        raw_hidden = self.config_task.get("hidden_metrics") or []
        self.hidden_metric_patterns: list[str] = [str(p) for p in raw_hidden]

        # --hide-hidden: when True, the default ReAct path additionally
        # withholds saved_metrics keys whose names contain a "hidden": true
        # test_cmd label. CSV writes are unaffected.
        self.hide_hidden: bool = bool(hide_hidden)
        self.hidden_test_labels: list[str] = [
            str(tc["label"]) for tc in self.config_task.get("test_cmds", [])
            if tc.get("hidden") and tc.get("label")
        ]

    def _filter_hidden_label_metrics(self, metrics: dict) -> dict:
        """Drop metric keys whose name contains any hidden test_cmd label.

        Substring/contains match against both the raw label and its
        hyphen→underscore variant, mirroring openevolve_agent._filter_hidden_metrics.
        Returns ``metrics`` unchanged when ``hide_hidden`` is off or no
        hidden labels exist.
        """
        if not self.hide_hidden or not self.hidden_test_labels or not metrics:
            return metrics
        labels = self.hidden_test_labels
        def _hides(k: str) -> bool:
            kn = str(k).replace("-", "_")
            for lab in labels:
                ln = lab.replace("-", "_")
                if lab in str(k) or ln in kn:
                    return True
            return False
        return {k: v for k, v in metrics.items() if not _hides(k)}

    # ------------------------------------------------------------------
    # Hidden-metric filtering
    # ------------------------------------------------------------------

    # Matches ``key=value`` where the key is identifier-like (allows
    # hyphen/underscore/dot/slash so column keys like ``piqa_lm-eval-345m``
    # are captured) and the value is a numeric literal or nan/inf sentinel.
    _HIDDEN_KV_RE = re.compile(
        r'([A-Za-z_][\w.\-/]*)='
        r'(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|inf|-inf|NaN|Inf)',
    )

    def _filter_hidden_metric_kvs(self, text: str) -> str:
        """Strip ``key=value`` segments whose key matches a hidden glob.

        Applied only when constructing agent-visible feedback for
        intermediate (non-final) tests. The leaderboard still receives
        the full metric dict — this only affects what the model sees.
        """
        if not self.hidden_metric_patterns or not text:
            return text

        patterns = self.hidden_metric_patterns

        def should_hide(key: str) -> bool:
            return any(fnmatch.fnmatchcase(key, p) for p in patterns)

        out_lines: list[str] = []
        for line in text.splitlines():
            new_line = self._HIDDEN_KV_RE.sub(
                lambda m: '' if should_hide(m.group(1)) else m.group(0),
                line,
            )
            # Collapse comma artefacts left behind by stripped kv pairs.
            new_line = re.sub(r',\s*,', ',', new_line)
            new_line = re.sub(r':\s*,\s*', ': ', new_line)
            new_line = re.sub(r',\s*$', '', new_line)
            new_line = new_line.rstrip()
            # Drop "Final metrics (...): " / "TEST_METRICS: " header stubs
            # whose payload was fully removed.
            if re.fullmatch(r'\s*(Final metrics \([^)]*\):|TEST_METRICS:?)\s*', new_line):
                continue
            out_lines.append(new_line)
        return '\n'.join(out_lines)

    # ------------------------------------------------------------------
    # CUDA / platform helpers
    # ------------------------------------------------------------------

    def _effective_use_cuda(self, pkg_config: dict) -> bool:
        """Return whether CUDA should be enabled for this run.

        Global override (from config.local.yaml ``use_cuda``) takes
        precedence over the per-package setting.
        """
        if self._use_cuda_override is not None:
            return self._use_cuda_override
        return pkg_config.get("use_cuda", False)

    @staticmethod
    def _container_bind_target(bind_spec: str) -> str:
        """Return the container-side target path from a bind specification."""
        parts = bind_spec.split(":")
        if len(parts) >= 2:
            return parts[1]
        return bind_spec

    # ------------------------------------------------------------------
    # test_cmd entry helpers
    # ------------------------------------------------------------------

    def _build_test_cmd_entries(self) -> list[dict]:
        """Build normalised list of test_cmd entries from config.

        Applies the host `compute_scale` via scale_test_cmd_entries: the
        llm-pretrain / llm-rl tasks (which carry an `h200` block) switch to their
        original H200 config; everything else keeps its baseline GPU count except
        fractional jobs, which pack denser.
        """
        entries = self.config_task.get("test_cmds", [])
        if not entries:
            # Backward compat: old single-string test_cmd
            cmd = self.config_task.get("test_cmd", "train.sh")
            entries = [{
                "cmd": cmd,
                "label": "test",
                "package": self.config_task.get("package", ""),
                "use_cuda": self.config_task.get("use_cuda", False),
                "workdir": self.config_task.get("workdir", "/app"),
            }]
        return scale_test_cmd_entries(entries, self.compute_scale, task_name=self.task_name)

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_pkg_name(name: str) -> str:
        """Normalize package name for case-insensitive, hyphen-insensitive matching."""
        return name.lower().replace("-", "").replace("_", "")

    @property
    def workspace_task_dir(self) -> Path:
        """Per-run workspace directory: workspace_root / task / exp_name."""
        return self.workspace_root / self.task_name / self.exp_name

    def _find_workspace_pkg(self, pkg_name: str) -> Path:
        """Find the workspace package directory, with normalized name matching."""
        wtd = self.workspace_task_dir
        if not wtd.exists():
            raise FileNotFoundError(f"Workspace task directory not found: {wtd}")
        norm = self._normalize_pkg_name(pkg_name)
        for d in wtd.iterdir():
            if d.is_dir() and self._normalize_pkg_name(d.name) == norm:
                return d
        raise FileNotFoundError(
            f"Package '{pkg_name}' not found in workspace: {wtd}"
        )

    def _resolve_workspace_path(self, filename: str) -> Path:
        """Resolve a filename to an absolute path.

        Resolution order:
        1. Package-relative: first path component is a workspace package name.
        2. Task-relative: fall back to ``tasks/<task_name>/<filename>``.
        """
        parts = filename.split("/")
        pkg_name = parts[0]
        rest = Path(*parts[1:]) if len(parts) > 1 else Path()
        try:
            pkg_dir = self._find_workspace_pkg(pkg_name)
            return pkg_dir / rest
        except FileNotFoundError:
            pass
        # Fall back to task directory
        task_path = self.project_root / "tasks" / self.task_name / filename
        if task_path.exists():
            return task_path
        raise FileNotFoundError(
            f"Cannot resolve '{filename}': not a workspace package nor a task file"
        )

    # ------------------------------------------------------------------
    # Allowed-to-protected conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _allowed_to_protected(allowed_ranges: list[list[int]]) -> list[list[int]]:
        """Convert allowed (editable) ranges to protected (non-editable) ranges.

        Returns the complement of *allowed_ranges*.  Convention for the
        sentinel value ``-1``:

        * In **allowed** ranges, ``[-1, -1]`` = entire file is editable.
        * In **protected** ranges, ``[-1, -1]`` = entire file is protected.
        * A range end of ``-1`` means "extends to EOF".

        Special cases (checked first for clarity):
        * ``[]``           → ``[[-1, -1]]``  (nothing editable → all protected)
        * ``[[-1, -1]]``   → ``[]``          (all editable → nothing protected)
        """
        # --- special cases ---------------------------------------------------
        if not allowed_ranges:
            return [[-1, -1]]  # nothing editable → entire file protected
        if any(r[0] == -1 and r[1] == -1 for r in allowed_ranges):
            return []  # entire file editable → nothing protected

        # --- general case: complement of sorted allowed ranges ----------------
        sorted_r = sorted(allowed_ranges, key=lambda x: x[0])
        protected = []
        # Before first allowed range
        if sorted_r[0][0] > 1:
            protected.append([1, sorted_r[0][0] - 1])
        # Gaps between consecutive allowed ranges
        for i in range(len(sorted_r) - 1):
            prev_end = sorted_r[i][1]
            if prev_end == -1:
                break  # previous range extends to EOF, no gap possible
            gap_start = prev_end + 1
            gap_end = sorted_r[i + 1][0] - 1
            if gap_start <= gap_end:
                protected.append([gap_start, gap_end])
        # After last allowed range → to EOF
        last_end = sorted_r[-1][1]
        if last_end != -1:
            protected.append([last_end + 1, -1])
        return protected

    # ------------------------------------------------------------------
    # Permission checking
    # ------------------------------------------------------------------

    def _check_edit_permission(self, filename: str, start_line: int, end_line: int) -> bool:
        """Return True if [start_line, end_line] does NOT overlap any protected range.

        A protected range of [-1, -1] means the entire file is protected.
        A protected range end of -1 means the range extends to EOF.
        """
        for r in self.live_protected_ranges.get(filename, []):
            if r[0] == -1 and r[1] == -1:
                return False  # entire file is protected
            prot_end = r[1] if r[1] != -1 else float('inf')
            if start_line <= prot_end and end_line >= r[0]:
                return False  # overlaps with a protected range
        return True

    def _compute_editable_ranges(self, filename: str, total_lines: int) -> list[tuple[int, int]]:
        """Compute editable line ranges as the complement of protected ranges.

        Returns a list of (start, end) tuples (1-indexed, inclusive).
        """
        protected = self.live_protected_ranges.get(filename, [])
        if not protected:
            return [(1, total_lines)]
        if any(r[0] == -1 and r[1] == -1 for r in protected):
            return []  # fully protected

        sorted_p = sorted(protected, key=lambda x: x[0])
        editable: list[tuple[int, int]] = []

        # Before the first protected range
        if sorted_p[0][0] > 1:
            editable.append((1, sorted_p[0][0] - 1))

        # Gaps between consecutive protected ranges
        for i in range(len(sorted_p) - 1):
            prev_end = sorted_p[i][1]
            if prev_end == -1:
                break  # extends to EOF, no gap after
            gap_start = prev_end + 1
            gap_end = sorted_p[i + 1][0] - 1
            if gap_start <= gap_end:
                editable.append((gap_start, gap_end))

        # After the last protected range (up to total_lines)
        last_end = sorted_p[-1][1]
        if last_end != -1 and last_end < total_lines:
            editable.append((last_end + 1, total_lines))

        return editable

    def _file_snapshot(self, filename: str) -> str:
        """Return a compact snapshot showing updated editable range boundaries.

        Shows only the first/last few lines of each editable region so the model
        can see the updated line numbers without re-reading its own edits.
        """
        try:
            path = self._resolve_workspace_path(filename)
            if not path.exists():
                return ""
            all_lines = path.read_text().splitlines()
        except Exception:
            return ""

        editable_ranges = self._compute_editable_ranges(filename, len(all_lines))
        if not editable_ranges:
            return ""

        range_strs = ", ".join(f"{s}–{e}" for s, e in editable_ranges)
        header = f"[Current file: {filename} | editable: {range_strs} | total: {len(all_lines)} lines]"

        # For short ranges (<=8 lines), show everything; otherwise show first/last 3
        peek = 3
        sections: list[str] = []
        N = len(all_lines)
        for start, end in editable_ranges:
            # Clamp to actual file length so freshly-created files (e.g. from
            # mid_edit) don't trip an IndexError on access beyond EOF.
            s_clamped = max(1, min(start, N))
            e_clamped = max(s_clamped - 1, min(end, N))
            if e_clamped < s_clamped:
                continue
            span = e_clamped - s_clamped + 1
            lines_out: list[str] = []
            if span <= peek * 2 + 2:
                # Short range — show all lines
                for i in range(s_clamped, e_clamped + 1):
                    lines_out.append(f"{i:6d}  {all_lines[i - 1]}")
            else:
                # Long range — show first/last few lines
                for i in range(s_clamped, s_clamped + peek):
                    lines_out.append(f"{i:6d}  {all_lines[i - 1]}")
                lines_out.append(f"       ... ({span - peek * 2} more lines) ...")
                for i in range(e_clamped - peek + 1, e_clamped + 1):
                    lines_out.append(f"{i:6d}  {all_lines[i - 1]}")
            sections.append("\n".join(lines_out))

        return header + "\n" + "\n...\n".join(sections)

    def _editable_range_str(self, filename: str) -> str:
        """Return a human-readable string describing the current editable line range."""
        protected = self.live_protected_ranges.get(filename, [])
        if not protected:
            return "entire file"
        if any(r[0] == -1 and r[1] == -1 for r in protected):
            return "(none — file is fully protected)"
        # Get actual file length for accurate tail-range reporting
        try:
            path = self._resolve_workspace_path(filename)
            total_lines = len(path.read_text().splitlines()) if path.exists() else 0
        except Exception:
            total_lines = 0
        editable = self._compute_editable_ranges(filename, total_lines)
        if not editable:
            return "(none)"
        return ", ".join(f"{s}–{e}" for s, e in editable)

    # ------------------------------------------------------------------
    # Live range updates after edits
    # ------------------------------------------------------------------

    def _update_ranges_after_insert(self, filename: str, after_line: int, num_new_lines: int):
        """Update live protected ranges after inserting num_new_lines after after_line."""
        for r in self.live_protected_ranges.get(filename, []):
            if r[0] == -1 and r[1] == -1:
                continue
            if r[0] > after_line:
                r[0] += num_new_lines
                if r[1] != -1:
                    r[1] += num_new_lines
            elif r[0] <= after_line < r[1]:
                r[1] += num_new_lines

    def _update_ranges_after_replace(
        self, filename: str, start_line: int, end_line: int, num_new_lines: int
    ):
        """Update live protected ranges after replacing [start_line, end_line] with num_new_lines lines."""
        delta = num_new_lines - (end_line - start_line + 1)
        for r in self.live_protected_ranges.get(filename, []):
            if r[0] == -1 and r[1] == -1:
                continue
            prot_end = r[1] if r[1] != -1 else float('inf')
            if prot_end < start_line:
                pass  # entirely before replaced region
            elif r[0] > end_line:
                r[0] += delta
                if r[1] != -1:
                    r[1] += delta
            else:
                raise ValueError(
                    f"Replace [{start_line},{end_line}] "
                    f"overlaps protected range [{r[0]},{r[1]}]"
                )

    def _shift_ranges_for_pre_edit(
        self, filename: str, change_after_line: int, delta: int
    ):
        """Shift protected range boundaries after a pre_edit op.

        Unlike _update_ranges_after_replace (which raises on overlap with
        protected zones), this method silently adjusts boundaries.  Pre-edit
        ops are system-level and are allowed to touch any part of the file.

        Args:
            filename: workspace-relative filename.
            change_after_line: boundaries strictly after this line are shifted.
            delta: positive = lines added, negative = lines removed.
        """
        if delta == 0:
            return
        for r in self.live_protected_ranges.get(filename, []):
            if r[0] == -1 and r[1] == -1:
                continue
            if r[0] > change_after_line:
                r[0] = max(1, r[0] + delta)
            if r[1] != -1 and r[1] > change_after_line:
                r[1] = max(1, r[1] + delta)

    # ------------------------------------------------------------------
    # Snapshot management (for undo)
    # ------------------------------------------------------------------

    def _save_snapshot(self, filename: str):
        """Save a file content + range snapshot before an edit."""
        path = self._resolve_workspace_path(filename)
        content = path.read_text() if path.exists() else None
        self._history.append({
            "filename": filename,
            "path": path,
            "content": content,
            "ranges": copy.deepcopy(self.live_protected_ranges),
        })

    # ------------------------------------------------------------------
    # Tool: edit
    # ------------------------------------------------------------------

    def edit(
        self,
        op: str,
        filename: str,
        content: str,
        after_line: int | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Unified file editing tool."""
        result = self._edit_impl(op, filename, content, after_line, start_line, end_line)

        # Always append the current file snapshot so the model sees the live state
        if filename in self.live_protected_ranges:
            snapshot = self._file_snapshot(filename)
            if snapshot:
                result = result + "\n\n" + snapshot

        return result

    def _edit_impl(
        self,
        op: str,
        filename: str,
        content: str,
        after_line: int | None,
        start_line: int | None,
        end_line: int | None,
    ) -> str:
        """Core edit logic (without snapshot appending)."""
        # Validate package
        parts = filename.split("/")
        pkg_name = parts[0]
        if self._normalize_pkg_name(pkg_name) not in [
            self._normalize_pkg_name(p) for p in self.all_external_packages
        ]:
            return f"ERROR: Package '{pkg_name}' is not in allowed packages"

        if op == "create":
            if not self.config_task.get("allow_create", False):
                return "ERROR: allow_create is false; cannot create new files"
            path = self._resolve_workspace_path(filename)
            if path.exists():
                return f"ERROR: File already exists: {filename}"
            self._save_snapshot(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return f"Created: {filename}"

        elif op == "insert":
            if after_line is None:
                return "ERROR: 'after_line' is required for op='insert'"
            if filename not in self.live_protected_ranges:
                return f"ERROR: File not editable: {filename}"
            # Insert places new lines AFTER after_line, so the new content
            # lands at line after_line+1.  Check that position, not after_line
            # itself (which may be the last line of a protected range).
            check_line = after_line + 1 if after_line > 0 else 1
            if not self._check_edit_permission(filename, check_line, check_line):
                return (
                    f"ERROR: Cannot insert after line {after_line} — target is outside the editable range. "
                    f"You may only edit lines {self._editable_range_str(filename)}."
                )
            path = self._resolve_workspace_path(filename)
            if not path.exists():
                return f"ERROR: File not found in workspace: {filename}"

            lines = path.read_text().splitlines(keepends=True)
            if after_line < 0 or after_line > len(lines):
                return f"ERROR: after_line={after_line} out of range (file has {len(lines)} lines)"

            new_lines = content.splitlines(keepends=True)
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

            self._save_snapshot(filename)
            lines[after_line:after_line] = new_lines
            path.write_text("".join(lines))
            self._update_ranges_after_insert(filename, after_line, len(new_lines))
            return (
                f"OK: Inserted {len(new_lines)} line(s) after line {after_line} in {filename}. "
                f"Editable range: {self._editable_range_str(filename)}."
            )

        elif op == "replace":
            if start_line is None or end_line is None:
                return "ERROR: 'start_line' and 'end_line' are required for op='replace'"
            if filename not in self.live_protected_ranges:
                return f"ERROR: File not editable: {filename}"
            if not self._check_edit_permission(filename, start_line, end_line):
                return (
                    f"ERROR: Lines {start_line}..{end_line} exceed the editable range. "
                    f"You may only edit lines {self._editable_range_str(filename)}."
                )
            path = self._resolve_workspace_path(filename)
            if not path.exists():
                return f"ERROR: File not found in workspace: {filename}"

            lines = path.read_text().splitlines(keepends=True)
            if start_line < 1 or end_line > len(lines) or start_line > end_line:
                return (
                    f"ERROR: Invalid line range {start_line}..{end_line} "
                    f"(file has {len(lines)} lines)"
                )

            new_lines = content.splitlines(keepends=True)
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

            self._save_snapshot(filename)
            lines[start_line - 1 : end_line] = new_lines
            path.write_text("".join(lines))
            self._update_ranges_after_replace(filename, start_line, end_line, len(new_lines))
            return (
                f"OK: Replaced lines {start_line}..{end_line} with {len(new_lines)} line(s) in {filename}. "
                f"Editable range: {self._editable_range_str(filename)}."
            )

        else:
            return f"ERROR: Unknown op '{op}'. Use 'create', 'insert', or 'replace'."

    # ------------------------------------------------------------------
    # Package config helpers
    # ------------------------------------------------------------------

    def _load_pkg_config(self, pkg_name: str) -> dict:
        """Load pkg config JSON. Convention: <dir>/config.json where dir name = package name."""
        from mlsbench.cli import load_pkg_config
        config, _ = load_pkg_config(pkg_name)
        return config

    def _find_ext_pkg_dir(self, pkg_name: str) -> Path:
        """Find the external_packages/<pkg> source directory, auto-fetching if needed."""
        from mlsbench.cli import find_ext_pkg_dir
        return find_ext_pkg_dir(pkg_name)

    # ------------------------------------------------------------------
    # Image build helpers
    # ------------------------------------------------------------------

    def _generate_def_content(self, pkg_config: dict, pkg_dir: Path) -> str:
        """Generate Apptainer definition file content from pkg config."""
        base_image = pkg_config["base_image"]
        workdir = pkg_config.get("workdir", "/app")
        install_cmds = pkg_config.get("install_cmds", [])

        files_section = f"    {pkg_dir.resolve()} {workdir}"
        post_section = "\n".join(f"    {line}" for line in install_cmds)
        env_section = "\n".join(
            f"    export {k}={v}" for k, v in pkg_config.get("env", {}).items()
        )

        return (
            f"Bootstrap: docker\n"
            f"From: {base_image}\n"
            f"\n"
            f"%files\n"
            f"{files_section}\n"
            f"\n"
            f"%post\n"
            f"    cd {workdir}\n"
            f"{post_section}\n"
            f"\n"
            f"%environment\n"
            f"{env_section}\n"
            f"\n"
            f"%runscript\n"
            f"    exec bash \"$@\"\n"
        )

    def _generate_dockerfile_content(
        self,
        pkg_config: dict,
        pkg_dir: Path,
        docker_extra_files: list[dict] | None = None,
    ) -> str:
        """Generate Dockerfile content from pkg config."""
        from mlsbench.cli import BUILD_PASSTHROUGH_ENV_VARS, docker_run_instruction_lines

        base_image = pkg_config["base_image"]
        workdir = pkg_config.get("workdir", "/app")
        pkg_workdir = f"{workdir.rstrip('/')}/{pkg_dir.name}"
        install_cmds = pkg_config.get("install_cmds", [])
        env = pkg_config.get("env", {})
        docker_extra_files = docker_extra_files or []

        lines = ["# syntax=docker/dockerfile:1.4", f"FROM {base_image}"]
        lines.append(f"COPY {pkg_dir.name} {pkg_workdir}")
        for ef in docker_extra_files:
            lines.append(f"COPY --from={ef['context_name']} {ef['copy_src']} {ef['dst']}")
        lines.append(f"WORKDIR {pkg_workdir}")
        # Only declare ARG for non-secret build-time env vars; API keys leak
        # into image layer history if forwarded via `--build-arg`, and that
        # metadata travels with `docker push`.
        for name in BUILD_PASSTHROUGH_ENV_VARS:
            lines.append(f"ARG {name}")
        for cmd in install_cmds:
            lines.extend(docker_run_instruction_lines(cmd))
        for k, v in env.items():
            lines.append(f"ENV {k}={v}")
        lines.append('ENTRYPOINT ["bash"]')
        return "\n".join(lines) + "\n"

    def _build_image(self, pkg_config: dict, pkg_dir: Path, image_path: Path) -> None:
        """Build a container image from pkg config (Apptainer or Docker)."""
        if "base_image" not in pkg_config:
            raise RuntimeError(
                f"pkg config for '{pkg_dir.name}' is missing 'base_image'"
            )

        if self.container_runtime == "docker":
            self._build_image_docker(pkg_config, pkg_dir, image_path)
        else:
            self._build_image_apptainer(pkg_config, pkg_dir, image_path)

    def _build_image_apptainer(self, pkg_config: dict, pkg_dir: Path, image_path: Path) -> None:
        """Build an Apptainer .sif image."""
        from mlsbench.cli import get_apptainer_build_cmd

        images_dir = self.project_root / "vendor" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        def_content = self._generate_def_content(pkg_config, pkg_dir)
        def_path = image_path.with_suffix(".def")
        def_path.write_text(def_content)
        print(f"[build] Definition file written: {def_path}")
        print(f"[build] --- Definition ---\n{def_content}[build] ----------------")

        build_cmd = get_apptainer_build_cmd()
        if pkg_config.get("use_cuda", False):
            build_cmd.append("--nv")
        build_cmd.extend([str(image_path), str(def_path)])

        # Use scratch-local tmpdir to avoid filling up /tmp on large images
        build_env = os.environ.copy()
        tmpdir = images_dir / "tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        build_env["TMPDIR"] = str(tmpdir)
        build_env["APPTAINER_TMPDIR"] = str(tmpdir)

        print(f"[build] Building: {' '.join(build_cmd)}")
        result = subprocess.run(build_cmd, text=True, cwd=str(self.project_root), env=build_env)
        if result.returncode != 0:
            raise RuntimeError(
                f"apptainer build failed (exit {result.returncode}) for {image_path}"
            )
        print(f"[build] Image ready: {image_path}")

    def _build_image_docker(self, pkg_config: dict, pkg_dir: Path, image_path: Path) -> None:
        """Build a Docker image using a Docker-compatible lowercase tag."""
        from mlsbench.cli import (
            BUILD_PASSTHROUGH_ENV_VARS,
            docker_image_tag,
            iter_passthrough_env_vars,
            resolve_docker_extra_files,
        )

        tag = docker_image_tag(image_path.stem)
        global_cfg = self._effective_global_config()
        data_root = global_cfg.get("data_root", str(self.project_root / "vendor" / "data"))
        docker_extra_files = resolve_docker_extra_files(pkg_config, data_root=data_root)
        dockerfile_content = self._generate_dockerfile_content(
            pkg_config,
            pkg_dir,
            docker_extra_files=docker_extra_files,
        )

        images_dir = self.project_root / "vendor" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        dockerfile_path = images_dir / f"{image_path.stem}.Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        print(f"[build] Dockerfile written: {dockerfile_path}")
        print(f"[build] --- Dockerfile ---\n{dockerfile_content}[build] ----------------")

        # Build context is pkg_dir's parent so COPY pkg_dir.name works
        context_dir = pkg_dir.parent
        build_cmd = ["docker", "build"]
        if self._platform:
            build_cmd.extend(["--platform", self._platform])
        # `--build-arg` values are persisted in image layer history, so only
        # forward non-secret env vars. Secrets are runtime-only.
        for name, value in iter_passthrough_env_vars(BUILD_PASSTHROUGH_ENV_VARS):
            build_cmd.extend(["--build-arg", f"{name}={value}"])
        for ef in docker_extra_files:
            build_cmd.extend(["--build-context", f"{ef['context_name']}={ef['context_path']}"])
        build_cmd.extend([
            "-t", tag,
            "-f", str(dockerfile_path),
            str(context_dir),
        ])

        print(f"[build] Building: {' '.join(build_cmd)}")
        result = subprocess.run(build_cmd, text=True, cwd=str(self.project_root))
        if result.returncode != 0:
            raise RuntimeError(
                f"docker build failed (exit {result.returncode}) for {tag}"
            )
        print(f"[build] Docker image ready: {tag}")

    def _ensure_data(self, pkg_name: str, pkg_config: dict) -> None:
        """Prepare data dependencies for a package if any are missing."""
        from mlsbench.cli import prepare_data_for_package
        deps = pkg_config.get("data_deps", [])
        if not deps:
            return
        global_cfg = self._effective_global_config()
        data_root = global_cfg.get(
            "data_root", str(self.project_root / "vendor" / "data"))
        prepare_data_for_package(
            pkg_name,
            pkg_config,
            data_root,
            global_config=global_cfg,
        )

    def _effective_global_config(self) -> dict:
        """Return the active run config, falling back to configs/config.yaml."""
        if self.global_config:
            return dict(self.global_config)

        from mlsbench.cli import load_global_config

        return load_global_config(str(self.project_root / "configs" / "config.yaml"))

    def _ensure_image(self, cmd_entry: dict) -> tuple[Path | str, dict]:
        """Return (image_ref, pkg_config) for this cmd_entry, building image if needed.

        For Apptainer, image_ref is a Path to the .sif file.
        For Docker, image_ref is the image tag string (e.g. "mlsbench/CORL:latest").
        For local execution, image_ref is the external package directory Path.
        Image is keyed by package name (shared across tasks that use the same package).
        Also ensures data dependencies are prepared.
        """
        pkg_name = cmd_entry.get("package")
        if not pkg_name:
            raise RuntimeError(
                f"cmd_entry '{cmd_entry.get('cmd')}' has no package — cannot resolve image"
            )
        pkg_config = self._load_pkg_config(pkg_name)
        pkg_dir = self._find_ext_pkg_dir(pkg_name)

        if self.container_runtime == "local":
            with self._build_lock:
                self._ensure_data(pkg_dir.name, pkg_config)
            return pkg_dir, pkg_config

        if self.container_runtime == "docker":
            from mlsbench.cli import docker_image_tag, try_pull_prebuilt

            tag = docker_image_tag(pkg_dir.name)
            # image_path used only for building (to derive tag/Dockerfile name)
            image_path = self.project_root / "vendor" / "images" / f"{pkg_dir.name}.sif"

            with self._build_lock:
                # Check if Docker image exists
                check = subprocess.run(
                    ["docker", "image", "inspect", tag],
                    capture_output=True, text=True,
                )
                if check.returncode != 0:
                    # Try the maintainer-published image on Docker Hub first.
                    if try_pull_prebuilt(pkg_dir.name, "docker"):
                        check = subprocess.run(
                            ["docker", "image", "inspect", tag],
                            capture_output=True, text=True,
                        )
                if check.returncode != 0:
                    print(f"[build] Docker image not found, building {tag} ...")
                    self._build_image(pkg_config, pkg_dir, image_path)
                self._ensure_data(pkg_dir.name, pkg_config)

            return tag, pkg_config
        else:
            from mlsbench.cli import try_pull_prebuilt

            # Apptainer: image named after the pkg config file stem
            image = self.project_root / "vendor" / "images" / f"{pkg_dir.name}.sif"

            with self._build_lock:
                if not image.exists():
                    # Try the maintainer-published image on Docker Hub first.
                    try_pull_prebuilt(pkg_dir.name, "apptainer", sif_path=image)
                if not image.exists():
                    print(f"[build] Image not found, building {image} ...")
                    self._build_image(pkg_config, pkg_dir, image)
                self._ensure_data(pkg_dir.name, pkg_config)

            return image, pkg_config

    @staticmethod
    def _expand_env_template(value: str, base_env: dict[str, str]) -> str:
        """Expand ``$VAR`` and ``${VAR}`` references against ``base_env``."""
        pattern = re.compile(r"\$(\w+)|\$\{([^}]+)\}")

        def repl(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(2) or ""
            return base_env.get(name, "")

        return pattern.sub(repl, value)

    @staticmethod
    def _translate_local_env_value(
        value: str,
        path_map: dict[str, str],
        base_env: dict[str, str],
    ) -> str:
        """Expand env templates and translate container paths to host paths."""
        translated = WorkspaceTools._expand_env_template(value, base_env)
        for container_path, host_path in sorted(path_map.items(), key=lambda item: len(item[0]), reverse=True):
            pattern = re.compile(
                rf"(?<![A-Za-z0-9._-]){re.escape(container_path)}(?=$|\s|['\"=:/])"
            )
            translated = pattern.sub(host_path, translated)
        return translated

    @staticmethod
    def _wrap_local_command(cmd: list[str], global_cfg: dict, *, pkg_name: str | None = None) -> list[str]:
        """Optionally wrap a local command in ``conda run`` based on config.

        When *pkg_name* is given and no explicit conda config is set, the
        per-package env ``mlsbench-<pkg>`` is used automatically.
        """
        from mlsbench.cli import wrap_with_conda
        return wrap_with_conda(cmd, global_cfg, pkg_name=pkg_name)

    def _build_local_exec_spec(
        self,
        cmd_entry: dict,
        seed: int,
        gpu_devices: str | None = None,
    ) -> tuple[list[str], str, dict[str, str]]:
        """Build ``(cmd, cwd, env)`` for direct local execution."""
        _, pkg_config = self._ensure_image(cmd_entry)
        workdir = pkg_config.get("workdir", "/app")
        task_dir = self.project_root / "tasks" / self.task_name
        task_mount = workdir.rstrip("/") + "/_task"

        pkg = cmd_entry.get("package")
        pkg_host_dir: Path | None = None
        pkg_workdir = str(task_dir.resolve())
        if pkg:
            try:
                pkg_host_dir = self._find_workspace_pkg(pkg)
            except FileNotFoundError:
                pkg_host_dir = self._find_ext_pkg_dir(pkg)
            pkg_workdir = str(pkg_host_dir.resolve())

        path_map: dict[str, str] = {task_mount: str(task_dir.resolve())}
        if pkg_host_dir is not None:
            path_map[f"{workdir.rstrip('/')}/{pkg_host_dir.name}"] = str(pkg_host_dir.resolve())
            # Map the container workdir root (e.g. /workspace) to the directory
            # that contains the package, so scripts doing `cd /workspace` then
            # `python pkg/script.py` work correctly in local mode.
            path_map.setdefault(workdir.rstrip("/"), str(pkg_host_dir.parent.resolve()))

        from mlsbench.cli import resolve_data_binds
        global_cfg = self._effective_global_config()
        data_root = global_cfg.get("data_root", str(self.project_root / "vendor" / "data"))
        resolved_data_root = str(Path(data_root).expanduser().resolve())
        # Standard data path mappings — must be added before the bare workdir
        # mapping so /workspace/data takes priority over /workspace.
        path_map.setdefault("/data", resolved_data_root)
        path_map.setdefault(f"{workdir.rstrip('/')}/data", resolved_data_root)
        for bind in resolve_data_binds(pkg_config, data_root):
            host_path, container_path = bind.split(":", 1)
            path_map[container_path] = str(Path(host_path).expanduser())
        # Task-level data_deps (same format as pkg config data_deps)
        if self.config_task.get("data_deps"):
            task_data_cfg = {"data_deps": self.config_task["data_deps"]}
            for bind in resolve_data_binds(task_data_cfg, data_root):
                host_path, container_path = bind.split(":", 1)
                path_map.setdefault(container_path, str(Path(host_path).expanduser()))
        # Map container HOME (/root) to real HOME so scripts with hard-coded
        # /root/data paths translate correctly in local mode.
        real_home = os.environ.get("HOME", "")
        if real_home:
            path_map.setdefault("/root", real_home)

        run_env = os.environ.copy()
        run_env["DATA_ROOT"] = resolved_data_root
        merged_env = dict(pkg_config.get("env", {}))
        merged_env.update(pkg_config.get("local_env", {}))
        # Per-baseline / per-run env (e.g. ALLOW_DENSE_FLAG=1 for dense oracle)
        merged_env.update(self.extra_env)
        for key, value in merged_env.items():
            if key == "HOME":
                continue
            expanded = str(value).replace("{project_root}", str(self.project_root))
            expanded = expanded.replace("{data_root}", str(data_root))
            run_env[key] = self._translate_local_env_value(expanded, path_map, run_env)

        if self.save_path:
            run_env["SAVE_PATH"] = self.save_path
            output_dir = f"{self.save_path}/{self.task_name}/{self.exp_name}/seed_{seed}"
            run_env["OUTPUT_DIR"] = output_dir
        run_env["SEED"] = str(seed)
        label = cmd_entry.get("label", "")
        if label:
            run_env["ENV"] = label
        # Per-entry env (e.g. H200 BATCH_SIZE/GRAD_ACCUM override); wins last.
        for k, v in (cmd_entry.get("env") or {}).items():
            run_env[k] = str(v)
        if gpu_devices:
            run_env["CUDA_VISIBLE_DEVICES"] = gpu_devices
            run_env["NVIDIA_VISIBLE_DEVICES"] = gpu_devices

        run_env["MLSBENCH_TASK_DIR"] = str(task_dir.resolve())
        if pkg_host_dir is not None:
            run_env["MLSBENCH_PKG_DIR"] = str(pkg_host_dir.resolve())
        run_env["MLSBENCH_LOCAL_PATH_MAP_JSON"] = json.dumps(path_map)

        if pkg and self._normalize_pkg_name(pkg) == self._normalize_pkg_name("VICON"):
            if pkg_host_dir is not None:
                run_env["VICON_WORKDIR"] = str(pkg_host_dir.resolve())
            vicon_data_root = path_map.get("/data/icon-data")
            if vicon_data_root:
                run_env["VICON_DATA_ROOT"] = vicon_data_root

        from mlsbench.cli import _has_conda_support
        conda_available = _has_conda_support(global_cfg)
        if pkg and not conda_available:
            from mlsbench.cli import local_python_target_dir

            local_site = local_python_target_dir(pkg, self.project_root).resolve()
            current_pythonpath = run_env.get("PYTHONPATH", "")
            run_env["PYTHONPATH"] = (
                f"{local_site}:{current_pythonpath}" if current_pythonpath else str(local_site)
            )

        from mlsbench.cli import apply_local_thread_limits

        apply_local_thread_limits(run_env, global_cfg)

        xdg_config_home = self.project_root / ".xdg-config"
        mpl_config_dir = self.project_root / ".mplconfig"
        xdg_config_home.mkdir(parents=True, exist_ok=True)
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        run_env.setdefault("XDG_CONFIG_HOME", str(xdg_config_home.resolve()))
        run_env.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
        run_env.setdefault("PYTHONUNBUFFERED", "1")

        script_src = (task_dir / cmd_entry["cmd"]).resolve()
        script_root = self.workspace_task_dir / ".local_scripts"
        script_root.mkdir(parents=True, exist_ok=True)
        script_dst = script_root / cmd_entry["cmd"]
        script_dst.parent.mkdir(parents=True, exist_ok=True)

        # Mirror sibling files (e.g. run_workflow.py) so $(dirname "$0")/foo.py
        # works the same as it would when bash-running task_dir/scripts/foo.sh.
        import shutil
        for sibling in script_src.parent.iterdir():
            if sibling.is_file() and sibling != script_src:
                dst = script_dst.parent / sibling.name
                if not dst.exists() or dst.stat().st_mtime < sibling.stat().st_mtime:
                    shutil.copy2(sibling, dst)

        translated_script = script_src.read_text()
        for container_path, host_path in sorted(path_map.items(), key=lambda item: len(item[0]), reverse=True):
            pattern = re.compile(
                rf"(?<![A-Za-z0-9._-]){re.escape(container_path)}(?=$|\s|['\"=:/])"
            )
            translated_script = pattern.sub(host_path, translated_script)
        script_dst.write_text(translated_script)
        os.chmod(script_dst, script_src.stat().st_mode)

        local_cmd = self._wrap_local_command(
            ["bash", str(script_dst.resolve())],
            global_cfg,
            pkg_name=pkg,
        )
        return local_cmd, pkg_workdir, run_env

    # ------------------------------------------------------------------
    # Tool: test — helpers
    # ------------------------------------------------------------------

    def _build_apptainer_cmd(
        self,
        cmd_entry: dict,
        seed: int,
        gpu_devices: str | None = None,
    ) -> list[str]:
        """Build the apptainer exec command for a single test_cmd entry."""
        from mlsbench.cli import iter_passthrough_env_vars

        image, pkg_config = self._ensure_image(cmd_entry)
        use_cuda = self._effective_use_cuda(pkg_config)
        workdir = pkg_config.get("workdir", "/app")
        task_dir = self.project_root / "tasks" / self.task_name
        task_mount = workdir.rstrip("/") + "/_task"

        apptainer_cmd = ["apptainer", "exec"]
        if use_cuda:
            apptainer_cmd.append("--nv")
        if pkg_config.get("fakeroot", False):
            apptainer_cmd.append("--fakeroot")
        if pkg_config.get("writable_tmpfs", False):
            apptainer_cmd.append("--writable-tmpfs")
        if pkg_config.get("no_home", False):
            apptainer_cmd.append("--no-home")
        for flag in pkg_config.get("apptainer_flags", []):
            apptainer_cmd.append(flag)

        # Inject environment variables
        env_vars: list[str] = []
        # Inject env from pkg config (e.g. MUJOCO_GL, PYOPENGL_PLATFORM)
        for k, v in pkg_config.get("env", {}).items():
            env_vars.append(f"{k}={v}")
        # Per-baseline / per-run env (e.g. ALLOW_DENSE_FLAG=1 for dense oracle)
        for k, v in self.extra_env.items():
            env_vars.append(f"{k}={v}")
        for name, value in iter_passthrough_env_vars():
            env_vars.append(f"{name}={value}")
        if self.save_path:
            env_vars.append(f"SAVE_PATH={self.save_path}")
            output_dir = f"{self.save_path}/{self.task_name}/{self.exp_name}/seed_{seed}"
            env_vars.append(f"OUTPUT_DIR={output_dir}")
        env_vars.append(f"SEED={seed}")
        if gpu_devices:
            env_vars.append(f"CUDA_VISIBLE_DEVICES={gpu_devices}")
            env_vars.append(f"NVIDIA_VISIBLE_DEVICES={gpu_devices}")
        # Inject ENV from cmd_entry label (e.g. "halfcheetah-medium-v2") so
        # baseline scripts that use ${ENV} get the correct environment name
        label = cmd_entry.get("label", "")
        if label:
            env_vars.append(f"ENV={label}")
        # Per-entry env (e.g. H200 BATCH_SIZE/GRAD_ACCUM override); wins last.
        for k, v in (cmd_entry.get("env") or {}).items():
            env_vars.append(f"{k}={v}")
        for ev in env_vars:
            apptainer_cmd.extend(["--env", ev])

        # Bind the cmd-specific packages
        binds: list[str] = []
        wtd = self.workspace_task_dir
        pkg = cmd_entry.get("package")
        pkg_dir: Path | None = None
        pkg_workdir = workdir  # default pwd
        if pkg and wtd.exists():
            norm = self._normalize_pkg_name(pkg)
            for d in wtd.iterdir():
                if d.is_dir() and self._normalize_pkg_name(d.name) == norm:
                    pkg_workdir = f"{workdir}/{d.name}"
                    binds.append(f"{d.resolve()}:{pkg_workdir}")
                    pkg_dir = d
                    break
        binds.append(f"{task_dir.resolve()}:{task_mount}")

        # Config-level data bind (with template expansion)
        from mlsbench.cli import resolve_data_binds
        global_cfg = self._effective_global_config()
        data_root = global_cfg.get(
            "data_root", str(self.project_root / "vendor" / "data"))
        binds.extend(resolve_data_binds(pkg_config, data_root))

        # Task-level data_deps (same format as pkg config data_deps)
        if self.config_task.get("data_deps"):
            task_data_cfg = {"data_deps": self.config_task["data_deps"]}
            for bind in resolve_data_binds(task_data_cfg, data_root):
                if bind not in binds:
                    binds.append(bind)

        if self.save_path:
            save_host = Path(self.save_path).expanduser().resolve()
            save_host.mkdir(parents=True, exist_ok=True)
            binds.append(f"{save_host}:{self.save_path}")

        # When using --contain, /tmp is emptied; bind a host tmpdir
        # Also bind /dev/shm for Ray/vLLM shared memory
        # Place the host tmpdir on shared scratch (visible from SLURM compute
        # nodes) rather than submit-node /tmp, so sbatch'd apptainer runs can
        # find the bind source.
        if "--contain" in apptainer_cmd:
            import tempfile
            tmp_root = pkg_dir.parent if pkg_dir is not None else self.project_root
            (tmp_root / "apptmp").mkdir(parents=True, exist_ok=True)
            host_tmp = tempfile.mkdtemp(prefix="aptmp-", dir=str(tmp_root / "apptmp"))
            binds.append(f"{host_tmp}:/tmp")
        if use_cuda:
            binds.append("/dev/shm")

        apptainer_cmd.extend(["--env", f"MLSBENCH_TASK_DIR={task_mount}"])
        if pkg:
            apptainer_cmd.extend(["--env", f"MLSBENCH_PKG_DIR={pkg_workdir}"])

        cmd = cmd_entry["cmd"]
        apptainer_cmd.extend(["--bind", ",".join(binds)])
        apptainer_cmd.extend(["--pwd", pkg_workdir, str(image)])
        apptainer_cmd.extend(["bash", f"{task_mount}/{cmd}"])

        return apptainer_cmd

    def _build_docker_cmd(
        self,
        cmd_entry: dict,
        seed: int,
        gpu_devices: str | None = None,
    ) -> list[str]:
        """Build the docker run command for a single test_cmd entry."""
        from mlsbench.cli import iter_passthrough_env_vars

        image_tag, pkg_config = self._ensure_image(cmd_entry)
        use_cuda = self._effective_use_cuda(pkg_config)
        workdir = pkg_config.get("workdir", "/app")
        task_dir = self.project_root / "tasks" / self.task_name
        task_mount = workdir.rstrip("/") + "/_task"

        # Generate a unique container name for log retrieval
        import uuid
        container_name = f"mlsbench-{self.task_name}-{uuid.uuid4().hex[:8]}"
        docker_cmd = ["docker", "run", "--name", container_name,
                       "--shm-size=16g", "--entrypoint", ""]
        if self._platform:
            docker_cmd.extend(["--platform", self._platform])
        if use_cuda:
            selected_gpus = gpu_devices or self.gpu_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if selected_gpus:
                docker_cmd.extend(["--gpus", self._docker_gpus_arg(selected_gpus)])
            else:
                docker_cmd.extend(["--gpus", "all"])

        # Inject environment variables
        for k, v in pkg_config.get("env", {}).items():
            docker_cmd.extend(["-e", f"{k}={v}"])
        for k, v in self.extra_env.items():
            docker_cmd.extend(["-e", f"{k}={v}"])
        for name, value in iter_passthrough_env_vars():
            docker_cmd.extend(["-e", f"{name}={value}"])
        if self.save_path:
            docker_cmd.extend(["-e", f"SAVE_PATH={self.save_path}"])
            output_dir = f"{self.save_path}/{self.task_name}/{self.exp_name}/seed_{seed}"
            docker_cmd.extend(["-e", f"OUTPUT_DIR={output_dir}"])
        docker_cmd.extend(["-e", f"SEED={seed}"])
        label = cmd_entry.get("label", "")
        if label:
            docker_cmd.extend(["-e", f"ENV={label}"])
        # Per-entry env (e.g. H200 BATCH_SIZE/GRAD_ACCUM override); wins last.
        for k, v in (cmd_entry.get("env") or {}).items():
            docker_cmd.extend(["-e", f"{k}={v}"])

        # Bind mounts
        wtd = self.workspace_task_dir
        pkg = cmd_entry.get("package")
        pkg_workdir = workdir  # default pwd
        if pkg and wtd.exists():
            norm = self._normalize_pkg_name(pkg)
            for d in wtd.iterdir():
                if d.is_dir() and self._normalize_pkg_name(d.name) == norm:
                    pkg_workdir = f"{workdir}/{d.name}"
                    docker_cmd.extend(["-v", f"{d.resolve()}:{pkg_workdir}"])
                    break
        docker_cmd.extend(["-v", f"{task_dir.resolve()}:{task_mount}"])
        docker_cmd.extend(["-e", f"MLSBENCH_TASK_DIR={task_mount}"])
        if pkg:
            docker_cmd.extend(["-e", f"MLSBENCH_PKG_DIR={pkg_workdir}"])

        if self.save_path:
            save_host = Path(self.save_path).expanduser().resolve()
            save_host.mkdir(parents=True, exist_ok=True)
            docker_cmd.extend(["-v", f"{save_host}:{self.save_path}"])

        # Config-level data bind (with template expansion)
        from mlsbench.cli import resolve_data_binds
        global_cfg = self._effective_global_config()
        data_root = global_cfg.get(
            "data_root", str(self.project_root / "vendor" / "data"))
        data_bind_targets: dict[str, str] = {}
        pkg_binds = resolve_data_binds(pkg_config, data_root)
        for db in pkg_binds:
            target = self._container_bind_target(db)
            previous = data_bind_targets.get(target)
            if previous and previous != db:
                raise RuntimeError(
                    f"Conflicting Docker data binds for container path {target}: "
                    f"{previous} vs {db}"
                )
            docker_cmd.extend(["-v", db])
            if target:
                data_bind_targets[target] = db

        # Task-level data_deps (same format as pkg config data_deps)
        if self.config_task.get("data_deps"):
            task_data_cfg = {"data_deps": self.config_task["data_deps"]}
            for bind in resolve_data_binds(task_data_cfg, data_root):
                target = self._container_bind_target(bind)
                previous = data_bind_targets.get(target)
                if previous is None:
                    docker_cmd.extend(["-v", bind])
                    if target:
                        data_bind_targets[target] = bind
                elif previous != bind:
                    raise RuntimeError(
                        f"Conflicting Docker data binds for container path {target}: "
                        f"{previous} vs {bind}"
                    )

        # Mount user home data directory for datasets
        home_data = Path.home() / "data"
        if home_data.exists() and "/root/data" not in data_bind_targets:
            docker_cmd.extend(["-v", f"{home_data.resolve()}:/root/data"])

        docker_cmd.extend(["-w", pkg_workdir])
        docker_cmd.append(str(image_tag))

        cmd = cmd_entry["cmd"]
        # docker_cmd.extend([f"{task_mount}/{cmd}"])
        docker_cmd.extend(["bash", f"{task_mount}/{cmd}"])

        return docker_cmd, container_name

    @staticmethod
    def _docker_gpus_arg(selected_gpus: str) -> str:
        selected_gpus = selected_gpus.strip()
        if selected_gpus in {"all", "none"}:
            return selected_gpus
        if selected_gpus.startswith(("device=", '"device=', "'device=")):
            spec = selected_gpus
        else:
            spec = f"device={selected_gpus}"
        if spec.startswith(("\"", "'")):
            return spec
        return f'"{spec}"'

    def _build_container_cmd(
        self,
        cmd_entry: dict,
        seed: int,
        gpu_devices: str | None = None,
    ) -> list[str] | tuple[list[str], str]:
        """Build the container command for a single test_cmd entry.

        Dispatches to _build_apptainer_cmd or _build_docker_cmd based on container_runtime.
        For Docker, returns (cmd_list, container_name) tuple.
        For Apptainer, returns cmd_list.
        """
        if self.container_runtime == "docker":
            return self._build_docker_cmd(cmd_entry, seed, gpu_devices=gpu_devices)
        return self._build_apptainer_cmd(cmd_entry, seed, gpu_devices=gpu_devices)

    @staticmethod
    def _as_docker_create_cmd(container_cmd: list[str]) -> list[str]:
        """Convert a ``docker run`` command into ``docker create``."""
        if len(container_cmd) < 2 or container_cmd[0] != "docker" or container_cmd[1] != "run":
            raise ValueError("expected a docker run command")
        return ["docker", "create", *container_cmd[2:]]

    def _docker_container_state(
        self,
        container_name: str,
        run_env: dict[str, str],
    ) -> tuple[str | None, int | None]:
        """Return ``(status, exit_code)`` for a Docker container, or ``(None, None)``."""
        try:
            inspect_result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", container_name],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=5,
                env=run_env,
            )
        except Exception:
            return None, None
        if inspect_result.returncode != 0:
            return None, None
        parts = (inspect_result.stdout or "").strip().split()
        if not parts:
            return None, None
        status = parts[0].strip().lower()
        exit_code = None
        if len(parts) > 1:
            try:
                exit_code = int(parts[1])
            except ValueError:
                exit_code = None
        return status, exit_code

    def _docker_container_exists(self, container_name: str, run_env: dict[str, str]) -> bool:
        """Return whether the Docker container object exists."""
        status, _ = self._docker_container_state(container_name, run_env)
        return status is not None

    def _wait_for_docker_create(
        self,
        create_cmd: list[str],
        container_name: str,
        run_env: dict[str, str],
        deadline: float,
    ) -> tuple[subprocess.CompletedProcess[str], str] | None:
        """Wait for a ``docker create`` client or the container object to materialize."""
        create_deadline = min(deadline, _time.time() + 60)
        proc = subprocess.Popen(
            create_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.project_root),
            env=run_env,
        )
        while _time.time() < create_deadline:
            ret = proc.poll()
            if ret is not None:
                stdout, stderr = proc.communicate()
                raw_output = (stdout or "") + (stderr or "")
                result = subprocess.CompletedProcess(create_cmd, ret, stdout=stdout, stderr=stderr)
                if ret == 0 or self._docker_container_exists(container_name, run_env):
                    return result, raw_output
                return result, raw_output
            if self._docker_container_exists(container_name, run_env):
                try:
                    proc.terminate()
                    stdout, stderr = proc.communicate(timeout=5)
                except Exception:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                raw_output = (stdout or "") + (stderr or "")
                return subprocess.CompletedProcess(create_cmd, 0, stdout=stdout, stderr=stderr), raw_output
            _time.sleep(0.2)

        try:
            proc.kill()
        except Exception:
            pass
        stdout, stderr = proc.communicate()
        raw_output = (stdout or "") + (stderr or "")
        if self._docker_container_exists(container_name, run_env):
            return subprocess.CompletedProcess(create_cmd, 0, stdout=stdout, stderr=stderr), raw_output
        return None

    def _wait_for_docker_start_attempt(
        self,
        start_cmd: list[str],
        container_name: str,
        run_env: dict[str, str],
        attempt_deadline: float,
    ) -> tuple[subprocess.CompletedProcess[str], str] | None:
        """Wait for one ``docker start`` attempt to move the container beyond ``Created``."""
        proc = subprocess.Popen(
            start_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.project_root),
            env=run_env,
        )
        while _time.time() < attempt_deadline:
            ret = proc.poll()
            if ret is not None:
                stdout, stderr = proc.communicate()
                raw_output = (stdout or "") + (stderr or "")
                result = subprocess.CompletedProcess(start_cmd, ret, stdout=stdout, stderr=stderr)
                status, _ = self._docker_container_state(container_name, run_env)
                if ret == 0 or status in ("running", "exited", "dead"):
                    return result, raw_output
                return result, raw_output
            status, _ = self._docker_container_state(container_name, run_env)
            if status in ("running", "exited", "dead"):
                try:
                    proc.terminate()
                    stdout, stderr = proc.communicate(timeout=5)
                except Exception:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                raw_output = (stdout or "") + (stderr or "")
                return subprocess.CompletedProcess(start_cmd, 0, stdout=stdout, stderr=stderr), raw_output
            _time.sleep(0.2)

        status, _ = self._docker_container_state(container_name, run_env)
        if status in ("running", "exited", "dead"):
            try:
                proc.terminate()
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                proc.kill()
                stdout, stderr = proc.communicate()
            raw_output = (stdout or "") + (stderr or "")
            return subprocess.CompletedProcess(start_cmd, 0, stdout=stdout, stderr=stderr), raw_output
        return None

    def _start_docker_container(
        self,
        container_name: str,
        run_env: dict[str, str],
        deadline: float,
    ) -> tuple[subprocess.CompletedProcess[str], str] | None:
        """Retry ``docker start`` until the container is running/exited or timeout."""
        start_cmd = ["docker", "start", container_name]
        attempt_budget = min(5.0, max(1.0, deadline - _time.time()))
        for _attempt in range(3):
            started = self._wait_for_docker_start_attempt(
                start_cmd,
                container_name,
                run_env,
                min(deadline, _time.time() + attempt_budget),
            )
            if started is not None:
                return started
            status, _ = self._docker_container_state(container_name, run_env)
            if status in ("running", "exited", "dead"):
                return subprocess.CompletedProcess(start_cmd, 0, stdout="", stderr=""), ""
        return None

    def _launch_docker_container(
        self,
        container_cmd: list[str],
        container_name: str,
        run_env: dict[str, str],
        timeout_secs: int,
    ) -> tuple[subprocess.CompletedProcess[str] | None, str, bool]:
        """Create and start a Docker container, returning ``(result, output, timed_out)``."""
        deadline = _time.time() + timeout_secs
        create_cmd = self._as_docker_create_cmd(container_cmd)
        launch_lock_path = self.project_root / ".docker-launch.lock"

        with open(launch_lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            waited = self._wait_for_docker_create(create_cmd, container_name, run_env, deadline)
            if waited is None:
                return None, "[TIMEOUT] docker create did not materialize a container before timeout.", True
            create_result, raw_output = waited
            if create_result.returncode != 0 and not self._docker_container_exists(container_name, run_env):
                return create_result, raw_output, False

            started = self._start_docker_container(container_name, run_env, deadline)
            if started is None:
                subprocess.run(
                    ["docker", "stop", container_name],
                    capture_output=True,
                    timeout=30,
                )
                raw_output = ""
                try:
                    logs_result = subprocess.run(
                        ["docker", "logs", container_name],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    raw_output = (logs_result.stdout or "") + (logs_result.stderr or "")
                except Exception:
                    pass
                raw_output = (
                    f"[TIMEOUT] docker start did not move '{container_name}' out of Created before timeout.\n"
                    f"{raw_output}"
                )
                return None, raw_output, True

            start_result, start_output = started
            if start_result.returncode != 0:
                raw_output = (start_result.stdout or "") + (start_result.stderr or "")
                return start_result, raw_output, False

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return start_result, start_output, False

    def _remove_docker_container(self, container_name: str, run_env: dict[str, str]) -> None:
        """Best-effort removal that tolerates hanging rootless Docker clients."""
        remove_cmd = ["docker", "rm", "-f", container_name]
        remove_deadline = _time.time() + 30
        proc = subprocess.Popen(
            remove_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.project_root),
            env=run_env,
        )
        while _time.time() < remove_deadline:
            ret = proc.poll()
            if ret is not None:
                proc.communicate()
                return
            if not self._docker_container_exists(container_name, run_env):
                try:
                    proc.terminate()
                    proc.communicate(timeout=5)
                except Exception:
                    proc.kill()
                    proc.communicate()
                return
            _time.sleep(0.2)
        try:
            proc.kill()
        except Exception:
            pass
        proc.communicate()

    def _run_docker_container(
        self,
        container_cmd: list[str],
        container_name: str,
        run_env: dict[str, str],
        timeout_secs: int,
    ) -> tuple[subprocess.CompletedProcess[str] | None, str, bool]:
        """Create, start, and clean up one Docker container.

        Returns ``(result, raw_output, timed_out)``.
        ``result`` is ``None`` only when the attached start phase timed out.
        """
        created = False
        start_output = ""
        deadline = _time.time() + timeout_secs
        try:
            launch_result, start_output, timed_out = self._launch_docker_container(
                container_cmd,
                container_name,
                run_env,
                timeout_secs=timeout_secs,
            )
            if timed_out:
                return None, start_output, True
            if launch_result is None or launch_result.returncode != 0:
                return launch_result, start_output, False
            created = True

            try:
                wait_result = subprocess.run(
                    ["docker", "wait", container_name],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root),
                    timeout=max(1.0, deadline - _time.time()),
                    env=run_env,
                )
            except subprocess.TimeoutExpired:
                subprocess.run(
                    ["docker", "stop", container_name],
                    capture_output=True,
                    timeout=30,
                )
                raw_output = ""
                try:
                    logs_result = subprocess.run(
                        ["docker", "logs", container_name],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    raw_output = (logs_result.stdout or "") + (logs_result.stderr or "")
                except Exception:
                    pass
                raw_output = (
                    f"[TIMEOUT] Command timed out after {timeout_secs}s. "
                    f"This result is INVALID and will not count. "
                    f"Your algorithm is too slow — reduce model size or computational complexity.\n{raw_output}"
                )
                return None, raw_output, True

            raw_output = ""
            if not raw_output:
                try:
                    logs_result = subprocess.run(
                        ["docker", "logs", container_name],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    raw_output = (logs_result.stdout or "") + (logs_result.stderr or "")
                except Exception:
                    pass
            if not raw_output:
                raw_output = start_output
            exit_code_str = (wait_result.stdout or "").strip().splitlines()
            try:
                exit_code = int(exit_code_str[-1]) if exit_code_str else wait_result.returncode
            except ValueError:
                exit_code = wait_result.returncode
            return (
                subprocess.CompletedProcess(
                    ["docker", "wait", container_name],
                    exit_code,
                    stdout=raw_output,
                    stderr=wait_result.stderr or "",
                ),
                raw_output,
                False,
            )
        finally:
            if created:
                try:
                    self._remove_docker_container(container_name, run_env)
                except Exception:
                    pass

    @staticmethod
    def _parse_time_to_seconds(time_str: str) -> int:
        """Parse HH:MM:SS, MM:SS, or plain seconds to total seconds."""
        parts = str(time_str).split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            else:
                return int(float(parts[0]))
        except (ValueError, IndexError):
            return 3600  # default 1 hour

    # ------------------------------------------------------------------
    # Budget check: parameter-cap enforcement (and post-hoc logging when
    # disabled, so the agent's chosen capacity is still recoverable).
    # ------------------------------------------------------------------

    _BUDGET_PARAM_RE = re.compile(r"agent model:\s*(\d+)\s*params", re.IGNORECASE)
    _BUDGET_BL_RE = re.compile(r"baseline\s+(\S+):\s*(\d+)\s*params", re.IGNORECASE)
    _BUDGET_CAP_RE = re.compile(r"budget:\s*(\d+)", re.IGNORECASE)

    def _log_budget_output(self, label: str, seed: int, output: str, returncode: int | None) -> None:
        """Parse budget_check.py stdout and persist a JSONL record.

        Captures: agent_params, per-baseline param counts, the cap, and the
        returncode (so post-hoc inspection can tell which runs would have
        been rejected by the cap had it been enforced).
        """
        m = self._BUDGET_PARAM_RE.search(output or "")
        agent_params = int(m.group(1)) if m else None
        baselines = {bn: int(bp) for bn, bp in self._BUDGET_BL_RE.findall(output or "")}
        cap_m = self._BUDGET_CAP_RE.search(output or "")
        budget_cap = int(cap_m.group(1)) if cap_m else None

        if agent_params is not None:
            self._last_agent_params[label] = agent_params

        # Resolve target log file once. Live alongside messages.jsonl in the
        # agent log dir so users can git-grep / inspect after the run.
        if self._param_log_path is None:
            log_dir = self.agent_log_dir or (
                self.project_root / "logs" / self.task_name
                / self._sanitize_for_path(self.model_name)
                / "agent"
            )
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                return
            self._param_log_path = log_dir / "param_counts.jsonl"

        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "task": self.task_name,
            "model": self._decorated_model_name(),
            "exp_name": self.exp_name,
            "test_count": self.test_count,
            "label": label,
            "seed": seed,
            "agent_params": agent_params,
            "baseline_params": baselines,
            "budget_cap": budget_cap,
            "returncode": returncode,
            "would_fail": (
                budget_cap is not None
                and agent_params is not None
                and agent_params > budget_cap
            ),
        }
        try:
            with open(self._param_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _sanitize_for_path(s: str) -> str:
        """Match BaseAgent's exp_name sanitizer for log directory names."""
        return re.sub(r"[/: ]", "_", s or "unknown")

    def _build_local_budget_exec_spec(
        self,
        cmd_entry: dict,
        seed: int,
        gpu_devices: str | None = None,
    ) -> tuple[list[str], str, dict[str, str]]:
        """Build the local/Conda command used to run budget_check.py.

        This mirrors the local branch of _run_budget_check(), and is also used
        by scheduler-backed execution paths that bypass _run_single_cmd().
        """
        task_dir = self.project_root / "tasks" / self.task_name
        budget_script = task_dir / "budget_check.py"
        _local_cmd, cwd, run_env = self._build_local_exec_spec(
            cmd_entry,
            seed,
            gpu_devices=gpu_devices,
        )
        pkg = cmd_entry.get("package")
        global_cfg = self._effective_global_config()
        budget_to_run = budget_script.resolve()
        path_map_raw = run_env.get("MLSBENCH_LOCAL_PATH_MAP_JSON", "")
        if path_map_raw:
            try:
                path_map = json.loads(path_map_raw)
            except json.JSONDecodeError:
                path_map = {}
            if path_map:
                budget_dst = self.workspace_task_dir / ".local_scripts" / "budget_check.py"
                budget_dst.parent.mkdir(parents=True, exist_ok=True)
                translated_budget = budget_script.read_text()
                for container_path, host_path in sorted(
                    path_map.items(), key=lambda item: len(item[0]), reverse=True
                ):
                    pattern = re.compile(
                        rf"(?<![A-Za-z0-9._-]){re.escape(container_path)}(?=$|\s|['\"=:/])"
                    )
                    translated_budget = pattern.sub(host_path, translated_budget)
                budget_dst.write_text(translated_budget)
                budget_to_run = budget_dst.resolve()
        budget_cmd = self._wrap_local_command(
            ["python", str(budget_to_run)],
            global_cfg,
            pkg_name=pkg,
        )
        return budget_cmd, cwd, run_env

    def _build_container_budget_cmd(
        self,
        cmd_entry: dict,
        seed: int,
        gpu_devices: str | None = None,
    ) -> list[str]:
        """Build a container command that runs budget_check.py instead of training."""
        _, pkg_config = self._ensure_image(cmd_entry)
        workdir = pkg_config.get("workdir", "/app")
        task_mount = workdir.rstrip("/") + "/_task"
        check_cmd = f"python {task_mount}/budget_check.py"

        container_result = self._build_container_cmd(
            cmd_entry,
            seed,
            gpu_devices=gpu_devices,
        )
        if isinstance(container_result, tuple):
            container_cmd = list(container_result[0])
        else:
            container_cmd = list(container_result)

        for i, arg in enumerate(container_cmd):
            if arg == "bash":
                return container_cmd[:i] + ["bash", "-c", check_cmd]
        return container_cmd + ["bash", "-c", check_cmd]

    def _run_budget_check(self, cmd_entry: dict, seed: int, gpu_devices: str | None = None) -> str | None:
        """Run budget_check.py before training. Returns error msg or None.

        The check is always enforced when tasks/<task>/budget_check.py exists.
        Its output is also logged for auditability.
        """
        task_dir = self.project_root / "tasks" / self.task_name
        budget_script = task_dir / "budget_check.py"
        if not budget_script.exists():
            return None
        label = cmd_entry.get("label", "test")

        def _finalize(output: str, returncode: int | None, error: str | None) -> str | None:
            self._log_budget_output(label, seed, output, returncode)
            return error

        if self.container_runtime == "local":
            budget_cmd, cwd, run_env = self._build_local_budget_exec_spec(
                cmd_entry,
                seed,
                gpu_devices=gpu_devices,
            )
            try:
                result = subprocess.run(
                    budget_cmd,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=120,
                    env=run_env,
                )
                output = (result.stdout or "") + (result.stderr or "")
                err = f"[BUDGET CHECK FAILED]\n{output}" if result.returncode != 0 else None
                return _finalize(output, result.returncode, err)
            except subprocess.TimeoutExpired:
                return _finalize("", None, "[BUDGET CHECK TIMEOUT] budget_check.py took >120s")
            except Exception as e:
                return _finalize("", None, f"[BUDGET CHECK ERROR] {e}")

        container_cmd = self._build_container_budget_cmd(
            cmd_entry,
            seed,
            gpu_devices=gpu_devices,
        )
        container_name = None
        if self.container_runtime == "docker":
            try:
                name_idx = container_cmd.index("--name")
                container_name = container_cmd[name_idx + 1]
            except (ValueError, IndexError):
                container_name = None

        run_env = os.environ.copy()
        if gpu_devices:
            run_env["CUDA_VISIBLE_DEVICES"] = gpu_devices
            run_env["NVIDIA_VISIBLE_DEVICES"] = gpu_devices

        try:
            if container_name:
                result, output, timed_out = self._run_docker_container(
                    container_cmd,
                    container_name,
                    run_env,
                    timeout_secs=120,
                )
                if timed_out:
                    return _finalize(output, None, output)
            else:
                result = subprocess.run(
                    container_cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root),
                    timeout=120,
                    env=run_env,
                )
                output = (result.stdout or "") + (result.stderr or "")
            if result is None or result.returncode != 0:
                return _finalize(output, getattr(result, "returncode", None),
                                 f"[BUDGET CHECK FAILED]\n{output}")
            return _finalize(output, result.returncode, None)
        except subprocess.TimeoutExpired:
            return _finalize("", None, "[BUDGET CHECK TIMEOUT] budget_check.py took >120s")
        except Exception as e:
            return _finalize("", None, f"[BUDGET CHECK ERROR] {e}")

    def _run_local_command(
        self,
        cmd: list[str],
        cwd: str,
        env: dict[str, str],
        timeout_secs: int,
    ) -> tuple[subprocess.CompletedProcess, str, bool]:
        """Run a local command and stream its combined output to stdout."""
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None

        deadline = _time.time() + timeout_secs
        output_parts: list[str] = []

        try:
            while True:
                if _time.time() > deadline:
                    proc.kill()
                    tail = proc.stdout.read() or ""
                    if tail:
                        print(tail, end="", flush=True)
                        output_parts.append(tail)
                    proc.wait(timeout=30)
                    return (
                        subprocess.CompletedProcess(cmd, proc.returncode or 124),
                        "".join(output_parts),
                        True,
                    )

                line = proc.stdout.readline()
                if line:
                    print(line, end="", flush=True)
                    output_parts.append(line)
                    continue

                ret = proc.poll()
                if ret is not None:
                    tail = proc.stdout.read() or ""
                    if tail:
                        print(tail, end="", flush=True)
                        output_parts.append(tail)
                    return (
                        subprocess.CompletedProcess(cmd, ret),
                        "".join(output_parts),
                        False,
                    )

                _time.sleep(0.1)
        finally:
            proc.stdout.close()

    def _build_docker_session_cmd(
        self,
        cmd_entry: dict,
        seed: int,
        gpu_devices: str | None = None,
    ) -> tuple[list[str], str, str, str]:
        """Build a long-lived Docker container command for repeated ``docker exec`` calls."""
        docker_cmd, container_name = self._build_docker_cmd(cmd_entry, seed, gpu_devices=gpu_devices)
        pkg_workdir = "/app"
        task_mount = "/app/_task"
        workdir_idx = -1
        if "-w" in docker_cmd:
            workdir_idx = docker_cmd.index("-w")
            pkg_workdir = docker_cmd[workdir_idx + 1]
            task_mount = pkg_workdir.rsplit("/", 1)[0] + "/_task" if "/" in pkg_workdir else "/_task"
        image_idx = workdir_idx + 2 if workdir_idx >= 0 else len(docker_cmd) - 2
        image = docker_cmd[image_idx]
        session_cmd = docker_cmd[:image_idx] + [
            image,
            "bash",
            "-lc",
            "while true; do sleep 3600; done",
        ]
        return session_cmd, container_name, pkg_workdir, task_mount

    def _is_rootless_docker(self) -> bool:
        """Best-effort detection for rootless Docker; defaults to False on uncertainty."""
        if self._docker_rootless_cache is not None:
            return self._docker_rootless_cache

        override = os.environ.get("MLSBENCH_ROOTLESS_DOCKER", "").strip().lower()
        if override in {"1", "true", "yes"}:
            self._docker_rootless_cache = True
            return True
        if override in {"0", "false", "no"}:
            self._docker_rootless_cache = False
            return False
        if self.container_runtime != "docker":
            self._docker_rootless_cache = False
            return False

        docker_host = os.environ.get("DOCKER_HOST", "")
        if docker_host.startswith("unix:///run/user/"):
            self._docker_rootless_cache = True
            return True

        try:
            info = subprocess.run(
                ["docker", "info", "--format", "{{json .SecurityOptions}}"],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=10,
            )
            if info.returncode == 0:
                self._docker_rootless_cache = "rootless" in (info.stdout or "").lower()
                return self._docker_rootless_cache
        except Exception:
            pass

        self._docker_rootless_cache = False
        return False

    def _docker_exec_env_args(
        self,
        seed: int,
        label: str,
        gpu_devices: str | None = None,
        env: dict | None = None,
    ) -> list[str]:
        """Build ``docker exec`` environment arguments for one command."""
        args: list[str] = []
        if self.save_path:
            args.extend(["-e", f"SAVE_PATH={self.save_path}"])
            output_dir = f"{self.save_path}/{self.task_name}/{self.exp_name}/seed_{seed}"
            args.extend(["-e", f"OUTPUT_DIR={output_dir}"])
        args.extend(["-e", f"SEED={seed}"])
        if label:
            args.extend(["-e", f"ENV={label}"])
        if gpu_devices:
            args.extend(["-e", f"CUDA_VISIBLE_DEVICES={gpu_devices}"])
            args.extend(["-e", f"NVIDIA_VISIBLE_DEVICES={gpu_devices}"])
        # Per-entry env (e.g. H200 BATCH_SIZE/GRAD_ACCUM override); wins last.
        for k, v in (env or {}).items():
            args.extend(["-e", f"{k}={v}"])
        return args

    def _run_docker_exec(
        self,
        container_name: str,
        pkg_workdir: str,
        exec_env_args: list[str],
        exec_cmd: list[str],
        run_env: dict[str, str],
        timeout_secs: int,
    ) -> tuple[subprocess.CompletedProcess[str] | None, str, bool]:
        """Run one command inside an existing Docker container."""
        cmd = ["docker", "exec", *exec_env_args, "-w", pkg_workdir, container_name, *exec_cmd]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=timeout_secs,
                env=run_env,
            )
            raw_output = (result.stdout or "") + (result.stderr or "")
            return result, raw_output, False
        except subprocess.TimeoutExpired:
            return None, (
                f"[TIMEOUT] Command timed out after {timeout_secs}s. "
                f"This result is INVALID and will not count. "
                f"Your algorithm is too slow — reduce model size or computational complexity."
            ), True

    def _run_docker_entries_in_session(
        self,
        cmd_entries: list[dict],
        seed: int,
        gpu_assignments: list[str | None] | None = None,
    ) -> list[tuple[str, dict, float]]:
        """Run a batch of Docker cmds sequentially inside one long-lived container."""
        if not cmd_entries:
            return []

        session_devices = gpu_assignments or self._allocate_group_gpu_assignments(cmd_entries)
        if session_devices is None:
            session_devices = [self._default_gpu_assignment(entry) for entry in cmd_entries]
        visible_devices = [d for d in self._get_visible_gpu_devices() if d]
        union_devices: list[str] = []
        for assignment in session_devices:
            if not assignment:
                continue
            for device in assignment.split(","):
                device = device.strip()
                if device and device not in union_devices:
                    union_devices.append(device)
        if not union_devices:
            union_devices = visible_devices
        session_gpu_devices = ",".join(union_devices) if union_devices else None

        session_cmd, container_name, pkg_workdir, task_mount = self._build_docker_session_cmd(
            cmd_entries[0],
            seed,
            gpu_devices=session_gpu_devices,
        )
        run_env = os.environ.copy()
        if session_gpu_devices:
            run_env["CUDA_VISIBLE_DEVICES"] = session_gpu_devices
            run_env["NVIDIA_VISIBLE_DEVICES"] = session_gpu_devices

        launch_result, launch_output, timed_out = self._launch_docker_container(
            session_cmd,
            container_name,
            run_env,
            timeout_secs=120,
        )
        if timed_out:
            return [
                (
                    f"### {entry.get('label', 'test')} ({entry.get('cmd', 'unknown')})\n{launch_output}",
                    {},
                    0.0,
                )
                for entry in cmd_entries
            ]
        if launch_result is None or launch_result.returncode != 0:
            output = launch_output or "[ERROR] Failed to start long-lived Docker container."
            return [
                (
                    f"### {entry.get('label', 'test')} ({entry.get('cmd', 'unknown')})\n{output}",
                    {},
                    0.0,
                )
                for entry in cmd_entries
            ]

        results: list[tuple[str, dict, float] | None] = [None] * len(cmd_entries)
        try:
            runnable: list[tuple[int, dict, list[str], int]] = []
            for idx, entry in enumerate(cmd_entries):
                label = entry.get("label", "test")
                cmd = entry["cmd"]
                entry_gpu = session_devices[idx] if idx < len(session_devices) else None
                exec_env_args = self._docker_exec_env_args(seed, label, gpu_devices=entry_gpu, env=entry.get("env"))

                budget_script = self.project_root / "tasks" / self.task_name / "budget_check.py"
                if budget_script.exists():
                    budget_start = _time.time()
                    _budget_result, budget_output, budget_timed_out = self._run_docker_exec(
                        container_name,
                        pkg_workdir,
                        exec_env_args,
                        ["bash", "-lc", f"python {task_mount}/budget_check.py"],
                        run_env,
                        timeout_secs=120,
                    )
                    rc = getattr(_budget_result, "returncode", None) if _budget_result else None
                    self._log_budget_output(label, seed, budget_output or "", rc)
                    if budget_timed_out:
                        results[idx] = (f"### {label} ({cmd})\n{budget_output}", {}, _time.time() - budget_start)
                        continue
                    if _budget_result is None or _budget_result.returncode != 0:
                        output = budget_output or "[BUDGET CHECK FAILED]"
                        results[idx] = (f"### {label} ({cmd})\n[BUDGET CHECK FAILED]\n{output}", {}, _time.time() - budget_start)
                        continue

                time_str = entry.get("time", "1:00:00")
                timeout_secs = self._parse_time_to_seconds(time_str) + 300
                runnable.append((idx, entry, exec_env_args, timeout_secs))

            def _run_one_session_entry(
                item: tuple[int, dict, list[str], int],
            ) -> tuple[int, tuple[str, dict, float]]:
                idx, entry, exec_env_args, timeout_secs = item
                label = entry.get("label", "test")
                cmd = entry["cmd"]
                t_start = _time.time()
                result, raw_output, cmd_timed_out = self._run_docker_exec(
                    container_name,
                    pkg_workdir,
                    exec_env_args,
                    ["bash", f"{task_mount}/{cmd}"],
                    run_env,
                    timeout_secs=timeout_secs,
                )
                elapsed = _time.time() - t_start

                if cmd_timed_out:
                    parse_result = self.parser.parse(label, raw_output)
                    return idx, (f"### {label} ({cmd})\n{raw_output}", parse_result.metrics or {}, elapsed)

                if result is None:
                    raw_output = raw_output or "[ERROR] docker exec failed unexpectedly."
                    return idx, (f"### {label} ({cmd})\n{raw_output}", {}, elapsed)

                if not raw_output:
                    raw_output = f"[exit code {result.returncode}, no output]"
                parse_result = self.parser.parse(label, raw_output)
                section_feedback = parse_result.feedback
                max_chars_per_cmd = 6000
                if len(section_feedback) > max_chars_per_cmd:
                    half = max_chars_per_cmd // 2
                    section_feedback = (
                        section_feedback[:half]
                        + "\n...[truncated]...\n"
                        + section_feedback[-half:]
                    )
                return idx, (f"### {label} ({cmd})\n{section_feedback}", parse_result.metrics or {}, elapsed)

            if len(runnable) == 1:
                idx, result = _run_one_session_entry(runnable[0])
                results[idx] = result
            elif runnable:
                with ThreadPoolExecutor(max_workers=len(runnable)) as executor:
                    future_to_idx = {
                        executor.submit(_run_one_session_entry, item): item[0]
                        for item in runnable
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            result_idx, result_tuple = future.result()
                        except Exception as exc:
                            entry = cmd_entries[idx]
                            label = entry.get("label", "test")
                            cmd = entry.get("cmd", "unknown")
                            result_idx = idx
                            result_tuple = (
                                f"### {label} ({cmd})\n[ERROR] Command failed with exception: {exc}",
                                {},
                                0.0,
                            )
                        results[result_idx] = result_tuple
        finally:
            try:
                self._remove_docker_container(container_name, run_env)
            except Exception:
                pass

        return [result if result is not None else ("", {}, 0.0) for result in results]

    def _get_visible_gpu_devices(self) -> list[str]:
        """Return the GPU ids visible to this process, if constrained."""
        devices = self.gpu_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not devices:
            return []
        return [d.strip() for d in devices.split(",") if d.strip()]

    def _default_gpu_assignment(self, cmd_entry: dict) -> str | None:
        """Return a best-effort GPU assignment for a single command."""
        devices = self._get_visible_gpu_devices()
        if not devices:
            return None
        compute = float(cmd_entry.get("compute", 1) or 1)
        need = max(1, math.ceil(compute))
        return ",".join(devices[:need])

    def _allocate_group_gpu_assignments(self, entries: list[dict]) -> list[str | None] | None:
        """Assign visible GPUs across a parallel direct-execution group.

        Returns a per-entry list of GPU id strings, or ``None`` if the visible GPU
        pool is insufficient for the group's concurrent compute requirements.
        """
        devices = self._get_visible_gpu_devices()
        if not devices:
            return [None] * len(entries)

        assignments: list[str | None] = [None] * len(entries)
        remaining = {device: 1.0 for device in devices}

        for idx, entry in enumerate(entries):
            compute = float(entry.get("compute", 1) or 1)
            if compute >= 1.0:
                need = max(1, math.ceil(compute))
                free = [device for device, cap in remaining.items() if cap >= 1.0]
                if len(free) < need:
                    return None
                chosen = free[:need]
                for device in chosen:
                    remaining[device] = 0.0
                assignments[idx] = ",".join(chosen)
            else:
                chosen = next((device for device, cap in remaining.items() if cap >= compute), None)
                if chosen is None:
                    return None
                remaining[chosen] -= compute
                assignments[idx] = chosen

        return assignments

    @staticmethod
    def _try_allocate_entry_to_remaining(
        entry: dict,
        remaining: dict[str, float],
    ) -> str | None:
        """Try to place one entry into the current visible-GPU capacity map."""
        compute = float(entry.get("compute", 1) or 1)
        if compute >= 1.0:
            need = max(1, math.ceil(compute))
            free = [device for device, cap in remaining.items() if cap >= 1.0]
            if len(free) < need:
                return None
            chosen = free[:need]
            for device in chosen:
                remaining[device] = 0.0
            return ",".join(chosen)

        chosen = next((device for device, cap in remaining.items() if cap >= compute), None)
        if chosen is None:
            return None
        remaining[chosen] -= compute
        return chosen

    def _partition_group_gpu_batches(
        self,
        entries: list[dict],
    ) -> list[tuple[list[dict], list[str | None]]] | None:
        """Partition a group into execution waves that fit visible GPUs.

        When a whole group cannot run at once, this keeps as much parallelism as
        possible instead of falling all the way back to fully sequential execution.
        """
        devices = self._get_visible_gpu_devices()
        if not devices:
            return [(list(entries), [None] * len(entries))]

        batches: list[tuple[list[dict], list[str | None]]] = []
        current_entries: list[dict] = []
        current_assignments: list[str | None] = []
        remaining = {device: 1.0 for device in devices}

        for entry in entries:
            assignment = self._try_allocate_entry_to_remaining(entry, remaining)
            if assignment is None:
                if current_entries:
                    batches.append((current_entries, current_assignments))
                    current_entries = []
                    current_assignments = []
                    remaining = {device: 1.0 for device in devices}
                    assignment = self._try_allocate_entry_to_remaining(entry, remaining)
                if assignment is None:
                    return None

            current_entries.append(entry)
            current_assignments.append(assignment)

        if current_entries:
            batches.append((current_entries, current_assignments))

        return batches

    def _run_single_cmd(self, cmd_entry: dict, seed: int, gpu_devices: str | None = None) -> tuple[str, dict, float]:
        """Run a single test_cmd entry and return (feedback_str, metrics_dict, elapsed_sec)."""
        label = cmd_entry.get("label", "test")
        cmd = cmd_entry["cmd"]

        # Run budget check before training (fail fast)
        budget_err = self._run_budget_check(cmd_entry, seed, gpu_devices=gpu_devices)
        if budget_err:
            self._current_test_had_failures = True
            return (
                f"### {label} ({cmd})\n{budget_err}",
                {},
                0.0,
            )

        if self.container_runtime == "local":
            local_cmd, cwd, run_env = self._build_local_exec_spec(cmd_entry, seed, gpu_devices=gpu_devices)
            time_str = cmd_entry.get("time", "1:00:00")
            timeout_secs = self._parse_time_to_seconds(time_str) + 300
            t_start = _time.time()
            result, raw_output, timed_out = self._run_local_command(
                local_cmd,
                cwd,
                run_env,
                timeout_secs,
            )
            if timed_out:
                self._current_test_had_failures = True
                raw_output = (
                    f"[TIMEOUT] Command timed out after {timeout_secs} seconds. "
                    f"This result is INVALID and will not count. "
                    f"Your algorithm is too slow — reduce model size or computational complexity."
                )
                elapsed = _time.time() - t_start
                parse_result = self.parser.parse(label, raw_output)
                feedback_str = f"### {label} ({cmd})\n{raw_output}"
                return feedback_str, parse_result.metrics or {}, elapsed

            elapsed = _time.time() - t_start
            if not raw_output:
                raw_output = f"[exit code {result.returncode}, no output]"
            status_line = ""
            if result.returncode != 0:
                self._current_test_had_failures = True
                status_line = f"[STATUS: FAILED exit={result.returncode}]\n"
                raw_output = f"[COMMAND FAILED exit={result.returncode}]\n{raw_output}"

            parse_result = self.parser.parse(label, raw_output)
            section_feedback = parse_result.feedback
            max_chars_per_cmd = 6000
            if len(section_feedback) > max_chars_per_cmd:
                half = max_chars_per_cmd // 2
                section_feedback = (
                    section_feedback[:half]
                    + "\n...[truncated]...\n"
                    + section_feedback[-half:]
                )
            feedback_str = f"### {label} ({cmd})\n{status_line}{section_feedback}"
            return feedback_str, parse_result.metrics or {}, elapsed

        container_result = self._build_container_cmd(cmd_entry, seed, gpu_devices=gpu_devices)

        # Docker returns (cmd_list, container_name); Apptainer returns cmd_list
        if isinstance(container_result, tuple):
            container_cmd, container_name = container_result
        else:
            container_cmd, container_name = container_result, None

        # Use config time field for timeout (+ 5min buffer), default 1hr
        time_str = cmd_entry.get("time", "1:00:00")
        timeout_secs = self._parse_time_to_seconds(time_str) + 300

        t_start = _time.time()
        run_env = os.environ.copy()
        if gpu_devices:
            run_env["CUDA_VISIBLE_DEVICES"] = gpu_devices
            run_env["NVIDIA_VISIBLE_DEVICES"] = gpu_devices
        try:
            if container_name:
                result, raw_output, timed_out = self._run_docker_container(
                    container_cmd,
                    container_name,
                    run_env,
                    timeout_secs=timeout_secs,
                )
                if timed_out:
                    self._current_test_had_failures = True
                    elapsed = _time.time() - t_start
                    parse_result = self.parser.parse(label, raw_output)
                    feedback_str = f"### {label} ({cmd})\n{raw_output}"
                    return feedback_str, parse_result.metrics or {}, elapsed
            else:
                result = subprocess.run(
                    container_cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root),
                    timeout=timeout_secs,
                    env=run_env,
                )
                raw_output = (result.stdout or "") + (result.stderr or "")
        except subprocess.TimeoutExpired:
            raw_output = (
                f"[TIMEOUT] Command timed out after {timeout_secs} seconds. "
                f"This result is INVALID and will not count. "
                f"Your algorithm is too slow — reduce model size or computational complexity."
            )
            self._current_test_had_failures = True
            elapsed = _time.time() - t_start
            parse_result = self.parser.parse(label, raw_output)
            feedback_str = f"### {label} ({cmd})\n{raw_output}"
            return feedback_str, parse_result.metrics or {}, elapsed

        elapsed = _time.time() - t_start

        if not raw_output:
            raw_output = f"[exit code {result.returncode}, no output]"
        status_line = ""
        if result.returncode != 0:
            self._current_test_had_failures = True
            status_line = f"[STATUS: FAILED exit={result.returncode}]\n"
            raw_output = f"[COMMAND FAILED exit={result.returncode}]\n{raw_output}"

        # Parse the output for feedback and metrics
        parse_result = self.parser.parse(label, raw_output)
        section_feedback = parse_result.feedback

        # Truncate per-cmd feedback if too long
        max_chars_per_cmd = 6000
        if len(section_feedback) > max_chars_per_cmd:
            half = max_chars_per_cmd // 2
            section_feedback = (
                section_feedback[:half]
                + "\n...[truncated]...\n"
                + section_feedback[-half:]
            )

        feedback_str = f"### {label} ({cmd})\n{status_line}{section_feedback}"
        return feedback_str, parse_result.metrics or {}, elapsed

    def _run_parallel_cmds(
        self,
        cmd_entries: list[dict],
        seed: int,
        gpu_assignments: list[str | None] | None = None,
    ) -> list[tuple[str, dict, float]]:
        """Run a list of cmd_entries in parallel and return list of (feedback_str, metrics_dict, elapsed_sec)."""
        results: list[tuple[int, str, dict, float]] = []
        with ThreadPoolExecutor(max_workers=len(cmd_entries)) as executor:
            future_to_idx = {
                executor.submit(
                    self._run_single_cmd,
                    entry,
                    seed,
                    gpu_assignments[i] if gpu_assignments else None,
                ): i
                for i, entry in enumerate(cmd_entries)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    feedback, metrics, elapsed = future.result()
                except Exception as exc:
                    entry = cmd_entries[idx]
                    label = entry.get("label", "test")
                    cmd = entry.get("cmd", "unknown")
                    print(f"[test] Parallel cmd failed ({label}): {exc}")
                    feedback = f"### {label} ({cmd})\n[ERROR] Command failed with exception: {exc}"
                    metrics = {}
                    elapsed = 0.0
                results.append((idx, feedback, metrics, elapsed))

        # Sort by original order
        results.sort(key=lambda x: x[0])
        return [(fb, met, el) for _, fb, met, el in results]

    def _group_entries(self) -> dict[int, list[dict]]:
        """Group test_cmd entries by their `group` field.

        Commands with the same `group` value are grouped together.
        Commands without a `group` each get a unique sequential group.
        """
        auto_group = 10000
        grouped: dict[int, list[dict]] = defaultdict(list)
        for entry in self.test_cmd_entries:
            g = entry.get("group")
            if g is None:
                grouped[auto_group].append(entry)
                auto_group += 1
            else:
                grouped[g].append(entry)
        return grouped

    def _run_all_cmds(self, seed: int) -> tuple[list[str], dict, list[bool]]:
        """Run all test_cmds grouped by `group`, return (feedback_parts, all_metrics, hidden_flags).

        Dispatches to SLURM or direct execution based on slurm_executor.
        Elapsed times are stored in all_metrics as ``elapsed_<label>`` keys.

        ``hidden_flags`` is a parallel list to ``feedback_parts`` indicating
        whether each feedback entry comes from a test_cmd with ``"hidden": true``.
        Hidden feedback is still executed and metrics are recorded, but the
        feedback string should be withheld from the agent during intermediate
        (non-final) test calls.
        """
        if self.slurm_executor:
            return self._run_all_cmds_slurm(seed)
        else:
            return self._run_all_cmds_direct(seed)

    def _run_all_cmds_direct(self, seed: int) -> tuple[list[str], dict, list[bool]]:
        """Run all test_cmds via direct apptainer execution (no SLURM).

        Commands with the same `group` value run in parallel.
        Different groups run sequentially in ascending order.

        Returns ``(feedback_parts, all_metrics, hidden_flags)`` where
        ``hidden_flags[i]`` is True when the corresponding test_cmd entry
        has ``"hidden": true``.
        """
        grouped = self._group_entries()

        feedback_parts: list[str] = []
        hidden_flags: list[bool] = []
        all_metrics: dict = {}

        for group_key in sorted(grouped.keys()):
            entries = grouped[group_key]
            if len(entries) == 1:
                fb, met, elapsed = self._run_single_cmd(
                    entries[0],
                    seed,
                    self._default_gpu_assignment(entries[0]),
                )
                feedback_parts.append(fb)
                hidden_flags.append(entries[0].get("hidden", False))
                all_metrics.update(met)
                label = entries[0].get("label", "test")
                all_metrics[f"elapsed_{label}"] = round(elapsed, 1)
            else:
                if self.container_runtime == "docker" and self._is_rootless_docker():
                    assignments = self._allocate_group_gpu_assignments(entries)
                    if assignments is None:
                        assignments = [self._default_gpu_assignment(entry) for entry in entries]
                    distinct_assignments = {
                        assignment for assignment in assignments
                        if assignment and "," not in assignment
                    }
                    if len(distinct_assignments) > 1:
                        results_by_idx: list[tuple[str, dict, float] | None] = [None] * len(entries)
                        with ThreadPoolExecutor(max_workers=len(entries)) as executor:
                            future_to_idx = {
                                executor.submit(
                                    self._run_docker_entries_in_session,
                                    [entry],
                                    seed,
                                    [assignments[idx]],
                                ): idx
                                for idx, entry in enumerate(entries)
                            }
                            for future in as_completed(future_to_idx):
                                idx = future_to_idx[future]
                                try:
                                    session_results = future.result()
                                    results_by_idx[idx] = session_results[0]
                                except Exception as exc:
                                    label = entries[idx].get("label", "test")
                                    cmd = entries[idx].get("cmd", "unknown")
                                    results_by_idx[idx] = (
                                        f"### {label} ({cmd})\n[ERROR] Command failed with exception: {exc}",
                                        {},
                                        0.0,
                                    )
                        results = [result if result is not None else ("", {}, 0.0) for result in results_by_idx]
                    else:
                        results = self._run_docker_entries_in_session(
                            entries,
                            seed,
                            gpu_assignments=assignments,
                        )
                    for (fb, met, elapsed), entry in zip(results, entries):
                        feedback_parts.append(fb)
                        hidden_flags.append(entry.get("hidden", False))
                        all_metrics.update(met)
                        label = entry.get("label", "test")
                        all_metrics[f"elapsed_{label}"] = round(elapsed, 1)
                    continue

                batches = self._partition_group_gpu_batches(entries)
                if batches is None:
                    print(
                        f"[test] Group {group_key} needs more GPUs than visible; "
                        "falling back to sequential execution.",
                    )
                    results = [
                        self._run_single_cmd(entry, seed, self._default_gpu_assignment(entry))
                        for entry in entries
                    ]
                else:
                    if len(batches) > 1:
                        print(
                            f"[test] Group {group_key} exceeds visible GPUs; "
                            f"running in {len(batches)} waves.",
                        )
                    results = []
                    for batch_entries, batch_assignments in batches:
                        if len(batch_entries) == 1:
                            results.append(
                                self._run_single_cmd(
                                    batch_entries[0],
                                    seed,
                                    batch_assignments[0] if batch_assignments else None,
                                )
                            )
                        else:
                            results.extend(
                                self._run_parallel_cmds(
                                    batch_entries,
                                    seed,
                                    gpu_assignments=batch_assignments,
                                )
                            )
                for (fb, met, elapsed), entry in zip(results, entries):
                    feedback_parts.append(fb)
                    hidden_flags.append(entry.get("hidden", False))
                    all_metrics.update(met)
                    label = entry.get("label", "test")
                    all_metrics[f"elapsed_{label}"] = round(elapsed, 1)

        return feedback_parts, all_metrics, hidden_flags

    @staticmethod
    def _split_into_sub_jobs(entries: list[dict]) -> list[list[dict]]:
        """Split group entries into sub-jobs based on compute requirements.

        - Entries with compute >= 1.0 each get their own SLURM job.
        - Entries with compute < 1.0 are bin-packed together to share GPUs.
        """
        sub_jobs: list[list[dict]] = []
        fractional: list[dict] = []

        for entry in entries:
            if entry.get("compute", 1) >= 1.0:
                sub_jobs.append([entry])
            else:
                fractional.append(entry)

        # Bin-pack fractional entries (fill up to ~1.0 compute per job)
        if fractional:
            current_bin: list[dict] = []
            current_compute = 0.0
            for entry in fractional:
                c = entry.get("compute", 1)
                if current_compute + c > 1.0 and current_bin:
                    sub_jobs.append(current_bin)
                    current_bin = [entry]
                    current_compute = c
                else:
                    current_bin.append(entry)
                    current_compute += c
            if current_bin:
                sub_jobs.append(current_bin)

        return sub_jobs

    def _find_recoverable_group_dir(self, group_key: int, suffix: str = "") -> Path | None:
        """Find the latest group dir with a recoverable SLURM job (has job_id.txt).

        First searches under the current ``exp_name``.  If nothing is found
        (e.g. the exp_name changed on resume), falls back to scanning ALL
        exp dirs for this task that share the same model prefix.

        Only returns dirs whose SLURM job is still active (PENDING/RUNNING)
        or already COMPLETED.  CANCELLED/FAILED jobs are skipped.
        """
        if not self.slurm_executor:
            return None
        target_name = f"group_{group_key}{suffix}"
        task_logs = self.slurm_executor.logs_dir / self.task_name

        def _is_recoverable(candidate: Path) -> bool:
            """Check that the SLURM job is still usable (not cancelled/failed)."""
            job_id = (candidate / "job_id.txt").read_text().strip()
            from mlsbench.agent.slurm import _run_slurm_query
            # Quick squeue check (fast for active jobs)
            sq = _run_slurm_query(["squeue", "-j", job_id, "--noheader", "-o", "%T"])
            if sq.returncode == 0 and sq.stdout.strip():
                state = sq.stdout.strip().split()[0].rstrip("+")
                if state in ("PENDING", "RUNNING"):
                    return True
            # sacct check (for completed jobs) — use -X to get job-level
            # state only; without -X, substeps like ".extern" may show
            # COMPLETED even when the overall job was CANCELLED.
            sa = _run_slurm_query(["sacct", "-j", job_id, "--format=State", "--noheader", "-P", "-X"])
            if sa.returncode == 0 and sa.stdout.strip():
                state = sa.stdout.strip().splitlines()[0].strip().split()[0].rstrip("+")
                if state == "COMPLETED":
                    return True
            return False

        # Phase 1: search under the current exp_name
        base = task_logs / self.exp_name
        if base.exists():
            for ts_dir in sorted(base.iterdir(), reverse=True):
                if not ts_dir.is_dir():
                    continue
                candidate = ts_dir / target_name
                if candidate.is_dir() and (candidate / "job_id.txt").exists():
                    if _is_recoverable(candidate):
                        return candidate

        # Phase 2 (fallback): search all exp dirs sharing the model prefix
        if task_logs.exists():
            model_prefix = self.exp_name.rsplit("_", 2)[0] if "_" in self.exp_name else self.exp_name
            for exp_dir in sorted(task_logs.iterdir(), reverse=True):
                if not exp_dir.is_dir() or exp_dir.name == self.exp_name:
                    continue
                if not exp_dir.name.startswith(model_prefix):
                    continue
                for ts_dir in sorted(exp_dir.iterdir(), reverse=True):
                    if not ts_dir.is_dir():
                        continue
                    candidate = ts_dir / target_name
                    if candidate.is_dir() and (candidate / "job_id.txt").exists():
                        if _is_recoverable(candidate):
                            print(f"[slurm-resume] Found job in older exp dir: {exp_dir.name}")
                            return candidate

        return None

    def _run_all_cmds_slurm(self, seed: int) -> tuple[list[str], dict, list[bool]]:
        """Run all test_cmds for a single seed via SLURM. Delegates to multi-seed method."""
        results = self._run_all_seeds_slurm([seed])
        return results[seed]

    def _run_all_seeds_slurm(self, seeds: list[int]) -> dict[int, tuple[list[str], dict, list[bool]]]:
        """Run all test_cmds for all seeds via SLURM with global bin-packing.

        Globally packs all (entry, seed) combinations across seeds to minimize
        the total number of SLURM jobs. Different groups still run sequentially.

        Returns per-seed ``(feedback_parts, metrics, hidden_flags)`` where
        ``hidden_flags[i]`` mirrors the ``"hidden"`` field of the originating
        test_cmd entry.
        """
        grouped = self._group_entries()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        seed_feedback: dict[int, list[str]] = {s: [] for s in seeds}
        seed_hidden: dict[int, list[bool]] = {s: [] for s in seeds}
        seed_metrics: dict[int, dict] = {s: {} for s in seeds}
        budget_script = self.project_root / "tasks" / self.task_name / "budget_check.py"
        budget_enabled = budget_script.exists()

        for group_key in sorted(grouped.keys()):
            entries = grouped[group_key]

            # Build ALL tasks across all seeds for this group
            # Iterate entry-first so same-dataset different-seeds are adjacent
            # for better bin-packing (similar runtime per job).
            all_tasks = []
            for entry in entries:
                # Resolve use_cuda for this entry's package
                pkg_name = entry.get("package", "")
                entry_use_cuda = False
                if pkg_name:
                    pkg_cfg = self._load_pkg_config(pkg_name)
                    entry_use_cuda = self._effective_use_cuda(pkg_cfg)
                for seed in seeds:
                    task_info = {
                        "entry": entry,
                        "seed": seed,
                        "orig_label": entry.get("label", "test"),
                        "label": f"{entry.get('label', 'test')}_s{seed}",
                        "compute": entry.get("compute", 1),
                        "time": entry.get("time", "1:00:00"),
                        "use_cuda": entry_use_cuda,
                    }
                    if self.container_runtime == "local":
                        local_cmd, local_cwd, local_env = self._build_local_exec_spec(entry, seed)
                        task_info["local_cmd"] = local_cmd
                        task_info["local_cwd"] = local_cwd
                        task_info["local_env"] = local_env
                        if budget_enabled:
                            budget_cmd, budget_cwd, budget_env = self._build_local_budget_exec_spec(entry, seed)
                            task_info["budget_local_cmd"] = budget_cmd
                            task_info["budget_local_cwd"] = budget_cwd
                            task_info["budget_local_env"] = budget_env
                    else:
                        container_result = self._build_container_cmd(entry, seed)
                        if isinstance(container_result, tuple):
                            # Docker returns (cmd_list, container_name)
                            task_info["apptainer_cmd"] = container_result[0]
                            task_info["docker_container_name"] = container_result[1]
                        else:
                            task_info["apptainer_cmd"] = container_result
                        if budget_enabled:
                            task_info["budget_apptainer_cmd"] = self._build_container_budget_cmd(entry, seed)
                    if "mem" in entry:
                        task_info["mem"] = entry["mem"]
                    all_tasks.append(task_info)

            # Global bin-packing across all seeds
            sub_jobs = self._split_into_sub_jobs(all_tasks)

            # Submit all sub-jobs (or recover existing ones)
            submitted: list[tuple] = []
            for sub_idx, sub_tasks in enumerate(sub_jobs):
                suffix = f"_{sub_idx}" if len(sub_jobs) > 1 else ""

                group_cmds = []
                for t in sub_tasks:
                    cmd_dict = {
                        "label": t["label"],
                        "compute": t["compute"],
                        "time": t["time"],
                        "use_cuda": t.get("use_cuda", True),
                        **({k: t[k] for k in ("mem",) if k in t}),
                    }
                    if "local_cmd" in t:
                        cmd_dict["local_cmd"] = t["local_cmd"]
                        cmd_dict["local_cwd"] = t["local_cwd"]
                        cmd_dict["local_env"] = t["local_env"]
                        if "budget_local_cmd" in t:
                            cmd_dict["budget_local_cmd"] = t["budget_local_cmd"]
                            cmd_dict["budget_local_cwd"] = t["budget_local_cwd"]
                            cmd_dict["budget_local_env"] = t["budget_local_env"]
                    else:
                        cmd_dict["apptainer_cmd"] = t["apptainer_cmd"]
                        if "budget_apptainer_cmd" in t:
                            cmd_dict["budget_apptainer_cmd"] = t["budget_apptainer_cmd"]
                    group_cmds.append(cmd_dict)

                # Recovery: reuse existing SLURM job instead of submitting
                recovered_dir = None
                if self._recover_pending_slurm:
                    recovered_dir = self._find_recoverable_group_dir(group_key, suffix)

                if recovered_dir:
                    out_dir = recovered_dir
                    job_id = (recovered_dir / "job_id.txt").read_text().strip()
                    print(f"[slurm-resume] Recovering job {job_id} from {out_dir}")
                else:
                    out_dir = (
                        self.slurm_executor.logs_dir
                        / self.task_name
                        / self.exp_name
                        / timestamp
                        / f"group_{group_key}{suffix}"
                    )
                    job_name = f"mls-{self.task_name}-g{group_key}{suffix}"
                    job_id = self.slurm_executor.submit_group(group_cmds, job_name, out_dir)

                submitted.append((job_id, sub_tasks, group_cmds, out_dir))

            # NOTE: Do NOT clear _recover_pending_slurm here — later
            # groups (e.g. group_2) may also have recoverable jobs from a
            # previous resume attempt.  The flag is cleared after ALL groups.

            # Wait for all sub-jobs and collect results
            for job_id, sub_tasks, group_cmds, out_dir in submitted:
                status = self.slurm_executor.wait_for_job(job_id)

                # Auto-resubmit if job was externally cancelled
                max_resubmit = 3 if getattr(
                    self.slurm_executor, "resubmit_cancelled_jobs", True
                ) else 0
                resubmit_count = 0
                while status == "CANCELLED" and resubmit_count < max_resubmit:
                    resubmit_count += 1
                    print(f"[slurm] Job {job_id} was CANCELLED externally — "
                          f"resubmitting (attempt {resubmit_count}/{max_resubmit})")
                    job_name = f"mls-{self.task_name}-g{group_key}"
                    job_id = self.slurm_executor.submit_group(group_cmds, job_name, out_dir)
                    status = self.slurm_executor.wait_for_job(job_id)

                labels = [cmd["label"] for cmd in group_cmds]
                outputs = self.slurm_executor.read_outputs(out_dir, labels)
                exit_codes = self.slurm_executor.read_exit_codes(out_dir)
                elapsed_map = self.slurm_executor.read_elapsed(out_dir)

                for task, cmd_info in zip(sub_tasks, group_cmds):
                    label = cmd_info["label"]
                    orig_label = task["orig_label"]
                    seed = task["seed"]
                    cmd = task["entry"]["cmd"]
                    raw_output = outputs.get(label, "")
                    label_had_exit_code = label in exit_codes
                    exit_code = exit_codes.get(label, -1)

                    budget_log = out_dir / f"{label}_budget_check.out"
                    if budget_log.exists():
                        try:
                            budget_output = budget_log.read_text()
                        except Exception:
                            budget_output = ""
                        budget_rc = (
                            exit_code
                            if raw_output.startswith("[BUDGET CHECK FAILED]")
                            else 0
                        )
                        self._log_budget_output(orig_label, seed, budget_output, budget_rc)

                    if not raw_output:
                        raw_output = f"[exit code {exit_code}, no output]"
                    status_line = ""
                    if not label_had_exit_code and status in {"TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "FAILED"}:
                        # SLURM killed the wrapper before it could write exit_codes.txt:
                        # this command was running when the SLURM job hit its wall/limit.
                        self._current_test_had_failures = True
                        status_line = (
                            f"[STATUS: SLURM job hit {status} — this command was killed mid-run "
                            f"and produced no aggregate metrics. The output below is partial. "
                            f"This result is INVALID and will not count.]\n"
                        )
                        raw_output = f"[COMMAND FAILED — slurm {status}]\n{raw_output}"
                    elif exit_code != 0:
                        self._current_test_had_failures = True
                        status_line = f"[STATUS: FAILED exit={exit_code}]\n"
                        raw_output = f"[COMMAND FAILED exit={exit_code}]\n{raw_output}"

                    parse_result = self.parser.parse(orig_label, raw_output)
                    section_feedback = parse_result.feedback

                    max_chars_per_cmd = 6000
                    if len(section_feedback) > max_chars_per_cmd:
                        half = max_chars_per_cmd // 2
                        section_feedback = (
                            section_feedback[:half]
                            + "\n...[truncated]...\n"
                            + section_feedback[-half:]
                        )

                    feedback_str = f"### {orig_label} ({cmd})\n{status_line}{section_feedback}"
                    seed_feedback[seed].append(feedback_str)
                    seed_hidden[seed].append(task["entry"].get("hidden", False))
                    seed_metrics[seed].update(parse_result.metrics or {})
                    # Per-cmd elapsed time
                    if label in elapsed_map:
                        seed_metrics[seed][f"elapsed_{orig_label}"] = elapsed_map[label]

        # Clear recovery flag after all groups have been processed
        if self._recover_pending_slurm:
            self._recover_pending_slurm = False

        return {s: (seed_feedback[s], seed_metrics[s], seed_hidden[s]) for s in seeds}

    @staticmethod
    def _aggregate_metrics(metrics_list: list[dict]) -> dict:
        """Aggregate metrics across seeds by computing numeric means."""
        if not metrics_list:
            return {}
        if len(metrics_list) == 1:
            return metrics_list[0]

        # Collect numeric values per key
        collected: dict[str, list[float]] = defaultdict(list)
        for m in metrics_list:
            for k, v in m.items():
                try:
                    collected[k].append(float(v))
                except (ValueError, TypeError):
                    pass

        aggregated: dict = {}
        for k, vals in collected.items():
            # Filter out NaN/Inf for robust aggregation
            finite_vals = [v for v in vals if math.isfinite(v)]
            if finite_vals:
                aggregated[k] = sum(finite_vals) / len(finite_vals)
            else:
                # All values are NaN/Inf — report NaN
                aggregated[k] = float("nan")

        return aggregated

    @staticmethod
    def _has_real_metrics(record: dict) -> bool:
        """Return True if a metrics dict contains at least one non-elapsed metric."""
        for key, value in record.items():
            if (
                key in {"timestamp", "model", "is_final", "seed"}
                or key.startswith("elapsed_")
                or key.endswith("_std")
            ):
                continue
            if value in ("", None):
                continue
            return True
        return False

    def _filter_valid_seed_metrics(
        self,
        seeds: list[int],
        metrics_list: list[dict],
    ) -> tuple[list[int], list[dict]]:
        """Keep only per-seed metric dicts that contain real task metrics."""
        valid_seeds: list[int] = []
        valid_metrics: list[dict] = []
        for seed, metrics in zip(seeds, metrics_list):
            if self._has_real_metrics(metrics):
                valid_seeds.append(seed)
                valid_metrics.append(metrics)
        return valid_seeds, valid_metrics

    def _decorated_model_name(self) -> str:
        """Append :ctx_<kind> and/or :web suffixes."""
        name = self.model_name
        if self.extra_context:
            name = f"{name}:ctx_{self.extra_context}"
        # Tag whenever the option was *available* — not whether the model
        # actually used it. Ablation analysis groups by the run condition we
        # set, and "web was offered, model didn't take it" is itself a result.
        if self.allow_web_search:
            name = f"{name}:web"
        return name

    def _write_leaderboard_records(
        self,
        *,
        is_final: bool,
        seeds: list[int],
        metrics_list: list[dict],
        always_record: bool = False,
    ) -> dict:
        """Write valid per-seed metrics to the leaderboard and return aggregated metrics.

        If always_record=True, writes a row even when no valid metrics exist
        (recording the attempt with empty metric columns).
        """
        if self.leaderboard is None:
            return {}

        model_name = self._decorated_model_name()

        # Single agent_params column = max across env labels (usually they
        # match — same model, different envs). The full per-label/per-seed
        # detail lives in <agent_log_dir>/param_counts.jsonl.
        params_col: dict = {}
        if self._last_agent_params:
            params_col["agent_params"] = max(self._last_agent_params.values())

        valid_seeds, valid_metrics = self._filter_valid_seed_metrics(seeds, metrics_list)
        if not valid_metrics:
            if always_record and seeds:
                # Record the attempt even without metrics (for vanilla tracking)
                for seed in seeds:
                    record = {"model": model_name, "is_final": is_final, "seed": str(seed)}
                    record.update(params_col)
                    self.leaderboard.add(record)
            return {}

        all_metrics = self._aggregate_metrics(valid_metrics)
        if not self._has_real_metrics(all_metrics):
            if always_record and seeds:
                for seed in seeds:
                    record = {"model": model_name, "is_final": is_final, "seed": str(seed)}
                    record.update(params_col)
                    self.leaderboard.add(record)
            return {}

        for seed, seed_metric in zip(valid_seeds, valid_metrics):
            record = {"model": model_name, "is_final": is_final, "seed": str(seed)}
            record.update(seed_metric)
            record.update(params_col)
            self.leaderboard.add(record)
        if len(valid_seeds) > 1:
            record = {"model": model_name, "is_final": is_final, "seed": "mean"}
            record.update(all_metrics)
            record.update(params_col)
            self.leaderboard.add(record)

        return all_metrics

    @staticmethod
    def _coerce_seed(seed_value) -> int | None:
        """Convert leaderboard seed values back to ints when possible."""
        if isinstance(seed_value, int):
            return seed_value
        if isinstance(seed_value, float) and math.isfinite(seed_value) and seed_value.is_integer():
            return int(seed_value)
        if isinstance(seed_value, str) and seed_value.isdigit():
            return int(seed_value)
        return None

    def _is_my_model_row(self, row_model) -> bool:
        """Match a leaderboard row to this run, allowing exactly the
        decoration suffixes we add (:ctx_<kind>, :web)."""
        if not isinstance(row_model, str):
            return False
        base = self.model_name
        if row_model == base:
            return True
        if not row_model.startswith(base + ":"):
            return False
        suffix = row_model[len(base) + 1:]
        for part in suffix.split(":"):
            if part == "web" or part.startswith("ctx_"):
                continue
            return False
        return True

    def _latest_valid_nonfinal_batch(self) -> tuple[list[int], list[dict]]:
        """Return the latest non-final leaderboard batch with real metrics for this model."""
        if self.leaderboard is None:
            return [], []

        non_final = [
            r for r in self.leaderboard.all_records()
            if self._is_my_model_row(r.get("model"))
            and str(r.get("is_final", "")) == "false"
            and r.get("seed") != "mean"
            and self._has_real_metrics(r)
        ]
        if not non_final:
            return [], []

        seen_timestamps: set[str] = set()
        timestamps: list[str] = []
        for record in reversed(non_final):
            ts = str(record.get("timestamp", ""))
            if ts and ts not in seen_timestamps:
                seen_timestamps.add(ts)
                timestamps.append(ts)

        for ts in timestamps:
            batch = [r for r in non_final if str(r.get("timestamp", "")) == ts]
            batch_seeds: list[int] = []
            batch_metrics: list[dict] = []
            for record in batch:
                seed = self._coerce_seed(record.get("seed"))
                if seed is None:
                    continue
                metrics = {
                    k: v for k, v in record.items()
                    if k not in ("timestamp", "model", "is_final", "seed")
                }
                if not self._has_real_metrics(metrics):
                    continue
                batch_seeds.append(seed)
                batch_metrics.append(metrics)
            if batch_metrics:
                return batch_seeds, batch_metrics

        return [], []

    def _resolve_submission_payload(self, entry: dict) -> tuple[list[int], list[dict], str]:
        """Resolve the metrics to use for a final submission."""
        seeds, metrics = self._filter_valid_seed_metrics(entry["seeds"], entry["seed_metrics"])
        if metrics:
            return seeds, metrics, "history"

        fallback_seeds, fallback_metrics = self._latest_valid_nonfinal_batch()
        if fallback_metrics:
            return fallback_seeds, fallback_metrics, "leaderboard"

        return [], [], "none"

    def _finalize_submission(self, entry: dict, *, test_num: int | None = None) -> tuple[str, dict]:
        """Write a final submission from a history entry or leaderboard fallback.

        Only submits results from the selected entry (same code version).
        Missing seeds are NOT filled from other tests (different code).
        ``record_zero_if_no_finals`` handles any seeds still missing after this.
        """
        seeds_to_run, all_seed_metrics, source = self._resolve_submission_payload(entry)
        if not all_seed_metrics:
            return "[submit] No valid metrics available to submit.", {}

        all_metrics = self._write_leaderboard_records(
            is_final=True,
            seeds=seeds_to_run,
            metrics_list=all_seed_metrics,
        )
        if not all_metrics:
            return "[submit] No valid metrics available to submit.", {}

        if source == "history":
            selected = f"test #{test_num}" if test_num is not None else "the selected test"
            prefix = f"[submit] Finalized {selected} as final."
        else:
            selected = f"test #{test_num}" if test_num is not None else "the selected test"
            prefix = (
                f"[submit] {selected} had no valid metrics; "
                "used the latest valid non-final leaderboard result instead."
            )
        visible_metrics = self._filter_hidden_label_metrics(all_metrics)
        if visible_metrics:
            return prefix + f"\n\n[Leaderboard] Results saved: {visible_metrics}", all_metrics
        return prefix, all_metrics

    def latest_test_history_entry(self) -> dict | None:
        """Return a copy of the latest test history entry for logging/resume."""
        if not self._test_history:
            return None
        entry = self._test_history[-1]
        return {
            "feedback": entry.get("feedback", ""),
            "visible_feedback": entry.get("visible_feedback", entry.get("feedback", "")),
            "seed_metrics": [dict(m) for m in entry.get("seed_metrics", [])],
            "seeds": list(entry.get("seeds", [])),
            "had_failures": bool(entry.get("had_failures", False)),
        }

    # ------------------------------------------------------------------
    # Tool: test
    # ------------------------------------------------------------------

    def test(self, **_kw) -> str:
        """Run all test_cmds and return combined feedback.

        Seed scheduling:
        - First test and last test (max_tests reached): all seeds.
        - Intermediate calls: single seed only.

        Hidden test_cmds:
        - If a test_cmd entry has ``"hidden": true``, its feedback is withheld
          from the agent during intermediate (non-final) test calls.
        - Metrics from hidden test_cmds are still collected and recorded to
          the leaderboard regardless.
        """
        if self.test_count >= self.max_tests:
            return (
                f"ERROR: Test budget exhausted ({self.test_count}/{self.max_tests}). "
                "You MUST call submit(n=N) to choose which test result to submit as final."
            )
        self.test_count += 1
        is_final = self.test_count >= self.max_tests
        self._current_test_had_failures = False

        # Always run all seeds for consistent multi-seed evaluation
        seeds_to_run = self.seeds

        all_feedback_parts: list[str] = []
        all_hidden_flags: list[bool] = []
        all_seed_metrics: list[dict] = []

        if len(seeds_to_run) == 1:
            feedback_parts, metrics, hidden_flags = self._run_all_cmds(seeds_to_run[0])
            all_feedback_parts.extend(feedback_parts)
            all_hidden_flags.extend(hidden_flags)
            all_seed_metrics.append(metrics)
        elif len(seeds_to_run) > 1 and self.slurm_executor:
            # SLURM multi-seed: global bin-packing across all seeds
            seed_results = self._run_all_seeds_slurm(seeds_to_run)
            for seed in seeds_to_run:
                feedback_parts, metrics, hidden_flags = seed_results[seed]
                all_feedback_parts.append(f"\n## Seed {seed}")
                all_hidden_flags.append(False)  # seed header is never hidden
                all_feedback_parts.extend(feedback_parts)
                all_hidden_flags.extend(hidden_flags)
                all_seed_metrics.append(metrics)
        else:
            # Sequential: Docker mode, each seed uses the same GPUs
            for seed in seeds_to_run:
                all_feedback_parts.append(f"\n## Seed {seed}")
                all_hidden_flags.append(False)  # seed header is never hidden
                feedback_parts, metrics, hidden_flags = self._run_all_cmds(seed)
                all_feedback_parts.extend(feedback_parts)
                all_hidden_flags.extend(hidden_flags)
                all_seed_metrics.append(metrics)

        # Build combined feedback: always include everything (for caching/leaderboard).
        combined_feedback = "\n\n".join(all_feedback_parts)

        # Build agent-visible feedback: hide entries from test_cmds with "hidden": true
        # during intermediate tests. With --hide-hidden, keep them hidden on
        # final tests as well. Also strip any metric key=value segments matching
        # ``hidden_metrics`` glob patterns.
        if is_final and not self.hide_hidden:
            visible_feedback = combined_feedback
        else:
            visible_parts = [
                self._filter_hidden_metric_kvs(part)
                for part, hidden in zip(all_feedback_parts, all_hidden_flags)
                if not hidden
            ]
            visible_feedback = "\n\n".join(visible_parts)

        # Store in history (1-indexed via len)
        self._test_history.append({
            "feedback": combined_feedback,
            "visible_feedback": visible_feedback,
            "seed_metrics": all_seed_metrics,
            "seeds": seeds_to_run,
            "had_failures": self._current_test_had_failures,
        })
        test_num = len(self._test_history)

        # Always write leaderboard records as non-final; final records are
        # written only when the agent explicitly calls submit().
        saved_metrics = self._write_leaderboard_records(
            is_final=False,
            seeds=seeds_to_run,
            metrics_list=all_seed_metrics,
            always_record=True,
        )

        if saved_metrics:
            visible_saved = self._filter_hidden_label_metrics(saved_metrics)
            if visible_saved:
                visible_feedback += f"\n\n[Leaderboard] Results saved: {visible_saved}"

        self._last_test_had_failures = self._current_test_had_failures

        remaining = self.max_tests - self.test_count
        header = f"[Test #{test_num}] ({remaining} test{'s' if remaining != 1 else ''} remaining"
        header += f"; call submit(n=N) to choose which test result to submit as final"
        header += ")\n\n"
        if is_final:
            header += (
                "[NOTE] This was your last test. You MUST now call submit(n=X) "
                "to choose which test result to submit as your final answer.\n\n"
            )
        return header + visible_feedback

    # ------------------------------------------------------------------
    # Tool: submit
    # ------------------------------------------------------------------

    def submit(self, n: int = -1, _force: bool = False, **_kw) -> str:
        """Submit a previous test result as final answer.

        n: test number (1-indexed). n=-1 means submit the latest test.
        _force: bypass min-test check (used internally for resume auto-complete).
        """
        if self.done:
            return "ERROR: Already submitted. Cannot submit again."

        if not self._test_history:
            return "ERROR: No test results yet. You must call test() first to run an experiment."

        if not _force and self.test_count < 2 and self.max_tests >= 2:
            return (
                "ERROR: You must iterate at least once (edit → test → review → edit → test) "
                f"before submitting. You have used {self.test_count}/{self.max_tests} tests. "
                "Please edit your solution and test again before submitting."
            )

        # n=-1 or n=0 → latest
        if n <= 0:
            n = len(self._test_history)

        if n > len(self._test_history):
            return f"ERROR: Invalid test number {n}. Valid range: 1–{len(self._test_history)}."

        entry = self._test_history[n - 1]
        feedback_key = "visible_feedback" if self.hide_hidden else "feedback"
        selected_feedback = entry.get(feedback_key) or entry.get("feedback", "")
        combined_feedback = f"[submit] Submitting result from test #{n} as final.\n\n" + selected_feedback
        submit_feedback, metrics = self._finalize_submission(entry, test_num=n)
        combined_feedback += f"\n\n{submit_feedback}"
        # Always mark done once submit passes validation — even if metrics
        # are empty (all tests crashed).  Without this, the model loops
        # calling submit forever since step_count isn't incremented.
        self.done = True
        return combined_feedback

    def record_zero_if_no_finals(self) -> None:
        """Record empty-metric final entries for seeds missing valid finals.

        Called at the end of a run.  Checks each seed individually: if a
        seed already has a final row with real metrics, it is skipped.
        Seeds without valid finals get an ``is_final=true`` row with empty
        metric columns, meaning "participated but produced no valid output".
        Empty metrics are naturally ranked worst regardless of direction.
        """
        if self.leaderboard is None:
            return
        records = self.leaderboard.all_records()
        # Find seeds that already have valid final entries
        final_seeds: set[int] = set()
        for r in records:
            if (self._is_my_model_row(r.get("model"))
                    and str(r.get("is_final", "")).lower() == "true"
                    and r.get("seed") != "mean"
                    and self._has_real_metrics(r)):
                s = self._coerce_seed(r.get("seed"))
                if s is not None:
                    final_seeds.add(s)
        missing = [s for s in self.seeds if s not in final_seeds]
        if not missing:
            return
        print(f"[agent] Missing final results for seeds {missing} — recording empty finals")
        decorated = self._decorated_model_name()
        for seed in missing:
            self.leaderboard.add({
                "model": decorated,
                "is_final": True,
                "seed": str(seed),
            })

    # ------------------------------------------------------------------
    # Tool: undo
    # ------------------------------------------------------------------

    def undo(self, n: int = 1) -> str:
        """Revert the last n file modification actions."""
        if not self._history:
            return "ERROR: Nothing to undo"

        restored = []
        for _ in range(min(n, len(self._history))):
            if not self._history:
                break
            snap = self._history.pop()
            if snap["content"] is None:
                snap["path"].unlink(missing_ok=True)
                restored.append(f"Deleted (created file): {snap['filename']}")
            else:
                snap["path"].write_text(snap["content"])
                restored.append(f"Restored: {snap['filename']}")
            self.live_protected_ranges = snap["ranges"]

        return "Undo complete:\n" + "\n".join(restored)

    # ------------------------------------------------------------------
    # Internal: apply pre_edit.json ops (no permission check, no history)
    # ------------------------------------------------------------------

    def apply_pre_edit(self, ops: list[dict], mutations_only_if_fresh: bool = False):
        """Apply pre_edit operations directly to workspace files (no permission checks).

        Args:
            ops: list of pre_edit operation dicts.
            mutations_only_if_fresh: if True, skip mutation ops (insert/replace/delete)
                because they were already applied in a previous run. 'create' ops
                always execute (overwrite) to reset editable files to template state.
        """
        for op in ops:
            filename = op["file"]
            path = self._resolve_workspace_path(filename)
            op_type = op["op"]

            if op_type == "create":
                # Always (re-)create: ensures editable files are reset to template
                path.parent.mkdir(parents=True, exist_ok=True)
                content = op["content"]
                if not content.endswith("\n"):
                    content += "\n"
                if path.exists():
                    print(f"[pre_edit] Overwriting (reset to template): {filename}")
                else:
                    print(f"[pre_edit] Creating: {filename}")
                path.write_text(content)
                continue

            # Mutation ops: skip if workspace was reused (already applied previously)
            if mutations_only_if_fresh:
                print(f"[pre_edit] Skipping {op_type} on {filename} (workspace reused)")
                continue

            if not path.exists():
                raise FileNotFoundError(f"Pre-edit target not found: {path}")

            lines = path.read_text().splitlines(keepends=True)

            if op_type == "replace":
                new_content = op["content"]
                if not new_content.endswith("\n"):
                    new_content += "\n"
                # start_line..end_line (1-indexed, inclusive)
                start = op["start_line"]
                end = op["end_line"]
                content_lines = new_content.splitlines(keepends=True)
                lines[start - 1 : end] = content_lines
                delta = len(content_lines) - (end - start + 1)
                self._shift_ranges_for_pre_edit(filename, end, delta)

            elif op_type == "insert":
                line_idx = op["after_line"]  # 1-indexed: insert after this line
                new_content = op["content"]
                if not new_content.endswith("\n"):
                    new_content += "\n"
                new_lines = new_content.splitlines(keepends=True)
                lines[line_idx:line_idx] = new_lines
                self._shift_ranges_for_pre_edit(filename, line_idx, len(new_lines))

            elif op_type == "delete":
                if "start_line" in op:
                    start = op["start_line"]
                    end = op.get("end_line", start)
                    num_deleted = end - start + 1
                    del lines[start - 1 : end]
                    self._shift_ranges_for_pre_edit(filename, start - 1, -num_deleted)
                else:
                    line_idx = op["line"] - 1
                    del lines[line_idx]
                    self._shift_ranges_for_pre_edit(filename, line_idx, -1)

            else:
                raise ValueError(f"Unknown pre_edit op: {op_type}")

            path.write_text("".join(lines))

    # ------------------------------------------------------------------
    # web_search (opt-in via --allow-web-search)
    # ------------------------------------------------------------------

    def web_search(self, query: str, max_results: int = 5,
                   include_answer: bool = False,
                   search_depth: str | None = None,
                   time_range: str | None = None,
                   include_domains: list | None = None,
                   exclude_domains: list | None = None,
                   include_raw_content: str | bool | None = None) -> str:
        if not self.allow_web_search:
            return "ERROR: web_search is not enabled for this run."
        if not isinstance(query, str) or not query.strip():
            return "ERROR: 'query' must be a non-empty string."
        try:
            n = max(1, min(int(max_results or 5), 10))
        except (TypeError, ValueError):
            return "ERROR: 'max_results' must be an integer."

        # Validate enum-style params before sending
        if search_depth is not None and search_depth not in ("basic", "advanced"):
            return "ERROR: 'search_depth' must be 'basic' or 'advanced'."
        if time_range is not None and time_range not in ("day", "week", "month", "year"):
            return "ERROR: 'time_range' must be one of day/week/month/year."
        if include_raw_content is not None and include_raw_content not in (
            True, False, "text", "markdown"):
            return "ERROR: 'include_raw_content' must be true, 'text', or 'markdown'."

        # Charge: basic = 1 credit, advanced = 2 credits.
        depth = search_depth or "basic"
        charge = 2 if depth == "advanced" else 1
        if self.max_web_credits:
            remaining = self.max_web_credits - self.web_credits_used
            if charge > remaining:
                return (f"ERROR: web budget exhausted — this call would cost "
                        f"{charge} credit(s) but only {remaining} remain "
                        f"({self.web_credits_used}/{self.max_web_credits} used). "
                        f"Try search_depth='basic' (1 credit).")

        api_key = self.tavily_api_key or os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return ("ERROR: Tavily API key not configured. Set "
                    "providers.tavily.api_key in the config or TAVILY_API_KEY env var.")
        try:
            from tavily import TavilyClient
        except ImportError:
            return "ERROR: web_search backend not installed. `pip install tavily-python`."

        kwargs: dict = {
            "query": query,
            "max_results": n,
            "include_answer": bool(include_answer),
            "search_depth": depth,
        }
        if time_range:
            kwargs["time_range"] = time_range
        if include_domains:
            kwargs["include_domains"] = list(include_domains)
        if exclude_domains:
            kwargs["exclude_domains"] = list(exclude_domains)
        if include_raw_content is not None:
            kwargs["include_raw_content"] = include_raw_content

        try:
            client = TavilyClient(api_key=api_key)
            resp = client.search(**kwargs)
        except Exception as e:
            return f"ERROR: web_search failed: {type(e).__name__}: {e}"

        # Successful API call — charge credits against the budget.
        self.web_credits_used += charge
        hits = resp.get("results") or []
        if not hits and not resp.get("answer"):
            return f"No results for {query!r}."
        lines = [f"Results for {query!r}:"]
        if resp.get("answer"):
            lines.append(f"Answer: {resp['answer'].strip()}")
            lines.append("")
        for i, h in enumerate(hits, 1):
            title = (h.get("title") or "").strip()
            url = h.get("url") or ""
            score = h.get("score")
            snippet = (h.get("content") or "").strip().replace("\n", " ")
            if len(snippet) > 500:
                snippet = snippet[:497] + "..."
            score_s = f" (score={score:.2f})" if isinstance(score, (int, float)) else ""
            lines.append(f"{i}. {title}{score_s}\n   {url}\n   {snippet}")
            raw = h.get("raw_content")
            if raw:
                raw_s = str(raw).strip()
                if len(raw_s) > 2000:
                    raw_s = raw_s[:1997] + "..."
                lines.append(f"   --- raw_content (truncated) ---\n{raw_s}")
        return "\n".join(lines)

    def web_extract(self, urls, query: str | None = None,
                    chunks_per_source: int = 3,
                    extract_depth: str | None = None,
                    format: str | None = None) -> str:
        if not self.allow_web_search:
            return "ERROR: web_extract is not enabled for this run."
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list) or not urls:
            return "ERROR: 'urls' must be a non-empty list of strings."
        if len(urls) > 5:
            return "ERROR: at most 5 URLs per web_extract call."
        urls = [str(u).strip() for u in urls if str(u).strip()]
        if not urls:
            return "ERROR: 'urls' contained no usable strings."
        try:
            cps = max(1, min(int(chunks_per_source or 3), 5))
        except (TypeError, ValueError):
            return "ERROR: 'chunks_per_source' must be an integer."
        if extract_depth is not None and extract_depth not in ("basic", "advanced"):
            return "ERROR: 'extract_depth' must be 'basic' or 'advanced'."
        if format is not None and format not in ("text", "markdown"):
            return "ERROR: 'format' must be 'text' or 'markdown'."

        # Charge: basic = 1 credit/URL, advanced = 2 credits/URL.
        depth = extract_depth or "basic"
        per_url = 2 if depth == "advanced" else 1
        charge = per_url * len(urls)
        if self.max_web_credits:
            remaining = self.max_web_credits - self.web_credits_used
            if charge > remaining:
                return (f"ERROR: web budget exhausted — this call would cost "
                        f"{charge} credit(s) ({per_url}/URL × {len(urls)} URLs) but "
                        f"only {remaining} remain "
                        f"({self.web_credits_used}/{self.max_web_credits} used). "
                        f"Try fewer URLs or extract_depth='basic'.")

        api_key = self.tavily_api_key or os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return ("ERROR: Tavily API key not configured. Set "
                    "providers.tavily.api_key in the config or TAVILY_API_KEY env var.")
        try:
            from tavily import TavilyClient
        except ImportError:
            return "ERROR: web_search backend not installed. `pip install tavily-python`."

        kwargs: dict = {
            "urls": urls,
            "extract_depth": depth,
            "format": format or "text",
        }
        if query and query.strip():
            kwargs["query"] = query.strip()
            kwargs["chunks_per_source"] = cps

        try:
            client = TavilyClient(api_key=api_key)
            resp = client.extract(**kwargs)
        except Exception as e:
            return f"ERROR: web_extract failed: {type(e).__name__}: {e}"

        self.web_credits_used += charge
        results = resp.get("results") or []
        failed = resp.get("failed_results") or []
        lines: list[str] = []
        if query:
            lines.append(f"Extracted (query={query!r}, chunks_per_source={cps}):")
        else:
            lines.append("Extracted (full page):")
        for i, r in enumerate(results, 1):
            url = r.get("url", "")
            title = (r.get("title") or "").strip()
            rc = r.get("raw_content") or ""
            rc_s = str(rc).strip()
            if len(rc_s) > 8000:
                rc_s = rc_s[:7997] + "..."
            header = f"{i}. {title} — {url}" if title else f"{i}. {url}"
            lines.append(header)
            if rc_s:
                lines.append(rc_s)
            else:
                lines.append("[no content extracted]")
        for r in failed:
            url = r.get("url", "?") if isinstance(r, dict) else str(r)
            err = r.get("error", "") if isinstance(r, dict) else ""
            lines.append(f"FAILED: {url}  {err}".rstrip())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, tool_name: str, tool_input: dict) -> str:
        """Dispatch a tool call by name. Increments step_count for non-submit calls."""
        # Handle parse errors forwarded from model client (e.g. malformed JSON)
        if "error" in tool_input and len(tool_input) == 1:
            result = f"ERROR: {tool_input['error']}"
            if tool_name != "submit":
                self.step_count += 1
            return result

        try:
            if tool_name == "edit":
                result = self.edit(**tool_input)
            elif tool_name == "test":
                result = self.test(**tool_input)
            elif tool_name == "submit":
                result = self.submit(**tool_input)
            elif tool_name == "undo":
                result = self.undo(**tool_input)
            elif tool_name == "web_search":
                result = self.web_search(**tool_input)
            elif tool_name == "web_extract":
                result = self.web_extract(**tool_input)
            else:
                result = f"ERROR: Unknown tool '{tool_name}'"
        except TypeError as e:
            result = f"ERROR: Invalid arguments for '{tool_name}': {e}"

        # submit doesn't count against the step budget
        if tool_name != "submit":
            self.step_count += 1
        return result


# ---------------------------------------------------------------------------
# Shared utility: load pre_edit ops for a task's packages
# ---------------------------------------------------------------------------

def load_pre_edit_ops(task_config: dict, pkg_configs_dir: Path) -> list[dict]:
    """Load pre_edit ops from pkg_configs for each package in the task.

    Convention: <dir>/pre_edit.py where dir name = package name.

    Shared by BaseAgent and the CLI baseline command.
    """
    import importlib.util

    def _normalize(s: str) -> str:
        return s.lower().replace("-", "").replace("_", "")

    seen: set[str] = set()
    packages: list[str] = []
    for entry in task_config.get("test_cmds", []):
        pkg = entry.get("package")
        if pkg:
            norm = _normalize(pkg)
            if norm not in seen:
                seen.add(norm)
                packages.append(pkg)

    def _load_module(path: Path) -> list[dict]:
        spec = importlib.util.spec_from_file_location("pre_edit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "OPS", [])

    ops: list[dict] = []
    for pkg in packages:
        norm = _normalize(pkg)
        for d in pkg_configs_dir.iterdir():
            if not d.is_dir():
                continue
            main_pe = d / "pre_edit.py"
            if main_pe.is_file() and _normalize(d.name) == norm:
                ops.extend(_load_module(main_pe))
                break
    return ops


def load_mid_edit_ops(task_name: str, tasks_dir: Path) -> list[dict]:
    """Load mid_edit ops from tasks/<task>/mid_edit.py if it exists.

    Mid-edit operations are task-specific workspace setup (template creation,
    task-specific TRAIN_METRICS injection) applied after pre_edit and before
    the agent starts.

    Shared by BaseAgent and the CLI baseline command.
    """
    import importlib.util

    mid_edit_file = tasks_dir / task_name / "edits" / "mid_edit.py"
    if not mid_edit_file.is_file():
        return []

    spec = importlib.util.spec_from_file_location("mid_edit", mid_edit_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "OPS", [])
