"""DiscoverAgent: MLS-Bench agent that delegates to ttt-discover's RL loop.

`ttt_discover.discover(config)` owns an asyncio on-policy RL loop that drives
Tinker-managed LoRA fine-tuning. Each rollout samples actions from the current
policy and receives reward from the MLS-Bench environment step override. We
bridge this to MLS-Bench by:

1. Building a dynamic ``Environment`` subclass at runtime that closes over
   ``self.tools`` and ``self._slot`` — the rollouts live in the same Python
   process as this agent, so direct calls into ``WorkspaceTools`` are correct.
2. Letting upstream ``do_single_rollout`` drive multiple env steps: each
   rollout may apply several code edits and terminates only when the model
   emits ``<test/>`` or the per-episode action cap forces a final test.
3. Monkey-patching ``tinker.SamplingClient.sample_async`` once, at ``run``
   time, to emit per-call token usage records to ``tokens.jsonl``.

ttt-discover's algorithmic core — rollouts, advantage estimation, two-phase
sampling, KL penalty, LoRA trainer — is not touched. W&B is gated off by
default via ``WANDB_MODE=disabled``.

Constraints (v1):
- Single editable file with a single contiguous edit range.
- Model must have a matching renderer registered in
  ``ttt_discover.tinker_utils.renderers`` (currently ``qwen3``,
  ``qwen3_instruct``, ``gpt_oss_*``). Anything Tinker can sample from with
  one of those renderers works; pick ``renderer_name`` accordingly.
- Workspace mutation and evaluation are serialized per pool slot; each slot
  owns one WorkspaceTools instance and workspace copy.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from mlsbench.agent.base import BaseAgent


class _AllHiddenScoreSpecError(ValueError):
    """Raised when hidden filtering leaves no reward settings."""


def _hidden_labels_from_test_cmds(config_task: dict) -> set[str]:
    return {
        str(tc["label"]).replace("-", "_")
        for tc in config_task.get("test_cmds", [])
        if tc.get("hidden") and "label" in tc
    }


def _filter_hidden_metrics(metrics: dict[str, Any], hidden_labels: set[str]) -> dict[str, Any]:
    if not hidden_labels:
        return metrics
    return {
        k: v
        for k, v in metrics.items()
        if k == "combined_score" or not any(
            h in str(k).replace("-", "_")
            or str(k).replace("-", "_").startswith(f"{h}_")
            or str(k).replace("-", "_").endswith(f"_{h}")
            for h in hidden_labels
        )
    }


def _safe_type_name(value: Any) -> str | None:
    if value is None:
        return None
    cls = value if isinstance(value, type) else type(value)
    module = getattr(cls, "__module__", "")
    qualname = getattr(cls, "__qualname__", getattr(cls, "__name__", cls.__name__))
    if module and module != "builtins":
        return f"{module}.{qualname}"
    return qualname


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _slot_hparam_dict(slot: "_EditSlot") -> dict[str, Any]:
    return {
        "filename": slot.filename,
        "start_line": slot.start_line,
        "end_line": slot.end_line,
        "is_full_file": slot.is_full_file,
    }


def _config_hparam_dict(config: Any) -> dict[str, Any]:
    keys = (
        "problem_type",
        "log_path",
        "eval_timeout",
        "num_cpus_per_task",
        "timeout",
        "convo_prefix",
    )
    return {
        key: _safe_scalar(getattr(config, key))
        for key in keys
        if hasattr(config, key)
    }


@dataclass
class _EditSlot:
    filename: str
    start_line: int
    end_line: int
    is_full_file: bool


@dataclass
class _PoolSlot:
    """One worker in the WorkspaceTools pool: own workspace, own edit slot, own lock."""
    tools: Any
    slot: _EditSlot
    task_name: str = ""
    pool_label: str = "train"
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __deepcopy__(self, memo):
        return {
            "_type": type(self).__name__,
            "task_name": self.task_name,
            "pool_label": self.pool_label,
            "slot": _slot_hparam_dict(self.slot),
            "tools_exp_name": _safe_scalar(getattr(self.tools, "exp_name", None)),
        }


@dataclass
class _TaskAssets:
    task_name: str
    config_task: dict
    config_edit: list[dict]
    pre_edit_ops: list[dict]
    mid_edit_ops: list[dict]
    leaderboard: Any
    slot_template: _EditSlot
    hidden_labels: set[str]


class _MLSBenchTaskDataset:
    """Small RLDataset implementation that can round-robin over task ids."""

    def __init__(
        self,
        *,
        task_ids: list[str],
        env_classes: dict[str, type],
        renderer: Any,
        samplers: dict[str, Any],
        configs: dict[str, Any],
        batch_size: int,
        group_size: int,
        pool_label: str,
    ):
        self.task_ids = list(task_ids)
        self.env_classes = env_classes
        self.renderer = renderer
        self.samplers = samplers
        self.configs = configs
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.pool_label = pool_label
        self.sampler = samplers[self.task_ids[0]] if self.task_ids else None

    def get_batch(self, index: int):
        from ttt_discover.rl.types import ProblemGroupBuilder

        builders = []
        for offset in range(self.batch_size):
            task_id = self.task_ids[(index * self.batch_size + offset) % len(self.task_ids)]
            sampler = self.samplers[task_id]
            state = sampler.sample_states(1)[0]
            env_cls = self.env_classes[task_id]
            cfg = self.configs[task_id]
            builders.append(
                ProblemGroupBuilder(
                    env_thunk=partial(
                        env_cls,
                        self.renderer,
                        initial_state=state,
                        sampler=sampler,
                        config=cfg,
                    ),
                    num_envs=self.group_size,
                    logging_name=f"{self.pool_label}/{task_id}",
                )
            )
        return builders

    def flush(self, step: int | None = None):
        for sampler in self.samplers.values():
            sampler.flush(step)

    def __len__(self) -> int:
        return 1

    def __deepcopy__(self, memo):
        return {
            "_type": type(self).__name__,
            "task_ids": list(self.task_ids),
            "pool_label": self.pool_label,
            "batch_size": self.batch_size,
            "group_size": self.group_size,
            "env_classes": {
                task_id: _safe_type_name(env_cls)
                for task_id, env_cls in self.env_classes.items()
            },
            "renderer": _safe_type_name(self.renderer),
            "samplers": {
                task_id: _safe_type_name(sampler)
                for task_id, sampler in self.samplers.items()
            },
            "configs": {
                task_id: _config_hparam_dict(config)
                for task_id, config in self.configs.items()
            },
        }


class _MLSBenchDatasetBuilder:
    """Build train/validation RL datasets for MLS-Bench task pools."""

    def __init__(
        self,
        *,
        train_task_id: str,
        train_env_cls: type,
        val_env_classes: dict[str, type],
        model_name: str,
        renderer_name: str,
        log_path: str,
        groups_per_batch: int,
        group_size: int,
        num_cpus_per_task: int,
        eval_timeout: int,
    ):
        self.train_task_id = train_task_id
        self.train_env_cls = train_env_cls
        self.val_env_classes = dict(val_env_classes)
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.log_path = log_path
        self.groups_per_batch = int(groups_per_batch)
        self.group_size = int(group_size)
        self.num_cpus_per_task = int(num_cpus_per_task)
        self.eval_timeout = int(eval_timeout)

    def __deepcopy__(self, memo):
        return {
            "_type": type(self).__name__,
            "train_task_id": self.train_task_id,
            "val_task_ids": list(self.val_env_classes),
            "train_env_cls": _safe_type_name(self.train_env_cls),
            "val_env_classes": {
                task_id: _safe_type_name(env_cls)
                for task_id, env_cls in self.val_env_classes.items()
            },
            "model_name": self.model_name,
            "renderer_name": self.renderer_name,
            "log_path": self.log_path,
            "groups_per_batch": self.groups_per_batch,
            "group_size": self.group_size,
            "num_cpus_per_task": self.num_cpus_per_task,
            "eval_timeout": self.eval_timeout,
        }

    async def __call__(self):
        from ttt_discover.tinker_utils import renderers
        from ttt_discover.tinker_utils.misc_utils import get_tokenizer
        from ttt_discover.tinker_utils.sampler import get_or_create_sampler_with_default

        tokenizer = get_tokenizer(self.model_name)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # We write this builder from scratch instead of subclassing
        # SingleProblemDatasetBuilder because the installed implementation is
        # tightly coupled to one env_type/problem_type pair and returns a single
        # dataset despite the abstract RLDatasetBuilder tuple contract.
        train_log_path = os.path.join(self.log_path, "train", self.train_task_id)
        os.makedirs(train_log_path, exist_ok=True)
        train_sampler = get_or_create_sampler_with_default(
            log_path=train_log_path,
            env_type=self.train_env_cls,
            batch_size=self.groups_per_batch,
            problem_type=self.train_task_id,
        )
        train_cfg = SimpleNamespace(
            timeout=8000.0,
            num_cpus_per_task=self.num_cpus_per_task,
            eval_timeout=self.eval_timeout,
            log_path=train_log_path,
            problem_type=self.train_task_id,
            convo_prefix=None,
        )
        train_ds = _MLSBenchTaskDataset(
            task_ids=[self.train_task_id],
            env_classes={self.train_task_id: self.train_env_cls},
            renderer=renderer,
            samplers={self.train_task_id: train_sampler},
            configs={self.train_task_id: train_cfg},
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            pool_label="train",
        )

        if not self.val_env_classes:
            return train_ds, None

        val_samplers = {}
        val_configs = {}
        for task_id, env_cls in self.val_env_classes.items():
            val_log_path = os.path.join(self.log_path, "val", task_id)
            os.makedirs(val_log_path, exist_ok=True)
            val_samplers[task_id] = get_or_create_sampler_with_default(
                log_path=val_log_path,
                env_type=env_cls,
                batch_size=1,
                problem_type=task_id,
            )
            val_configs[task_id] = SimpleNamespace(
                timeout=8000.0,
                num_cpus_per_task=self.num_cpus_per_task,
                eval_timeout=self.eval_timeout,
                log_path=val_log_path,
                problem_type=task_id,
                convo_prefix=None,
            )
        val_ds = _MLSBenchTaskDataset(
            task_ids=list(self.val_env_classes),
            env_classes=self.val_env_classes,
            renderer=renderer,
            samplers=val_samplers,
            configs=val_configs,
            batch_size=max(1, min(self.groups_per_batch, len(self.val_env_classes))),
            group_size=self.group_size,
            pool_label="val",
        )
        return train_ds, val_ds


class DiscoverAgent(BaseAgent):
    """Drive ttt-discover as an MLS-Bench agent.

    Overrides :meth:`run` because ttt-discover owns the training loop; the
    standard :meth:`get_action` tool-use interface is not used.
    """

    agent_label = "discover"

    def __init__(
        self,
        task_name: str,
        global_config: dict,
        workspace_root=None,
        val_tasks: list[str] | None = None,
    ):
        super().__init__(task_name, global_config, workspace_root)

        self._task_assets: dict[str, _TaskAssets] = {}
        primary_assets = self._get_task_assets(task_name)
        self._slot = primary_assets.slot_template

        disc_cfg = dict(global_config.get("discover") or {})
        self._iterations: int | None = (
            int(disc_cfg["iterations"]) if disc_cfg.get("iterations") is not None else None
        )
        self._disc_config_path: str | None = disc_cfg.get("config_path")
        self._disc_overrides: dict = disc_cfg.get("overrides") or {}
        self._extra_tasks: list[str] = list(disc_cfg.get("extra_tasks") or [])
        configured_val_tasks = (
            val_tasks
            if val_tasks is not None
            else disc_cfg.get("val_tasks")
            or disc_cfg.get("discover_val_tasks")
            or []
        )
        self._val_tasks: list[str] = []
        seen_val_tasks: set[str] = set()
        for task in configured_val_tasks:
            task_name_str = str(task).strip()
            if task_name_str and task_name_str not in seen_val_tasks:
                self._val_tasks.append(task_name_str)
                seen_val_tasks.add(task_name_str)
        self._max_actions_per_episode: int = max(
            1, int(disc_cfg.get("max_actions_per_episode", 8))
        )
        # Optional shaping only for malformed edit turns. Default 0 preserves
        # sparse terminal-reward behavior.
        self._edit_format_penalty: float = float(disc_cfg.get("edit_format_penalty", 0.0))
        if self._extra_tasks:
            raise NotImplementedError(
                "--discover-tasks (multi-task training) is not yet supported "
                "in v1. Run with a single task for now."
            )
        # Pool size: number of parallel WorkspaceTools the reward evaluator can
        # pick from. Each pool slot has its own workspace dir (unique exp_name)
        # so concurrent ttt-discover rollouts don't clobber each other's edits.
        self._pool_size: int = max(1, int(disc_cfg.get("pool_size", 4)))

        # Filled after YAML defaults and CLI overrides are merged. Discover's
        # true rollout count depends on num_epochs * groups_per_batch *
        # group_size, so constructor-time guesses can under-budget badly.
        self._base_max_tests = int(self.tools.max_tests)
        self._max_tests_per_pool = self._base_max_tests

        self._tokens_log: Path | None = None
        self._iter_counter = 0
        self._iter_lock = threading.Lock()
        self._best_code: str | None = None
        self._best_score: float = float("-inf")
        self._best_codes: dict[str, str] = {}
        self._best_scores: dict[str, float] = {self.task_name: float("-inf")}
        self._token_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "calls": 0,
        }
        self._train_token_totals = dict(self._token_totals)
        self._val_token_totals = dict(self._token_totals)
        self._sample_pool: ContextVar[str] = ContextVar("mlsbench_discover_sample_pool", default="train")

        # Pool is materialised lazily in run() (after setup_workspace on
        # primary self.tools completes). self.tools stays the "primary" slot,
        # used for final is_final=true submit.
        self._pool: queue.Queue[_PoolSlot] = queue.Queue()
        self._train_pools: dict[str, queue.Queue[_PoolSlot]] = {}
        self._val_pools: dict[str, queue.Queue[_PoolSlot]] = {}
        self._train_pool_slots: dict[str, list[_PoolSlot]] = {}
        self._val_pool_slots: dict[str, list[_PoolSlot]] = {}

        # Lazy: a present-but-failing score_spec fails closed at reward time.
        # The sorted-key heuristic is only valid for tasks with no score_spec.py.
        self._score_spec = None
        self._score_anchors = None
        self._score_spec_error: str | None = None
        self._task_score_specs: dict[str, Any] = {}
        self._task_score_anchors: dict[str, Any] = {}
        self._task_score_spec_errors: dict[str, str | None] = {}
        self._task_hidden_labels: dict[str, set[str]] = {
            task_name: primary_assets.hidden_labels
        }
        self._hidden_labels: set[str] = primary_assets.hidden_labels
        print("[discover-agent] hidden metrics excluded from candidate response: "
              f"{sorted(self._hidden_labels)}")

    # ------------------------------------------------------------------
    def get_action(self, messages: list) -> dict | None:  # pragma: no cover
        raise NotImplementedError(
            "DiscoverAgent drives its own loop via ttt_discover.discover; "
            "get_action is not called."
        )

    # ------------------------------------------------------------------
    # Editable-slot resolution (parallel to OpenEvolveAgent)
    # ------------------------------------------------------------------
    def _get_task_assets(self, task_name: str) -> _TaskAssets:
        cached = self._task_assets.get(task_name)
        if cached is not None:
            return cached

        if task_name == self.task_name:
            config_task = self.config_task
            config_edit = self.config_edit
            pre_edit_ops = self.pre_edit_ops
            mid_edit_ops = self.mid_edit_ops
            leaderboard = self.leaderboard
        else:
            task_dir = self.project_root / "tasks" / task_name
            config_path = task_dir / "config.json"
            with open(config_path) as f:
                config_task = json.load(f)
            config_edit = config_task.get("files", [])

            from mlsbench.agent.leaderboard import Leaderboard
            from mlsbench.agent.tools import load_mid_edit_ops, load_pre_edit_ops

            pre_edit_ops = load_pre_edit_ops(
                config_task, self.project_root / "vendor" / "pkg_configs"
            )
            mid_edit_ops = load_mid_edit_ops(task_name, self.project_root / "tasks")
            leaderboard = Leaderboard(task_dir / "leaderboard.csv")

        assets = _TaskAssets(
            task_name=task_name,
            config_task=config_task,
            config_edit=config_edit,
            pre_edit_ops=pre_edit_ops,
            mid_edit_ops=mid_edit_ops,
            leaderboard=leaderboard,
            slot_template=self._resolve_editable_slot(config_edit, task_name),
            hidden_labels=_hidden_labels_from_test_cmds(config_task),
        )
        self._task_assets[task_name] = assets
        if hasattr(self, "_task_hidden_labels"):
            self._task_hidden_labels[task_name] = assets.hidden_labels
        return assets

    def _resolve_editable_slot(
        self,
        config_edit: list[dict] | None = None,
        task_name: str | None = None,
    ) -> _EditSlot:
        config_edit = self.config_edit if config_edit is None else config_edit
        task_name = self.task_name if task_name is None else task_name
        editable = [f for f in config_edit if f.get("edit")]
        if len(editable) != 1:
            raise ValueError(
                f"DiscoverAgent requires exactly 1 editable file, "
                f"got {len(editable)} in task {task_name!r}"
            )
        entry = editable[0]
        ranges = entry["edit"]
        if len(ranges) != 1:
            raise ValueError(
                f"DiscoverAgent requires a single contiguous edit range, "
                f"got {len(ranges)} ranges on {entry['filename']!r}"
            )
        r = ranges[0]
        start, end = int(r["start"]), int(r["end"])
        is_full = (start == -1 and end == -1)
        return _EditSlot(
            filename=entry["filename"],
            start_line=start if not is_full else 1,
            end_line=end if not is_full else -1,
            is_full_file=is_full,
        )

    def _read_workspace_file(self) -> str:
        return self.tools._resolve_workspace_path(self._slot.filename).read_text()

    def _slot_content(self) -> str:
        text = self._read_workspace_file()
        if self._slot.is_full_file:
            return text
        lines = text.splitlines()
        return "\n".join(lines[self._slot.start_line - 1 : self._slot.end_line])

    def _refresh_slot_end(self) -> None:
        text = self._read_workspace_file()
        total = len(text.splitlines())
        if self._slot.is_full_file:
            self._slot.end_line = total
        else:
            if self._slot.end_line < 0 or self._slot.end_line > total:
                self._slot.end_line = total

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------
    def _build_slot_for_tools(
        self,
        tools,
        base_slot: _EditSlot | None = None,
    ) -> _EditSlot:
        """Compute an _EditSlot against `tools`'s workspace copy of the file."""
        base = base_slot or self._slot
        if base.is_full_file:
            text = tools._resolve_workspace_path(base.filename).read_text()
            return _EditSlot(
                filename=base.filename,
                start_line=1,
                end_line=len(text.splitlines()),
                is_full_file=True,
            )
        path = tools._resolve_workspace_path(base.filename)
        total = len(path.read_text().splitlines())
        return _EditSlot(
            filename=base.filename,
            start_line=base.start_line,
            end_line=min(base.end_line, total) if base.end_line > 0 else total,
            is_full_file=False,
        )

    def _pool_exp_name(self, task_name: str, pool_label: str, slot_index: int) -> str:
        base_model_name = self.global_config.get("model", "openai/gpt-oss-20b")
        sanitized = base_model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if pool_label == "train":
            return f"{sanitized}_{ts}_pool{slot_index}"
        safe_task = task_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        return f"discover_{sanitized}_val_{safe_task}_{ts}"

    def _build_pool_slot(
        self,
        task_name: str,
        pool_label: str,
        *,
        tools=None,
        slot_index: int = 0,
    ) -> _PoolSlot:
        from mlsbench.agent.tools import WorkspaceTools

        assets = self._get_task_assets(task_name)
        if tools is None:
            use_cuda_override = self.global_config.get("use_cuda")
            if use_cuda_override is not None:
                use_cuda_override = bool(use_cuda_override)
            tools = WorkspaceTools(
                task_name=task_name,
                config_task=assets.config_task,
                config_edit=assets.config_edit,
                workspace_root=self.workspace_root,
                project_root=self.project_root,
                max_tests=self._max_tests_per_pool,
                model_name=self.tools.model_name,
                leaderboard=assets.leaderboard,
                save_path=self.global_config.get("save_path", ""),
                seeds=assets.config_task.get("seeds") or self.global_config.get("seeds"),
                slurm_config=self.global_config.get("slurm"),
                exp_name=self._pool_exp_name(task_name, pool_label, slot_index),
                container_runtime=self.global_config.get("container_runtime", "apptainer"),
                use_cuda=use_cuda_override,
                platform=self.global_config.get("platform", ""),
                gpu_devices=self.global_config.get("gpu_devices", ""),
                global_config=self.global_config,
            )
            self._setup_tools_workspace(tools, assets)

        return _PoolSlot(
            tools=tools,
            slot=self._build_slot_for_tools(tools, assets.slot_template),
            task_name=task_name,
            pool_label=pool_label,
        )

    def _build_pool(self) -> None:
        """Create pool_size additional WorkspaceTools, each with own exp_name.

        The "primary" WorkspaceTools (``self.tools``) is also wrapped as a pool
        slot so the pool lifecycle is uniform. Primary stays reserved for the
        final ``is_final=true`` submit at the end of the run.
        """
        # Primary slot
        train_pool: queue.Queue[_PoolSlot] = queue.Queue()
        primary = self._build_pool_slot(
            self.task_name, "train", tools=self.tools, slot_index=0
        )
        train_pool.put(primary)
        train_slots: list[_PoolSlot] = [primary]

        # Peer slots share the exact primary model label, so leaderboard rows
        # are attributed consistently across the pool.
        for i in range(self._pool_size - 1):
            peer_slot = self._build_pool_slot(
                self.task_name, "train", slot_index=i + 1
            )
            train_pool.put(peer_slot)
            train_slots.append(peer_slot)

        self._pool = train_pool
        self._train_pools[self.task_name] = train_pool
        self._train_pool_slots[self.task_name] = train_slots
        self._pool_slots: list[_PoolSlot] = train_slots

        merged = getattr(self, "_merged_disc_cfg", {}) or {}
        val_slots_per_task = max(1, int(merged.get("group_size", 1)))
        for val_task in self._val_tasks:
            val_pool: queue.Queue[_PoolSlot] = queue.Queue()
            val_slots: list[_PoolSlot] = []
            for slot_index in range(val_slots_per_task):
                val_slot = self._build_pool_slot(
                    val_task, "val", slot_index=slot_index + 1
                )
                val_pool.put(val_slot)
                val_slots.append(val_slot)
            self._val_pools[val_task] = val_pool
            self._val_pool_slots[val_task] = val_slots

    def _setup_peer_workspace(self, tools) -> None:
        self._setup_tools_workspace(tools, self._get_task_assets(self.task_name))

    def _setup_tools_workspace(self, tools, assets: _TaskAssets) -> None:
        """Replicate BaseAgent.setup_workspace for a peer tools instance.

        The primary self.tools has already been set up by self.setup_workspace()
        in run(); peers need the same pipeline against their own exp_name dir.
        """
        import shutil

        workspace_task_dir = tools.workspace_task_dir
        workspace_task_dir.mkdir(parents=True, exist_ok=True)
        ext_dir = self.project_root / "vendor" / "external_packages"

        all_packages: list[str] = []
        seen_norm: set[str] = set()
        for entry in tools.test_cmd_entries:
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
                continue
            src = None
            norm = pkg.lower().replace("-", "").replace("_", "")
            for d in ext_dir.iterdir():
                if d.is_dir() and d.name.lower().replace("-", "").replace("_", "") == norm:
                    src = d
                    break
            if src is None:
                raise FileNotFoundError(f"External package '{pkg}' not found in {ext_dir}")
            shutil.copytree(src, dst, symlinks=True)
            any_copied = True

        if assets.pre_edit_ops:
            tools.apply_pre_edit(assets.pre_edit_ops, mutations_only_if_fresh=not any_copied)
        if assets.mid_edit_ops:
            tools.apply_pre_edit(assets.mid_edit_ops, mutations_only_if_fresh=not any_copied)

    def _compute_max_tests_per_pool(self, merged: dict) -> int:
        num_epochs = int(merged.get("num_epochs", 2))
        groups_per_batch = int(merged.get("groups_per_batch", 1))
        group_size = int(merged.get("group_size", 2))
        final_eval_pad = 1
        needed = num_epochs * groups_per_batch * group_size + final_eval_pad
        return max(self._base_max_tests, needed)

    def _apply_pool_test_budget(self, max_tests: int) -> None:
        self._max_tests_per_pool = int(max_tests)
        self.tools.max_tests = self._max_tests_per_pool
        for pool_slot in getattr(self, "_pool_slots", []):
            pool_slot.tools.max_tests = self._max_tests_per_pool
        for slots_by_task in (
            getattr(self, "_train_pool_slots", {}),
            getattr(self, "_val_pool_slots", {}),
        ):
            for slots in slots_by_task.values():
                for pool_slot in slots:
                    pool_slot.tools.max_tests = self._max_tests_per_pool

    def _pool_queue(self, pool_label: str, task_id: str | None) -> queue.Queue[_PoolSlot]:
        if pool_label == "val":
            task_key = task_id or (self._val_tasks[0] if self._val_tasks else "")
            if task_key in self._val_pools:
                return self._val_pools[task_key]
            raise KeyError(f"validation pool for task {task_key!r} is not available")
        task_key = task_id or getattr(self, "task_name", "")
        if task_key and task_key in getattr(self, "_train_pools", {}):
            return self._train_pools[task_key]
        return self._pool

    def _acquire(
        self,
        timeout: float = 3600.0,
        *,
        pool_label: str = "train",
        task_id: str | None = None,
    ) -> _PoolSlot:
        return self._pool_queue(pool_label, task_id).get(timeout=timeout)

    def _release(
        self,
        slot: _PoolSlot,
        *,
        pool_label: str | None = None,
        task_id: str | None = None,
    ) -> None:
        label = pool_label or slot.pool_label or "train"
        task_key = task_id or slot.task_name or getattr(self, "task_name", "")
        self._pool_queue(label, task_key).put(slot)

    # ------------------------------------------------------------------
    # Primary-metric helper: uses score_spec.py when available
    # ------------------------------------------------------------------
    def _load_score_spec_safely(self, task_name: str | None = None) -> None:
        """Load the task's score_spec.py + baseline anchors.

        Strips settings backed by hidden test_cmds so the RL reward isn't
        peeking at held-out evaluations. Final leaderboard scoring (done by
        external tooling) still uses the full spec including hidden settings.
        """
        task_name = task_name or self.task_name
        assets = self._get_task_assets(task_name)
        task_dir = self.project_root / "tasks" / task_name
        spec_path = task_dir / "score_spec.py"
        if not spec_path.exists():
            self._task_score_specs[task_name] = None
            self._task_score_anchors[task_name] = None
            self._task_score_spec_errors[task_name] = None
            return

        try:
            import copy as _copy
            from mlsbench.scoring.anchors import BaselineAnchors
            from mlsbench.scoring.evaluate import load_expanded_spec

            anchors = BaselineAnchors(task_dir)
            spec = load_expanded_spec(task_dir, anchors)
            if spec is None:
                score_spec_error = (
                    f"{spec_path} exists but expanded to no score settings"
                )
                self._task_score_spec_errors[task_name] = score_spec_error
                if task_name == self.task_name:
                    self._score_spec_error = score_spec_error
                print(f"[discover-agent] ERROR: {score_spec_error}; "
                      "candidates will receive fail-closed reward")
                return

            hidden_labels = {
                tc["label"] for tc in assets.config_task.get("test_cmds", [])
                if tc.get("hidden") and "label" in tc
            }
            if hidden_labels:
                visible = _copy.deepcopy(spec)
                visible.settings = {
                    name: s for name, s in visible.settings.items()
                    if name not in hidden_labels
                }
                if not visible.settings:
                    raise _AllHiddenScoreSpecError(
                        "score_spec.py has no visible settings after excluding "
                        f"hidden labels {sorted(hidden_labels)}. Widen score_spec "
                        "to include a visible setting, or remove the hidden flag "
                        "from at least one scored test_cmd."
                    )
                score_spec = visible
                print(f"[discover-agent] using score_spec.py (excluding hidden labels: "
                      f"{sorted(hidden_labels)}) for {task_name}")
            else:
                score_spec = spec
                print(f"[discover-agent] using score_spec.py from {task_dir}")
            self._task_score_specs[task_name] = score_spec
            self._task_score_anchors[task_name] = anchors
            self._task_score_spec_errors[task_name] = None
            if task_name == self.task_name:
                self._score_spec = score_spec
                self._score_anchors = anchors
                self._score_spec_error = None
        except _AllHiddenScoreSpecError:
            raise
        except Exception as exc:
            score_spec_error = f"score_spec load failed: {exc!r}"
            self._task_score_spec_errors[task_name] = score_spec_error
            if task_name == self.task_name:
                self._score_spec_error = score_spec_error
            print(f"[discover-agent] ERROR: {score_spec_error}; "
                  "candidates will receive fail-closed reward")

    def _primary_metric_from_entry(
        self,
        entry: dict,
        task_name: str | None = None,
    ) -> tuple[float, dict]:
        task_name = task_name or getattr(self, "task_name", "")
        seed_metrics: list[dict] = entry.get("seed_metrics") or []
        if not seed_metrics:
            return (0.0, {})

        # Compute an averaged record up front; score_spec failures return this
        # alongside a fail-closed scalar instead of rewarding arbitrary metrics.
        agg: dict[str, float] = {}
        counts: dict[str, int] = {}
        for m in seed_metrics:
            for k, v in m.items():
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        avg = {k: agg[k] / counts[k] for k in agg}
        if not avg:
            return (0.0, {})

        score_spec_error = getattr(self, "_task_score_spec_errors", {}).get(task_name)
        if score_spec_error is None and task_name == getattr(self, "task_name", task_name):
            score_spec_error = self._score_spec_error
        if score_spec_error:
            print(f"[discover-agent] ERROR: {score_spec_error}; "
                  "returning fail-closed reward")
            return (-1e9, avg)

        # score_spec.py path — build a seed=mean record and call score_record
        score_spec = getattr(self, "_task_score_specs", {}).get(task_name)
        score_anchors = getattr(self, "_task_score_anchors", {}).get(task_name)
        if task_name == getattr(self, "task_name", task_name) and score_spec is None:
            score_spec = self._score_spec
            score_anchors = self._score_anchors
        if score_spec is not None and score_anchors is not None:
            try:
                from mlsbench.scoring.evaluate import score_record

                # score_record expects str-keyed record that may include
                # is_final/seed/model; those don't affect metric resolution.
                record = {"seed": "mean", **{k: v for k, v in avg.items()}}
                spec_score = float(score_record(score_spec, record, score_anchors))
                return (spec_score, avg)
            except Exception as exc:
                print(f"[discover-agent] ERROR: score_record failed: {exc!r}; "
                      "returning fail-closed reward")
                return (-1e9, avg)

        # Fallback: alphabetical-first metric across seed-averaged values
        primary_key = sorted(avg.keys())[0]
        return (avg[primary_key], avg)

    # ------------------------------------------------------------------
    # Token observer — writes per-sample usage to tokens.jsonl
    # ------------------------------------------------------------------
    def _totals_from_jsonl(self, pool: str | None = None) -> dict:
        """Replay tokens.jsonl into a totals dict. Resilient to missing file."""
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "calls": 0,
        }
        if self._tokens_log is None or not self._tokens_log.exists():
            return totals
        try:
            with self._tokens_log.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if pool is not None and rec.get("pool", "train") != pool:
                        continue
                    for k in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "cached_tokens",
                        "cache_creation_tokens",
                    ):
                        v = rec.get(k) or 0
                        if isinstance(v, (int, float)):
                            totals[k] += int(v)
                    totals["calls"] += 1
        except Exception:
            pass
        return totals

    def _emit_tokens(self, record: dict) -> None:
        if self._tokens_log is None:
            return
        try:
            with self._tokens_log.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            return
        for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                  "cached_tokens", "cache_creation_tokens"):
            v = record.get(k) or 0
            if isinstance(v, (int, float)):
                self._token_totals[k] += int(v)
                pool = record.get("pool", "train")
                if pool == "val":
                    self._val_token_totals[k] += int(v)
                else:
                    self._train_token_totals[k] += int(v)
        self._token_totals["calls"] += 1
        if record.get("pool", "train") == "val":
            self._val_token_totals["calls"] += 1
        else:
            self._train_token_totals["calls"] += 1

    def _append_action_protocol(self, prompt: str, slot: _EditSlot | None = None) -> str:
        slot = slot or self._slot
        return (
            prompt.rstrip()
            + f"""

## Action protocol (multi-turn)

Each turn you may either:

1. Propose a code edit by emitting a single ```python ... ``` block. The block replaces lines {slot.start_line}-{slot.end_line} of {slot.filename} in your workspace. You will see whether the edit applied and any syntax errors.
2. Run evaluation by emitting `<test/>` on a line by itself. This runs the benchmark on your current code and ends this rollout. You will be scored by the test result.

You have at most {self._max_actions_per_episode} turns. If you reach the limit without testing, the system will run a final test for you.
"""
        )

    def _build_initial_prompt_for_task(
        self,
        task_name: str,
        tools,
        assets: _TaskAssets,
    ) -> str:
        if task_name == self.task_name and tools is self.tools:
            return self.build_initial_prompt()

        saved = (
            self.task_name,
            self.config_task,
            self.config_edit,
            self.tools,
            self._extra_context_request,
            self._extra_context_text,
        )
        try:
            self.task_name = task_name
            self.config_task = assets.config_task
            self.config_edit = assets.config_edit
            self.tools = tools
            self._extra_context_request = None
            self._extra_context_text = ""
            return BaseAgent.build_initial_prompt(self)
        finally:
            (
                self.task_name,
                self.config_task,
                self.config_edit,
                self.tools,
                self._extra_context_request,
                self._extra_context_text,
            ) = saved

    def _build_dataset_builder(
        self,
        *,
        train_env_cls: type | None = None,
        val_env_classes: dict[str, type] | None = None,
        model_name: str | None = None,
        renderer_name: str | None = None,
        log_path: str | Path | None = None,
        groups_per_batch: int | None = None,
        group_size: int | None = None,
        num_cpus_per_task: int | None = None,
        eval_timeout: int | None = None,
    ):
        merged = getattr(self, "_merged_disc_cfg", None)
        if merged is None:
            if hasattr(self, "_disc_config_path"):
                merged = {**self._load_yaml_defaults(), **getattr(self, "_disc_overrides", {})}
            else:
                merged = dict(getattr(self, "_disc_overrides", {}) or {})
        if train_env_cls is None:
            try:
                from ttt_discover import Environment, State
            except ImportError as exc:
                raise RuntimeError(
                    "DiscoverAgent requires the `discover` extra: "
                    "pip install -e '.[discover]'"
                ) from exc
            train_prompt = self._append_action_protocol(self.build_initial_prompt())
            train_env_cls = self._build_environment_cls(
                Environment, State, train_prompt, task_id=self.task_name, pool_label="train"
            )
        val_env_classes = dict(val_env_classes or {})
        model_name = model_name or self.global_config.get("model", "openai/gpt-oss-20b")
        renderer_name = renderer_name or merged.get("renderer_name", "gpt_oss_high_reasoning")
        if log_path is None:
            exp = merged.get("experiment_name", f"mlsbench-{self.task_name}-{self.tools.exp_name}")
            log_path = Path("./tinker_log") / exp
        return _MLSBenchDatasetBuilder(
            train_task_id=self.task_name,
            train_env_cls=train_env_cls,
            val_env_classes=val_env_classes,
            model_name=model_name,
            renderer_name=renderer_name,
            log_path=str(log_path),
            groups_per_batch=(
                groups_per_batch
                if groups_per_batch is not None
                else int(merged.get("groups_per_batch", 1))
            ),
            group_size=(
                group_size
                if group_size is not None
                else int(merged.get("group_size", 2))
            ),
            num_cpus_per_task=(
                num_cpus_per_task
                if num_cpus_per_task is not None
                else int(merged.get("num_cpus_per_task", 0))
            ),
            eval_timeout=(
                eval_timeout
                if eval_timeout is not None
                else int(merged.get("eval_timeout", 1000))
            ),
        )

    def _record_best_code(self, task_name: str, score: float, code: str) -> None:
        if not hasattr(self, "_best_scores"):
            self._best_scores = {getattr(self, "task_name", task_name): getattr(self, "_best_score", float("-inf"))}
        if not hasattr(self, "_best_codes"):
            self._best_codes = {}
        current = self._best_scores.get(task_name, float("-inf"))
        if score <= current:
            return
        self._best_scores[task_name] = float(score)
        if code.strip():
            self._best_codes[task_name] = code
        if task_name == getattr(self, "task_name", task_name):
            self._best_score = float(score)
            if code.strip():
                self._best_code = code

    def _copy_train_config(self, cfg, dataset_builder):
        return type(cfg)(
            env_type=cfg.env_type,
            problem_type=cfg.problem_type,
            learning_rate=cfg.learning_rate,
            dataset_builder=dataset_builder,
            model_name=cfg.model_name,
            num_epochs=cfg.num_epochs,
            temperature=cfg.temperature,
            lora_rank=cfg.lora_rank,
            adv_estimator=cfg.adv_estimator,
            adv_estimator_beta=cfg.adv_estimator_beta,
            kl_penalty_coef=cfg.kl_penalty_coef,
            loss_fn=cfg.loss_fn,
            num_substeps=cfg.num_substeps,
            wandb_project=cfg.wandb_project,
            wandb_name=cfg.wandb_name,
            log_path=cfg.log_path,
            enable_trace=cfg.enable_trace,
            remove_constant_reward_groups=cfg.remove_constant_reward_groups,
            save_every=cfg.save_every,
            load_checkpoint_path=cfg.load_checkpoint_path,
            phase1_max_tokens=cfg.phase1_max_tokens,
            local_model_path=cfg.local_model_path,
        )

    async def _run_validation_rollouts(
        self,
        cfg,
        train_mod,
        sampling_client,
        val_dataset,
        i_batch: int,
    ) -> dict:
        import asyncio

        from ttt_discover.rl.metric_util import compute_trajectory_metrics

        env_group_builders = val_dataset.get_batch(i_batch % max(1, len(val_dataset)))
        token = self._sample_pool.set("val")
        try:
            trajectory_groups = await asyncio.gather(
                *[
                    asyncio.create_task(
                        train_mod.do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            temperature=cfg.temperature,
                            do_remove_constant_reward_groups=False,
                            step_idx=i_batch,
                            model_name=cfg.local_model_path or cfg.model_name,
                            phase1_max_tokens=cfg.phase1_max_tokens,
                        ),
                        name=f"val_task_{i}",
                    )
                    for i, builder in enumerate(env_group_builders)
                ],
            )
        finally:
            self._sample_pool.reset(token)

        if hasattr(val_dataset, "flush"):
            val_dataset.flush(step=i_batch + 1)

        trajectory_groups = [tg for tg in trajectory_groups if tg is not None]
        if not trajectory_groups:
            return {}
        taglist = [builder.logging_tags() for builder in env_group_builders[: len(trajectory_groups)]]
        metrics = compute_trajectory_metrics(trajectory_groups, taglist)
        return {
            f"val/{k}": v
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }

    def _install_discover_val_adapter(self, discovery_mod, train_mod, dataset_builder) -> None:
        self._orig_discovery_get_builder = discovery_mod.get_single_problem_dataset_builder
        self._orig_discovery_main = discovery_mod.main
        agent_self = self

        class _TrainOnlyBuilder:
            def __init__(self, dataset):
                self.dataset = dataset

            def __deepcopy__(self, memo):
                if hasattr(self.dataset, "__deepcopy__"):
                    dataset = self.dataset.__deepcopy__(memo)
                else:
                    dataset = _safe_type_name(self.dataset)
                return {
                    "_type": type(self).__name__,
                    "dataset": dataset,
                }

            async def __call__(self):
                return self.dataset

        async def _main_with_val(cfg):
            built = await cfg.dataset_builder()
            if isinstance(built, tuple):
                train_dataset, val_dataset = built
            else:
                train_dataset, val_dataset = built, None
            train_cfg = agent_self._copy_train_config(
                cfg, _TrainOnlyBuilder(train_dataset)
            )
            orig_train_step = train_mod.do_train_step_and_get_sampling_client

            async def _wrapped_train_step(*args, **kwargs):
                sampling_client, metrics = await orig_train_step(*args, **kwargs)
                if val_dataset is not None:
                    i_batch = args[1] if len(args) > 1 else kwargs.get("i_batch", 0)
                    val_metrics = await agent_self._run_validation_rollouts(
                        train_cfg, train_mod, sampling_client, val_dataset, int(i_batch)
                    )
                    metrics.update(val_metrics)
                return sampling_client, metrics

            train_mod.do_train_step_and_get_sampling_client = _wrapped_train_step
            try:
                await agent_self._orig_discovery_main(train_cfg)
            finally:
                train_mod.do_train_step_and_get_sampling_client = orig_train_step

        discovery_mod.get_single_problem_dataset_builder = lambda _dataset_config: dataset_builder
        discovery_mod.main = _main_with_val
        self._discovery_mod = discovery_mod

    def _restore_discover_val_adapter(self) -> None:
        mod = getattr(self, "_discovery_mod", None)
        if mod is None:
            return
        if hasattr(self, "_orig_discovery_get_builder"):
            mod.get_single_problem_dataset_builder = self._orig_discovery_get_builder
        if hasattr(self, "_orig_discovery_main"):
            mod.main = self._orig_discovery_main
        self._discovery_mod = None

    def _finalize_task_results(self, task_names: list[str]) -> None:
        for task_name in task_names:
            if task_name == self.task_name:
                slots = self._train_pool_slots.get(task_name) or getattr(self, "_pool_slots", [])
            else:
                slots = self._val_pool_slots.get(task_name) or []
            if not slots:
                continue
            pool_slot = slots[0]
            tools = pool_slot.tools
            best_code = self._best_codes.get(task_name)
            if best_code is None and task_name == self.task_name:
                best_code = self._best_code
            if best_code and not tools.done:
                slot = pool_slot.slot
                path = tools._resolve_workspace_path(slot.filename)
                total = len(path.read_text().splitlines())
                if slot.is_full_file:
                    slot.end_line = total
                elif slot.end_line < 0 or slot.end_line > total:
                    slot.end_line = total
                tools.edit(
                    op="replace",
                    filename=slot.filename,
                    content=best_code,
                    start_line=slot.start_line,
                    end_line=slot.end_line,
                )
                slot.end_line = slot.start_line + len(best_code.splitlines()) - 1

            if not tools.done and tools.test_count < tools.max_tests:
                try:
                    tools.test()
                except Exception:
                    traceback.print_exc()

            if not tools.done and tools._test_history:
                try:
                    tools.submit(n=-1, _force=True)
                except Exception:
                    traceback.print_exc()

            tools.record_zero_if_no_finals()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, resume: bool = False) -> dict:
        if resume:
            print("[discover-agent] resume not supported yet; starting fresh")

        self.setup_workspace()

        run_dir = Path(self.logger.log_dir) / "discover"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._tokens_log = run_dir / "tokens.jsonl"
        if not resume and self._tokens_log.exists():
            try:
                self._tokens_log.unlink()
            except Exception:
                pass

        self._refresh_slot_end()

        # Merge YAML + CLI overrides early so pool_size from the YAML config
        # takes effect before we build the pool.
        yaml_defaults = self._load_yaml_defaults()
        self._merged_disc_cfg = {**yaml_defaults, **self._disc_overrides}
        if self._iterations is not None:
            self._merged_disc_cfg["num_epochs"] = self._iterations
        self._max_actions_per_episode = max(
            1,
            int(
                self._merged_disc_cfg.get(
                    "max_actions_per_episode", self._max_actions_per_episode
                )
            ),
        )
        self._edit_format_penalty = float(
            self._merged_disc_cfg.get("edit_format_penalty", self._edit_format_penalty)
        )
        yaml_pool = yaml_defaults.get("pool_size")
        if yaml_pool is not None and "pool_size" not in self._disc_overrides:
            self._pool_size = max(1, int(yaml_pool))
        self._apply_pool_test_budget(self._compute_max_tests_per_pool(self._merged_disc_cfg))

        # Build the reward-evaluator pool: primary self.tools + (pool_size - 1)
        # peers, each with its own workspace dir keyed by exp_name. Each pool
        # slot maintains its own edit_slot book-keeping.
        print(f"[discover-agent] building WorkspaceTools pool of size {self._pool_size} "
              f"with max_tests={self._max_tests_per_pool} per slot")
        self._build_pool()
        self._apply_pool_test_budget(self._max_tests_per_pool)

        # Load score_spec.py for the task. Reward evaluator uses it if present,
        # else falls back to alphabetical-first metric heuristic.
        self._load_score_spec_safely()
        for val_task in self._val_tasks:
            self._load_score_spec_safely(val_task)

        initial_prompt = self._append_action_protocol(self.build_initial_prompt())
        self.logger.reset()
        self.logger.log_initial_prompt(initial_prompt)
        self.logger._append({"role": "_meta", "exp_name": self.tools.exp_name})

        # Default W&B off unless user opted in.
        os.environ.setdefault("WANDB_MODE", "disabled")

        # Ensure Tinker key is in env for the tinker SDK.
        tk_key = (self.global_config.get("providers") or {}).get("tinker", {}).get("api_key")
        tk_key = tk_key or self.global_config.get("tinker_api_key")
        if tk_key and "PLACEHOLDER" not in str(tk_key):
            os.environ.setdefault("TINKER_API_KEY", tk_key)
        if not os.environ.get("TINKER_API_KEY"):
            raise RuntimeError(
                "TINKER_API_KEY is not set. Pass --tinker-api-key, set the env var, "
                "or add providers.tinker.api_key to your config."
            )

        # Lazy imports so MLS-Bench can load this module even if ttt-discover
        # or tinker aren't installed.
        try:
            import tinker  # noqa: F401
            import ttt_discover.discovery as discovery_mod
            import ttt_discover.rl.train as train_mod
            from ttt_discover import (
                DiscoverConfig,
                Environment,
                State,
                discover,
            )
        except ImportError as exc:
            raise RuntimeError(
                "DiscoverAgent requires the `discover` extra: "
                "pip install -e '.[discover]'"
            ) from exc

        try:
            self._install_tinker_observer(tinker)
            self._install_wandb_stub()

            env_cls = self._build_environment_cls(
                Environment, State, initial_prompt, task_id=self.task_name, pool_label="train"
            )
            val_env_classes: dict[str, type] = {}
            for val_task in self._val_tasks:
                val_slots = self._val_pool_slots.get(val_task) or []
                if not val_slots:
                    continue
                val_assets = self._get_task_assets(val_task)
                val_prompt = self._append_action_protocol(
                    self._build_initial_prompt_for_task(
                        val_task, val_slots[0].tools, val_assets
                    ),
                    slot=val_slots[0].slot,
                )
                val_env_classes[val_task] = self._build_environment_cls(
                    Environment,
                    State,
                    val_prompt,
                    task_id=val_task,
                    pool_label="val",
                )

            model_name = self.global_config.get("model", "openai/gpt-oss-20b")
            problem_type = self.task_name  # used only as a label by ttt-discover

            disc_cfg = self._build_discover_config(
                DiscoverConfig, env_cls, model_name, problem_type, run_dir
            )
            if self._val_tasks:
                dataset_builder = self._build_dataset_builder(
                    train_env_cls=env_cls,
                    val_env_classes=val_env_classes,
                    model_name=model_name,
                    renderer_name=disc_cfg.renderer_name,
                    log_path=Path("./tinker_log") / disc_cfg.experiment_name,
                    groups_per_batch=disc_cfg.groups_per_batch,
                    group_size=disc_cfg.group_size,
                    num_cpus_per_task=disc_cfg.num_cpus_per_task,
                    eval_timeout=disc_cfg.eval_timeout,
                )
                self._install_discover_val_adapter(
                    discovery_mod, train_mod, dataset_builder
                )

            print(f"[discover-agent] launching ttt_discover.discover "
                  f"model={model_name} epochs={disc_cfg.num_epochs} "
                  f"group_size={disc_cfg.group_size} groups_per_batch={disc_cfg.groups_per_batch}")

            discover(disc_cfg)
        except Exception:
            traceback.print_exc()
        finally:
            self._restore_tinker()
            self._restore_wandb_stub()
            self._restore_discover_val_adapter()

        # Apply best candidate (if we saw one) and run the final all-seeds test
        # on the PRIMARY pool slot (self.tools), so the final is_final=true row
        # is attributed to the canonical exp_name.
        if self._val_tasks:
            self._finalize_task_results([self.task_name, *self._val_tasks])
        else:
            primary_slot = self._pool_slots[0] if getattr(self, "_pool_slots", None) else None
            if self._best_code and not self.tools.done:
                target_slot = primary_slot.slot if primary_slot else self._slot
                # Refresh end against primary tools workspace file
                path = self.tools._resolve_workspace_path(target_slot.filename)
                total = len(path.read_text().splitlines())
                if target_slot.is_full_file:
                    target_slot.end_line = total
                elif target_slot.end_line < 0 or target_slot.end_line > total:
                    target_slot.end_line = total
                self.tools.edit(
                    op="replace",
                    filename=target_slot.filename,
                    content=self._best_code,
                    start_line=target_slot.start_line,
                    end_line=target_slot.end_line,
                )
                target_slot.end_line = target_slot.start_line + len(self._best_code.splitlines()) - 1

            if not self.tools.done and self.tools.test_count < self.tools.max_tests:
                try:
                    self.tools.test()
                except Exception:
                    traceback.print_exc()

            if not self.tools.done and self.tools._test_history:
                try:
                    self.tools.submit(n=-1, _force=True)
                except Exception:
                    traceback.print_exc()

            self.tools.record_zero_if_no_finals()

        totals = self._totals_from_jsonl()
        if not totals.get("calls"):
            totals = dict(self._token_totals)
        train_totals = self._totals_from_jsonl(pool="train") if self._val_tasks else dict(self._train_token_totals)
        val_totals = self._totals_from_jsonl(pool="val") if self._val_tasks else dict(self._val_token_totals)
        print(f"[discover-agent] token totals: {totals}")
        if self._val_tasks:
            print(f"[discover-agent] train token totals: {train_totals}")
            print(f"[discover-agent] val token totals: {val_totals}")
        print(f"[discover-agent] best reward seen: {self._best_score}")

        self._token_totals = dict(totals)
        # Snapshot the merged Discover hyperparams so downstream analysis can
        # tell two runs apart even if their --discover-config differed.
        disc_config_snapshot = dict(getattr(self, "_merged_disc_cfg", None) or {})
        self._write_run_summary(extra={
            "discover_best_reward": self._best_score,
            "iter_counter": self._iter_counter,
            "discover_config": disc_config_snapshot,
        })

        summary = {
            "steps": self._iter_counter,
            "tests": self.tools.test_count,
            "done": self.tools.done,
            "tokens": totals,
            "discover_best_reward": self._best_score,
        }
        if self._val_tasks:
            summary["train_tokens"] = train_totals
            summary["val_tokens"] = val_totals
        return summary

    # ------------------------------------------------------------------
    # Tinker sampling-client monkey-patch (token accounting)
    # ------------------------------------------------------------------
    def _install_tinker_observer(self, tinker_mod) -> None:
        self._orig_sample_async = tinker_mod.SamplingClient.sample_async
        agent_self = self
        model_name = self.global_config.get("model", "openai/gpt-oss-20b")

        async def wrapped(self, prompt, num_samples, sampling_params):
            result = await agent_self._orig_sample_async(self, prompt, num_samples, sampling_params)
            try:
                prompt_tokens = getattr(prompt, "length", 0) or 0
                completion_tokens = 0
                for seq in getattr(result, "sequences", []) or []:
                    toks = getattr(seq, "tokens", None) or []
                    completion_tokens += len(toks)
                record = {
                    "timestamp": time.time(),
                    "step": agent_self._iter_counter,
                    "model": model_name,
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens),
                    "cached_tokens": 0,
                    "cache_creation_tokens": 0,
                }
                if agent_self._val_tasks:
                    record["pool"] = agent_self._sample_pool.get("train")
                agent_self._emit_tokens(record)
            except Exception:
                pass
            return result

        tinker_mod.SamplingClient.sample_async = wrapped
        self._tinker_mod = tinker_mod

    # ------------------------------------------------------------------
    # ml_log.setup_logging stub (pad loggers to len>=3 when wandb disabled)
    # ------------------------------------------------------------------
    # Upstream ttt_discover/rl/train.py:587-592 unconditionally indexes
    # ml_logger.loggers[2] under a `len >= 2` guard (off-by-one). With
    # WANDB_MODE=disabled or wandb_project=null, setup_logging only
    # registers JsonLogger + PrettyPrintLogger (length 2), so the third
    # access raises IndexError and aborts RL training. We pad with a
    # no-op Logger that fails the `isinstance(_, WandbLogger)` guards
    # cleanly.
    def _install_wandb_stub(self) -> None:
        try:
            from ttt_discover.tinker_utils import ml_log
        except ImportError:
            self._ml_log_mod = None
            return

        from abc import ABC

        class _NoOpLogger(ml_log.Logger):  # type: ignore[misc]
            def log_hparams(self, config):  # noqa: D401
                return None

            def log_metrics(self, metrics, step=None):  # noqa: D401
                return None

        self._orig_setup_logging = ml_log.setup_logging

        def _padded_setup_logging(*args, **kwargs):
            ml_logger = self._orig_setup_logging(*args, **kwargs)
            try:
                if len(ml_logger.loggers) < 3:
                    ml_logger.loggers.append(_NoOpLogger())
            except Exception:
                pass
            return ml_logger

        ml_log.setup_logging = _padded_setup_logging
        self._ml_log_mod = ml_log

    def _restore_wandb_stub(self) -> None:
        mod = getattr(self, "_ml_log_mod", None)
        if mod is not None and hasattr(self, "_orig_setup_logging"):
            try:
                mod.setup_logging = self._orig_setup_logging
            except Exception:
                pass

    def _restore_tinker(self) -> None:
        mod = getattr(self, "_tinker_mod", None)
        if mod is not None and hasattr(self, "_orig_sample_async"):
            try:
                mod.SamplingClient.sample_async = self._orig_sample_async
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Dynamic Environment + RewardEvaluator classes
    # ------------------------------------------------------------------
    def _build_environment_cls(
        self,
        Environment,
        State,
        initial_prompt: str,
        *,
        task_id: str | None = None,
        pool_label: str = "train",
    ):
        agent_self = self
        env_task_id = task_id or getattr(self, "task_name", "")
        env_pool_label = pool_label

        # Environment: static single-problem env. get_question returns the
        # same MLS-Bench initial prompt every rollout. Reward/test execution
        # lives in step() so upstream do_single_rollout can run multi-turn
        # episodes without invoking Environment._run_verification.
        prompt_text = initial_prompt

        class MLSBenchEnv(Environment):
            state_type = State

            def __init__(self, renderer, initial_state, sampler, config):
                super().__init__(
                    renderer,
                    initial_state=initial_state,
                    sampler=sampler,
                    config=config,
                )
                self.convo = [{"role": "user", "content": prompt_text}]
                self._pool_slot: _PoolSlot | None = None
                self._released_slot = False
                self._actions_taken = 0

            def get_question(self) -> str:
                return prompt_text

            async def initial_observation(self):
                self._ensure_pool_slot()
                self.convo = [{"role": "user", "content": prompt_text}]
                return self.renderer.build_generation_prompt(self.convo), self.stop_condition

            def _ensure_pool_slot(self) -> _PoolSlot:
                if self._pool_slot is None:
                    try:
                        self._pool_slot = agent_self._acquire(
                            pool_label=env_pool_label,
                            task_id=env_task_id,
                        )
                    except TypeError:
                        self._pool_slot = agent_self._acquire()
                    self._released_slot = False
                return self._pool_slot

            def _release_pool_slot(self) -> None:
                if self._pool_slot is not None and not self._released_slot:
                    try:
                        agent_self._release(
                            self._pool_slot,
                            pool_label=env_pool_label,
                            task_id=env_task_id,
                        )
                    except TypeError:
                        agent_self._release(self._pool_slot)
                    self._released_slot = True
                    self._pool_slot = None

            def __del__(self):  # pragma: no cover - defensive cleanup only
                try:
                    self._release_pool_slot()
                except Exception:
                    pass

            def close(self) -> None:
                self._release_pool_slot()

            def _refresh_slot_end_for_pool(self) -> None:
                pool_slot = self._ensure_pool_slot()
                tools = pool_slot.tools
                slot = pool_slot.slot
                path = tools._resolve_workspace_path(slot.filename)
                total = len(path.read_text().splitlines())
                if slot.is_full_file:
                    slot.end_line = total
                elif slot.end_line < slot.start_line or slot.end_line > total:
                    slot.end_line = total

            def _empty_observation(self):
                import tinker

                return tinker.ModelInput.empty()

            def _next_observation(self):
                return self.renderer.build_generation_prompt(self.convo)

            def _logged_step_idx(self, message_text: str, action_kind: str) -> int:
                with agent_self._iter_lock:
                    agent_self._iter_counter += 1
                    logged_idx = agent_self._iter_counter
                pool_slot = self._ensure_pool_slot()
                agent_self.logger.log_assistant(
                    logged_idx,
                    {
                        "name": "discover_rollout_step",
                        "input": {
                            "iteration": logged_idx,
                            "rollout_step": self._actions_taken,
                            "action_kind": action_kind,
                            "pool_slot_exp_name": pool_slot.tools.exp_name,
                            "message": message_text[:4000],
                            **(
                                {"pool": env_pool_label, "task_id": env_task_id}
                                if getattr(agent_self, "_val_tasks", [])
                                or env_pool_label != "train"
                                else {}
                            ),
                        },
                        "thinking": None,
                    },
                )
                return logged_idx

            def _log_tool_result(self, logged_idx: int, result: str, meta: dict) -> None:
                if getattr(agent_self, "_val_tasks", []) or env_pool_label != "train":
                    meta = {
                        **meta,
                        "pool": env_pool_label,
                        "task_id": env_task_id,
                    }
                agent_self.logger.log_tool_result(logged_idx, result, meta=meta)

            def _terminal_step_result(
                self,
                reward: float,
                metrics: dict[str, Any],
                feedback: str,
                logged_idx: int,
                meta: dict,
            ):
                hidden = getattr(agent_self, "_task_hidden_labels", {}).get(
                    env_task_id, getattr(agent_self, "_hidden_labels", set())
                )
                visible_metrics = _filter_hidden_metrics(metrics, hidden)
                meta = dict(meta)
                meta["discover_metrics"] = visible_metrics
                self._log_tool_result(logged_idx, feedback, meta=meta)
                self._release_pool_slot()
                from ttt_discover.rl.types import StepResult

                return StepResult(
                    reward=float(reward),
                    episode_done=True,
                    next_observation=self._empty_observation(),
                    next_stop_condition=self.stop_condition,
                    metrics=visible_metrics,
                )

            def _nonterminal_step_result(
                self,
                reward: float,
                metrics: dict[str, Any],
                feedback: str,
                logged_idx: int,
                meta: dict,
            ):
                self._log_tool_result(logged_idx, feedback, meta=meta)
                from ttt_discover.rl.types import StepResult

                return StepResult(
                    reward=float(reward),
                    episode_done=False,
                    next_observation=self._next_observation(),
                    next_stop_condition=self.stop_condition,
                    metrics=metrics,
                )

            def _budget_exhausted_result(self, message_text: str):
                logged_idx = self._logged_step_idx(message_text, "budget_exhausted")
                metrics = {
                    "action_kind": "budget_exhausted",
                    "edit_ok": False,
                    "budget_exhausted": 1,
                }
                return self._terminal_step_result(
                    0.0,
                    metrics,
                    "budget exhausted",
                    logged_idx,
                    {"action_kind": "budget_exhausted", "edit_ok": False},
                )

            def _test_result(
                self,
                message_text: str,
                logged_idx: int,
                action_kind: str,
                forced_test: bool = False,
                edit_ok: bool | None = None,
                prefix: str | None = None,
            ):
                pool_slot = self._ensure_pool_slot()
                tools = pool_slot.tools
                try:
                    test_out = tools.test()
                except Exception as exc:
                    metrics = {
                        "action_kind": action_kind,
                        "edit_ok": bool(edit_ok) if edit_ok is not None else False,
                        "test_error": 1,
                    }
                    if forced_test:
                        metrics["forced_test"] = 1
                    return self._terminal_step_result(
                        0.0,
                        metrics,
                        f"test raised: {exc}",
                        logged_idx,
                        {
                            "action_kind": action_kind,
                            "edit_ok": edit_ok,
                            "forced_test": forced_test,
                        },
                    )

                entry = tools.latest_test_history_entry()
                if entry is None:
                    metrics = {
                        "action_kind": action_kind,
                        "edit_ok": bool(edit_ok) if edit_ok is not None else False,
                        "no_test_history": 1,
                    }
                    if forced_test:
                        metrics["forced_test"] = 1
                    return self._terminal_step_result(
                        0.0,
                        metrics,
                        "no test history",
                        logged_idx,
                        {
                            "action_kind": action_kind,
                            "edit_ok": edit_ok,
                            "forced_test": forced_test,
                        },
                    )

                try:
                    primary, avg = agent_self._primary_metric_from_entry(
                        entry, task_name=env_task_id
                    )
                except TypeError:
                    primary, avg = agent_self._primary_metric_from_entry(entry)
                reward_metrics: dict[str, Any] = {
                    "action_kind": action_kind,
                    "combined_score": float(primary),
                }
                if edit_ok is not None:
                    reward_metrics["edit_ok"] = bool(edit_ok)
                if forced_test:
                    reward_metrics["forced_test"] = 1
                for k, v in avg.items():
                    if k not in reward_metrics:
                        reward_metrics[k] = float(v)
                if entry.get("had_failures"):
                    reward_metrics["_had_failures"] = 1.0

                best_code = self._current_slot_content()
                if hasattr(agent_self, "_record_best_code"):
                    agent_self._record_best_code(env_task_id, float(primary), best_code)
                elif primary > agent_self._best_score:
                    agent_self._best_score = float(primary)
                    if best_code.strip():
                        agent_self._best_code = best_code

                feedback_parts = []
                if prefix:
                    feedback_parts.append(prefix)
                feedback_parts.append(
                    f"reward={primary}\n\n"
                    + (test_out[:1500] if isinstance(test_out, str) else str(test_out))
                )
                return self._terminal_step_result(
                    float(primary),
                    reward_metrics,
                    "\n\n".join(feedback_parts),
                    logged_idx,
                    {
                        "action_kind": action_kind,
                        "edit_ok": edit_ok,
                        "forced_test": forced_test,
                        "test_history_entry": entry,
                    },
                )

            def _current_slot_content(self) -> str:
                pool_slot = self._ensure_pool_slot()
                tools = pool_slot.tools
                slot = pool_slot.slot
                path = tools._resolve_workspace_path(slot.filename)
                text = path.read_text()
                if slot.is_full_file:
                    return text
                lines = text.splitlines()
                return "\n".join(lines[slot.start_line - 1 : slot.end_line])

            async def step(self, action, step_idx: int, *args, **kwargs):
                try:
                    pool_slot = self._ensure_pool_slot()
                    tools = pool_slot.tools
                    if tools.done or tools.test_count >= tools.max_tests:
                        return self._budget_exhausted_result("budget exhausted")

                    message, _parse_success = self.renderer.parse_response(action)
                    message_text = _ensure_text(message.get("content", ""))
                    self.convo.append({"role": "assistant", "content": message_text})
                    self._actions_taken += 1

                    if _TEST_TAG_RE.search(message_text):
                        logged_idx = self._logged_step_idx(message_text, "test")
                        return self._test_result(message_text, logged_idx, "test")

                    if _has_python_block(message_text):
                        block = _extract_python(message_text)
                        logged_idx = self._logged_step_idx(message_text, "edit")
                        with pool_slot.lock:
                            self._refresh_slot_end_for_pool()
                            slot = pool_slot.slot
                            edit_res = tools.edit(
                                op="replace",
                                filename=slot.filename,
                                content=block,
                                start_line=slot.start_line,
                                end_line=slot.end_line,
                            )
                            edit_ok = not (
                                isinstance(edit_res, str) and edit_res.startswith("ERROR")
                            )
                            if edit_ok:
                                slot.end_line = slot.start_line + len(block.splitlines()) - 1

                        self.convo.append({"role": "user", "content": edit_res})
                        edit_reward = (
                            0.0 if edit_ok else -float(agent_self._edit_format_penalty)
                        )
                        metrics = {"action_kind": "edit", "edit_ok": bool(edit_ok)}
                        meta = {"action_kind": "edit", "edit_ok": bool(edit_ok)}
                        if self._actions_taken >= agent_self._max_actions_per_episode:
                            forced_prefix = (
                                f"{edit_res}\n\nAction limit reached; running final test."
                            )
                            return self._test_result(
                                message_text,
                                logged_idx,
                                "edit",
                                forced_test=True,
                                edit_ok=bool(edit_ok),
                                prefix=forced_prefix,
                            )
                        return self._nonterminal_step_result(
                            edit_reward,
                            metrics,
                            edit_res,
                            logged_idx,
                            meta,
                        )

                    feedback = (
                        "No <code> or <test/> action detected. Reply with a ```python ...``` "
                        "block to edit the editable region, or `<test/>` on its own line "
                        "to run evaluation."
                    )
                    self.convo.append({"role": "user", "content": feedback})
                    logged_idx = self._logged_step_idx(message_text, "noop")
                    metrics = {"action_kind": "noop", "edit_ok": False}
                    meta = {"action_kind": "noop", "edit_ok": False}
                    if self._actions_taken >= agent_self._max_actions_per_episode:
                        forced_prefix = (
                            f"{feedback}\n\nAction limit reached; running final test."
                        )
                        return self._test_result(
                            message_text,
                            logged_idx,
                            "noop",
                            forced_test=True,
                            edit_ok=False,
                            prefix=forced_prefix,
                        )
                    return self._nonterminal_step_result(
                        0.0,
                        metrics,
                        feedback,
                        logged_idx,
                        meta,
                    )
                except Exception:
                    err = traceback.format_exc()
                    try:
                        logged_idx = self._logged_step_idx(str(action), "error")
                        return self._terminal_step_result(
                            0.0,
                            {"action_kind": "error", "edit_ok": False, "step_error": 1},
                            err,
                            logged_idx,
                            {"action_kind": "error", "edit_ok": False},
                        )
                    except Exception:
                        self._release_pool_slot()
                        raise

        return MLSBenchEnv

    # ------------------------------------------------------------------
    # DiscoverConfig construction
    # ------------------------------------------------------------------
    def _load_yaml_defaults(self) -> dict:
        """Parse the YAML at discover.config_path, if provided."""
        if not self._disc_config_path:
            return {}
        import yaml
        with open(self._disc_config_path) as f:
            return yaml.safe_load(f) or {}

    def _build_discover_config(self, DiscoverConfig, env_cls, model_name: str,
                                problem_type: str, run_dir: Path):
        # Use the already-merged config assembled in run(); fall back to
        # loading here if _merged_disc_cfg wasn't populated (e.g., tests).
        merged = getattr(self, "_merged_disc_cfg", None)
        if merged is None:
            merged = {**self._load_yaml_defaults(), **self._disc_overrides}

        cfg_kwargs = dict(
            env_type=env_cls,
            problem_type=problem_type,
            model_name=model_name,
            experiment_name=merged.get("experiment_name", f"mlsbench-{self.task_name}-{self.tools.exp_name}"),
            wandb_project=merged.get("wandb_project"),  # None by default; WANDB_MODE=disabled anyway
            num_cpus_per_task=int(merged.get("num_cpus_per_task", 0)),  # 0 → no ray dispatch
            eval_timeout=int(merged.get("eval_timeout", 1000)),
            lora_rank=int(merged.get("lora_rank", 32)),
            renderer_name=merged.get("renderer_name", "gpt_oss_high_reasoning"),
            save_every=int(merged.get("save_every", 2)),
            group_size=int(merged.get("group_size", 2)),
            groups_per_batch=int(merged.get("groups_per_batch", 1)),
            learning_rate=float(merged.get("learning_rate", 4e-5)),
            num_epochs=int(merged.get("num_epochs", 2)),
            temperature=float(merged.get("temperature", 1.0)),
            kl_penalty_coef=float(merged.get("kl_penalty_coef", 0.1)),
            phase1_max_tokens=int(merged.get("phase1_max_tokens", 8000)),
        )

        return DiscoverConfig(**cfg_kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
import re as _re

_PY_BLOCK_RE = _re.compile(r"```python\s+([\s\S]*?)\s*```", _re.MULTILINE)
_TEST_TAG_RE = _re.compile(r"<test\s*/?>", _re.IGNORECASE)


def _extract_python(text: str) -> str:
    """Extract the last ```python ... ``` block from a raw response.

    If no fenced block is present, fall back to the raw text.
    """
    if not isinstance(text, str):
        return ""
    matches = list(_PY_BLOCK_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    return text.strip()


def _has_python_block(text: str) -> bool:
    return isinstance(text, str) and _PY_BLOCK_RE.search(text) is not None


def _ensure_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and len(content) == 1:
        part = content[0]
        if isinstance(part, dict) and part.get("type") == "text":
            return str(part.get("text", ""))
    return str(content)
