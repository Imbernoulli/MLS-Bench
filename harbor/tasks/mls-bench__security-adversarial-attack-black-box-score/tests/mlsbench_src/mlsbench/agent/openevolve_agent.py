"""OpenEvolveAgent: MLS-Bench agent that delegates to OpenEvolve's evolutionary loop.

OpenEvolve drives its own loop (MAP-Elites + islands + LLM mutation) via
``openevolve.run_evolution``. Each evolution iteration evaluates a candidate
program by calling a user-supplied ``evaluate(program_path) -> dict`` function.

Because OpenEvolve runs evaluations in a ProcessPoolExecutor, the evaluator
cannot close over ``self.tools`` directly. We use a filesystem broker:

1. The evaluator file we hand to OpenEvolve writes a request JSON into
   ``<run_dir>/broker/requests/<id>.json`` and polls for a matching
   ``<run_dir>/broker/responses/<id>.json``.
2. A background thread in the main process (this agent) watches the requests
   dir, applies each candidate to the workspace via ``tools.edit``, runs
   ``tools.test()``, parses metrics, and writes the response.

This keeps all of OpenEvolve's algorithmic machinery untouched — we only
interpose on the evaluator contract.

Constraints (v1):
- Task must have exactly one editable file with one contiguous edit range.
- Task's ``score_spec.py`` primary metric is mapped to ``combined_score`` for
  OpenEvolve; additional metrics are passed through unchanged.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import textwrap
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlsbench.agent.base import BaseAgent


EVOLVE_START = "# EVOLVE-BLOCK-START"
EVOLVE_END = "# EVOLVE-BLOCK-END"


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


@dataclass
class _EditSlot:
    """Describes the single editable slot in the workspace file."""

    filename: str
    start_line: int  # 1-indexed, fixed across iterations
    end_line: int    # 1-indexed, advances as content length changes
    is_full_file: bool


class OpenEvolveAgent(BaseAgent):
    """Drive OpenEvolve as an MLS-Bench agent.

    Overrides :meth:`run` because OpenEvolve owns the iteration loop; the
    standard :meth:`get_action` tool-use interface doesn't fit.
    """

    agent_label = "openevolve"

    def __init__(self, task_name: str, global_config: dict, workspace_root=None):
        super().__init__(task_name, global_config, workspace_root)

        self._slot = self._resolve_editable_slot()

        # All OpenEvolve-agent-specific config lives under global_config["openevolve"]
        oe_cfg = dict(global_config.get("openevolve") or {})
        # iterations: only override the YAML if the user explicitly set it via
        # --openevolve-iterations / oe_knobs["iterations"]. Otherwise we keep
        # whatever max_iterations the YAML config declares so authors can tune
        # iteration count in calls64.yaml etc. without per-CLI plumbing.
        self._iterations: int | None = (
            int(oe_cfg["iterations"]) if oe_cfg.get("iterations") is not None else None
        )
        self._oe_config_path: str | None = oe_cfg.get("config_path")
        self._oe_overrides: dict = oe_cfg.get("overrides") or {}

        self._broker_stop = threading.Event()
        self._broker_thread: threading.Thread | None = None
        self._broker_dir: Path | None = None
        self._tokens_log: Path | None = None
        self._score_spec = None
        self._score_anchors = None
        self._score_spec_error: str | None = None
        self._hidden_labels: set[str] = _hidden_labels_from_test_cmds(self.config_task)
        print("[openevolve-agent] hidden metrics excluded from candidate response: "
              f"{sorted(self._hidden_labels)}")
        self._token_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "calls": 0,
        }
        self._iter_counter = 0  # broker-side iteration index for logging

    # ------------------------------------------------------------------
    # get_action: not used — we override run() — but BaseAgent is abstract
    # ------------------------------------------------------------------
    def get_action(self, messages: list) -> dict | None:  # pragma: no cover
        raise NotImplementedError(
            "OpenEvolveAgent drives its own loop via OpenEvolve; "
            "get_action is not called."
        )

    # ------------------------------------------------------------------
    # Editable-slot resolution
    # ------------------------------------------------------------------
    def _resolve_editable_slot(self) -> _EditSlot:
        editable = [f for f in self.config_edit if f.get("edit")]
        if len(editable) != 1:
            raise ValueError(
                f"OpenEvolveAgent requires exactly 1 editable file, "
                f"got {len(editable)} in task {self.task_name!r}"
            )
        entry = editable[0]
        ranges = entry["edit"]
        if len(ranges) != 1:
            raise ValueError(
                f"OpenEvolveAgent requires a single contiguous edit range, "
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
        path = self.tools._resolve_workspace_path(self._slot.filename)
        return path.read_text()

    def _slot_content(self) -> str:
        """Extract the current content of the editable slot from the workspace file."""
        text = self._read_workspace_file()
        if self._slot.is_full_file:
            return text
        lines = text.splitlines()
        return "\n".join(lines[self._slot.start_line - 1 : self._slot.end_line])

    def _refresh_slot_end(self) -> None:
        """Recompute end_line from the workspace file.

        Used after setup (so the file as-copied dictates the initial end) and
        after each successful edit. For full-file edits, end is total line count.
        """
        text = self._read_workspace_file()
        total = len(text.splitlines())
        if self._slot.is_full_file:
            self._slot.end_line = total
        else:
            # On first call end_line comes from config. Afterwards we track it.
            # If config's end exceeds file length, clamp (edge case for new files).
            if self._slot.end_line < 0 or self._slot.end_line > total:
                self._slot.end_line = total

    # ------------------------------------------------------------------
    # Primary-metric helpers — score_spec.py-based reward
    # ------------------------------------------------------------------
    def _load_score_spec_safely(self) -> None:
        """Load the task's score_spec.py + baseline anchors.

        Builds a *visible-only* spec that strips settings backed by hidden
        test_cmds. The agent uses this as its evolutionary reward so the
        search isn't peeking at held-out evaluations; final leaderboard
        scoring (which is done by external tooling, not the agent) still
        uses the full spec including hidden settings.
        """
        if hasattr(self, "_score_spec") and self._score_spec is not None:
            return
        self._score_spec = None
        self._score_anchors = None
        task_dir = self.project_root / "tasks" / self.task_name
        spec_path = task_dir / "score_spec.py"
        if not spec_path.exists():
            return

        try:
            import copy as _copy
            from mlsbench.scoring.anchors import BaselineAnchors
            from mlsbench.scoring.evaluate import load_expanded_spec

            anchors = BaselineAnchors(task_dir)
            spec = load_expanded_spec(task_dir, anchors)
            if spec is None:
                self._score_spec_error = (
                    f"{spec_path} exists but expanded to no score settings"
                )
                print(f"[openevolve-agent] ERROR: {self._score_spec_error}; "
                      "candidates will receive fail-closed reward")
                return

            hidden_labels = {
                tc["label"] for tc in self.config_task.get("test_cmds", [])
                if tc.get("hidden") and "label" in tc
            }
            if hidden_labels:
                visible = _copy.deepcopy(spec)
                # Setting names mirror test_cmd labels by convention.
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
                self._score_spec = visible
                print(f"[openevolve-agent] using score_spec.py (excluding hidden labels: "
                      f"{sorted(hidden_labels)})")
            else:
                self._score_spec = spec
                print(f"[openevolve-agent] using score_spec.py from {task_dir}")
            self._score_anchors = anchors
        except _AllHiddenScoreSpecError:
            raise
        except Exception as exc:
            self._score_spec_error = f"score_spec load failed: {exc!r}"
            print(f"[openevolve-agent] ERROR: {self._score_spec_error}; "
                  "candidates will receive fail-closed reward")

    def _primary_metric_from_entry(self, entry: dict) -> tuple[float, dict]:
        """Reduce a test history entry to (combined_score, all_metrics dict).

        Prefers score_spec.py when the task has one — produces the canonical
        benchmark scalar that OpenEvolve should be optimizing. Falls back to
        the alphabetical-first heuristic when score_spec isn't available.
        """
        seed_metrics: list[dict] = entry.get("seed_metrics") or []
        if not seed_metrics:
            return (0.0, {})

        # Aggregate seeds → mean record (mirrors the leaderboard's seed=mean row)
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

        # score_spec path — preferred. Without this, OpenEvolve's mutation
        # selection picks whatever metric happens to sort first alphabetically,
        # which can be elapsed_* (timing) for many tasks → catastrophic
        # mis-optimization (rewarding slow code).
        if getattr(self, "_score_spec_error", None):
            print(f"[openevolve-agent] ERROR: {self._score_spec_error}; "
                  "returning fail-closed reward")
            return (-1e9, avg)

        if getattr(self, "_score_spec", None) is not None and getattr(self, "_score_anchors", None) is not None:
            try:
                from mlsbench.scoring.evaluate import score_record
                record = {"seed": "mean", **{k: v for k, v in avg.items()}}
                spec_score = float(score_record(self._score_spec, record, self._score_anchors))
                return (spec_score, avg)
            except Exception as exc:
                print(f"[openevolve-agent] ERROR: score_record failed: {exc!r}; "
                      "returning fail-closed reward")
                return (-1e9, avg)

        primary_key = sorted(avg.keys())[0]
        return (avg[primary_key], avg)

    # ------------------------------------------------------------------
    # Broker: receive eval requests from evaluator workers, dispatch to tools
    # ------------------------------------------------------------------
    def _start_broker(self, run_dir: Path) -> Path:
        broker = run_dir / "broker"
        if broker.exists():
            shutil.rmtree(broker)
        (broker / "requests").mkdir(parents=True, exist_ok=True)
        (broker / "responses").mkdir(parents=True, exist_ok=True)
        self._broker_dir = broker
        self._broker_stop.clear()
        t = threading.Thread(
            target=self._broker_loop, name="mlsbench-evo-broker", daemon=True
        )
        t.start()
        self._broker_thread = t
        return broker

    def _stop_broker(self) -> bool:
        self._broker_stop.set()
        if self._broker_thread is None:
            return True
        self._broker_thread.join(timeout=300)
        if self._broker_thread.is_alive():
            print("[openevolve-agent] WARNING: broker thread did not stop within "
                  "300s; skipping final eval to avoid concurrent WorkspaceTools use")
            return False
        self._broker_thread = None
        return True

    def _broker_loop(self) -> None:
        assert self._broker_dir is not None
        req_dir = self._broker_dir / "requests"
        res_dir = self._broker_dir / "responses"
        while not self._broker_stop.is_set():
            try:
                reqs = sorted(req_dir.glob("*.json"))
            except FileNotFoundError:
                reqs = []
            for req_path in reqs:
                if self._broker_stop.is_set():
                    break
                res_path = res_dir / req_path.name
                if res_path.exists():
                    continue  # stale double-write
                try:
                    data = json.loads(req_path.read_text())
                    if self._broker_stop.is_set():
                        metrics = {"combined_score": -1e9, "error": "broker stopping"}
                    else:
                        metrics = self._handle_candidate(data.get("code", ""))
                except Exception as exc:
                    metrics = {
                        "combined_score": -1e9,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                metrics = _filter_hidden_metrics(metrics, self._hidden_labels)
                # Atomic write of response, then delete request
                tmp = res_dir / f".{req_path.stem}.tmp"
                tmp.write_text(json.dumps({"metrics": metrics}))
                tmp.replace(res_path)
                try:
                    req_path.unlink()
                except FileNotFoundError:
                    pass
            time.sleep(0.2)

    def _handle_candidate(self, code: str) -> dict[str, float]:
        """Apply candidate code to the workspace, run test(), return metrics."""
        block = _strip_evolve_markers(code)
        if not block.strip():
            return {"combined_score": -1e9, "error": "empty candidate"}

        # Test budget guard: if tools are done or budget exhausted, fast-fail.
        if self.tools.done or self.tools.test_count >= self.tools.max_tests:
            return {"combined_score": -1e9, "error": "test budget exhausted"}

        self._refresh_slot_end()
        edit_result = self.tools.edit(
            op="replace",
            filename=self._slot.filename,
            content=block,
            start_line=self._slot.start_line,
            end_line=self._slot.end_line,
        )
        if edit_result.startswith("ERROR"):
            return {"combined_score": -1e9, "error": f"edit rejected: {edit_result[:200]}"}

        new_lines = block.splitlines()
        self._slot.end_line = self._slot.start_line + len(new_lines) - 1

        if self._broker_stop.is_set():
            return {"combined_score": -1e9, "error": "broker stopping"}

        test_output = self.tools.test()
        entry = self.tools.latest_test_history_entry()
        if entry is None:
            return {"combined_score": -1e9, "error": "no test history"}

        combined, avg = self._primary_metric_from_entry(entry)
        metrics: dict[str, float] = {"combined_score": float(combined)}
        for k, v in avg.items():
            if k not in metrics:
                metrics[k] = float(v)
        if entry.get("had_failures"):
            metrics["_had_failures"] = 1.0

        self._iter_counter += 1
        self.logger.log_assistant(
            self._iter_counter,
            {
                "name": "openevolve_iteration",
                "input": {"iteration": self._iter_counter},
                "thinking": None,
            },
        )
        self.logger.log_tool_result(
            self._iter_counter,
            f"metrics={metrics}\n\n" + (test_output[:2000] if isinstance(test_output, str) else str(test_output)),
            meta={"test_history_entry": entry, "openevolve_metrics": metrics},
        )
        return metrics

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------
    def _totals_from_jsonl(self) -> dict:
        """Replay tokens.jsonl into a totals dict. Resilient to missing file."""
        totals = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cached_tokens": 0, "cache_creation_tokens": 0, "calls": 0,
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
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                              "cached_tokens", "cache_creation_tokens"):
                        v = rec.get(k) or 0
                        if isinstance(v, (int, float)):
                            totals[k] += int(v)
                    totals["calls"] += 1
        except Exception:
            pass
        return totals

    def _token_observer(self, record: dict) -> None:
        if self._tokens_log is None:
            return
        try:
            with self._tokens_log.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            return
        for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                  "cached_tokens", "cache_creation_tokens"):
            v = record.get(k) or 0
            if isinstance(v, (int, float)):
                self._token_totals[k] += int(v)
        self._token_totals["calls"] += 1

    # ------------------------------------------------------------------
    # OpenEvolve config construction
    # ------------------------------------------------------------------
    def _resolve_llm_endpoint(self, cfg, default_cfg) -> tuple[str, str | None]:
        """Pick the base_url and api_key for OpenEvolve's LLM calls.

        Priority:
        1. openevolve.overrides.api_base / .api_key
        2. openevolve.provider (name) → global_config.providers[<name>]
        3. First provider in global_config.providers that has a non-PLACEHOLDER
           key AND a non-default base_url (a LiteLLM proxy counts, an empty
           field does not). Preference order: litellm, openrouter, anthropic,
           openai, qwen, deepseek, gemini.
        4. OPENROUTER_API_KEY / OPENAI_API_KEY env vars.

        The base URL falls through to OpenRouter only if nothing else resolves.
        """
        overrides = self._oe_overrides
        providers = self.global_config.get("providers") or {}

        def _clean(v):
            if not v or "PLACEHOLDER" in str(v):
                return None
            return v

        api_base = _clean(overrides.get("api_base"))
        api_key = _clean(overrides.get("api_key"))

        if (not api_base or not api_key) and overrides.get("provider"):
            entry = providers.get(overrides["provider"]) or {}
            api_base = api_base or _clean(entry.get("base_url"))
            api_key = api_key or _clean(entry.get("api_key"))

        if not api_key:
            for name in ("litellm", "openrouter", "anthropic", "openai",
                         "qwen", "deepseek", "gemini"):
                entry = providers.get(name) or {}
                k = _clean(entry.get("api_key"))
                b = _clean(entry.get("base_url"))
                if k:
                    api_key = k
                    if not api_base:
                        api_base = b
                    break

        api_key = api_key or _clean(os.environ.get("OPENROUTER_API_KEY")) \
            or _clean(os.environ.get("OPENAI_API_KEY"))

        default_base = default_cfg.llm.api_base
        if not api_base or api_base == default_base:
            api_base = cfg.llm.api_base
        if not api_base or api_base == default_base:
            api_base = "https://openrouter.ai/api/v1"

        return api_base, api_key

    def _build_openevolve_config(self, broker_dir: Path):
        # Local import so MLS-Bench can import this module even if openevolve
        # isn't installed (lazy dependency).
        from openevolve.config import Config, LLMModelConfig, load_config

        if self._oe_config_path:
            cfg = load_config(str(self._oe_config_path))
        else:
            cfg = Config()

        # Apply overrides (shallow merge) from global_config["openevolve"]["overrides"]
        _apply_overrides(cfg, self._oe_overrides)

        # Ensure sequential evaluation (our broker serializes anyway).
        cfg.evaluator.parallel_evaluations = 1

        # Iteration count: only override YAML when CLI explicitly set it.
        if self._iterations is not None:
            cfg.max_iterations = self._iterations
        # Reflect the resolved iteration count back so downstream code (test
        # budget, run_evolution arg, log line) uses the right number.
        self._iterations = int(cfg.max_iterations)
        # Test budget headroom: OpenEvolve evaluates the initial program,
        # then one candidate per iteration, then MLS-Bench re-evaluates the
        # restored best program as the explicit final submission candidate.
        needed = self._iterations + 2
        if self.tools.max_tests < needed:
            self.tools.max_tests = needed

        # Model: prefer explicit global_config["model"]. Default api_base to OpenRouter.
        model_name = self.global_config.get("model", "qwen-3.6-plus")
        api_base, api_key = self._resolve_llm_endpoint(cfg, Config())
        if not api_key:
            raise RuntimeError(
                "No API key found. Set an 'openrouter' provider key in config.yaml, "
                "or export OPENROUTER_API_KEY / OPENAI_API_KEY, or pass "
                "openevolve.overrides.api_key."
            )

        cfg.llm.api_base = api_base
        cfg.llm.api_key = api_key
        # Minimal ensemble: single model unless user overrode via config file
        if not cfg.llm.models:
            cfg.llm.models = [LLMModelConfig(name=model_name, weight=1.0)]
        if not cfg.llm.evaluator_models:
            cfg.llm.evaluator_models = [LLMModelConfig(name=model_name, weight=1.0)]
        # CLI --model always wins over the yaml's model list.  The yaml may
        # enumerate weights/order for an ensemble; we keep the list shape but
        # rewrite the first entry's name so the user's CLI choice routes.
        if model_name and cfg.llm.models:
            cfg.llm.models[0].name = model_name
        if model_name and cfg.llm.evaluator_models:
            cfg.llm.evaluator_models[0].name = model_name
        # Always overwrite per-model endpoint with the resolved one — load_config
        # push-down from the openevolve YAML may have populated an outdated
        # api_base (e.g. openrouter) that our provider resolver has superseded.
        for m in list(cfg.llm.models) + list(cfg.llm.evaluator_models):
            m.api_base = api_base
            m.api_key = api_key

        # Broker env var picked up by the generated evaluator file
        os.environ["MLSBENCH_EVO_BROKER_DIR"] = str(broker_dir)
        return cfg

    def _write_evaluator_file(self, run_dir: Path) -> Path:
        path = run_dir / "mlsbench_evaluator.py"
        path.write_text(_EVALUATOR_TEMPLATE)
        return path

    def _write_initial_program(self, run_dir: Path) -> Path:
        self._refresh_slot_end()
        body = self._slot_content()
        wrapped = f"{EVOLVE_START}\n{body}\n{EVOLVE_END}\n"
        path = run_dir / "initial_program.py"
        path.write_text(wrapped)
        return path

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, resume: bool = False) -> dict:
        if resume:
            # Resume is not yet supported for OpenEvolve; advertise that clearly
            # instead of silently running fresh.
            print("[openevolve-agent] resume not supported yet; starting fresh")

        self.setup_workspace()

        # Load score_spec.py — the broker uses it for the reward, so OpenEvolve
        # actually optimizes the benchmark scalar instead of an arbitrary
        # alphabetical-first metric.
        self._load_score_spec_safely()

        # Per-run dir for OpenEvolve state + broker
        run_dir = Path(self.logger.log_dir) / "openevolve"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._tokens_log = run_dir / "tokens.jsonl"
        # Reset tokens.jsonl at run start so totals don't accumulate across
        # re-runs against the same log dir (replay-from-JSONL needs a clean file).
        if not resume and self._tokens_log.exists():
            try:
                self._tokens_log.unlink()
            except Exception:
                pass

        # Reset slot end to match the as-copied workspace file
        self._refresh_slot_end()

        # Log synthetic initial prompt for audit parity with InteractiveAgent
        initial_prompt = self.build_initial_prompt()
        self.logger.reset()
        self.logger.log_initial_prompt(initial_prompt)
        self.logger._append({"role": "_meta", "exp_name": self.tools.exp_name})

        broker_dir = self._start_broker(run_dir)
        evaluator_path = self._write_evaluator_file(run_dir)
        initial_program_path = self._write_initial_program(run_dir)

        # Install token observer. Also wire the env-var fallback so that
        # OpenEvolve's ProcessPoolExecutor workers (which lose the in-process
        # observer on spawn) still append usage records to the same log.
        from openevolve.llm import openai as oe_openai
        oe_openai.set_token_observer(self._token_observer)
        os.environ["MLSBENCH_OE_TOKENS_LOG"] = str(self._tokens_log)

        broker_stopped = True
        try:
            from openevolve.api import run_evolution

            oe_config = self._build_openevolve_config(broker_dir)
            print(f"[openevolve-agent] launching {self._iterations} iterations "
                  f"with model={oe_config.llm.models[0].name}")
            result = run_evolution(
                initial_program=str(initial_program_path),
                evaluator=str(evaluator_path),
                config=oe_config,
                iterations=self._iterations,
                output_dir=str(run_dir / "evolve_output"),
                cleanup=False,
            )
            best_code = result.best_code if result else ""
            best_metrics = result.metrics if result else {}
        except Exception:
            traceback.print_exc()
            best_code = ""
            best_metrics = {}
        finally:
            broker_stopped = self._stop_broker()
            try:
                oe_openai.set_token_observer(None)
            except Exception:
                pass
            try:
                del os.environ["MLSBENCH_OE_TOKENS_LOG"]
            except KeyError:
                pass

        # Restore best candidate and run one final all-seeds evaluation.
        # If a best program is available, submit exactly the history entry
        # created by this post-restore test; do not accidentally finalize the
        # last broker rollout.
        final_submit_n: int | None = None
        best_candidate_available = False
        if broker_stopped and best_code and not self.tools.done:
            block = _strip_evolve_markers(best_code)
            if block.strip():
                best_candidate_available = True
                self._refresh_slot_end()
                edit_result = self.tools.edit(
                    op="replace",
                    filename=self._slot.filename,
                    content=block,
                    start_line=self._slot.start_line,
                    end_line=self._slot.end_line,
                )
                if isinstance(edit_result, str) and edit_result.startswith("ERROR"):
                    print(f"[openevolve-agent] best-code edit rejected: {edit_result[:200]}")
                    continue_final_eval = False
                else:
                    continue_final_eval = True
                    self._slot.end_line = self._slot.start_line + len(block.splitlines()) - 1

                if continue_final_eval:
                    if self.tools.test_count < self.tools.max_tests:
                        before_history = len(self.tools._test_history)
                        try:
                            self.tools.test()  # post-best-code evaluation
                        except Exception:
                            traceback.print_exc()
                        else:
                            if len(self.tools._test_history) > before_history:
                                final_submit_n = len(self.tools._test_history)
                    else:
                        print("[openevolve-agent] no test budget left for post-best-code final eval")

        if (
            broker_stopped
            and not self.tools.done
            and final_submit_n is None
            and not best_candidate_available
            and self.tools._test_history
        ):
            final_submit_n = len(self.tools._test_history)

        if broker_stopped and not self.tools.done and final_submit_n is not None:
            try:
                self.tools.submit(n=final_submit_n, _force=True)
            except Exception:
                traceback.print_exc()

        if broker_stopped:
            self.tools.record_zero_if_no_finals()
        else:
            print("[openevolve-agent] skipped final zero-recording because broker is still active")

        # Replay tokens.jsonl as ground truth — the in-process self._token_totals
        # is unreliable when OpenEvolve dispatches LLM calls from
        # ProcessPoolExecutor workers (each worker has its own copy of
        # _token_totals; only the env-var fallback writes to the JSONL). The
        # JSONL is shared via filesystem so it captures every call.
        totals = self._totals_from_jsonl()
        if not totals.get("calls"):
            # Fallback to in-memory totals if JSONL missing for any reason.
            totals = dict(self._token_totals)
        print(f"[openevolve-agent] token totals: {totals}")
        print(f"[openevolve-agent] best OpenEvolve score: "
              f"{best_metrics.get('combined_score', 'n/a')}")

        self._token_totals = dict(totals)
        oe_config_snapshot = {
            "config_path": str(self._oe_config_path) if self._oe_config_path else None,
            "overrides": dict(self._oe_overrides or {}),
            "iterations": self._iterations,
        }
        self._write_run_summary(extra={
            "openevolve_best": best_metrics,
            "iter_counter": self._iter_counter,
            "openevolve_config": oe_config_snapshot,
        })

        return {
            "steps": self._iter_counter,
            "tests": self.tools.test_count,
            "done": self.tools.done,
            "tokens": totals,
            "openevolve_best": best_metrics,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_EVOLVE_BLOCK_RE = re.compile(
    r"#\s*EVOLVE-BLOCK-START\s*\n(.*?)\n\s*#\s*EVOLVE-BLOCK-END",
    re.DOTALL,
)


def _strip_evolve_markers(code: str) -> str:
    """Extract the content between the first EVOLVE-BLOCK-START/END markers.

    If markers aren't present, return the code unchanged (OpenEvolve may emit
    un-wrapped programs if the initial program wasn't wrapped).
    """
    m = _EVOLVE_BLOCK_RE.search(code)
    if m:
        return m.group(1).rstrip("\n")
    return code.rstrip("\n")


def _apply_overrides(cfg, overrides: dict) -> None:
    """Shallow dotted-key override (e.g. 'database.num_islands': 1).

    Keeps the implementation minimal; users who want deep config control should
    pass a full YAML file via ``openevolve.config_path``.
    """
    for dotted, value in (overrides or {}).items():
        parts = dotted.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)


# Evaluator template written to disk at runtime. It is re-imported in each
# OpenEvolve worker subprocess, so it cannot reference anything from the main
# process. Communicates with the main process via ``MLSBENCH_EVO_BROKER_DIR``.
_EVALUATOR_TEMPLATE = textwrap.dedent('''\
    """MLS-Bench evaluator bridge for OpenEvolve.

    Writes a request JSON into the broker requests/ dir and polls the
    responses/ dir for a matching result. The main MLS-Bench agent process
    runs the actual edit+test pipeline and posts the response.
    """
    import json
    import os
    import time
    import uuid
    from pathlib import Path

    BROKER_DIR = Path(os.environ["MLSBENCH_EVO_BROKER_DIR"])
    REQ_DIR = BROKER_DIR / "requests"
    RES_DIR = BROKER_DIR / "responses"
    REQ_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    POLL_SECS = float(os.environ.get("MLSBENCH_EVO_POLL_SECS", "0.5"))
    TIMEOUT_SECS = float(os.environ.get("MLSBENCH_EVO_TIMEOUT_SECS", "7200"))


    def evaluate(program_path):
        rid = uuid.uuid4().hex
        with open(program_path, "r") as f:
            code = f.read()
        payload = json.dumps({"id": rid, "program_path": str(program_path), "code": code})
        tmp = REQ_DIR / f".{rid}.tmp"
        tmp.write_text(payload)
        req = REQ_DIR / f"{rid}.json"
        tmp.replace(req)

        res = RES_DIR / f"{rid}.json"
        start = time.time()
        while not res.exists():
            if time.time() - start > TIMEOUT_SECS:
                try:
                    req.unlink()
                except FileNotFoundError:
                    pass
                return {"combined_score": -1e9, "error": "mlsbench broker timeout"}
            time.sleep(POLL_SECS)
        try:
            data = json.loads(res.read_text())
        except Exception as exc:
            return {"combined_score": -1e9, "error": f"mlsbench broker parse: {exc}"}
        try:
            res.unlink()
        except FileNotFoundError:
            pass
        return data.get("metrics") or {"combined_score": -1e9, "error": "empty response"}
''')
