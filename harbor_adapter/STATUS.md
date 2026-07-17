# MLS-Bench → Harbor adapter — status

Branch: `harbor-adapter` (this branch). Last updated: 2026-05-13.

## What this branch does

Publishes MLS-Bench's 140 ML-research tasks as a [Harbor](https://github.com/harbor-framework/harbor)
dataset. Once the per-package base images are on Docker Hub and the dataset
is rendered, any Harbor-supported agent (`claude-code`, `codex`,
`terminus-2`, `openhands`, `aider`, …) can run the suite via:

```
harbor run -p <dataset>/<task-dir> -a <agent> -m <model>
```

End-to-end already validated against Harbor v0.6.6 with the `oracle` agent
on `causal-observational-linear-gaussian`: oracle applies the `pc`
baseline, reward = 0.026, metrics byte-identical to MLS-Bench's
`baseline:pc` leaderboard row across all 5 eval cmds including the hidden
ER20-Noisy one. Hostile out-of-range solve.sh → reward 0 + violation file
populated.

## Architecture

Two-tier images + verifier-side pristine baseline:

1. **Per-package harbor base**
   `bohanlyu2022/mlsbench-harbor-<pkg>:latest`. Built once by
   `harbor_adapter/scripts/build_base_image.py`. Layers the existing
   `bohanlyu2022/mlsbench-<pkg>:latest` (which carries the package source
   + data deps from MLS-Bench's native image set) with package-level
   `pre_edit.py` applied. Pushed to Docker Hub.

2. **Per-task Dockerfile** — emitted by `harbor_adapter/src/mls_bench/adapter.py`
   into each rendered task dir. 1-line `FROM <harbor base>` + a single
   `COPY _scaffold/` line that overlays the task's `mid_edit` scaffolding
   onto `/workspace/<pkg>/`. Built locally by `harbor run`.

3. **Pristine baseline** for the edit-range guard is **not** in the image.
   The adapter ships per-task pristines under `tests/meta/pristine/<rel>`
   (full bytes for declared files; needed for content-based fixed-segment
   matching) plus `tests/meta/pristine_manifest.json` (sha256 of every
   file under any guarded prefix). Harbor uploads `tests/` to `/tests/`
   only at verify time, so a root agent cannot tamper with the baseline.

4. **Edit-range guard** (`tests/score_task.py::cmd_guard`) is
   content-based, not line-number-based: pristine is split into
   alternating fixed/editable segments per `config.json::files[].edit`,
   then every fixed segment is anchored verbatim in the agent's
   submission (rightmost feasible occurrence between surrounding
   anchors, robust to duplicate separators). Files under guarded
   prefixes but not declared are hash-checked against the manifest;
   missing manifest → fail closed.

5. **Hidden eval scripts** — all eval scripts (visible + hidden) live in
   `tests/eval/scripts/*.sh`. Agent shell never sees `tests/` during the
   work session (Harbor uploads it only at verify time). `instruction.md`
   lists labels for all `test_cmds` since native MLS-Bench's default
   (`hide_hidden=False`) returns hidden metrics to the agent.

6. **In-container mini-scheduler** (`tests/score_task.py::cmd_run_evals`)
   ports MLS-Bench's native parallel-scheduling logic into the Harbor
   container. Harbor's docker env is not GPU-native (it allocates one
   container per trial with a static GPU reservation; nothing
   schedules processes across GPUs inside the container), so we ported
   the relevant pieces of `mlsbench.scheduler.peak_gpus` and
   `mlsbench.agent.tools._run_all_cmds_direct` /
   `_allocate_group_gpu_assignments` / `_partition_group_gpu_batches`
   into the verifier-side Python. Behavior:
   - seeds run sequentially (outer loop)
   - groups within a seed run sequentially in ascending order
   - within a group, test_cmds launch in parallel via
     `subprocess.Popen(start_new_session=True, …)` with a per-entry
     `CUDA_VISIBLE_DEVICES` assignment (whole-GPU jobs first, then
     pack fractional jobs into remaining per-GPU capacity)
   - if a group's peak demand exceeds the reservation, wave-partition
     into `_partition_group_gpu_batches`-style batches
   - if any single test_cmd needs more GPUs than reserved, fail fast
     for that entry only (rc=125); other entries still run
   - per-wave deadline = `max(time) + 300s`; on expiry,
     `os.killpg(pid, SIGTERM)` then `SIGKILL` after a 30s grace
   - budget checks run serially before each group's parallel launch;
     a failing budget check excludes that entry only

7. **DockerGPUEnvironment plugin** (`mls_bench.harbor_env:
   DockerGPUEnvironment`) is a subclass of Harbor's stock
   `DockerEnvironment` that flips `capabilities.gpus = True`.
   Loaded via Harbor's documented
   `EnvironmentFactory.create_environment_from_import_path` extension
   point (no Harbor fork, no site-packages patch). For each task with
   `gpus > 0`, the adapter emits an `environment/docker-compose.yaml`
   reserving nvidia devices via the standard
   `deploy.resources.reservations.devices` block. `count` matches the
   per-task peak from the new `_resources()`.

## Where files come from

| Adapter file | Source |
|---|---|
| `task.toml.j2` → `task.toml` | per-task budgets from `tasks/<t>/config.json::test_cmds[].time`, package + data resource lookup |
| `instruction.md.j2` → `instruction.md` | `tasks/<t>/task_description.md` + every baseline's editable region as read-only reference (mirrors native `base.py:425-498`) |
| `environment/Dockerfile.j2` | 1-line `FROM` + scaffold COPY |
| `environment/_scaffold/` | `tasks/<t>/edits/mid_edit.py` (create + replace + insert + delete ops) applied to materialized package source |
| `solution/solve.sh.j2` | Strongest baseline from `tasks/<t>/leaderboard.csv`, replays its `edit_ops` from `tasks/<t>/edits/<name>.edit.py` |
| `tests/meta/*` | per-task verifier metadata (parser.py, score_spec.py, leaderboard.csv, budget_check.py if present) |
| `tests/eval/scripts/*` | every `tasks/<t>/scripts/*.sh` |
| `tests/mlsbench_src/` | `src/mlsbench/` Python tree (verifier-only, never in image) |
| `tests/meta/pristine/` | declared-file bytes from materialized package source |
| `tests/meta/pristine_manifest.json` | sha256 of every file under any guarded prefix |

## Security posture

| Threat | Mitigation |
|---|---|
| Agent reads eval script content | Scripts live in `tests/`, uploaded at verify time only |
| Agent edits non-editable lines | Content-based guard in `score_task.py::cmd_guard` |
| Agent overwrites pristine baseline | Pristine lives in `tests/`, not in image |
| Agent shadows `python3` via PATH | `test.sh` resets PATH + unsets `PYTHON*`/`LD_*` envs + invokes `${PYTHON_BIN}` with `-I` (isolated) |
| Agent plants `sitecustomize.py` | `PYTHONNOUSERSITE=1` + `-I` flag |
| Agent replaces `/usr/bin/python3` | Logged via sha256 audit to `/logs/verifier/python_audit.txt`; cannot prevent without Harbor `[verifier_environment]` feature |
| Hostile config.json points eval at `../../workspace/payload.sh` | `_safe_join` rejects path traversal in `cmd_run_evals` |

Same caveat as PostTrainBench's adapter (PR #8): a root agent that
`pip install`s a tampered package the verifier imports is currently not
defeatable. Tracked upstream as Harbor's `[verifier_environment]`
feature.

## Build progress (as of last update)

Image build:`harbor_adapter/scripts/build_base_image.py --all --push`
is running in the background, building all 65 per-package harbor base
images. Current state:

- ✅ **All 65 harbor base images pushed to Docker Hub** under
  `bohanlyu2022/mlsbench-harbor-<pkg>:latest`.
- The `dbim-codebase` data dir was originally 463 GB because a 400 GB
  scratch `datasets/DIODE/` (raw tarball extract used only as input to a
  one-time preprocessing step that produces the 4.1 GB `DIODE-256/`)
  shipped under `vendor/data/dbim_data/`. After deleting it, the actual
  data is 33 GB and the image is ~34 GB — well within Hub's per-layer
  limit. `dbim-codebase` is on the Hub like the rest.
- One MLS-Bench config bug fixed during the build:
  `vendor/pkg_configs/dLLM-cache/config.json` had its
  `dlm-dkv-hf-datasets` data_dep `host_path` pointing at a 6-byte
  sentinel file (`.dlm-dkv-datasets-ready`) rather than the actual
  `datasets/` directory. Repointed at the directory and moved the
  sentinel under `ready_files`.

Dataset render: validated end-to-end against the live build at the time
of validation. After fresh build completes, re-render with:

```
cd harbor_adapter && PYTHONPATH=src python3 -m mls_bench.main \
  --output-dir datasets/mls-bench \
  --mls-bench-root <mls-bench-checkout> --overwrite
```

Expect `Generated 140/140 tasks; 0 failed`.

## Open questions for review

1. **Should the agent be able to run eval mid-session?**
   PostTrainBench bakes `evaluate.py` into the agent's workspace and
   lets it iterate. Frontier-CS drops iteration entirely (single-shot
   `.cpp` submission). MLS-Bench's native `WorkspaceTools.test()` is
   somewhere between. Current adapter behavior: **agent cannot run any
   eval during its session** (eval scripts only exist in `/tests/`
   which is uploaded at verify time). This is stricter than native MLS.

2. (Resolved) `dbim-codebase`'s 463 GB was a stale 400 GB
   preprocessing scratch (`datasets/DIODE/` — raw tarball extract used
   only as input to a one-time `_preprocess_diode` step that emits the
   4.1 GB `DIODE-256/`). Deleted; real ready_files total ~30 GB; the
   harbor base image is ~34 GB and now on Hub.

3. **Multi-package tasks** (`tasks/llm-pretrain-attention/` etc.) —
   each task currently picks ONE primary package for the image
   (`test_cmds[0].package`). Secondary packages are staged into the
   scaffold but not their full source. For tasks that genuinely need
   multiple package envs simultaneously, this may need richer handling.

4. **Hidden test_cmds visibility in `instruction.md`** — currently lists
   labels for ALL test_cmds since native default (`hide_hidden=False`)
   exposes them. If we want to operate under `hide_hidden=True`
   semantics for Harbor, adapter needs a flag.

## Files to read first

- `harbor_adapter/README.md` — short overview + critical invariants
- `harbor_adapter/src/mls_bench/adapter.py::render_task` — entry point
  for per-task rendering (line ~702)
- `harbor_adapter/src/mls_bench/adapter.py::_resources` — ports
  native `peak_gpus` (per-group whole+ceil(fractional), max across
  groups) — replaces the old `gpus = 1 if use_cuda else 0`
- `harbor_adapter/src/mls_bench/task-template/tests/score_task.py::cmd_run_evals`
  — the in-container mini-scheduler (seeds outer, group-sequential,
  within-group parallel, first-fit GPU packing, wave fallback,
  per-wave timeout with SIGTERM→SIGKILL)
- `harbor_adapter/src/mls_bench/task-template/tests/score_task.py::cmd_guard`
  — the edit-range guard
- `harbor_adapter/src/mls_bench/harbor_env.py` — the
  `DockerGPUEnvironment` plugin (loaded via Harbor's
  `--environment-import-path` extension point)
- `harbor_adapter/scripts/build_base_image.py::build_one` — per-package
  harbor base build
- `harbor_adapter/tests/test_scheduler.py` — 7 scheduler unit tests
  (4 GPU-allocation fixtures + budget-fail + timeout + oversized-compute)

## Validation done so far

- AST OK across every `.py` under `harbor_adapter/`
- Full render: 140 / 140 tasks, 0 failures
- E2E with real Harbor binary (v0.6.6) on the oracle agent:
  - `causal-observational-linear-gaussian` → reward 0.026, metrics
    byte-match `baseline:pc` leaderboard row across all 5 eval cmds
    (visible + hidden ER20-Noisy)
  - `ml-clustering-algorithm` → reward 0.388, metrics byte-match
    `baseline:kmeans` seed-42 row (< 0.5 because adapter's
    `_pick_strongest_baseline` chose kmeans while `mlsbench.scoring`'s
    bounded_power reference for this task is a different baseline)
  - `ml-anomaly-detection` → reward 0.5016, metrics byte-match
    `baseline:isolation_forest` seed-42 row (≈ 0.5 because oracle's
    pick happens to coincide with `mlsbench.scoring`'s ref baseline;
    `ref_score=0.5` anchors the strongest baseline at 0.5 by design)
  - Hostile out-of-range solve.sh → reward 0, violation file populated
- Codex review against the diff: 9 substantive findings (1 critical,
  3 high, 5 medium), all addressed in this branch.

## Real bugs caught during validation

- **`test.sh` used `/usr/bin/python3`** as verifier interpreter. On
  pytorch-based base images that's a bare Debian python with no numpy
  / torch / pandas, so `budget_check.py` crashed
  (`ModuleNotFoundError: numpy`) and oracle returned reward 0 with "no
  metrics extracted from logs". Fixed by walking
  `/opt/conda/bin/python3` → `/opt/miniconda3/bin/python3` →
  `/usr/local/bin/python3` → `/usr/bin/python3` and adding the chosen
  interpreter's bin dir back onto PATH so eval scripts that spawn
  `python` inherit the right one. `-I` (isolated mode) still hardens
  the verifier from agent-planted PYTHON* envs and user site.
- **Harbor `[environment].gpus` is metadata only**: it doesn't attach
  the nvidia runtime to the container. Tasks declared with
  `use_cuda=true` now ship an `environment/docker-compose.yaml`
  reserving GPU devices, which Harbor merges over its base compose
  via `harbor/environments/docker/docker.py:292`.
