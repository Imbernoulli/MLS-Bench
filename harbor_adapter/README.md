# MLS-Bench → Harbor adapter

Converts the 140 [MLS-Bench](https://github.com/Bohan22/MLS-Bench) tasks into
[Harbor](https://github.com/harbor-framework/harbor) tasks so any Harbor agent
(`claude-code`, `openhands`, `codex`, `terminus-2`, …) can be evaluated against
the MLS-Bench problem set.

## Quick start

```bash
# 1. Render the 140 tasks (CPU-only; this is fast)
cd harbor_adapter
PYTHONPATH=src python3 -m mls_bench.main \
    --output-dir ./datasets/mls-bench \
    --mls-bench-root /path/to/MLS-Bench    # default: auto-detect from cwd

# 2. Run any Harbor agent against the rendered dataset.
#    `run_mls-bench.yaml` configures the GPU-capable env + oracle agent;
#    swap `agents:` for any other Harbor agent.
harbor run -c run_mls-bench.yaml
```

To run the rendered tasks on Daytona (the default config selects the 138
non-API tasks):

```bash
uv tool install "harbor[daytona]"
export DAYTONA_API_KEY="<your-daytona-key>"
PYTHONPATH=src harbor run -c run-daytona.yaml
```

`run-daytona.yaml` uses `mls_bench.harbor_env:DaytonaEnvironment`. The
adapter recognises its GPU-only Docker Compose overlays and sends those GPU
counts to Daytona's direct sandbox resource request; this avoids Daytona's
unsupported GPU+DinD combination while leaving CPU-only real multi-container
tasks untouched (Daytona rejects GPU+Compose). The provider key must be set in the host environment and is never
embedded in task files. Daytona account limits still apply: the smoke runner
caps CPU sandboxes at 10 GiB and allows a larger disk override for GPU
sandboxes; images that cannot be built by the provider are retained as
explicit errors in the CSV report.

The only two tasks whose benchmark commands call external model APIs are
`agent-tool-reasoning` and `mas-topology`; supply their DeepSeek/DashScope
credentials separately if evaluating them. They are not used as Daytona smoke
tests.

For isolated environment checks, run the helper in the sibling `harbor`
directory (from the repository root):

```bash
DAYTONA_API_KEY="<your-daytona-key>" \
  python harbor/scripts/daytona_smoke.py --scope environment
```

This selects one representative (GPU-preferred) task for each of the 63
remaining package environments and launches a fresh Harbor process/sandbox for
each. `--scope task` instead covers all 138 non-API tasks. The default uses a
`nop` agent with verification disabled; add `--verify --agent oracle` for full
verifier runs. A CSV report is written under `harbor/jobs-daytona-smoke/`.

Per-package base images are already on Docker Hub
(`bohanlyu2022/mlsbench-harbor-<pkg>:latest`, 65 packages). The per-task
Dockerfile each rendered task ships is a 5-line layer on top of those.

## Architecture

Two-tier image build + verifier-side pristine baseline:

1. **Per-package harbor base** — `bohanlyu2022/mlsbench-harbor-<pkg>:latest`
   layers MLS-Bench's existing `bohanlyu2022/mlsbench-<pkg>:latest` (package
   source + data deps) with package-level `pre_edit.py` applied. Built by
   `scripts/build_base_image.py`; pushed once.
2. **Per-task Dockerfile** — emitted by the adapter: `FROM <harbor-base>` +
   `COPY _scaffold/` which overlays the task's `mid_edit` create/replace
   scaffolding onto `/workspace/<pkg>/`.
3. **Pristine baseline is NOT in the image** — it ships under
   `tests/meta/pristine/<rel>` + `tests/meta/pristine_manifest.json`. Harbor
   mounts `tests/` at `/tests/` only at verify time, so a root agent cannot
   tamper with the diff baseline before scoring.

Each rendered task directory:

```
mls-bench__<task-id>/
├── task.toml                 # name, timeouts, cpus/memory/gpus
├── instruction.md            # task description + editable-range list + baseline read-only references
├── environment/
│   ├── Dockerfile            # 5-line FROM <harbor-base> + COPY _scaffold/
│   ├── _scaffold/            # mid_edit create/replace files
│   └── docker-compose.yaml   # only when gpus > 0 (per-task device reservation)
├── solution/
│   ├── solve.sh              # oracle: apply baseline_edit_ops.json then exit
│   └── baseline_edit_ops.json
└── tests/
    ├── test.sh               # PATH-hardened verifier entry
    ├── score_task.py         # mini-scheduler + edit-range guard + native scoring
    ├── meta/
    │   ├── config.json, parser.py, score_spec.py, leaderboard.csv, budget_check.py (if any)
    │   ├── pristine/<rel>             # pristine bytes of declared files
    │   └── pristine_manifest.json     # sha256 of every file under any guarded prefix
    ├── eval/scripts/*.sh     # every eval cmd (visible + hidden alike)
    └── mlsbench_src/         # verifier-only src/mlsbench/ tree
```

## Critical invariants preserved

- **Editable line ranges** — `tests/score_task.py::cmd_guard` is
  content-based, not line-number-based. Pristine is split into alternating
  fixed/editable segments per `config.json::files[].edit`; every fixed
  segment must appear verbatim in the agent's submission (rightmost-feasible
  match between surrounding anchors, robust to duplicate separators and to
  baseline edits that change line counts). Files under guarded prefixes but
  not declared are hash-checked against `pristine_manifest.json`. Violation
  → reward 0 + `/logs/verifier/violation.txt`.
- **`allow_create`** — documents whether a task expects the agent to author new
  files. The guard does not act on it: creating files is neither a violation nor
  reverted. It protects the fixed baseline against *modification* and *deletion*,
  which is what an agent cannot otherwise do. Note the asymmetry with the native
  harness, where the agent has no shell and `edit(op="create")` is refused
  outright when the flag is false.
- **`budget_check.py`** (e.g. `llm-pretrain-normalization`) runs as part of
  the eval scripts; no extra wrapping needed.
- **Hidden eval scripts** — all eval scripts (visible + hidden) live in
  `tests/eval/scripts/*.sh`, outside `/workspace`. Agent shell session
  never sees `/tests/`; Harbor uploads it at verify time only.
- **Per-task budgets** — `[environment].{cpus, memory_mb, gpus}` and
  `[agent]/[verifier].timeout_sec` are derived from `test_cmds[].compute`
  and `test_cmds[].time` in `config.json`, matching native
  `mlsbench.scheduler.peak_gpus` semantics: per-group
  `whole + ceil(fractional)` GPUs × `n_seeds`, max across groups.
- **GPU cap** — that peak is clamped to `MAX_PARALLEL_GPUS = 8`
  (`adapter.py`), because docker refuses to start a container reserving more
  devices than the host has. A group that wants more than 8 GPUs at once is
  not rejected: the verifier runs it as sequential waves of ≤ 8 GPUs (see
  below) and `[verifier].timeout_sec` charges wall clock per wave, so the
  budget covers the serialized run. `task.toml` carries a comment saying so
  whenever the cap bites. Only a single `(test_cmd, seed)` job whose own
  `compute` exceeds 8 GPUs raises the reservation above the cap — it has no
  smaller schedule. Native MLS-Bench runs are unaffected; SLURM and the
  local GPU scheduler already queue whatever does not fit.

## In-container mini-scheduler

`tests/score_task.py::cmd_run_evals` ports MLS-Bench's native parallel
scheduling (`_run_all_seeds_slurm` / `_allocate_group_gpu_assignments` /
`_partition_group_gpu_batches`) into the verifier, because Harbor's docker
env allocates one container with a static GPU reservation and doesn't itself
schedule processes across GPUs. Behavior:

- groups run sequentially in ascending order;
- within a group every `(test_cmd, seed)` pair is its own job and they launch
  in parallel via `subprocess.Popen(start_new_session=True, …)` with
  per-entry `CUDA_VISIBLE_DEVICES` (whole-GPU jobs first, fractional jobs
  packed into remaining per-GPU capacity);
- if a group's peak demand exceeds the reservation — always the case once the
  `MAX_PARALLEL_GPUS` cap bites — wave-partition into feasible batches and
  run the waves back to back;
- per-wave deadline = `max(time) + 300s`; on expiry,
  `os.killpg(pid, SIGTERM)` then `SIGKILL` after 30 s.

## GPU support

Harbor's stock `type: docker` environment refuses any task with `gpus > 0`
(it sets `EnvironmentCapabilities.gpus = False`). The adapter ships
`mls_bench.harbor_env:DockerGPUEnvironment`, a subclass that flips that flag,
loaded via Harbor's documented
`EnvironmentFactory.create_environment_from_import_path` extension point.
For each task with `gpus > 0` the adapter emits a per-task
`docker-compose.yaml` reserving nvidia devices via the standard
`deploy.resources.reservations.devices` block. No Harbor fork, no
site-packages patch.

`run_mls-bench.yaml` wires this up:

```yaml
environment:
  import_path: mls_bench.harbor_env:DockerGPUEnvironment
  force_build: true
  delete: true
```

Switch to `type: docker` if you don't need GPU and don't want to install
this adapter as a package.

### IsaacGym on Hopper needs a >= 570 user-space driver

`mls-bench__robo-humanoid-sim2real-algo` runs IsaacGym Preview 4, whose
`libPhysXGpu_64.so` tops out at an `sm_80` cubin plus PTX targeting `sm_86`.
On `sm_90` the driver has to JIT that PTX at `create_sim`, and user-space
driver branches older than 570 segfault doing it — a bare `Segmentation fault`
with no python traceback, scored 0 (issue #59).

Only the *user-space* driver matters; the host kernel driver can stay on
535/550. The task's `scripts/gpu_env_preflight.sh` (rendered into
`tests/eval/scripts/`) logs a `GPU_ENV_PREFLIGHT` line with the GPU compute
capability, both driver versions and the PhysX device-code inventory, and on
`sm_90`+ with a pre-570 user-space driver it activates a CUDA
forward-compatibility stack from `/opt/mlsbench/cuda-compat` — baked into the
`humanoid-gym` base image — rolling back if the swapped `libcuda` does not load
or `cuInit` fails. Operators running an older base image can supply their own:

```bash
docker run -v /path/to/cuda-compat-12-8:/opt/mlsbench/cuda-compat ...
# or point MLSBENCH_CUDA_COMPAT_DIR at it
```

## Baseline picking & oracle solve

`solution/solve.sh` is an oracle that replays the strongest declared
baseline's `edit_ops`. The picker delegates to native
`mlsbench.scoring`:

- `BaselineAnchors(task_dir)` discovers baseline rows in `leaderboard.csv`
  (handles the `baseline:<name>` model-column convention);
- `load_expanded_spec` reads `tasks/<t>/score_spec.py` (the canonical
  per-task metric declaration);
- `score_record(spec, row, anchors)` scores each baseline's preferred row
  (`is_final + seed=mean` > `seed=mean` > any, by metric-completeness
  + timestamp);
- the baseline with the highest gmean wins.

This is identical to how MLS-Bench scores real agent submissions; no
adapter-side fuzzy column-matching, no silent fallbacks. Tasks without a
`score_spec.py` (e.g. deprecated tasks) fall back to the first declared
baseline with a warning.

## Vendor-drift safety

`pre_edit.py` and `mid_edit.py` reference upstream source by line number,
which is brittle to upstream package drift (a re-fetch off the pinned
commit can shift line numbers). The adapter parse-checks every `.py` file
written by either op pass (`_stage_task_scaffold` for mid_edit,
`_apply_pre_edit_ops` for pre_edit at base-image build time) and raises with
file:line + offending snippet + a drift hint if the post-edit file isn't
valid Python. Bugs that would otherwise ship silently and only surface at
agent runtime now fail at render/build time.

## Building base images

You only need this if you're regenerating the harbor base images (otherwise
the adapter pulls `bohanlyu2022/mlsbench-harbor-<pkg>:latest` from Hub
automatically when Harbor builds the per-task layer):

```bash
python3 scripts/build_base_image.py --all --push
# or single package:
python3 scripts/build_base_image.py --pkg Time-Series-Library --push
# skip already-pushed packages:
python3 scripts/build_base_image.py --all --skip-pkg
```

## CLI

```bash
PYTHONPATH=src python3 -m mls_bench.main \
    --output-dir ./datasets/mls-bench \
    [--task-ids t1,t2,…] [--limit N] \
    [--overwrite] [--continue-on-error] \
    [--mls-bench-root /path/to/MLS-Bench]
```

`--mls-bench-root` defaults to: `$MLS_BENCH_ROOT` → nearest ancestor of cwd
containing `tasks/` + `vendor/packages.yaml` → `~/MLS-Bench/`.

## Status & known limitations

- All 65 per-package base images are pushed to Docker Hub.
- Agent cannot run eval mid-session (eval scripts only exist in `/tests/`
  which Harbor uploads at verify time). Stricter than native MLS-Bench's
  `WorkspaceTools.test()`.
- Multi-package tasks (e.g. `tasks/llm-pretrain-attention/`) pick one
  primary package for the image; secondary packages are staged into the
  scaffold but not their full source.

## Files to read first

- `src/mls_bench/adapter.py::render_task` — per-task rendering entry point
- `src/mls_bench/task-template/tests/score_task.py::cmd_run_evals` — the
  in-container mini-scheduler
- `src/mls_bench/task-template/tests/score_task.py::cmd_guard` — the
  edit-range guard
- `src/mls_bench/harbor_env.py` — `DockerGPUEnvironment` plugin
- `scripts/build_base_image.py::build_one` — per-package harbor base build
- `tests/test_scheduler.py`, `tests/test_adapter.py`,
  `tests/test_score_task.py` — 28 unit tests
