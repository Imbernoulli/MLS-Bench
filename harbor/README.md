# MLS-Bench on Harbor

140 algorithmic ML-research tasks from [MLS-Bench](https://github.com/Bohan22/MLS-Bench),
packaged as a [Harbor](https://github.com/harbor-framework/harbor) dataset.
Any Harbor agent (`claude-code`, `codex`, `openhands`, `terminus-2`, …) can be
evaluated on the suite with a single command.

## Quick start

Prerequisites:
- [Harbor](https://github.com/harbor-framework/harbor) installed.
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  (for GPU tasks; about half the suite). CPU-only mode also works.
- ≥ 80 GB free disk for harbor base images (pulled on demand from Docker Hub).

Run the oracle agent (replays each task's strongest baseline; useful for
smoke-testing):

```bash
PYTHONPATH=. harbor run -c run.yaml
```

## Run on Daytona

```bash
uv tool install "harbor[daytona]"
export DAYTONA_API_KEY="<your-daytona-key>"
PYTHONPATH=.:../harbor_adapter/src harbor run -c run-daytona.yaml
```

`harbor_env:DaytonaEnvironment` routes all 140 tasks through Daytona's direct
GPU sandbox API, including the 116 GPU tasks whose Compose file is only a
local-Docker NVIDIA reservation. Daytona does not support GPU + DinD/Compose.

Select one task with `--path`, and an agent with `--agent`:

```bash
harbor run -c run-daytona.yaml --path tasks/mls-bench__robo-diffusion-policy \
  --agent oracle                    # strongest declared baseline
harbor run -c run-daytona.yaml --path tasks/mls-bench__TASK \
  --agent claude-code --model anthropic/claude-opus-4-7 \
  --agent-env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
```

`--agent-env` reaches the agent, `--verifier-env` only the verifier, and
`--ek KEY=VALUE` the Daytona environment. For an environment-only check use
`--agent nop --no-verifier`; the 140 tasks share 65 base images, so one task
per image covers every environment. `--path` takes one task *or* a dataset
directory and cannot be repeated — select a subset with `task_names` /
`exclude_task_names` under the config's `datasets:` block.

### The 5-hour agent budget

Every rendered task ships `[agent] timeout_sec = 18000` and Harbor enforces it
per trial, so the recommended budget needs no flag. Three things silently
change it, and a run that used any of them is not comparable to published
numbers: `--agent-timeout-multiplier X`, `--timeout-multiplier X` (which also
scales the verifier and build timeouts), and `agents: [{override_timeout_sec:
N}]` in the config. A finished run records what it used in each trial's
`result.json` under `config.agent` and `config.timeout_multiplier`. The
verifier timeout is separate and task-specific — sized per task from its own
eval cost — and is deliberately not flattened to one value.

### Options

`--ek` keys, with the defaults `run-daytona.yaml` already sets:

| Key | Default | Purpose |
| --- | --- | --- |
| `gpu_type` | `H100` | Daytona's pool also holds Blackwell cards the pinned CUDA wheels cannot run. Use `H200` only for tasks with a native `h200` profile. |
| `spot` | `false` | Spot capacity is cheaper but frequently unavailable. |
| `gpu_memory_gb` / `gpu_cpus` | `64` / `16` | Floors. Daytona enforces `task.toml` resources as hard cgroup limits, which local Docker effectively does not. verl RL validation needs `128`. |
| `eval_time_scale` | `2.0` | Multiplies the verifier's per-eval budgets for the slower remote CPUs. `1` restores native budgets. |
| `toolbox_ready_retries` | `3` | Recreates a sandbox whose toolbox never answers. |
| `snapshot_salt` | unset | Appends a no-op `RUN` layer, forcing a fresh snapshot. Use when retries keep landing on the same bad runner — Daytona prefers whichever runner already caches the snapshot, and a cached copy can be broken. |

The adapter also caps `OMP_NUM_THREADS` and friends at the CPU quota (the
container reports every host core and would otherwise oversubscribe ~20x), and
sets `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` on multi-GPU sandboxes,
where both NCCL paths fail with `cudaErrorIllegalState`.

**GPU counts come from each task's own declaration** (up to 8). Size
`--n-concurrent` against those counts, not against the trial count. Do not
lower them with `--override-gpus`: that changes the parallelism the task was
calibrated for. Reserve `--override-*` for an organization whose per-sandbox
limits genuinely cannot hold the request; CPU sandboxes cap at 8 GB RAM and
10 GB disk.

H200 is defined by the native MLS-Bench configs, not by the provider layer: 15
`llm-pretrain-*`/`llm-rl-*` tasks declare validated `h200` command/compute/env
blocks in `tasks/*/config.json`. `gpu_type=H200` passes `MLSBENCH_GPU_TYPE` to
the verifier, which selects those blocks and derives the GPU reservation from
their `compute` values. No batch size, TP size, or command is invented here,
and tasks without an `h200` block are never scaled.

`mls-bench/agent-tool-reasoning` and `mls-bench/mas-topology` call DeepSeek /
DashScope during evaluation and need those keys too; `DAYTONA_API_KEY` only
authenticates the sandbox provider.

## What's in this directory

```
.
├── README.md          this file
├── run.yaml           reference Harbor config (GPU-enabled environment + oracle agent)
├── harbor_env.py      DockerGPUEnvironment — Harbor's docker env with the GPU flag flipped
└── tasks/             140 rendered task directories + dataset.toml manifest
    ├── dataset.toml
    ├── mls-bench__causal-observational-linear-gaussian/
    ├── mls-bench__ts-classification/
    └── ...
```

Each task dir is a self-contained Harbor task:

```
mls-bench__<task-id>/
├── task.toml                 budgets (cpus, memory, gpus, timeouts)
├── instruction.md            task description + editable-range list + baseline references
├── environment/
│   ├── Dockerfile            FROM bohanlyu2022/mlsbench-harbor-<pkg>:latest + scaffold COPY
│   ├── _scaffold/            mid_edit create/replace files
│   └── docker-compose.yaml   (only when gpus > 0) per-task device reservation
├── solution/                 oracle: replays the strongest baseline's edits
└── tests/                    PATH-hardened verifier + edit-range guard + native scoring
```

## What each task expects of an agent

`instruction.md` in each task spells out:

- which file(s) the agent may edit (line ranges enforced by a content-based diff guard);
- the eval commands that score the submission;
- any parameter budget (e.g. `llm-pretrain-normalization` caps parameter count at 1.05× baseline);
- a read-only excerpt of the strongest declared baselines' implementations for reference.

The agent has shell access in a container with the relevant package source
pre-staged at its workdir. Harbor uploads the verifier scripts only at scoring
time, so the eval scripts themselves stay out of the agent's view.

## Time limits

Each `task.toml` carries two independent budgets:

- **`[agent] timeout_sec`** — how long the agent may explore. Every task in
  this dataset ships **5 hours** (`18000`), the recommended budget and the one
  behind every result on the
  [MLS-Bench-Lite leaderboard](https://mls-bench.com/leaderboard). It is
  deliberately uniform: the exploration budget is part of the protocol, not a
  per-task property. To deviate, scale all tasks at once with
  `timeout_multiplier` in `run.yaml`.
- **`[verifier] timeout_sec`** — how long scoring may take. This one is
  genuinely task-specific: it is sized to each task's own training and
  evaluation cost and should be left alone.

A run whose agent budget differs substantially from 5 hours is still valid,
but its scores are not directly comparable to the published leaderboard.

## Scoring

Each task uses MLS-Bench's native `score_spec.py` declaration to compute a
single `combined_score ∈ [0, 1]` written to `/logs/verifier/reward.txt`. Per-
test-cmd raw metrics also land in `/logs/verifier/metrics.json` for analysis.

Edit-range violations short-circuit to `reward = 0` with a populated
`/logs/verifier/violation.txt`.

## GPU support

Tasks declare per-task GPU requirements in `task.toml` (`[environment].gpus`)
and ship a `docker-compose.yaml` reserving nvidia devices when needed. Harbor
merges that compose file with its base; you just need NVIDIA Container Toolkit
on the host. The `DockerGPUEnvironment` in `harbor_env.py` is what lets Harbor
accept the `gpus > 0` declaration in the first place — the stock `docker`
environment refuses it.

If your host has fewer GPUs than the task requests, Harbor will fail the task
at container start. Run with `--limit` or `--task-ids` to subset.

## License

MLS-Bench tasks: see the upstream MLS-Bench repository for license. Harbor:
see the upstream Harbor repository.
