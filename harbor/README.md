# MLS-Bench on Harbor

140 algorithmic ML-research tasks from [MLS-Bench](https://github.com/Bohan22/MLS-Bench),
packaged as a [Harbor](https://github.com/harbor-framework/harbor) dataset.
Any Harbor agent (`claude-code`, `codex`, `openhands`, `terminus-2`, …) can be
evaluated on the suite with a single command.

## Quick start

Prerequisites:
- [Harbor](https://github.com/harbor-framework/harbor) installed. The bundles
  need nothing beyond stock Harbor; `harbor_env.py` is optional (see below).
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
export PYTHONPATH=.:../harbor_adapter/src
harbor run -c run-daytona-lite.yaml      # the 30 MLS-Bench-Lite tasks
harbor run -c run-daytona.yaml           # all 138 non-API tasks
```

`tasks-daytona/` is Dockerfile-only and sized for Daytona, so stock Harbor's
`daytona` environment (`--env daytona`, no `PYTHONPATH`) runs it as shipped —
that is the path the bundles are validated on. `run-daytona*.yaml` load
`harbor_env:DaytonaEnvironment` on top: it defaults `gpu_type` to H100, clamps
a request the organization refuses instead of failing the trial, recreates a
sandbox whose toolbox never answers, and exports a task's `h200` profile to
the agent's shell. Daytona does not support GPU + DinD/Compose, which is why
this variant carries no compose file.

Select one task with `--path`, and an agent with `--agent`:

```bash
harbor run -c run-daytona.yaml --path tasks-daytona/mls-bench__robo-diffusion-policy \
  --agent oracle                    # strongest declared baseline
harbor run -c run-daytona.yaml --path tasks-daytona/mls-bench__TASK \
  --agent claude-code --model anthropic/claude-opus-4-7 \
  --agent-env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
```

`--agent-env` reaches the agent, `--verifier-env` only the verifier, and
`--ek KEY=VALUE` the Daytona environment. For an environment-only check use
`--agent nop --disable-verification`; the 140 tasks share 65 base images, so one task
per image covers every environment. `--path` takes one task *or* a dataset
directory and cannot be repeated; a directory replaces the config's
`datasets:` block, exclude list included. `run-daytona-lite.yaml` selects the
30 MLS-Bench-Lite tasks by `task_names`; copy it for any other subset.

## Run on Modal

```bash
uv tool install "harbor[modal]"          # Harbor >= 0.22
modal token new                          # or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
export PYTHONPATH=.
harbor run -c run-modal-lite.yaml        # the 30 MLS-Bench-Lite tasks
harbor run -c run-modal.yaml             # all 138 non-API tasks
harbor run -e modal --path tasks-modal/mls-bench__TASK --agent oracle   # stock Harbor, no PYTHONPATH
```

`tasks-modal/` is the variant sized for Modal — the native calibration
(CPU-only tasks 8 CPUs / 32 GB), CPUs capped at Modal's 64 per sandbox
(only the four 8-GPU tasks: 96 → 64), no compose file — and stock Harbor's
`modal` environment runs it as shipped: Harbor builds the Dockerfile with
`Image.from_dockerfile`, sends `cpus`/`memory_mb` as the sandbox request and
`gpu_types[0]:gpus` (`h100:<n>`) as the GPU. `harbor_env:ModalEnvironment`
adds only `--ek gpu_type=H200` (see below). Modal's other limits: a sandbox
lives at most 24 hours (`sandbox_timeout_secs`, default and maximum 86400),
so a 5-hour agent budget plus a verifier deadline above 19 hours does not fit
in one sandbox — that is `robo-humanoid-sim2real-algo` and
`robo-diffusion-policy`, whose oracle runs are far shorter. Modal reports the
granted cores as `nproc`, and its GPU sandboxes have a large `/dev/shm`.

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
| `gpu_memory_gb` / `gpu_cpus` | unset | Opt-in floors. Sandbox CPU/RAM normally come straight from `task.toml`; set these only when a task needs more than it declares. |
| `toolbox_ready_retries` | `3` | Recreates a sandbox whose toolbox never answers. |
| `snapshot_salt` | unset | Appends a no-op `RUN` layer, forcing a fresh snapshot. Use when retries keep landing on the same bad runner — Daytona prefers whichever runner already caches the snapshot, and a cached copy can be broken. |

### Container resources

Two rendered variants ship, one per provider, because the providers' hard
constraints do not intersect cleanly and a single bundle would be right for
neither:

| | `tasks-docker/` (local Docker, `run.yaml`) | `tasks-daytona/` (Daytona, `run-daytona*.yaml`) | `tasks-modal/` (Modal, `run-modal*.yaml`) |
| --- | --- | --- | --- |
| GPU task | 12 CPUs per GPU (16 min), 64 GB (128 GB verl) — the native SLURM calibration | same | same, CPUs capped at Modal's 64 per sandbox (the four 8-GPU tasks: 96 → 64) |
| CPU-only task | 8 CPUs / 32 GB / 30 GB — native | 4 CPUs / 8 GB / 10 GB — Daytona's CPU-sandbox ceiling; stock Harbor cannot clamp, and the 22 CPU-only tasks whose oracle needs no model API key pass at full scale at this shape | 8 CPUs / 32 GB / 30 GB — native |
| `environment/docker-compose.yaml` | NVIDIA device reservation + `shm_size: 16gb` | none — stock Harbor's Daytona provider refuses GPU tasks that carry one | none — Harbor's Modal provider builds the Dockerfile directly |

Everything else — Dockerfile, `tests/`, `solution/`, `instruction.md` — is
byte-identical across the three (git stores each blob once). Both providers
treat `task.toml` as hard limits: Daytona provisions the numbers, local Docker
applies them through Compose's `deploy.resources.limits`. GPU tasks in both
variants declare `gpu_types = ["h100"]`, which newer Harbor's Daytona provider
turns into a placement constraint (the images' CUDA wheels have no Blackwell
kernels). Render with `python -m mls_bench.main --provider docker|daytona|modal`.

A provider that cannot supply the declared amount clamps to its own ceiling
and logs a warning rather than the bundle understating the need. Daytona
answers an oversized request with `CPU request 96 exceeds maximum allowed per
sandbox (16)`; the adapter reads the ceiling out of that message, clamps, and
retries, so the organization's limits (currently 16 CPUs per GPU on a GPU
sandbox and 4 CPUs / 8 GB / 10 GB on a CPU sandbox — every bundle in
`tasks-daytona/` fits, so nothing is clamped there today) are never
hard-coded here. Local Docker
refuses a `--cpus` above the host's core count, so `DockerGPUEnvironment`
clamps to it. In both cases the thread budget follows the cores actually
granted, and evals sized for the declared budget run closer to their deadline.
The one place that margin is thin: `causal-discovery-discrete` runs five
single-threaded structure learners at once and its Hailfinder eval finishes
at 83–87 % of its 59-minute deadline on a 4-CPU Daytona sandbox (two runs;
native and local Docker give it 8). A solution slower than the baseline there can time
out on Daytona while passing elsewhere.

Task images pin `OMP_NUM_THREADS` and friends to that CPU budget, because a
container reports every core the *host* has while its cgroup grants only the
task's own. Unpinned, a 4-CPU task on a 384-core host runs 192 BLAS threads and
a fixed matmul benchmark takes 34.3 s instead of 2.6 s — enough to push a
CPU-bound eval past its deadline. `tests/score_task.py` recomputes the budget
from the live cgroup at verify time and splits it across the commands a wave
runs concurrently; `MLSBENCH_LOCAL_THREADS` overrides it, as it does natively.

Multi-GPU images also set `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0`, since
both NCCL paths fail with `cudaErrorIllegalState` inside a GPU sandbox.

The `shm_size: 16gb` in the docker variant's overlay matches the
`--shm-size=16g` MLS-Bench's own docker path uses: PyTorch DataLoader workers
pass batches through `/dev/shm`, and docker's 64 MB default kills them mid-eval
with `Bus error ... out of shared memory`. `DockerGPUEnvironment` writes its
own copy of that overlay per trial (so `--ek gpu_ids=2,3` can pin devices on a
shared host) and uses it instead of the bundle's; pointing `run.yaml` at
`tasks-daytona/` therefore also works locally. Daytona sandboxes get their
GPUs from `[environment].gpus` and come with a 64 GB `/dev/shm`.

When an eval command does not exit 0 the verifier writes `eval_failures.txt`
next to `reward.txt` and repeats it in `score_error.txt` and `metrics.json`,
naming the command, its exit code and — for rc=124 — the deadline it missed,
so a harness timeout is never mistaken for a model that produced nothing.

### Using the bundles from another harness

Everything a task needs travels in its bundle; nothing in this repository's
environment classes is required to run one. Copy the variant for your
provider — `harbor/tasks-daytona/` for Daytona, `harbor/tasks-modal/` for
Modal, `harbor/tasks-docker/` for local Docker — and honour `task.toml`'s `cpus` / `memory_mb` / `storage_mb` /
`gpus` / `gpu_types` as the container's resources and the `[agent]` /
`[verifier]` timeouts; the image's `ENV` carries the thread pinning and NCCL
settings, and the docker variant's compose overlay carries the device
reservation and `/dev/shm` size. Verified: **stock Harbor 0.22.0's `daytona`
environment runs `tasks-daytona/` as it ships**, and its `docker` environment
runs the CPU-only tasks of `tasks-docker/` as they ship (it reports no GPU
capability and refuses tasks with `gpus > 0` — `harbor/harbor_env.py:
DockerGPUEnvironment` is what lifts that). H200 is opt-in
(`MLSBENCH_GPU_TYPE=H200` in the environment, see below); without it every
task runs its H100 profile.

**GPU counts come from each task's own declaration** (up to 8). Size
`--n-concurrent` against those counts, not against the trial count. Do not
lower them with `--override-gpus`: that changes the parallelism the task was
calibrated for. Reserve `--override-*` for an organization whose per-sandbox
limits genuinely cannot hold the request; CPU sandboxes cap at 8 GB RAM and
10 GB disk. A run interrupted while its snapshot builds leaves the sandbox,
and its share of the GPU quota, allocated until you delete it in the Daytona
dashboard.

H200 is defined by the native MLS-Bench configs, not by the provider layer: 15
`llm-pretrain-*`/`llm-rl-*` tasks declare validated `h200` command/compute/env
blocks in `tasks/*/config.json`. `gpu_type=H200` passes `MLSBENCH_GPU_TYPE` to
the verifier, which selects those blocks and derives the GPU reservation from
their `compute` values; the blocks' env is exported to the agent's shell too,
so exploration runs match the scored one. `run.yaml` (local Docker) takes the
same `--ek gpu_type=H200`. No batch size, TP size, or command is invented
here, and tasks without an `h200` block are never scaled. Without the switch,
H200 hardware runs the H100 profile, which is valid there.

`mls-bench/agent-tool-reasoning` and `mls-bench/mas-topology` call DeepSeek /
DashScope during evaluation and need those keys too; `DAYTONA_API_KEY` only
authenticates the sandbox provider.

## What's in this directory

```
.
├── README.md          this file
├── run.yaml           reference Harbor config (GPU-enabled environment + oracle agent)
├── run-daytona.yaml   the same run on Daytona sandboxes (138 non-API tasks)
├── run-daytona-lite.yaml  Daytona, restricted to the 30 MLS-Bench-Lite tasks
├── run-modal.yaml     the same run on Modal sandboxes (138 non-API tasks)
├── run-modal-lite.yaml    Modal, restricted to the 30 MLS-Bench-Lite tasks
├── harbor_env.py      optional: re-exports DockerGPUEnvironment / DaytonaEnvironment / ModalEnvironment from harbor_adapter
├── tasks-docker/      140 rendered bundles for local Docker (+ dataset.toml)
├── tasks-modal/       the same 140 for Modal: native shape, CPUs capped at 64, no compose file
└── tasks-daytona/     the same 140 for Daytona: Daytona CPU-sandbox shape, no compose file
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
│   └── docker-compose.yaml   tasks-docker only: shm_size 16gb, plus the NVIDIA reservation when gpus > 0 (absent in tasks-daytona / tasks-modal)
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

Tasks declare their GPU count in `task.toml` (`[environment].gpus`). Stock
Harbor's `docker` environment refuses any `gpus > 0`; `DockerGPUEnvironment`
in `harbor_env.py` lifts that and writes a per-trial compose overlay with the
NVIDIA device reservation and `shm_size: 16gb` (stock Harbor uses the
bundle's own overlay in `tasks-docker/`, which is enough for the CPU-only
tasks). You need the NVIDIA Container Toolkit on the host. `--ek gpu_ids=2,3`
pins a trial to specific devices on a shared host; a CPU request above the
host's core count is clamped with a warning.

A task that declares more GPUs than the host has fails at container start.
Subset with Harbor's `-i/--include-task-name` and `-x/--exclude-task-name`
(globs) and `-l/--n-tasks`, or with `task_names` in the config.

## License

MLS-Bench tasks: see the upstream MLS-Bench repository for license. Harbor:
see the upstream Harbor repository.
