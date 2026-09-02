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

Harbor's built-in Daytona provider is wired into the rendered tasks. Install
the provider extra, export a Daytona API key on the host, and use the
dedicated configuration:

```bash
uv tool install "harbor[daytona]"
export DAYTONA_API_KEY="<your-daytona-key>"
PYTHONPATH=. harbor run -c run-daytona.yaml
```

The repository's `harbor_env:DaytonaEnvironment` compatibility layer handles
the 116 GPU tasks whose Compose file is only a local-Docker NVIDIA reservation:
Daytona receives the declared GPU count through its direct GPU sandbox API.
The 24 CPU tasks use the same direct Daytona path. A genuine CPU-only
multi-container Compose task, if added later, can remain delegated to Harbor's
Daytona DinD implementation; Daytona does not support GPU+DinD/Compose.

Daytona resource limits are provider/account specific. The task files request
up to 4 CPUs, 16 GiB RAM, 60 GiB disk, and 8 GPUs; use Harbor's
`--override-*` flags when your Daytona organization has lower limits. The
smoke helper automatically caps CPU sandboxes at 10 GiB (the limit of the
current Daytona organization) and can use a larger disk for GPU sandboxes.
Some published base images are larger than Daytona can build within the
available limits; those environments are recorded as provider errors rather
than reported as passes.

Two tasks intentionally call model APIs during evaluation:
`mls-bench/agent-tool-reasoning` (DeepSeek/DashScope) and
`mls-bench/mas-topology` (DeepSeek/DashScope). They also require their
corresponding model API keys; `DAYTONA_API_KEY` only authenticates the sandbox
provider and is not a substitute. The remaining 138 tasks use local/offline
evaluation inputs and are suitable for a Daytona smoke run without additional
model-provider credentials.

To test environments independently, use the smoke-test helper from this
directory. Its default `--scope environment` runs one representative task per
package image (63 environments after excluding the two API tasks), with a
`nop` agent and verification disabled. Set `--concurrency N` to match the
available Daytona GPU quota (for example, `--concurrency 10`):

```bash
DAYTONA_API_KEY="<your-daytona-key>" \
  python scripts/daytona_smoke.py --scope environment --concurrency 10
```

By default `--scope environment` selects one representative task for each of
the 63 remaining package environments. With `--resource gpu`, 53 GPU
environments are selected; use `--resource cpu` for the remaining CPU
environments. Use `--scope task` to run all 138 non-API task environments one by
one. Each invocation creates its own Harbor job and Daytona sandbox; results
are written to `jobs-daytona-smoke/report.csv`. Add `--verify --agent oracle`
when the objective is a full baseline/verifier run rather than an environment
startup smoke test. `--dry-run` prints the exact commands without contacting
Daytona.

Replace the agent with a real one by editing `run.yaml` or via CLI:

```bash
PYTHONPATH=. harbor run -c run.yaml -a claude-code -m anthropic/claude-opus-4-7
PYTHONPATH=. harbor run -c run.yaml -a codex       -m openai/gpt-5
```

Pick specific tasks:

```bash
PYTHONPATH=. harbor run -c run.yaml -p tasks/mls-bench__causal-observational-linear-gaussian
```

### Single-task Oracle and Agent commands

`--path` accepts a rendered task directory, so this command runs one task only
and uses Harbor's Oracle harness (the strongest declared baseline):

```bash
export DAYTONA_API_KEY="<your-daytona-key>"
export PYTHONPATH=.
harbor run -c run-daytona.yaml \
  --path tasks/mls-bench__robo-diffusion-policy \
  --agent oracle --n-concurrent 1
```

GPU selection is explicit. Spot GPU requests default to H100 in the adapter;
request H200 only when the task has an H200-compatible configuration. For a
task that supports both, either command is valid:

```bash
# H100 Spot
harbor run -c run-daytona.yaml --path tasks/mls-bench__TASK \
  --agent oracle --ek spot=true --ek gpu_type=H100

# H200 Spot
harbor run -c run-daytona.yaml --path tasks/mls-bench__TASK \
  --agent oracle --ek spot=true --ek gpu_type=H200

# H100 on-demand (does count against the normal GPU quota)
harbor run -c run-daytona.yaml --path tasks/mls-bench__TASK \
  --agent oracle --ek spot=false --ek gpu_type=H100
```

H200 is implemented by the native MLS-Bench configs, not by a second Daytona
configuration: the 15 supported `llm-pretrain-*`/`llm-rl-*` tasks declare their
validated `h200` command/compute/env blocks in native `tasks/*/config.json`.
When `gpu_type=H200` is requested, the adapter passes `MLSBENCH_GPU_TYPE` to the
verifier, which selects those existing blocks, and derives the Daytona GPU
reservation from their existing `compute` values. No batch size, TP size, or
training command is invented by the provider layer. H100 requests retain the
original baseline values; tasks without an `h200` block are never scaled.

The standard Agent harnesses use `--agent` and `--model`; credentials and
OpenAI-compatible endpoints are passed to the Agent process with repeated
`--agent-env` options:

```bash
harbor run -c run-daytona.yaml --path tasks/mls-bench__TASK \
  --agent codex --model openai/gpt-5.6 \
  --agent-env OPENAI_API_KEY="$OPENAI_API_KEY" \
  --agent-env OPENAI_BASE_URL="https://api.openai.com/v1"

harbor run -c run-daytona.yaml --path tasks/mls-bench__TASK \
  --agent claude-code --model anthropic/claude-opus-4-7 \
  --agent-env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
```

Use `--agent-import-path package.module:AgentClass` for a custom Harbor Agent
and repeat `--agent-kwarg KEY=VALUE` for its constructor. `--agent-env` reaches
the Agent; `--verifier-env KEY=VALUE` reaches only the verifier (needed when a
verifier itself calls an API); and `--ek KEY=VALUE` (`--environment-env`) is
forwarded to Daytona (`spot`, `gpu_type`, and provider-specific options).
The task's `task.toml`, `instruction.md`, `tests/eval/scripts/`, and
`tests/meta/{parser.py,score_spec.py}` together define the Harbor harness.

For all tasks, use the quota-aware runner rather than launching an unbounded
set of Harbor processes. It queues by declared GPU count and can keep ten
available GPUs full:

```bash
python scripts/daytona_smoke.py --scope task --resource gpu \
  --gpu-limit 10 --concurrency 10 --spot --gpu-type H100 \
  --agent oracle --verify
```

`PYTHONPATH=.` is needed because `harbor_env.py` (the GPU-enabled
`DockerEnvironment` subclass) lives next to `run.yaml`. Drop it if you
replace the `environment` block in `run.yaml` with `type: docker` and run
only CPU tasks.

## What's in this directory

```
.
├── README.md          this file
├── run.yaml           reference Harbor config (GPU-enabled environment + oracle agent)
├── scripts/daytona_smoke.py  isolated per-environment Daytona smoke runner
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
