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

Replace the agent with a real one by editing `run.yaml` or via CLI:

```bash
PYTHONPATH=. harbor run -c run.yaml -a claude-code -m anthropic/claude-opus-4-7
PYTHONPATH=. harbor run -c run.yaml -a codex       -m openai/gpt-5
```

Pick specific tasks:

```bash
PYTHONPATH=. harbor run -c run.yaml -p tasks/mls-bench__causal-observational-linear-gaussian
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
