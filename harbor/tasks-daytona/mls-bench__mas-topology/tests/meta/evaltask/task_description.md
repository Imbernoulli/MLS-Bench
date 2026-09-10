# Multi-Agent Collaboration Topology Design

## Research Question
Design a novel multi-agent collaboration topology that maximizes the quality of LLM-generated code. Implement a `generate_topology(node_num)` function that returns directed edges forming a DAG. Agents are organized according to your topology: each agent receives a predecessor's solution, reviews it, and produces an improved version; when a node has multiple predecessors, solutions are aggregated. The topology determines the balance between depth, diversity, and synthesis.

## Background
MacNet (Qian et al., "Scaling Large-Language-Model-based Multi-Agent Collaboration", ICLR 2025, arXiv:2406.07155, code at https://github.com/OpenBMB/ChatDev) organizes LLM agents as nodes in a DAG; the topology orchestrates their interactive reasoning. The MacNet paper shows that:
- collaboration scales effectively to 1000+ agents;
- irregular topologies can outperform regular ones;
- overall performance follows a logistic growth pattern as agents scale ("collaborative scaling law").

Reference topology families (the typical baselines):
- **Chain** (`0→1→2→…→N-1`) — deep iterative refinement, no diversity.
- **Star** — all source agents feed a single hub — broad parallel exploration, no depth.
- **Layered (MLP-like)** — partition nodes into layers, fully connect adjacent layers — balances breadth and depth.

## What you can modify
The `generate_topology` function in `chatdev-macnet/custom_topology.py`:

```python
def generate_topology(node_num: int) -> list[tuple[int, int]]:
    """Return directed edges (source, target) forming a DAG over nodes 0..node_num-1."""
```

### Constraints
- Must return a valid DAG (no cycles).
- All nodes `0` to `node_num-1` must be reachable from the input sentinel.
- Edges should respect topological order (lower-numbered → higher-numbered is the safest convention).
- The MacNet runtime automatically adds an input sentinel (`-1`) connecting to source nodes and an output sentinel (`-2`) connecting from sink nodes.
- The topology is deterministic in `node_num`; cross-seed variability comes from LLM API responses.

The underlying LLM backbones, prompts, aggregation machinery, and evaluators are fixed.

## Reference baselines
- `chain` — `0→1→…→N-1`.
- `star` — all nodes feed into one hub.
- `layered` — MLP-like layered DAG.

## Fixed Pipeline / Evaluation
Each topology is evaluated with **4 agent nodes** across three settings (2 benchmarks × different MacNet backbone LLMs):

| # | Benchmark | MacNet backbone | label |
|---|-----------|----------------|-------|
| 1 | HumanEval (33 problems) | deepseek-chat | `humaneval-4-deepseek` |
| 2 | HumanEval (33 problems) | qwen2.5-72b-instruct | `humaneval-4-qwen` |
| 3 | SRDD (20 prompts) | deepseek-chat | `srdd-4-deepseek` |

### Metrics
- **`pass_at_1_deepseek` / `pass_at_1_qwen`** (HumanEval, higher is better) — fraction of problems whose generated code passes all unit tests on the first attempt.
- **`srdd_exec_rate`** (SRDD, higher is better) — fraction of generated software projects whose entry point (`main.py`) executes without crashing (exit code 0 / running at timeout / no Traceback).

The SRDD prompts come from the SRDD (Software Requirement Description Dataset) released with ChatDev (Qian et al., "ChatDev: Communicative Agents for Software Development", arXiv:2307.07924); 20 prompts are sampled across the 5 SRDD categories. A good topology should generalize across all three settings rather than over-specializing to one model or benchmark.

### Network requirement
This task requires internet access at runtime to call LLM APIs. Set both `DEEPSEEK_API_KEY` (for `humaneval-4-deepseek` and `srdd-4-deepseek`) and `QWEN_API_KEY` (for `humaneval-4-qwen` via DashScope) before running. Not compatible with offline / air-gapped compute nodes.
