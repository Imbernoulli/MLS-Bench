# MLS-Bench: mas-topology

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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/chatdev-macnet/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `chatdev-macnet/custom_topology.py`
- editable lines **16–42**


Other files you may **read** for context (do not modify):
- `chatdev-macnet/generate_graph.py`
- `chatdev-macnet/graph.py`


## Readable Context


### `chatdev-macnet/custom_topology.py`  [EDITABLE — lines 16–42 only]

```python
     1: """Custom multi-agent collaboration topology.
     2: 
     3: This module defines the DAG topology for multi-agent code generation.
     4: The function generate_topology(node_num) returns a list of directed edges
     5: that determine how LLM agents collaborate to produce code.
     6: 
     7: The system will automatically add:
     8:   - An input sentinel node (-1) connecting to all source nodes (no predecessors)
     9:   - An output sentinel node (-2) connecting from all sink nodes (no successors)
    10: """
    11: 
    12: 
    13: # ── Editable topology function ───────────────────────────────────────
    14: # EDITABLE REGION START
    15: 
    16: def generate_topology(node_num: int) -> list[tuple[int, int]]:
    17:     """Design the multi-agent collaboration topology.
    18: 
    19:     Given N agent nodes (numbered 0 to node_num-1), return a list of
    20:     directed edges (source, target) forming a DAG. The graph will
    21:     automatically get input/output sentinel nodes (-1, -2) added.
    22: 
    23:     Constraints:
    24:     - Must form a valid DAG (no cycles)
    25:     - All nodes 0..node_num-1 should be reachable from at least one path
    26:     - Edges must go from lower-indexed to higher-indexed nodes
    27: 
    28:     Args:
    29:         node_num: Number of agent nodes (e.g., 4, 8, 16)
    30: 
    31:     Returns:
    32:         edges: List of (source, target) tuples
    33:     """
    34:     # Default: chain topology (sequential pipeline)
    35:     # Each agent improves upon the previous agent's solution.
    36:     # 0 -> 1 -> 2 -> ... -> (node_num - 1)
    37:     edges = []
    38:     for i in range(node_num - 1):
    39:         edges.append((i, i + 1))
    40:     return edges
    41: 
    42: # EDITABLE REGION END
    43: # ── End of editable region ───────────────────────────────────────────
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `chain` baseline — editable region  [READ-ONLY — reference implementation]

In `chatdev-macnet/custom_topology.py`:

```python
Lines 16–42:
    13: # ── Editable topology function ───────────────────────────────────────
    14: # EDITABLE REGION START
    15: 
    16: def generate_topology(node_num: int) -> list[tuple[int, int]]:
    17:     """Design the multi-agent collaboration topology.
    18: 
    19:     Given N agent nodes (numbered 0 to node_num-1), return a list of
    20:     directed edges (source, target) forming a DAG. The graph will
    21:     automatically get input/output sentinel nodes (-1, -2) added.
    22: 
    23:     Constraints:
    24:     - Must form a valid DAG (no cycles)
    25:     - All nodes 0..node_num-1 should be reachable from at least one path
    26:     - Edges must go from lower-indexed to higher-indexed nodes
    27: 
    28:     Args:
    29:         node_num: Number of agent nodes (e.g., 4, 8, 16)
    30: 
    31:     Returns:
    32:         edges: List of (source, target) tuples
    33:     """
    34:     # Chain topology: sequential pipeline
    35:     # Each agent improves upon the previous agent's solution.
    36:     # 0 -> 1 -> 2 -> ... -> (node_num - 1)
    37:     edges = []
    38:     for i in range(node_num - 1):
    39:         edges.append((i, i + 1))
    40:     return edges
    41: 
    42: # EDITABLE REGION END
    43: # ── End of editable region ───────────────────────────────────────────
```

### `star` baseline — editable region  [READ-ONLY — reference implementation]

In `chatdev-macnet/custom_topology.py`:

```python
Lines 16–42:
    13: # ── Editable topology function ───────────────────────────────────────
    14: # EDITABLE REGION START
    15: 
    16: def generate_topology(node_num: int) -> list[tuple[int, int]]:
    17:     """Design the multi-agent collaboration topology.
    18: 
    19:     Given N agent nodes (numbered 0 to node_num-1), return a list of
    20:     directed edges (source, target) forming a DAG. The graph will
    21:     automatically get input/output sentinel nodes (-1, -2) added.
    22: 
    23:     Constraints:
    24:     - Must form a valid DAG (no cycles)
    25:     - All nodes 0..node_num-1 should be reachable from at least one path
    26:     - Edges must go from lower-indexed to higher-indexed nodes
    27: 
    28:     Args:
    29:         node_num: Number of agent nodes (e.g., 4, 8, 16)
    30: 
    31:     Returns:
    32:         edges: List of (source, target) tuples
    33:     """
    34:     # Star topology: hub-and-spoke
    35:     # Node 0 broadcasts to all other nodes in parallel.
    36:     # Good for generating diverse solutions simultaneously.
    37:     edges = []
    38:     for i in range(1, node_num):
    39:         edges.append((0, i))
    40:     return edges
    41: 
    42: # EDITABLE REGION END
    43: # ── End of editable region ───────────────────────────────────────────
```

### `layered` baseline — editable region  [READ-ONLY — reference implementation]

In `chatdev-macnet/custom_topology.py`:

```python
Lines 16–60:
    13: # ── Editable topology function ───────────────────────────────────────
    14: # EDITABLE REGION START
    15: 
    16: def generate_topology(node_num: int) -> list[tuple[int, int]]:
    17:     """Design the multi-agent collaboration topology.
    18: 
    19:     Given N agent nodes (numbered 0 to node_num-1), return a list of
    20:     directed edges (source, target) forming a DAG. The graph will
    21:     automatically get input/output sentinel nodes (-1, -2) added.
    22: 
    23:     Constraints:
    24:     - Must form a valid DAG (no cycles)
    25:     - All nodes 0..node_num-1 should be reachable from at least one path
    26:     - Edges must go from lower-indexed to higher-indexed nodes
    27: 
    28:     Args:
    29:         node_num: Number of agent nodes (e.g., 4, 8, 16)
    30: 
    31:     Returns:
    32:         edges: List of (source, target) tuples
    33:     """
    34:     # Layered (MLP-like) topology
    35:     # Split nodes into layers, fully connect adjacent layers.
    36:     # Balances depth (iterative refinement) and width (diversity).
    37:     import math
    38:     layer_num = max(2, int(math.log(node_num, 2)))
    39:     layers = [node_num // layer_num] * layer_num
    40:     # Distribute remainder to first layer
    41:     layers[0] += node_num % layer_num
    42: 
    43:     # Compute start/end indices for each layer
    44:     start_ids = []
    45:     end_ids = []
    46:     cur = 0
    47:     for size in layers:
    48:         start_ids.append(cur)
    49:         end_ids.append(cur + size)
    50:         cur += size
    51: 
    52:     # Fully connect adjacent layers
    53:     edges = []
    54:     for i in range(len(layers) - 1):
    55:         for u in range(start_ids[i], end_ids[i]):
    56:             for v in range(start_ids[i + 1], end_ids[i + 1]):
    57:                 edges.append((u, v))
    58:     return edges
    59: 
    60: # EDITABLE REGION END
    61: # ── End of editable region ───────────────────────────────────────────
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
