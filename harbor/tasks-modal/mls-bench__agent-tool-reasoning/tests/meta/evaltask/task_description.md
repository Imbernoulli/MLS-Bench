# LLM Agent Tool-Use Reasoning Strategy

## Research Question
Design a better search/reasoning strategy for an LLM-based tool-use agent on multi-step API tasks. The strategy controls how the agent explores the action space (which tool to call next, when to backtrack, when to give up) and trades off task success against the number of LLM queries spent.

## Background
StableToolBench (Guo et al., 2024, arXiv:2403.07714) is a stabilized version of ToolBench (Qin et al., 2023, arXiv:2307.16789, the ToolLLM paper). It evaluates LLM agents on multi-step tool use over RapidAPI tools, replacing unstable real APIs with a virtual API server (cache + simulator) and a GPT-4-based judge that produces a Solvable Pass Rate / Stable Pass Rate. Given a user query and a set of tool APIs, the agent decides which tools to call, with what arguments, and in what order to arrive at a final answer.

## Fixed Pipeline
- Benchmark subset, tool environment (virtual API server), agent backbones, and answer judge are all fixed and must not be modified.
- The agent backbones include both DeepSeek and Qwen models; the same `search()` policy is run across all backbones.
- Datasets, prompts, and per-call decoding parameters are fixed.

## What you can modify
The `search(self, root_node)` method in `custom_search.py`. You have access to:

- `self._step(node)` — one LLM call + tool execution; returns new leaf nodes.
- `self._add_diversity_prompt(node)` — encourages different actions when re-expanding.
- `self._rank_nodes(candidates)` — LLM pairwise ranking (costs extra queries).
- Tree state: `self.query_count`, `self.max_query_count`, `self.terminal_node`, etc.
- Node properties: `node.is_terminal`, `node.pruned`, `node.observation_code`, `node.get_depth()`.

## Reference baselines (algorithmic templates)
- **Greedy chain (CoT/ReAct-style)**: call LLM, execute tool, repeat. No backtracking.
- **DFS with ranking**: generate multiple children, use LLM to rank them, expand best first; backtracks on failure (extra LLM calls for ranking).
- **DFSDT** (Qin et al., ToolLLM, 2023): generate one child, recurse depth-first; on failure or "Finish by Giving Up", backtrack a fixed number of steps and expand a new node.

## Evaluation
Per-task feedback reports:
- **pass_rate** — fraction of queries with a valid final answer (higher is better).
- **avg_queries** — average LLM queries per task (lower is better, efficiency signal).
- **give_up_rate** — fraction of queries where the agent gives up (lower is better).

The score emphasizes answer quality (pass rate / Stable Pass Rate from the GPT-4 judge); query count and give-up rate serve as efficiency and diagnostic signals. The same `search()` policy is evaluated across multiple agent backbones on the I1-instruction subset.
