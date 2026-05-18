"""Greedy chain baseline — simple sequential reasoning with no backtracking.

Replaces the editable region (search method) in custom_search.py.
This is identical to the default template implementation and equivalent
to the CoT@1 strategy in StableToolBench.
"""

_FILE = "stabletoolbench/toolbench/inference/Algorithms/custom_search.py"

_GREEDY_CHAIN = """\
    def search(self, root_node):
        \"\"\"Greedy chain: follow one path, no backtracking.\"\"\"
        now_node = root_node
        for step in range(self.single_chain_max_step):
            if self.query_count >= self.max_query_count:
                break
            if len(self.terminal_node) >= self.answer_count:
                break

            new_leaves = self._step(now_node)
            if not new_leaves:
                break

            now_node = new_leaves[-1]

            if now_node.is_terminal:
                self.status = 1
                self.terminal_node.append(now_node)
                break

            if now_node.pruned:
                if now_node.observation_code == 4:
                    self.give_up_node.append(now_node)
                break

            if now_node.get_depth() >= self.single_chain_max_step:
                now_node.pruned = True
                break
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 368,
        "end_line": 439,
        "content": _GREEDY_CHAIN,
    },
]
