"""DFSDT baseline — depth-first search with decision tree (no ranking).

Replaces the editable region (search method) in custom_search.py.
Generates one child, immediately recurses. Backtracks `back_length`
steps on pruning or terminal nodes. Adds diversity prompts when
re-expanding a node. Equivalent to the DFSDT (with_filter=False)
strategy in StableToolBench.
"""

_FILE = "stabletoolbench/toolbench/inference/Algorithms/custom_search.py"

_DFSDT = """\
    def search(self, root_node):
        \"\"\"DFSDT: generate one child, recurse immediately, backtrack on failure.\"\"\"
        self._dfsdt(root_node)

    def _dfsdt(self, now_node):
        \"\"\"Recursive DFSDT. Returns number of levels to backtrack.\"\"\"
        final_answer_back_length = 2
        prune_back_length = 2

        now_node.expand_num = self.now_expand_num
        self.now_expand_num += 1

        # Base cases
        if now_node.get_depth() >= self.single_chain_max_step or now_node.pruned or now_node.is_terminal:
            if now_node.is_terminal:
                self.status = 1
                self.terminal_node.append(now_node)
                return final_answer_back_length
            else:
                now_node.pruned = True
                if now_node.observation_code == 4:
                    self.give_up_node.append(now_node)
                    return prune_back_length
                return 1

        # Try beam_size times (each time generates one child and recurses)
        for i in range(self.tree_beam_size):
            if self.query_count >= self.max_query_count:
                return 100000

            # Add diversity prompt if node already has children
            added_diversity = self._add_diversity_prompt(now_node)

            new_leaves = self._step(now_node)

            # Mark diversity message as invalid
            if added_diversity:
                now_node.messages[-1]["valid"] = False

            if not new_leaves:
                continue

            leaf = new_leaves[-1]

            # Immediately recurse (no ranking)
            result = self._dfsdt(leaf)
            if len(self.terminal_node) >= self.answer_count:
                return 10000
            elif result > 1:
                return result - 1

        return 1
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 368,
        "end_line": 439,
        "content": _DFSDT,
    },
]
