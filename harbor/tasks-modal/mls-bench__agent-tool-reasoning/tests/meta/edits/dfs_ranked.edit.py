"""DFS with LLM ranking baseline.

Replaces the editable region (search method) in custom_search.py.
Generates `tree_beam_size` children per node, ranks them using LLM
pairwise comparison, and expands the best first. Backtracks on failure.
Equivalent to the DFS (with_filter=True) strategy in StableToolBench.
"""

_FILE = "stabletoolbench/toolbench/inference/Algorithms/custom_search.py"

_DFS_RANKED = """\
    def search(self, root_node):
        \"\"\"DFS with LLM ranking: expand best child first, backtrack on failure.\"\"\"
        self._dfs(root_node)

    def _dfs(self, now_node):
        \"\"\"Recursive DFS. Returns number of levels to backtrack.\"\"\"
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

        # Generate beam_size children
        candidates = []
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
            candidates.append(new_leaves[-1])

        if not candidates:
            return 1

        # Rank candidates using LLM pairwise comparison
        if len(candidates) > 1:
            scores = self._rank_nodes(candidates)
            for score, node in zip(scores, candidates):
                node.prior_score = score
            candidates.sort(key=lambda x: x.prior_score, reverse=True)

        # Expand best candidates in order
        for cand in candidates:
            result = self._dfs(cand)
            if len(self.terminal_node) >= self.answer_count:
                return 10000
            elif result > 1:
                now_node.make_finish(2)
                return result - 1

        return 1
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 368,
        "end_line": 439,
        "content": _DFS_RANKED,
    },
]
