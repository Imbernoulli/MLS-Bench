"""CustomSearch: Editable search algorithm for StableToolBench.

This module implements a search strategy for LLM-based tool-use agents.
The agent receives a user query and a set of tool APIs, then must decide
which tools to call, with what arguments, and in what order.

The `search()` method is the editable region — modify it to implement
your own search/reasoning strategy (e.g., beam search, MCTS, adaptive
backtracking, best-first search, iterative deepening, etc.).

Helper methods `_step()` and `_rank_nodes()` are provided and should
NOT be modified.
"""

import re
import json
import os, random; random.seed(int(os.environ.get("SEED", "42")))
from copy import deepcopy

from Tree.Tree import my_tree, tree_node
from Prompts.ReAct_prompts import (
    FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION,
    FORMAT_INSTRUCTIONS_USER_FUNCTION,
)
from Prompts.Tree_search_prompts import DIVERSITY_PROMPT
from Algorithms.base_search import base_search_method
from LLM_rank.rank_candidate import sum_based_rankn, rank2_subfix


class CustomSearch(base_search_method):
    """A configurable tree-search strategy for tool-use reasoning.

    Parameters
    ----------
    llm : ChatGPTFunction
        The LLM interface for generating thoughts and actions.
    io_func : rapidapi_wrapper
        The environment interface for executing tool calls.
    process_id : int
        Process identifier for logging (0 = verbose).
    callbacks : list
        Optional callbacks for tracing.
    """

    def __init__(self, llm, io_func, process_id=0, callbacks=None):
        super().__init__(llm, io_func, process_id, callbacks)
        self.io_func = io_func
        self.llm = _ledger(self, llm).llm  # counts every LLM call (trusted)
        self.process_id = process_id
        self.callbacks = callbacks if callbacks is not None else []
        self.restart()

    def restart(self):
        self.status = 0
        self.terminal_node = []
        self.give_up_node = []
        self.now_expand_num = 0
        self.query_count = 0
        self.total_tokens = 0

    # ------------------------------------------------------------------
    # JSON serialization (do NOT modify)
    # ------------------------------------------------------------------
    def to_json(self, answer=False, process=True):
        if process:
            json_obj = {
                "win": _trusted_win(self),
                "tree": self.tree.to_json_recursive(),
                "forward_args": self.forward_args,
                "compare_candidates": [],
            }
            for node in _answer_nodes(self):
                if not node.pruned:
                    json_obj["compare_candidates"].append(
                        node.get_chain_result_from_this_node(use_messages=False)
                    )
        else:
            json_obj = {}

        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "query_count": _ledger(self).llm_calls,
                "total_tokens": self.total_tokens,
                "final_answer": "",
                "finish_type": "give_answer",
                "function": self.io_func.functions,
                "chain": [],
            }
            for node in _answer_nodes(self):
                if not node.pruned:
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_answer"
                    json_obj["answer_generation"]["final_answer"] = node.description
                    json_obj["answer_generation"]["train_messages"] = (
                        node.get_train_messages_from_this_node()
                    )
                    break
            if not json_obj["answer_generation"]["valid_data"]:
                if len(self.give_up_node) > 0:
                    pick = self.give_up_node[random.randint(0, len(self.give_up_node) - 1)]
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_up"
                    json_obj["answer_generation"]["final_answer"] = pick.description
                    json_obj["answer_generation"]["train_messages"] = (
                        pick.get_train_messages_from_this_node()
                    )
        return json_obj

    # ------------------------------------------------------------------
    # Entry point (do NOT modify)
    # ------------------------------------------------------------------
    def start(self, single_chain_max_step, tree_beam_size=1,
              max_query_count=60, answer=1, with_filter=True):
        """Initialize the search tree and launch the search.

        Parameters
        ----------
        single_chain_max_step : int
            Maximum depth (in tree nodes) before pruning a chain.
        tree_beam_size : int
            Number of children to expand per node.
        max_query_count : int
            Budget of LLM queries before the search terminates.
        answer : int
            Stop after finding this many terminal (give_answer) nodes.
        with_filter : bool
            Legacy flag for DFS vs DFSDT. Can be repurposed.
        """
        self.forward_args = {
            "single_chain_max_step": single_chain_max_step,
            "tree_beam_size": tree_beam_size,
            "max_query_count": max_query_count,
            "answer": answer,
            "with_filter": with_filter,
        }
        self.single_chain_max_step = single_chain_max_step
        self.tree_beam_size = tree_beam_size
        self.max_query_count = _ledger(self).open_budget(max_query_count)
        self.answer_count = answer

        # Build the root node
        self.tree = my_tree()
        self.tree.root.node_type = "Action Input"
        self.tree.root.io_state = deepcopy(self.io_func)

        system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION.replace(
            "{task_description}", self.io_func.task_description
        )
        self.tree.root.messages.append({"role": "system", "content": system})

        user = FORMAT_INSTRUCTIONS_USER_FUNCTION.replace(
            "{input_description}", self.io_func.input_description
        )
        self.tree.root.messages.append({"role": "user", "content": user})

        # Run the search
        self.search(self.tree.root)

        return _finalize(self)

    # ------------------------------------------------------------------
    # Helper: single LLM step (do NOT modify)
    # ------------------------------------------------------------------
    def _step(self, now_node):
        """Perform one LLM call from *now_node*, execute any tool calls,
        and return a list of new leaf nodes created.

        The returned leaves are 'Action Input' nodes (with observations)
        or 'Thought' nodes (if the LLM produced no tool call).

        Side-effects
        -------------
        - Increments ``self.query_count`` and ``self.total_tokens``.
        - Appends children to *now_node*.
        - Sets ``is_terminal`` / ``pruned`` / ``observation_code`` on new nodes.

        Returns
        -------
        list[tree_node]
            The deepest new leaf nodes produced by this step.
        """
        if _ledger(self).exhausted(): return []  # budget spent: no LLM call
        new_message, error_code, total_tokens = _ledger(self).chat(now_node.messages,
            self.io_func.functions, process_id=self.process_id
        )
        new_message = {k: v for k, v in new_message.items() if v is not None}
        self.query_count = _ledger(self).llm_calls
        self.total_tokens += total_tokens

        if _ledger(self).exhausted():
            return []

        assert new_message["role"] == "assistant"

        temp_now_node = now_node

        # --- Thought node ---
        if "content" in new_message and new_message["content"] is not None:
            temp_node = tree_node()
            temp_node.node_type = "Thought"
            temp_node.description = new_message["content"]
            child_io_state = deepcopy(temp_now_node.io_state)
            child_io_state.retriever = None
            temp_node.io_state = child_io_state
            temp_node.is_terminal = child_io_state.check_success() != 0
            temp_node.messages = deepcopy(temp_now_node.messages)
            temp_node.father = temp_now_node
            temp_now_node.children.append(temp_node)
            temp_node.print(self.process_id)
            temp_now_node = temp_node

            if error_code != 0:
                temp_now_node.observation_code = error_code
                temp_now_node.pruned = True

        # --- Tool-call nodes ---
        if (
            "tool_calls" in new_message
            and new_message["tool_calls"] is not None
            and len(new_message["tool_calls"]) > 0
        ):
            tool_calls = new_message["tool_calls"]
            if self.process_id == 0:
                print("number of parallel calls:", len(tool_calls))

            for i in range(len(tool_calls)):
                function_name = tool_calls[i]["function"]["name"]
                temp_node = tree_node()
                temp_node.node_type = "Action"
                temp_node.description = function_name
                child_io_state = deepcopy(temp_now_node.io_state)
                child_io_state.retriever = None
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = deepcopy(temp_now_node.messages)
                temp_node.father = temp_now_node
                temp_now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                temp_now_node = temp_node

                function_input = tool_calls[i]["function"]["arguments"]
                temp_node = tree_node()
                temp_node.node_type = "Action Input"
                temp_node.description = function_input
                child_io_state = deepcopy(temp_now_node.io_state)
                child_io_state.retriever = None

                observation, status = child_io_state.step(
                    action_name=temp_now_node.description,
                    action_input=function_input,
                )
                temp_node.observation = observation
                temp_node.observation_code = status

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = deepcopy(temp_now_node.messages)
                temp_node.father = temp_now_node
                temp_now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                temp_now_node = temp_node

                if status != 0:
                    if status == 4:
                        temp_now_node.pruned = True
                    elif status == 1:  # hallucination api name
                        assert (
                            "tool_calls" in new_message
                            and len(new_message["tool_calls"]) > 0
                        )
                        tool_calls[i]["function"]["name"] = (
                            "invalid_hallucination_function_name"
                        )
                    elif status == 3:  # final answer
                        temp_now_node.is_terminal = True
                        temp_now_node.make_finish(2)

                if i == 0:
                    temp_now_node.messages.append(new_message)
                if temp_now_node.node_type == "Action Input":
                    temp_now_node.messages.append(
                        {
                            "role": "tool",
                            "name": tool_calls[i]["function"]["name"],
                            "content": temp_now_node.observation,
                            "tool_call_id": tool_calls[i]["id"],
                        }
                    )
        else:
            temp_now_node.messages.append(new_message)

        return [_ledger(self).record_step(now_node, temp_now_node)]

    # ------------------------------------------------------------------
    # Helper: add diversity prompt (do NOT modify)
    # ------------------------------------------------------------------
    def _add_diversity_prompt(self, node):
        """If *node* already has children, append a diversity prompt to
        encourage the LLM to try a different action. Returns True if a
        diversity message was appended (caller should mark it invalid
        after the LLM call)."""
        if len(node.children) == 0:
            return False

        former_candidates_des = ""
        js_list = []
        for child in node.children:
            temp_node = child
            while (
                not temp_node.is_terminal
                and temp_node.node_type != "Action Input"
                and len(temp_node.children) > 0
            ):
                temp_node = temp_node.children[0]
            if temp_node.node_type == "Action Input":
                obj_dict = {
                    "name": temp_node.father.description,
                    "arguments": temp_node.description,
                    "function_output": temp_node.observation,
                    "mento-carlo-action-value": temp_node.compute_weight(),
                }
                js_list.append(obj_dict)

        if len(js_list) > 0:
            former_candidates_des += f"{json.dumps(js_list, indent=2)}\n"
            if node.observation != "":
                former_candidates_des += (
                    f"again, your former observation: {node.observation}\n"
                )
            diverse_prompt = DIVERSITY_PROMPT.replace(
                "{previous_candidate}", former_candidates_des
            )
            node.messages.append({"role": "user", "content": diverse_prompt})
            return True
        return False

    # ------------------------------------------------------------------
    # Helper: LLM-based pairwise ranking (do NOT modify)
    # ------------------------------------------------------------------
    def _rank_nodes(self, candidates):
        """Rank a list of candidate nodes using LLM pairwise comparison.

        Returns
        -------
        list[float]
            Scores for each candidate (higher is better).
        """
        if len(candidates) <= 1:
            return [0.0] * len(candidates)

        LLM_rank_args = {
            "functions": self.io_func.functions,
            "process_id": self.process_id,
            "task_description": self.io_func.task_description,
            "rank_func": rank2_subfix,
        }
        scores, rank_query_count, total_tokens = sum_based_rankn(
            _ledger(self).llm, LLM_rank_args=LLM_rank_args, candidates=candidates
        )
        self.query_count = _ledger(self).llm_calls
        self.total_tokens += total_tokens
        return scores

    # ==================================================================
    # EDITABLE REGION START
    # ==================================================================
    def search(self, root_node):
        """Core search logic. Modify this method to implement your strategy.

        Available helpers (do NOT modify them, just call them):
        --------------------------------------------------------
        self._step(node) -> list[tree_node]
            One LLM call + tool execution from *node*.
            Returns list of new leaf nodes (usually length 1).
            Returns [] if query budget is exhausted.

        self._add_diversity_prompt(node) -> bool
            Appends a diversity prompt if node already has children.
            Returns True if prompt was added (mark it invalid after step).

        self._rank_nodes(candidates) -> list[float]
            LLM pairwise ranking of candidate nodes. Costs extra queries.

        Available state:
        ----------------
        self.query_count       LLM queries used so far (read-only mirror).
        self.max_query_count   Budget limit (_step enforces its own copy).
        self.terminal_node     Register answer nodes here (verified, see end of file).
        self.give_up_node      List of nodes that gave up.
        self.status            Informational; recomputed after search().
        self.answer_count      Stop after this many answers.
        self.single_chain_max_step   Max tree depth before pruning.
        self.tree_beam_size    Number of children per expansion.

        Node properties:
        ----------------
        node.is_terminal       True if the node produced a final answer.
        node.pruned            True if the node is pruned (dead end).
        node.observation_code  Status code (0=ok, 1=hallucination, 3=finish, 4=give_up).
        node.get_depth()       Depth in the tree.
        node.children          List of child nodes.
        node.messages          OpenAI message history up to this node.

        Default implementation: simple greedy chain (no backtracking).
        """
        now_node = root_node
        for step in range(self.single_chain_max_step):
            # Check budget
            if self.query_count >= self.max_query_count:
                break

            # Check if we already have enough answers
            if len(self.terminal_node) >= self.answer_count:
                break

            # Take one step
            new_leaves = self._step(now_node)
            if not new_leaves:
                break

            now_node = new_leaves[-1]

            # Check if the new node is terminal (gave a final answer)
            if now_node.is_terminal:
                self.status = 1
                self.terminal_node.append(now_node)
                break

            # Check if the new node is pruned (dead end / give up)
            if now_node.pruned:
                if now_node.observation_code == 4:
                    self.give_up_node.append(now_node)
                break

            # Check depth limit
            if now_node.get_depth() >= self.single_chain_max_step:
                now_node.pruned = True
                break
    # ==================================================================
    # EDITABLE REGION END
    # ==================================================================


# ======================================================================
# Trusted accounting (do NOT modify -- fixed code, outside the class and
# outside the editable region)
# ----------------------------------------------------------------------
# The query budget and the pass/fail outcome live here, not in attributes
# that search() can rewrite:
#   * Every LLM call made through ``self.llm`` (by _step, _rank_nodes or by
#     search() itself) is counted by _CountingLLM. _step and _rank_nodes use
#     the ledger's LLM, and _step enforces the budget from the ledger (it
#     makes no LLM call and returns [] once the budget is spent), so rewriting
#     ``self.llm``, ``self.query_count`` or ``self.max_query_count`` has no
#     effect; ``self.query_count`` is only a mirror of the ledger count.
#   * "win" is derived from the trajectory: a node that search() registered
#     in ``self.terminal_node`` counts only if _step created it from a
#     ``Finish`` call with ``return_type="give_answer"`` that the environment
#     accepted (status 3), and it is not pruned. ``self.status`` is recomputed
#     from this after search() returns; setting it inside search() does
#     nothing.
# ======================================================================
import sys as _sys
import weakref as _weakref


class _CountingLLM:
    """Forwards to the backbone LLM and counts every ``parse`` call."""

    def __init__(self, llm, ledger):
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_ledger", ledger)

    def parse(self, *args, **kwargs):
        out = self._llm.parse(*args, **kwargs)
        self._ledger.llm_calls += 1
        return out

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def __setattr__(self, name, value):
        setattr(self._llm, name, value)


class _Ledger:
    def __init__(self):
        self.llm = None
        self.llm_calls = 0
        self.max_query_count = None
        self.answer_nodes = []  # nodes _step built from an accepted give_answer

    def open_budget(self, max_query_count):
        self.max_query_count = max_query_count
        return max_query_count

    def exhausted(self):
        return self.llm_calls >= self.max_query_count

    def chat(self, messages, *args, **kwargs):
        self.llm.change_messages(messages)
        return self.llm.parse(*args, **kwargs)

    def record_step(self, start_node, leaf):
        """Record the nodes one _step call created from an accepted
        Finish(give_answer): the Finish 'Action Input' node and every node
        the same step created after it. Returns *leaf* unchanged."""
        chain = []
        node = leaf
        while node is not None and node is not start_node:
            chain.append(node)
            node = node.father
        finished = False
        for node in reversed(chain):
            if (not finished
                    and node.node_type == "Action Input"
                    and node.observation_code == 3
                    and node.father is not None
                    and node.father.node_type == "Action"
                    and node.father.description == "Finish"):
                finished = True
            if finished:
                self.answer_nodes.append(node)
        return leaf


_LEDGERS = _weakref.WeakKeyDictionary()


def _ledger(search, llm=None):
    led = _LEDGERS.get(search)
    if led is None:
        led = _LEDGERS[search] = _Ledger()
    if llm is not None:
        led.llm = _CountingLLM(llm, led)
    return led


def _answer_nodes(search):
    """Nodes registered in search.terminal_node that _step really produced
    from an accepted give_answer, in registration order."""
    genuine = {id(n) for n in _ledger(search).answer_nodes}
    return [n for n in list(search.terminal_node) if id(n) in genuine]


def _trusted_win(search):
    return any(not n.pruned for n in _answer_nodes(search))


def _finalize(search):
    led = _ledger(search)
    win = _trusted_win(search)
    if getattr(search, "status", 0) == 1 and not win:
        print("CustomSearch: search() set status=1, but no registered terminal "
              "node came from an accepted Finish(give_answer); this query is "
              "scored as a failure.", file=_sys.stderr, flush=True)
    search.status = 1 if win else 0
    search.query_count = led.llm_calls
    return search.status


# The fixed methods above must not be redefined from the editable region
# (a later ``def _step`` / ``to_json = ...`` in the class body would silently
# replace them), and the region must not add special methods.
_EDIT_REGION_FIRST_LINE = 366
_FIXED_METHODS = ("__init__", "restart", "to_json", "start", "_step",
                  "_add_diversity_prompt", "_rank_nodes")
_ALLOWED_DUNDERS = {"__module__", "__qualname__", "__doc__", "__dict__",
                    "__weakref__", "__init__", "__firstlineno__",
                    "__static_attributes__", "__annotations__"}
_THIS_FILE = (lambda: 0).__code__.co_filename


def _check_fixed_methods():
    for name in _FIXED_METHODS:
        code = getattr(CustomSearch.__dict__.get(name), "__code__", None)
        if (code is None or code.co_name != name
                or code.co_filename != _THIS_FILE
                or code.co_firstlineno >= _EDIT_REGION_FIRST_LINE):
            raise RuntimeError(
                f"CustomSearch.{name} is fixed code and must not be redefined "
                f"in the editable region")
    for name in CustomSearch.__dict__:
        if (name.startswith("__") and name.endswith("__")
                and name not in _ALLOWED_DUNDERS):
            raise RuntimeError(
                f"CustomSearch must not define {name} in the editable region")


_check_fixed_methods()
