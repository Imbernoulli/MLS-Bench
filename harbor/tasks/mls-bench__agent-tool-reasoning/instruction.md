# MLS-Bench: agent-tool-reasoning

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


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/stabletoolbench/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `stabletoolbench/toolbench/inference/Algorithms/custom_search.py`
- editable lines **368–439**


Other files you may **read** for context (do not modify):
- `stabletoolbench/toolbench/inference/Tree/Tree.py`


## Readable Context


### `stabletoolbench/toolbench/inference/Algorithms/custom_search.py`  [EDITABLE — lines 368–439 only]

```python
     1: """CustomSearch: Editable search algorithm for StableToolBench.
     2: 
     3: This module implements a search strategy for LLM-based tool-use agents.
     4: The agent receives a user query and a set of tool APIs, then must decide
     5: which tools to call, with what arguments, and in what order.
     6: 
     7: The `search()` method is the editable region — modify it to implement
     8: your own search/reasoning strategy (e.g., beam search, MCTS, adaptive
     9: backtracking, best-first search, iterative deepening, etc.).
    10: 
    11: Helper methods `_step()` and `_rank_nodes()` are provided and should
    12: NOT be modified.
    13: """
    14: 
    15: import re
    16: import json
    17: import os, random; random.seed(int(os.environ.get("SEED", "42")))
    18: from copy import deepcopy
    19: 
    20: from Tree.Tree import my_tree, tree_node
    21: from Prompts.ReAct_prompts import (
    22:     FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION,
    23:     FORMAT_INSTRUCTIONS_USER_FUNCTION,
    24: )
    25: from Prompts.Tree_search_prompts import DIVERSITY_PROMPT
    26: from Algorithms.base_search import base_search_method
    27: from LLM_rank.rank_candidate import sum_based_rankn, rank2_subfix
    28: 
    29: 
    30: class CustomSearch(base_search_method):
    31:     """A configurable tree-search strategy for tool-use reasoning.
    32: 
    33:     Parameters
    34:     ----------
    35:     llm : ChatGPTFunction
    36:         The LLM interface for generating thoughts and actions.
    37:     io_func : rapidapi_wrapper
    38:         The environment interface for executing tool calls.
    39:     process_id : int
    40:         Process identifier for logging (0 = verbose).
    41:     callbacks : list
    42:         Optional callbacks for tracing.
    43:     """
    44: 
    45:     def __init__(self, llm, io_func, process_id=0, callbacks=None):
    46:         super().__init__(llm, io_func, process_id, callbacks)
    47:         self.io_func = io_func
    48:         self.llm = llm
    49:         self.process_id = process_id
    50:         self.callbacks = callbacks if callbacks is not None else []
    51:         self.restart()
    52: 
    53:     def restart(self):
    54:         self.status = 0
    55:         self.terminal_node = []
    56:         self.give_up_node = []
    57:         self.now_expand_num = 0
    58:         self.query_count = 0
    59:         self.total_tokens = 0
    60: 
    61:     # ------------------------------------------------------------------
    62:     # JSON serialization (do NOT modify)
    63:     # ------------------------------------------------------------------
    64:     def to_json(self, answer=False, process=True):
    65:         if process:
    66:             json_obj = {
    67:                 "win": self.status == 1,
    68:                 "tree": self.tree.to_json_recursive(),
    69:                 "forward_args": self.forward_args,
    70:                 "compare_candidates": [],
    71:             }
    72:             for node in self.terminal_node:
    73:                 if not node.pruned:
    74:                     json_obj["compare_candidates"].append(
    75:                         node.get_chain_result_from_this_node(use_messages=False)
    76:                     )
    77:         else:
    78:             json_obj = {}
    79: 
    80:         if answer:
    81:             json_obj["answer_generation"] = {
    82:                 "valid_data": False,
    83:                 "query_count": self.query_count,
    84:                 "total_tokens": self.total_tokens,
    85:                 "final_answer": "",
    86:                 "finish_type": "give_answer",
    87:                 "function": self.io_func.functions,
    88:                 "chain": [],
    89:             }
    90:             for node in self.terminal_node:
    91:                 if not node.pruned:
    92:                     json_obj["answer_generation"]["valid_data"] = True
    93:                     json_obj["answer_generation"]["finish_type"] = "give_answer"
    94:                     json_obj["answer_generation"]["final_answer"] = node.description
    95:                     json_obj["answer_generation"]["train_messages"] = (
    96:                         node.get_train_messages_from_this_node()
    97:                     )
    98:                     break
    99:             if not json_obj["answer_generation"]["valid_data"]:
   100:                 if len(self.give_up_node) > 0:
   101:                     pick = self.give_up_node[random.randint(0, len(self.give_up_node) - 1)]
   102:                     json_obj["answer_generation"]["valid_data"] = True
   103:                     json_obj["answer_generation"]["finish_type"] = "give_up"
   104:                     json_obj["answer_generation"]["final_answer"] = pick.description
   105:                     json_obj["answer_generation"]["train_messages"] = (
   106:                         pick.get_train_messages_from_this_node()
   107:                     )
   108:         return json_obj
   109: 
   110:     # ------------------------------------------------------------------
   111:     # Entry point (do NOT modify)
   112:     # ------------------------------------------------------------------
   113:     def start(self, single_chain_max_step, tree_beam_size=1,
   114:               max_query_count=60, answer=1, with_filter=True):
   115:         """Initialize the search tree and launch the search.
   116: 
   117:         Parameters
   118:         ----------
   119:         single_chain_max_step : int
   120:             Maximum depth (in tree nodes) before pruning a chain.
   121:         tree_beam_size : int
   122:             Number of children to expand per node.
   123:         max_query_count : int
   124:             Budget of LLM queries before the search terminates.
   125:         answer : int
   126:             Stop after finding this many terminal (give_answer) nodes.
   127:         with_filter : bool
   128:             Legacy flag for DFS vs DFSDT. Can be repurposed.
   129:         """
   130:         self.forward_args = {
   131:             "single_chain_max_step": single_chain_max_step,
   132:             "tree_beam_size": tree_beam_size,
   133:             "max_query_count": max_query_count,
   134:             "answer": answer,
   135:             "with_filter": with_filter,
   136:         }
   137:         self.single_chain_max_step = single_chain_max_step
   138:         self.tree_beam_size = tree_beam_size
   139:         self.max_query_count = max_query_count
   140:         self.answer_count = answer
   141: 
   142:         # Build the root node
   143:         self.tree = my_tree()
   144:         self.tree.root.node_type = "Action Input"
   145:         self.tree.root.io_state = deepcopy(self.io_func)
   146: 
   147:         system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION.replace(
   148:             "{task_description}", self.io_func.task_description
   149:         )
   150:         self.tree.root.messages.append({"role": "system", "content": system})
   151: 
   152:         user = FORMAT_INSTRUCTIONS_USER_FUNCTION.replace(
   153:             "{input_description}", self.io_func.input_description
   154:         )
   155:         self.tree.root.messages.append({"role": "user", "content": user})
   156: 
   157:         # Run the search
   158:         self.search(self.tree.root)
   159: 
   160:         return 1 if self.status == 1 else 0
   161: 
   162:     # ------------------------------------------------------------------
   163:     # Helper: single LLM step (do NOT modify)
   164:     # ------------------------------------------------------------------
   165:     def _step(self, now_node):
   166:         """Perform one LLM call from *now_node*, execute any tool calls,
   167:         and return a list of new leaf nodes created.
   168: 
   169:         The returned leaves are 'Action Input' nodes (with observations)
   170:         or 'Thought' nodes (if the LLM produced no tool call).
   171: 
   172:         Side-effects
   173:         -------------
   174:         - Increments ``self.query_count`` and ``self.total_tokens``.
   175:         - Appends children to *now_node*.
   176:         - Sets ``is_terminal`` / ``pruned`` / ``observation_code`` on new nodes.
   177: 
   178:         Returns
   179:         -------
   180:         list[tree_node]
   181:             The deepest new leaf nodes produced by this step.
   182:         """
   183:         self.llm.change_messages(now_node.messages)
   184:         new_message, error_code, total_tokens = self.llm.parse(
   185:             self.io_func.functions, process_id=self.process_id
   186:         )
   187:         new_message = {k: v for k, v in new_message.items() if v is not None}
   188:         self.query_count += 1
   189:         self.total_tokens += total_tokens
   190: 
   191:         if self.query_count >= self.max_query_count:
   192:             return []
   193: 
   194:         assert new_message["role"] == "assistant"
   195: 
   196:         temp_now_node = now_node
   197: 
   198:         # --- Thought node ---
   199:         if "content" in new_message and new_message["content"] is not None:
   200:             temp_node = tree_node()
   201:             temp_node.node_type = "Thought"
   202:             temp_node.description = new_message["content"]
   203:             child_io_state = deepcopy(temp_now_node.io_state)
   204:             child_io_state.retriever = None
   205:             temp_node.io_state = child_io_state
   206:             temp_node.is_terminal = child_io_state.check_success() != 0
   207:             temp_node.messages = deepcopy(temp_now_node.messages)
   208:             temp_node.father = temp_now_node
   209:             temp_now_node.children.append(temp_node)
   210:             temp_node.print(self.process_id)
   211:             temp_now_node = temp_node
   212: 
   213:             if error_code != 0:
   214:                 temp_now_node.observation_code = error_code
   215:                 temp_now_node.pruned = True
   216: 
   217:         # --- Tool-call nodes ---
   218:         if (
   219:             "tool_calls" in new_message
   220:             and new_message["tool_calls"] is not None
   221:             and len(new_message["tool_calls"]) > 0
   222:         ):
   223:             tool_calls = new_message["tool_calls"]
   224:             if self.process_id == 0:
   225:                 print("number of parallel calls:", len(tool_calls))
   226: 
   227:             for i in range(len(tool_calls)):
   228:                 function_name = tool_calls[i]["function"]["name"]
   229:                 temp_node = tree_node()
   230:                 temp_node.node_type = "Action"
   231:                 temp_node.description = function_name
   232:                 child_io_state = deepcopy(temp_now_node.io_state)
   233:                 child_io_state.retriever = None
   234:                 temp_node.io_state = child_io_state
   235:                 temp_node.is_terminal = child_io_state.check_success() != 0
   236:                 temp_node.messages = deepcopy(temp_now_node.messages)
   237:                 temp_node.father = temp_now_node
   238:                 temp_now_node.children.append(temp_node)
   239:                 temp_node.print(self.process_id)
   240:                 temp_now_node = temp_node
   241: 
   242:                 function_input = tool_calls[i]["function"]["arguments"]
   243:                 temp_node = tree_node()
   244:                 temp_node.node_type = "Action Input"
   245:                 temp_node.description = function_input
   246:                 child_io_state = deepcopy(temp_now_node.io_state)
   247:                 child_io_state.retriever = None
   248: 
   249:                 observation, status = child_io_state.step(
   250:                     action_name=temp_now_node.description,
   251:                     action_input=function_input,
   252:                 )
   253:                 temp_node.observation = observation
   254:                 temp_node.observation_code = status
   255: 
   256:                 temp_node.io_state = child_io_state
   257:                 temp_node.is_terminal = child_io_state.check_success() != 0
   258:                 temp_node.messages = deepcopy(temp_now_node.messages)
   259:                 temp_node.father = temp_now_node
   260:                 temp_now_node.children.append(temp_node)
   261:                 temp_node.print(self.process_id)
   262:                 temp_now_node = temp_node
   263: 
   264:                 if status != 0:
   265:                     if status == 4:
   266:                         temp_now_node.pruned = True
   267:                     elif status == 1:  # hallucination api name
   268:                         assert (
   269:                             "tool_calls" in new_message
   270:                             and len(new_message["tool_calls"]) > 0
   271:                         )
   272:                         tool_calls[i]["function"]["name"] = (
   273:                             "invalid_hallucination_function_name"
   274:                         )
   275:                     elif status == 3:  # final answer
   276:                         temp_now_node.is_terminal = True
   277:                         temp_now_node.make_finish(2)
   278: 
   279:                 if i == 0:
   280:                     temp_now_node.messages.append(new_message)
   281:                 if temp_now_node.node_type == "Action Input":
   282:                     temp_now_node.messages.append(
   283:                         {
   284:                             "role": "tool",
   285:                             "name": tool_calls[i]["function"]["name"],
   286:                             "content": temp_now_node.observation,
   287:                             "tool_call_id": tool_calls[i]["id"],
   288:                         }
   289:                     )
   290:         else:
   291:             temp_now_node.messages.append(new_message)
   292: 
   293:         return [temp_now_node]
   294: 
   295:     # ------------------------------------------------------------------
   296:     # Helper: add diversity prompt (do NOT modify)
   297:     # ------------------------------------------------------------------
   298:     def _add_diversity_prompt(self, node):
   299:         """If *node* already has children, append a diversity prompt to
   300:         encourage the LLM to try a different action. Returns True if a
   301:         diversity message was appended (caller should mark it invalid
   302:         after the LLM call)."""
   303:         if len(node.children) == 0:
   304:             return False
   305: 
   306:         former_candidates_des = ""
   307:         js_list = []
   308:         for child in node.children:
   309:             temp_node = child
   310:             while (
   311:                 not temp_node.is_terminal
   312:                 and temp_node.node_type != "Action Input"
   313:                 and len(temp_node.children) > 0
   314:             ):
   315:                 temp_node = temp_node.children[0]
   316:             if temp_node.node_type == "Action Input":
   317:                 obj_dict = {
   318:                     "name": temp_node.father.description,
   319:                     "arguments": temp_node.description,
   320:                     "function_output": temp_node.observation,
   321:                     "mento-carlo-action-value": temp_node.compute_weight(),
   322:                 }
   323:                 js_list.append(obj_dict)
   324: 
   325:         if len(js_list) > 0:
   326:             former_candidates_des += f"{json.dumps(js_list, indent=2)}\n"
   327:             if node.observation != "":
   328:                 former_candidates_des += (
   329:                     f"again, your former observation: {node.observation}\n"
   330:                 )
   331:             diverse_prompt = DIVERSITY_PROMPT.replace(
   332:                 "{previous_candidate}", former_candidates_des
   333:             )
   334:             node.messages.append({"role": "user", "content": diverse_prompt})
   335:             return True
   336:         return False
   337: 
   338:     # ------------------------------------------------------------------
   339:     # Helper: LLM-based pairwise ranking (do NOT modify)
   340:     # ------------------------------------------------------------------
   341:     def _rank_nodes(self, candidates):
   342:         """Rank a list of candidate nodes using LLM pairwise comparison.
   343: 
   344:         Returns
   345:         -------
   346:         list[float]
   347:             Scores for each candidate (higher is better).
   348:         """
   349:         if len(candidates) <= 1:
   350:             return [0.0] * len(candidates)
   351: 
   352:         LLM_rank_args = {
   353:             "functions": self.io_func.functions,
   354:             "process_id": self.process_id,
   355:             "task_description": self.io_func.task_description,
   356:             "rank_func": rank2_subfix,
   357:         }
   358:         scores, rank_query_count, total_tokens = sum_based_rankn(
   359:             self.llm, LLM_rank_args=LLM_rank_args, candidates=candidates
   360:         )
   361:         self.query_count += rank_query_count
   362:         self.total_tokens += total_tokens
   363:         return scores
   364: 
   365:     # ==================================================================
   366:     # EDITABLE REGION START
   367:     # ==================================================================
   368:     def search(self, root_node):
   369:         """Core search logic. Modify this method to implement your strategy.
   370: 
   371:         Available helpers (do NOT modify them, just call them):
   372:         --------------------------------------------------------
   373:         self._step(node) -> list[tree_node]
   374:             One LLM call + tool execution from *node*.
   375:             Returns list of new leaf nodes (usually length 1).
   376:             Returns [] if query budget is exhausted.
   377: 
   378:         self._add_diversity_prompt(node) -> bool
   379:             Appends a diversity prompt if node already has children.
   380:             Returns True if prompt was added (mark it invalid after step).
   381: 
   382:         self._rank_nodes(candidates) -> list[float]
   383:             LLM pairwise ranking of candidate nodes. Costs extra queries.
   384: 
   385:         Available state:
   386:         ----------------
   387:         self.query_count       Current number of LLM queries used.
   388:         self.max_query_count   Budget limit.
   389:         self.terminal_node     List of nodes that produced a final answer.
   390:         self.give_up_node      List of nodes that gave up.
   391:         self.status            Set to 1 when a valid answer is found.
   392:         self.answer_count      Stop after this many answers.
   393:         self.single_chain_max_step   Max tree depth before pruning.
   394:         self.tree_beam_size    Number of children per expansion.
   395: 
   396:         Node properties:
   397:         ----------------
   398:         node.is_terminal       True if the node produced a final answer.
   399:         node.pruned            True if the node is pruned (dead end).
   400:         node.observation_code  Status code (0=ok, 1=hallucination, 3=finish, 4=give_up).
   401:         node.get_depth()       Depth in the tree.
   402:         node.children          List of child nodes.
   403:         node.messages          OpenAI message history up to this node.
   404: 
   405:         Default implementation: simple greedy chain (no backtracking).
   406:         """
   407:         now_node = root_node
   408:         for step in range(self.single_chain_max_step):
   409:             # Check budget
   410:             if self.query_count >= self.max_query_count:
   411:                 break
   412: 
   413:             # Check if we already have enough answers
   414:             if len(self.terminal_node) >= self.answer_count:
   415:                 break
   416: 
   417:             # Take one step
   418:             new_leaves = self._step(now_node)
   419:             if not new_leaves:
   420:                 break
   421: 
   422:             now_node = new_leaves[-1]
   423: 
   424:             # Check if the new node is terminal (gave a final answer)
   425:             if now_node.is_terminal:
   426:                 self.status = 1
   427:                 self.terminal_node.append(now_node)
   428:                 break
   429: 
   430:             # Check if the new node is pruned (dead end / give up)
   431:             if now_node.pruned:
   432:                 if now_node.observation_code == 4:
   433:                     self.give_up_node.append(now_node)
   434:                 break
   435: 
   436:             # Check depth limit
   437:             if now_node.get_depth() >= self.single_chain_max_step:
   438:                 now_node.pruned = True
   439:                 break
   440:     # ==================================================================
   441:     # EDITABLE REGION END
   442:     # ==================================================================
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `greedy_chain` baseline — editable region  [READ-ONLY — reference implementation]

In `stabletoolbench/toolbench/inference/Algorithms/custom_search.py`:

```python
Lines 368–395:
   365:     # ==================================================================
   366:     # EDITABLE REGION START
   367:     # ==================================================================
   368:     def search(self, root_node):
   369:         """Greedy chain: follow one path, no backtracking."""
   370:         now_node = root_node
   371:         for step in range(self.single_chain_max_step):
   372:             if self.query_count >= self.max_query_count:
   373:                 break
   374:             if len(self.terminal_node) >= self.answer_count:
   375:                 break
   376: 
   377:             new_leaves = self._step(now_node)
   378:             if not new_leaves:
   379:                 break
   380: 
   381:             now_node = new_leaves[-1]
   382: 
   383:             if now_node.is_terminal:
   384:                 self.status = 1
   385:                 self.terminal_node.append(now_node)
   386:                 break
   387: 
   388:             if now_node.pruned:
   389:                 if now_node.observation_code == 4:
   390:                     self.give_up_node.append(now_node)
   391:                 break
   392: 
   393:             if now_node.get_depth() >= self.single_chain_max_step:
   394:                 now_node.pruned = True
   395:                 break
   396:     # ==================================================================
   397:     # EDITABLE REGION END
   398:     # ==================================================================
```

### `dfs_ranked` baseline — editable region  [READ-ONLY — reference implementation]

In `stabletoolbench/toolbench/inference/Algorithms/custom_search.py`:

```python
Lines 368–431:
   365:     # ==================================================================
   366:     # EDITABLE REGION START
   367:     # ==================================================================
   368:     def search(self, root_node):
   369:         """DFS with LLM ranking: expand best child first, backtrack on failure."""
   370:         self._dfs(root_node)
   371: 
   372:     def _dfs(self, now_node):
   373:         """Recursive DFS. Returns number of levels to backtrack."""
   374:         final_answer_back_length = 2
   375:         prune_back_length = 2
   376: 
   377:         now_node.expand_num = self.now_expand_num
   378:         self.now_expand_num += 1
   379: 
   380:         # Base cases
   381:         if now_node.get_depth() >= self.single_chain_max_step or now_node.pruned or now_node.is_terminal:
   382:             if now_node.is_terminal:
   383:                 self.status = 1
   384:                 self.terminal_node.append(now_node)
   385:                 return final_answer_back_length
   386:             else:
   387:                 now_node.pruned = True
   388:                 if now_node.observation_code == 4:
   389:                     self.give_up_node.append(now_node)
   390:                     return prune_back_length
   391:                 return 1
   392: 
   393:         # Generate beam_size children
   394:         candidates = []
   395:         for i in range(self.tree_beam_size):
   396:             if self.query_count >= self.max_query_count:
   397:                 return 100000
   398: 
   399:             # Add diversity prompt if node already has children
   400:             added_diversity = self._add_diversity_prompt(now_node)
   401: 
   402:             new_leaves = self._step(now_node)
   403: 
   404:             # Mark diversity message as invalid
   405:             if added_diversity:
   406:                 now_node.messages[-1]["valid"] = False
   407: 
   408:             if not new_leaves:
   409:                 continue
   410:             candidates.append(new_leaves[-1])
   411: 
   412:         if not candidates:
   413:             return 1
   414: 
   415:         # Rank candidates using LLM pairwise comparison
   416:         if len(candidates) > 1:
   417:             scores = self._rank_nodes(candidates)
   418:             for score, node in zip(scores, candidates):
   419:                 node.prior_score = score
   420:             candidates.sort(key=lambda x: x.prior_score, reverse=True)
   421: 
   422:         # Expand best candidates in order
   423:         for cand in candidates:
   424:             result = self._dfs(cand)
   425:             if len(self.terminal_node) >= self.answer_count:
   426:                 return 10000
   427:             elif result > 1:
   428:                 now_node.make_finish(2)
   429:                 return result - 1
   430: 
   431:         return 1
   432:     # ==================================================================
   433:     # EDITABLE REGION END
   434:     # ==================================================================
```

### `dfsdt` baseline — editable region  [READ-ONLY — reference implementation]

In `stabletoolbench/toolbench/inference/Algorithms/custom_search.py`:

```python
Lines 368–419:
   365:     # ==================================================================
   366:     # EDITABLE REGION START
   367:     # ==================================================================
   368:     def search(self, root_node):
   369:         """DFSDT: generate one child, recurse immediately, backtrack on failure."""
   370:         self._dfsdt(root_node)
   371: 
   372:     def _dfsdt(self, now_node):
   373:         """Recursive DFSDT. Returns number of levels to backtrack."""
   374:         final_answer_back_length = 2
   375:         prune_back_length = 2
   376: 
   377:         now_node.expand_num = self.now_expand_num
   378:         self.now_expand_num += 1
   379: 
   380:         # Base cases
   381:         if now_node.get_depth() >= self.single_chain_max_step or now_node.pruned or now_node.is_terminal:
   382:             if now_node.is_terminal:
   383:                 self.status = 1
   384:                 self.terminal_node.append(now_node)
   385:                 return final_answer_back_length
   386:             else:
   387:                 now_node.pruned = True
   388:                 if now_node.observation_code == 4:
   389:                     self.give_up_node.append(now_node)
   390:                     return prune_back_length
   391:                 return 1
   392: 
   393:         # Try beam_size times (each time generates one child and recurses)
   394:         for i in range(self.tree_beam_size):
   395:             if self.query_count >= self.max_query_count:
   396:                 return 100000
   397: 
   398:             # Add diversity prompt if node already has children
   399:             added_diversity = self._add_diversity_prompt(now_node)
   400: 
   401:             new_leaves = self._step(now_node)
   402: 
   403:             # Mark diversity message as invalid
   404:             if added_diversity:
   405:                 now_node.messages[-1]["valid"] = False
   406: 
   407:             if not new_leaves:
   408:                 continue
   409: 
   410:             leaf = new_leaves[-1]
   411: 
   412:             # Immediately recurse (no ranking)
   413:             result = self._dfsdt(leaf)
   414:             if len(self.terminal_node) >= self.answer_count:
   415:                 return 10000
   416:             elif result > 1:
   417:                 return result - 1
   418: 
   419:         return 1
   420:     # ==================================================================
   421:     # EDITABLE REGION END
   422:     # ==================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
