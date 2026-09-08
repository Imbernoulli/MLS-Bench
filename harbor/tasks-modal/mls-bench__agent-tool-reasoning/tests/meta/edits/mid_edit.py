"""Mid-edit operations for the agent-tool-reasoning task.

Applied to the stabletoolbench workspace after pre_edit, before the agent starts.
1. Creates custom_search.py — the agent's editable search algorithm.
2. Patches rapidapi_multithread.py to register CustomSearch as a valid method.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# Patch for method_converter in rapidapi_multithread.py:
# Add import for CustomSearch and an elif branch in method_converter.
# The import goes after the DFS import (line 16), and the elif goes
# before the "else: print('invalid method')" block (line 504).

_IMPORT_PATCH = """\
from toolbench.inference.Algorithms.DFS import DFS_tree_search
from toolbench.inference.Algorithms.custom_search import CustomSearch
"""

_METHOD_PATCH = """\
        elif method.startswith("CustomSearch"):
            chain = CustomSearch(llm=llm_forward, io_func=env, process_id=process_id, callbacks=callbacks)
            result = chain.start(
                single_chain_max_step=single_chain_max_step,
                tree_beam_size=3,
                max_query_count=max_query_count,
                answer=1,
                with_filter=True,
            )
        else:
            print("invalid method")
            raise NotImplementedError
"""

# Ops ordered: create first, then patches bottom-to-top within the same file.
OPS = [
    # 1. Create the custom search module
    {
        "op": "create",
        "file": "stabletoolbench/toolbench/inference/Algorithms/custom_search.py",
        "content": _CUSTOM_PY,
    },
    # 2. Add CustomSearch branch in method_converter (replace lines 504-506)
    #    Original:
    #        else:
    #            print("invalid method")
    #            raise NotImplementedError
    #    Applied first (bottom) to keep line numbers stable.
    {
        "op": "replace",
        "file": "stabletoolbench/toolbench/inference/Downstream_tasks/rapidapi_multithread.py",
        "start_line": 504,
        "end_line": 506,
        "content": _METHOD_PATCH,
    },
    # 3. Add import for CustomSearch (replace line 16 which has the DFS import)
    {
        "op": "replace",
        "file": "stabletoolbench/toolbench/inference/Downstream_tasks/rapidapi_multithread.py",
        "start_line": 16,
        "end_line": 16,
        "content": _IMPORT_PATCH,
    },
]
