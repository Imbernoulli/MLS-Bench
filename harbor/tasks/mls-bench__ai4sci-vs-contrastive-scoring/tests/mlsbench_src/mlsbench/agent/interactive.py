"""InteractiveAgent: LLM-powered agent using the tool_use API."""

from mlsbench.agent.base import BaseAgent
from mlsbench.agent.models import build_client
from mlsbench.agent.tools import TOOL_SCHEMAS, WEB_SEARCH_SCHEMA, WEB_EXTRACT_SCHEMA

SYSTEM_PROMPT_SCI = """\
You are an ML scientist. Your goal is to propose and implement a novel algorithmic \
contribution that improves performance on the given task.

What counts as a good contribution:
- A new loss function or objective formulation
- A new policy update rule or gradient estimation method
- A novel exploration or regularization strategy
- A new way to parameterize or combine components, with clear motivation

What does NOT count:
- Trivially increasing network capacity to brute-force a metric (a per-task parameter cap is enforced before each test)
- Hyperparameter tuning (learning rates, batch sizes, etc.)
- Copying a reference baseline with cosmetic changes
- Pure engineering tricks without algorithmic novelty

Parameter count is capped (enforced before each test); architectural changes within that budget are encouraged.

IMPORTANT workflow:
1. FIRST call edit() to implement your improved algorithm. Do NOT call test() before making edits.
2. THEN call test() to run the experiment. Each run is numbered (#1, #2, ...).
3. Review the metrics, then edit() to improve your solution based on the feedback.
4. Call test() again to verify the improvement. You MUST iterate at least once \
(edit → test → review → edit → test) before submitting, unless only 1 test is allowed.
5. When satisfied, call submit(n=N) to submit your best test #N as final.
You have a limited number of test() calls, so make each one count by editing first.

Available tools:
- edit(op, filename, content, ...): Modify files in the workspace.
  - op='replace': replace lines start_line..end_line with content
  - op='insert': insert content after after_line
  - op='create': create a new file (only if allow_create=true)
- test(): Run a new experiment. Executes training and evaluation. Each run is
  numbered #1, #2, etc. The first test runs all seeds; intermediate tests run one seed.
  You have a limited budget of test() calls, so make each one count by editing first.
  If max tests is reached, the last test is auto-submitted.
- submit(n=N): Submit the result from test #N as your final answer (1-indexed).
  This does NOT re-run anything — it picks a previous result. Use n=-1 for the latest.
  You must call test() at least once before you can submit.
- undo(n=1): Revert the last n edit operations.

Constraints:
- Each file shown in the prompt is labeled [READ-ONLY] or [EDITABLE — lines X–Y only].
  Only edit files and line ranges marked EDITABLE. Do not touch READ-ONLY files.
- When a file has multiple editable regions, editing one region may shift line numbers \
in subsequent regions. Edit from the last (bottom-most) region first, or check the \
updated editable ranges shown after each edit.
- You MUST call test() at least once before you can call submit().
- When you are done, call submit(n=N) to submit your best test result.
- If your algorithm requires new hyperparameters (e.g., cql_alpha, expectile_tau) that are not
  in the existing config, hardcode their values directly in your code (e.g., in __init__).
  You cannot modify the training loop or config to pass them via command line.
"""


class InteractiveAgent(BaseAgent):
    """LLM-powered agent that uses tool_use API to drive the modify->test loop."""

    def __init__(
        self,
        task_name: str,
        global_config: dict,
        workspace_root=None,
    ):
        super().__init__(task_name, global_config, workspace_root)
        self.client = build_client(global_config)
        self.system_prompt = SYSTEM_PROMPT_SCI
        self.allow_web_search = bool(global_config.get("allow_web_search", False))
        self._tool_schemas = list(TOOL_SCHEMAS)
        if self.allow_web_search:
            self._tool_schemas.append(WEB_SEARCH_SCHEMA)
            self._tool_schemas.append(WEB_EXTRACT_SCHEMA)
            cap = int(global_config.get("max_web_credits", 20) or 0)
            cap_note = (f"You have a budget of {cap} Tavily credits for this run, "
                        f"shared between web_search and web_extract."
                        if cap else
                        "There is no per-run credit cap on web_search/web_extract.")
            web_tools_block = (
                "- web_search(query, ...): Search the web (Tavily). Returns title/URL/"
                "score/snippet per result, plus optional synthesized answer. Use it to "
                "find candidate URLs.\n"
                "- web_extract(urls, query, chunks_per_source): Read the actual content "
                "of specific URLs. When you pass `query` together with `chunks_per_source`, "
                "Tavily returns ONLY the chunks of each page most relevant to that query "
                "— this is how you read a paper's derivation, a doc's specific section, or "
                "a discussion of a method, without dumping the whole page.\n"
                "  Recommended pattern: web_search(...) → pick 1-3 promising URLs → "
                "web_extract(urls=[...], query='your specific question', chunks_per_source=3, "
                "extract_depth='advanced' for arxiv PDFs).\n"
                "  Budget — calls cost Tavily *credits*, not just call count:\n"
                "    • web_search basic = 1 credit; advanced = 2 credits.\n"
                "    • web_extract basic = 1 credit per URL; advanced = 2 credits per URL.\n"
                "    • Example: web_extract(urls=[a,b,c], extract_depth='advanced') = 6 credits.\n"
                f"  {cap_note} Pre-flight check rejects any call that would overrun the cap "
                "— the error message tells you the cost and what's left, so adjust depth or "
                "url count and retry. Each call also counts against your step budget.\n"
            )
            # Insert web tools INTO the existing Available tools list (between
            # undo() and the Constraints block) so the model sees them as
            # peer tools, not a separate aside.
            anchor = "- undo(n=1): Revert the last n edit operations.\n\nConstraints:"
            replacement = (
                "- undo(n=1): Revert the last n edit operations.\n"
                + web_tools_block
                + "\nConstraints:"
            )
            if anchor in self.system_prompt:
                self.system_prompt = self.system_prompt.replace(anchor, replacement, 1)
            else:
                # Defensive fallback: append at end.
                self.system_prompt += "\n\n" + web_tools_block

            # Workflow nudge — explicit "use them when uncertain" beats a
            # passive "they're available". Append after the existing closing
            # workflow line about test() budget.
            workflow_anchor = (
                "You have a limited number of test() calls, so make each one "
                "count by editing first."
            )
            workflow_nudge = (
                workflow_anchor
                + " If you're unsure about a paper, derivation, library API, "
                "or whether a method exists, USE web_search/web_extract before "
                "implementing — it's much cheaper than burning a test() on a "
                "broken try."
            )
            self.system_prompt = self.system_prompt.replace(
                workflow_anchor, workflow_nudge, 1,
            )

    @staticmethod
    def _is_transient_api_error(exc: Exception) -> bool:
        err_name = type(exc).__name__.lower()
        status_code = getattr(exc, "status_code", 0)
        text = str(exc).lower()
        return (
            "timeout" in err_name
            or "ratelimit" in err_name
            or "rate_limit" in err_name
            or "jsondecode" in err_name          # malformed API response body
            or "timeout" in text
            or "rate limit" in text
            or "too many requests" in text
            or "malformed api response" in text   # guard in models.py
            or status_code in (408, 409, 425, 429, 500, 502, 503, 504, 529)
        )

    def _uses_openrouter_free_model(self) -> bool:
        base_url = getattr(self.client, "base_url", "") or ""
        model = getattr(self.client, "model", "") or ""
        return "openrouter" in base_url and ":free" in model

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        retry_after = headers.get("retry-after")
        if retry_after is None:
            return None
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            return None

    def get_action(self, messages: list) -> dict | None:
        """Call the LLM and return the next tool_use action, with retry on transient errors."""
        import random
        import time
        is_free_openrouter = self._uses_openrouter_free_model()
        max_retries = 8 if is_free_openrouter else 5
        for attempt in range(max_retries):
            try:
                return self.client.call(messages, tools=self._tool_schemas, system=self.system_prompt)
            except Exception as e:
                if not self._is_transient_api_error(e):
                    raise
                if attempt == max_retries - 1:
                    raise
                retry_after = self._retry_after_seconds(e)
                if retry_after is not None:
                    wait = retry_after
                else:
                    base_wait = 60 if is_free_openrouter else 20
                    wait = min(900, base_wait * (2 ** attempt))
                    if is_free_openrouter:
                        wait = max(wait, 90)
                jitter = random.uniform(0, min(30.0, max(1.0, wait * 0.2)))
                total_wait = wait + jitter
                err_name = type(e).__name__
                print(
                    f"[agent] {err_name}: {e} — retrying in {total_wait:.1f}s "
                    f"({attempt + 1}/{max_retries})"
                )
                time.sleep(total_wait)
        return None  # unreachable: last attempt raises above
