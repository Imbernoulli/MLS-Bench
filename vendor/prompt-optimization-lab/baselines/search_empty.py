"""Weak baseline: EMPTY (zero) instruction — no automatic prompt optimization.

Return the empty instruction string: the frozen LM executes the task with no task
guidance at all (only the input field). This is the naive floor an APE search must
beat; a small instruction LM left without a task description sits near the class
prior. Reference: vendor/prompt-optimization-lab/solution/search.py (default).
"""


def optimize(ctx) -> str:
    return ""
