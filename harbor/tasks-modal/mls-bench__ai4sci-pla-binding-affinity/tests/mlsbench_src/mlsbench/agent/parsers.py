"""Output parsers for MLS-Bench task feedback.

Each task defines its own Parser subclass in tasks/<task>/parser.py.
The base OutputParser provides pass-through behavior (raw output, no metrics).
"""

# Defer annotation evaluation so PEP 585 builtin generics (e.g. ``tuple[str,
# float]``) in function signatures don't crash at import under Python 3.8 —
# several task images (cleanrl/CORL Py3.8.10, humanoid-gym Py3.8.13) run the
# verifier under their own 3.8 interpreter.
from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path


@dataclass
class ParseResult:
    """Result of parsing a command's output."""
    feedback: str        # What to return to the model (filtered/formatted output)
    metrics: dict = field(default_factory=dict)  # Structured data for the leaderboard


class OutputParser:
    """Base parser. Subclasses override parse() for task-specific handling."""

    @staticmethod
    def parse_metric_assignment(parts: str) -> tuple[str, float] | None:
        """Parse a single ``metric_name=value`` assignment.

        Uses the last ``=`` as the separator so metric names may themselves
        contain ``=`` characters, e.g. ``blur_psnr (f=10)=21.7``.
        """
        metric_name, sep, raw_value = parts.rpartition("=")
        if not sep:
            return None

        metric_name = metric_name.strip()
        raw_value = raw_value.strip()
        if not metric_name or not raw_value:
            return None

        try:
            value = float(raw_value)
        except ValueError:
            return None

        return metric_name, value

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        """Parse the raw output of a command.

        Args:
            cmd_label: The label of the test_cmd entry (e.g. 'train', 'evaluate').
            raw_output: The combined stdout+stderr from the command.

        Returns:
            ParseResult with feedback (shown to model) and metrics (for leaderboard).
        """
        return ParseResult(feedback=raw_output, metrics={})


_FAILTAIL_RE = re.compile(r"^--- .*failed; last lines of its log:", re.MULTILINE)


def extract_failure_tail(raw_output: str, max_chars: int = 2000) -> str:
    """The block a hardened run script prints when a stage dies.

    ``scripts/_run.sh`` in the hardened task family ends a failed run with::

        failtail() { echo "--- $1 failed; last lines of its log:"; \
                     grep -v -E '^(FINAL|EVAL)_METRICS' "$2" | tail -n 25; }

    i.e. it already prints the cause, with metric markers stripped, before it emits
    ``FINAL_METRICS[...]: INVALID ...``. Task parsers build their feedback from a
    whitelist of progress prefixes (``SETUP:``, ``TRAIN_METRICS:`` ...), which this
    block matches none of, so the diagnosis is dropped and the caller sees only
    ``INVALID train_rc=1``.

    Returns the block, or "" if the output holds none.
    """
    m = _FAILTAIL_RE.search(raw_output or "")
    if not m:
        return ""
    block = []
    for line in raw_output[m.start():].splitlines():
        # Stop at the terminal marker: the diagnosis is what precedes it, and the
        # caller already reports the INVALID reason itself.
        if line.startswith(("FINAL_METRICS", "EVAL_METRICS")):
            break
        block.append(line)
    return "\n".join(block)[:max_chars].rstrip()


class _WithFailureDiagnostics(OutputParser):
    """Delegate to a task parser, but never swallow the run script's own diagnosis.

    Only fires when the parse produced NO metrics -- a successful run's feedback is
    left exactly as the task wrote it. The appended text is the run script's output,
    already scrubbed of ``FINAL_METRICS``/``EVAL_METRICS`` lines by ``failtail``
    itself, so this discloses nothing the script did not already print.
    """

    def __init__(self, inner: OutputParser):
        self._inner = inner

    def __getattr__(self, name):  # keep any task-specific attributes reachable
        return getattr(self._inner, name)

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        result = self._inner.parse(cmd_label, raw_output)
        if result.metrics:
            return result
        tail = extract_failure_tail(raw_output)
        if not tail or tail[:60] in (result.feedback or ""):
            return result
        return ParseResult(feedback=((result.feedback or "").rstrip() + "\n" + tail).strip(),
                           metrics=result.metrics)


def with_failure_diagnostics(parser: OutputParser) -> OutputParser:
    """Wrap a parser so a failed run reports why it failed."""
    if isinstance(parser, _WithFailureDiagnostics):
        return parser
    return _WithFailureDiagnostics(parser)


def load_parser(task_name: str, project_root: Path) -> OutputParser:
    """Load the task-specific Parser from tasks/<task>/parser.py, or return base OutputParser."""
    parser_path = project_root / "tasks" / task_name / "parser.py"
    if parser_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("task_parser", parser_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "Parser"):
            return module.Parser()
    return OutputParser()
