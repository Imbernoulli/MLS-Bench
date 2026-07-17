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
