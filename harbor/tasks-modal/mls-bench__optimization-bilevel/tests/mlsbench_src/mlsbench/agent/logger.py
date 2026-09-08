"""RunLogger: persist agent conversation and file snapshots to disk."""

# Defer annotation evaluation: imported (via base.py) by the Harbor verifier's
# score_task.py, which several task images run under Python 3.8 — without this,
# PEP 585 builtin generics in signatures crash at import.
from __future__ import annotations

import json
import re
from pathlib import Path


# Conservative secret patterns. Tool results capture stdout from third-party
# packages, which occasionally print provider config dicts containing API
# keys (e.g. stabletoolbench prints `{'api_key': 'sk-...'}` on startup).
# Without scrubbing, those keys land in messages.jsonl, which is then mirrored
# to the public website by scripts/build_site_data.py. Patterns here only
# match well-known secret prefixes and require enough length to avoid false
# positives on real data.
_SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-[a-f0-9]{48,}"),     # OpenRouter
    re.compile(r"sk-ant-(?:api|admin)[A-Za-z0-9_-]{30,}"),  # Anthropic
    re.compile(r"sk-proj-[A-Za-z0-9_-]{40,}"),  # OpenAI project
    re.compile(r"sk-svcacct-[A-Za-z0-9_-]{40,}"),  # OpenAI service-account
    re.compile(r"sk-[a-f0-9]{32}"),            # DeepSeek/Qwen-style hex
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),     # Google API key
]
_REDACTED = "<REDACTED>"


def _redact_secrets(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub(_REDACTED, text)
    return text


class RunLogger:
    """Append-only logger that writes JSONL messages and file snapshots.

    Directory layout::

        <log_dir>/
            messages.jsonl   – one JSON object per line
            files/           – file snapshots after edits
                step_3_trainer.py
                ...
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir = self.log_dir / "files"
        self.files_dir.mkdir(exist_ok=True)
        self._messages_path = self.log_dir / "messages.jsonl"
        self._tokens_path = self.log_dir / "tokens.jsonl"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append(self, record: dict) -> None:
        """Append a single JSON record to messages.jsonl."""
        with open(self._messages_path, "a") as f:
            f.write(_redact_secrets(json.dumps(record, default=str)) + "\n")

    @staticmethod
    def _sanitize(name: str) -> str:
        """Turn a file path into a safe filename (replace / and special chars)."""
        return re.sub(r"[^\w.\-]", "_", name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear existing messages and file snapshots for a fresh run."""
        if self._messages_path.exists():
            self._messages_path.unlink()
        if self._tokens_path.exists():
            self._tokens_path.unlink()
        for p in self.files_dir.iterdir():
            if p.is_file():
                p.unlink()

    def rewrite_messages(self, records: list[dict]) -> None:
        """Atomically replace messages.jsonl with the given list of records.

        Used by resume to drop orphan/cancelled records from disk so repeated
        resume cycles don't accumulate duplicate assistant entries.
        """
        tmp_path = self._messages_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w") as f:
            for rec in records:
                f.write(_redact_secrets(json.dumps(rec, default=str)) + "\n")
        tmp_path.replace(self._messages_path)

    def log_initial_prompt(self, prompt: str) -> None:
        """Log the initial prompt sent to the model (step 0)."""
        self._append({"step": 0, "role": "user", "content": prompt})

    def log_user_message(self, content: str) -> None:
        """Log a synthetic user message injected mid-loop (e.g. nudge/reminder)."""
        self._append({"role": "user", "content": content})

    def log_assistant(self, step: int, tool_use: dict) -> None:
        """Log an assistant tool-use action.

        Extracts thinking, tool name, and tool input from the tool_use dict.
        """
        record: dict = {
            "step": step,
            "role": "assistant",
            "tool_name": tool_use.get("name"),
            "tool_input": tool_use.get("input", {}),
        }
        thinking = tool_use.get("thinking")
        if thinking:
            record["thinking"] = thinking
        self._append(record)

    def log_tool_result(self, step: int, result: str, meta: dict | None = None) -> None:
        """Log a tool result."""
        record = {"step": step, "role": "tool_result", "result": result}
        if meta:
            record["meta"] = meta
        self._append(record)

    def log_tokens(self, step: int, usage: dict) -> None:
        """Append one per-call LLM usage record to tokens.jsonl.

        Contract: ``usage`` is the normalized dict returned by
        ``models._extract_usage`` (keys: model, prompt_tokens, completion_tokens,
        total_tokens, cached_tokens, cache_creation_tokens). We add step +
        timestamp. Swallows write errors so logging issues never break the
        agent loop.
        """
        from datetime import datetime, timezone
        record = {
            "step": step,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **usage,
        }
        try:
            with open(self._tokens_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            return

    def log_file_snapshot(self, step: int, filename: str, content: str) -> None:
        """Save a copy of a file after an edit."""
        safe = self._sanitize(filename)
        snapshot_path = self.files_dir / f"step_{step}_{safe}"
        snapshot_path.write_text(content)

    # ------------------------------------------------------------------
    # Resume helpers
    # ------------------------------------------------------------------

    def has_messages(self) -> bool:
        """Check if there is an existing messages.jsonl to resume from."""
        return self._messages_path.exists() and self._messages_path.stat().st_size > 0

    def read_messages(self) -> list[dict]:
        """Read all records from messages.jsonl."""
        records: list[dict] = []
        with open(self._messages_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def get_latest_snapshots(self) -> dict[str, Path]:
        """Return the latest file snapshot path for each unique filename.

        Scans files/ dir for step_N_<sanitized_name> files and returns
        a mapping of sanitized_name -> Path for the highest step N.
        """
        snapshots: dict[str, tuple[int, Path]] = {}
        for p in self.files_dir.iterdir():
            if not p.is_file():
                continue
            # Parse step_N_<rest>
            m = re.match(r"step_(\d+)_(.*)", p.name)
            if not m:
                continue
            step_num = int(m.group(1))
            fname_key = m.group(2)
            if fname_key not in snapshots or step_num > snapshots[fname_key][0]:
                snapshots[fname_key] = (step_num, p)
        return {k: v[1] for k, v in snapshots.items()}
