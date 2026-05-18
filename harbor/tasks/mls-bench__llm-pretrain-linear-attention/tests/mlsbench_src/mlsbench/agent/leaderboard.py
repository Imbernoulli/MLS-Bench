"""Leaderboard for MLS-Bench evaluation results.

Per-task CSV stored at tasks/<task>/leaderboard.csv.

Includes a write-ahead log (WAL) to protect against data loss when
external processes (e.g. git checkout) replace the CSV file.
"""

import fcntl
import csv
import hashlib
import json
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class Leaderboard:
    """Append-only leaderboard stored as CSV, with WAL protection."""

    META_COLS = {"timestamp", "model", "is_final", "seed"}
    INFORMATIONAL_PREFIXES = ("elapsed_", "n_samples", "n_prompts")

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self._build_lock_path(self.path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # WAL lives next to the lock file (outside the repo)
        self._wal_path = self.lock_path.with_suffix(".wal")

    @staticmethod
    def _build_lock_path(path: Path) -> Path:
        # Keep lock files out of task directories so leaderboard reads/writes do not
        # pollute the repo working tree. Hashing the resolved CSV path prevents
        # collisions across checkouts and worktrees that share task names.
        resolved = path.resolve(strict=False)
        digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
        safe_parent = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.parent.name)
        safe_name = safe_parent or "task"
        import os
        lock_dir = Path(tempfile.gettempdir()) / f"mlsbench-leaderboard-locks-{os.getuid()}"
        return lock_dir / f"{safe_name}-{path.stem}-{digest}.lock"

    @contextmanager
    def _locked(self, lock_type: int):
        with self._lock:
            with open(self.lock_path, "a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), lock_type)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_existing(self) -> tuple[list[str], list[dict]]:
        """Read existing CSV, return (fieldnames, rows)."""
        if not self.path.exists():
            return [], []
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        return fieldnames, rows

    # --- WAL helpers ---

    def _append_wal(self, entry: dict) -> None:
        """Append a JSON record to the WAL file."""
        with open(self._wal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _read_wal(self) -> list[dict]:
        """Read all records from the WAL file."""
        if not self._wal_path.exists():
            return []
        records = []
        with open(self._wal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _replay_wal(self, fieldnames: list[str], rows: list[dict]) -> tuple[list[str], list[dict]]:
        """Replay WAL entries that are missing from the CSV rows."""
        wal_records = self._read_wal()
        if not wal_records:
            return fieldnames, rows

        # Build set of existing (timestamp, model, seed) for dedup
        existing = set()
        for r in rows:
            key = (r.get("timestamp", ""), r.get("model", ""), r.get("seed", ""))
            existing.add(key)

        recovered = 0
        for entry in wal_records:
            key = (entry.get("timestamp", ""), entry.get("model", ""), entry.get("seed", ""))
            if key not in existing:
                for k in entry:
                    if k not in fieldnames:
                        fieldnames.append(k)
                rows.append(entry)
                existing.add(key)
                recovered += 1

        if recovered:
            print(f"[leaderboard] WAL replay: recovered {recovered} rows lost by external file replacement")

        return fieldnames, rows

    def compact_wal(self, *, removed_keys: set[tuple[str, str, str]] | None = None) -> int:
        """Prune the WAL so it never resurrects rows that are no longer in the CSV.

        After any out-of-band CSV edit (manual cleanup, ``atomic_swap_final``,
        cleanup_leaderboards.py, etc.) this MUST be called while holding the
        lock. Otherwise the next ``Leaderboard.add()`` call will see WAL
        entries whose ``(timestamp, model, seed)`` key is missing from the CSV
        and reinsert them as "lost" rows — silently undoing the cleanup.

        Two pruning rules are applied:
          1. Drop any WAL entry whose key is not in the current CSV.
          2. If ``removed_keys`` is provided, also drop entries whose key is
             in that set (used by ``atomic_swap_final`` to drop the WAL copies
             of finals it just deleted, even if the CSV happens to still hold
             a row with the same key).

        Returns the number of WAL entries dropped. Acquires its own exclusive
        lock if not already held by the caller.
        """
        wal_records = self._read_wal()
        if not wal_records:
            return 0

        csv_keys: set[tuple[str, str, str]] = set()
        if self.path.exists():
            with open(self.path, newline="") as f:
                for r in csv.DictReader(f):
                    csv_keys.add((r.get("timestamp", ""), r.get("model", ""), r.get("seed", "")))

        kept: list[dict] = []
        dropped = 0
        for entry in wal_records:
            key = (entry.get("timestamp", ""), entry.get("model", ""), entry.get("seed", ""))
            if removed_keys and key in removed_keys:
                dropped += 1
                continue
            if key not in csv_keys:
                dropped += 1
                continue
            kept.append(entry)

        if dropped == 0:
            return 0

        tmp_path = self._wal_path.with_suffix(self._wal_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for entry in kept:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self._wal_path)
        return dropped

    def add(self, record: dict) -> None:
        """Append a record to the leaderboard. Adds timestamp if not present.

        Records are written to a WAL first, then merged into the CSV.
        If git (or another process) replaces the CSV between writes,
        the WAL ensures no data is lost.
        """
        entry = {"timestamp": datetime.now(timezone.utc).isoformat()}
        entry.update(record)
        # Ensure is_final is stored as lowercase string for CSV consistency
        if "is_final" in entry:
            entry["is_final"] = str(entry["is_final"]).lower()

        # Write to WAL first (outside the repo, safe from git)
        self._append_wal(entry)

        with self._locked(fcntl.LOCK_EX):
            fieldnames, rows = self._read_existing()
            # Replay any WAL entries missing from the CSV (recovers git-lost rows)
            fieldnames, rows = self._replay_wal(fieldnames, rows)
            # Merge new keys into fieldnames (preserve order, append new ones)
            for k in entry:
                if k not in fieldnames:
                    fieldnames.append(k)
            # Dedup: don't re-add if WAL replay already included this entry
            entry_key = (entry.get("timestamp", ""), entry.get("model", ""), entry.get("seed", ""))
            already_present = any(
                (r.get("timestamp", ""), r.get("model", ""), r.get("seed", "")) == entry_key
                for r in rows
            )
            if not already_present:
                rows.append(entry)
            # Rewrite entire CSV with updated fieldnames
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

    def all_records(self) -> list[dict]:
        """Return all leaderboard records."""
        if not self.path.exists():
            return []
        with self._locked(fcntl.LOCK_SH):
            with open(self.path, newline="") as f:
                reader = csv.DictReader(f)
                records = []
                for row in reader:
                    # Convert numeric-looking values back to floats
                    parsed: dict = {}
                    for k, v in row.items():
                        if v == "":
                            continue
                        try:
                            parsed[k] = float(v)
                        except (ValueError, TypeError):
                            parsed[k] = v
                    records.append(parsed)
                return records

    @classmethod
    def has_real_metrics(cls, record: dict) -> bool:
        """Return True if a record contains at least one non-elapsed metric."""
        for key, value in record.items():
            if (
                key in cls.META_COLS
                or key.startswith(cls.INFORMATIONAL_PREFIXES)
                or key.endswith("_std")
            ):
                continue
            if value in ("", None):
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            return True
        return False

    def to_markdown(self, metric_cols: list[str] | None = None) -> str:
        """Render the leaderboard as a Markdown table."""
        records = self.all_records()
        if not records:
            return "_No results yet._"

        base_cols = ["timestamp", "model", "seed"]
        if metric_cols is None:
            seen = set()
            metric_cols = []
            for r in records:
                for k, v in r.items():
                    if k not in base_cols and k not in seen and isinstance(v, (int, float)):
                        seen.add(k)
                        metric_cols.append(k)
        cols = base_cols + metric_cols

        header = " | ".join(cols)
        sep = " | ".join(["---"] * len(cols))
        rows = []
        for r in records:
            row = " | ".join(str(r.get(c, "")) for c in cols)
            rows.append(row)

        return "| " + header + " |\n| " + sep + " |\n" + "\n".join("| " + r + " |" for r in rows)

    def save_markdown(self, out_path: Path, metric_cols: list[str] | None = None) -> None:
        """Write the leaderboard as a Markdown file."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.to_markdown(metric_cols=metric_cols))
