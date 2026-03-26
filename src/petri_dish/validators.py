"""Hidden validators for agent file output.

Validators watch /env/outgoing/ for processed files and score them against
hidden criteria. The agent NEVER sees validation logic or scoring rules.

Credit rewards are loaded from config:
  - easy files:  credit_rewards.easy (default 0.3)
  - hard files:  credit_rewards.hard (default 2.0)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from typing import TYPE_CHECKING, Callable

from petri_dish.config import Settings

if TYPE_CHECKING:
    from petri_dish.sandbox import SandboxManager

logger = logging.getLogger(__name__)

_TIMESTAMP_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"),
    re.compile(r"\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}"),
]
_LOG_LEVEL_PATTERN = re.compile(r"\b(INFO|WARN|ERROR|DEBUG|WARNING|CRITICAL)\b")
_LOG_MESSAGE_PATTERN = re.compile(r"(GET|POST|PUT|DELETE|PATCH)\s+/\S+\s+\d{3}")


def _detect_family(filename: str) -> str | None:
    """Extract file family from filename pattern data_{ts}_{family}_{difficulty}.{ext}."""
    parts = filename.rsplit(".", 1)
    if len(parts) < 2:
        return None
    stem = parts[0]
    segments = stem.split("_")
    if len(segments) >= 4 and segments[0] == "data":
        family = segments[2]
        if family in ("csv", "json", "log"):
            return family
    return None


def _detect_difficulty(filename: str) -> str | None:
    """Extract difficulty from filename pattern data_{ts}_{family}_{difficulty}.{ext}."""
    parts = filename.rsplit(".", 1)
    if len(parts) < 2:
        return None
    stem = parts[0]
    segments = stem.split("_")
    if len(segments) >= 4 and segments[0] == "data":
        difficulty = segments[3]
        if difficulty in ("easy", "hard"):
            return difficulty
    return None


def _validate_csv(content: str) -> tuple[bool, str]:
    """Validate CSV output: parseable, consistent columns, rows > 0."""
    try:
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
    except csv.Error as e:
        return False, f"CSV parse error: {e}"

    if len(rows) < 2:
        return False, f"Too few rows: {len(rows)} (need header + at least 1 data row)"

    header_len = len(rows[0])
    if header_len == 0:
        return False, "Empty header row"

    mismatched = 0
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != header_len:
            mismatched += 1

    if mismatched > len(rows) * 0.3:
        return (
            False,
            f"Column count inconsistency: {mismatched}/{len(rows) - 1} rows differ from header ({header_len} cols)",
        )

    return True, f"Valid CSV: {header_len} columns, {len(rows) - 1} data rows"


def _validate_json(content: str) -> tuple[bool, str]:
    """Validate JSON output: parseable, preserves all keys from records."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"

    if isinstance(data, list):
        if len(data) == 0:
            return False, "Empty JSON array"
        for i, record in enumerate(data):
            if not isinstance(record, dict):
                return False, f"Record {i} is not an object: {type(record).__name__}"
        all_keys: set[str] = set()
        for record in data:
            all_keys.update(record.keys())
        if len(all_keys) == 0:
            return False, "No keys found in records"
        return True, f"Valid JSON: {len(data)} records, {len(all_keys)} unique keys"

    elif isinstance(data, dict):
        if len(data) == 0:
            return False, "Empty JSON object"
        return True, f"Valid JSON object: {len(data)} keys"

    return False, f"Unexpected JSON root type: {type(data).__name__}"


def _validate_log(content: str) -> tuple[bool, str]:
    """Validate log output: extracts timestamp/level/message, coverage >= 80%."""
    lines = [line for line in content.strip().split("\n") if line.strip()]
    if len(lines) == 0:
        return False, "No log lines found"

    timestamp_hits = 0
    level_hits = 0
    message_hits = 0

    for line in lines:
        has_ts = any(p.search(line) for p in _TIMESTAMP_PATTERNS)
        has_level = bool(_LOG_LEVEL_PATTERN.search(line))
        has_msg = bool(_LOG_MESSAGE_PATTERN.search(line))
        if has_ts:
            timestamp_hits += 1
        if has_level:
            level_hits += 1
        if has_msg:
            message_hits += 1

    total = len(lines)
    ts_coverage = timestamp_hits / total
    level_coverage = level_hits / total
    msg_coverage = message_hits / total
    overall_coverage = (ts_coverage + level_coverage + msg_coverage) / 3

    if overall_coverage < 0.8:
        return False, (
            f"Low coverage ({overall_coverage:.0%}): "
            f"timestamps={ts_coverage:.0%}, levels={level_coverage:.0%}, "
            f"messages={msg_coverage:.0%}"
        )

    return True, (
        f"Valid log: {total} lines, coverage={overall_coverage:.0%} "
        f"(ts={ts_coverage:.0%}, lvl={level_coverage:.0%}, msg={msg_coverage:.0%})"
    )


_VALIDATORS: dict[str, Callable[[str], tuple[bool, str]]] = {
    "csv": _validate_csv,
    "json": _validate_json,
    "log": _validate_log,
}


class FileValidator:
    """Hidden validator that scores agent file output.

    The agent never sees validation criteria. Validators check files
    placed in /env/outgoing/ and award credits based on difficulty.

    Args:
        settings: Configuration. Uses defaults from config.yaml if None.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        if settings is None:
            settings = Settings.from_yaml()
        self._settings = settings
        self._credit_rewards: dict[str, float] = getattr(
            settings, "credit_rewards", {"easy": 0.3, "hard": 2.0}
        )
        self._scoring_weights: dict[str, float] = settings.validator_scoring_weights

    def validate(self, filename: str, content: str) -> tuple[bool, float]:
        """Validate a processed file and return (passed, credits_earned).

        Args:
            filename: Original input filename (used to detect family/difficulty).
            content: The agent's processed output content.

        Returns:
            (passed, credits_earned): Whether validation passed, and credits awarded.
        """
        family = _detect_family(filename)
        difficulty = _detect_difficulty(filename)

        if family is None or difficulty is None:
            logger.warning(
                "Cannot detect family/difficulty from filename: %s", filename
            )
            return False, 0.0

        validator_fn = _VALIDATORS.get(family)
        if validator_fn is None:
            logger.warning("No validator for family: %s", family)
            return False, 0.0

        passed, detail = validator_fn(content)
        weight = self._scoring_weights.get(family, 1.0)
        base_reward = self._credit_rewards.get(difficulty, 0.0)
        credits_earned = base_reward * weight if passed else 0.0

        logger.info(
            "Validation [%s/%s] %s: %s | credits=%.2f",
            family,
            difficulty,
            "PASS" if passed else "FAIL",
            detail,
            credits_earned,
        )
        return passed, credits_earned

    def collect_outputs(
        self,
        sandbox: SandboxManager,
        container_id: str,
    ) -> list[tuple[str, str]]:
        """List files in /env/outgoing/ and read their contents.

        Args:
            sandbox: SandboxManager for container access.
            container_id: Docker container ID.

        Returns:
            List of (filename, content) tuples from /env/outgoing/.
        """
        listing = sandbox.exec_in_container(
            container_id, "ls /env/outgoing/ 2>/dev/null"
        )
        if not listing.strip() or "No such file" in listing:
            return []

        results: list[tuple[str, str]] = []
        for line in listing.strip().split("\n"):
            fname = line.strip()
            if not fname or fname.startswith("total") or fname.startswith("["):
                continue
            try:
                content = sandbox.read_file(container_id, f"/env/outgoing/{fname}")
                results.append((fname, content))
            except Exception as exc:
                logger.warning("Failed to read /env/outgoing/%s: %s", fname, exc)
        return results
