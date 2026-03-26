"""Resource ecology: generates raw unlabeled files and drops them into containers.

File families: csv (tabular), json (nested records), log (server logs).
Each family has two difficulties: easy (well-formed) and hard (noisy/mixed).
Filenames intentionally give NO hints about expected processing.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import random
import string
import time
from typing import TYPE_CHECKING, Callable

from petri_dish.config import Settings

if TYPE_CHECKING:
    from petri_dish.sandbox import SandboxManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Internal data generators
# ---------------------------------------------------------------------------

_FIRST_NAMES = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "hank"]
_LAST_NAMES = ["smith", "jones", "lee", "garcia", "chen", "patel", "kim", "wilson"]
_LOG_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
_HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
_PATHS = ["/api/users", "/api/orders", "/health", "/api/items", "/login", "/api/data"]
_STATUS_CODES = [200, 201, 204, 301, 400, 401, 403, 404, 500, 502, 503]


def _rand_str(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _rand_ip() -> str:
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


# ---------------------------------------------------------------------------
#  CSV generators
# ---------------------------------------------------------------------------


def _generate_csv_easy(rows: int = 20) -> str:
    """Well-formed CSV with consistent columns."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    headers = ["id", "name", "email", "age", "score"]
    writer.writerow(headers)
    for i in range(1, rows + 1):
        first = random.choice(_FIRST_NAMES)
        last = random.choice(_LAST_NAMES)
        writer.writerow(
            [
                i,
                f"{first} {last}",
                f"{first}.{last}@example.com",
                random.randint(18, 80),
                round(random.uniform(0, 100), 2),
            ]
        )
    return buf.getvalue()


def _generate_csv_hard(rows: int = 25) -> str:
    """Noisy CSV: mixed delimiters, missing values, extra whitespace, quoted commas."""
    buf = io.StringIO()
    headers = ["id", "name", "email", "age", "score", "notes"]
    buf.write(",".join(headers) + "\n")
    for i in range(1, rows + 1):
        first = random.choice(_FIRST_NAMES)
        last = random.choice(_LAST_NAMES)
        age = str(random.randint(18, 80)) if random.random() > 0.2 else ""
        score = (
            str(round(random.uniform(0, 100), 2)) if random.random() > 0.15 else "N/A"
        )
        notes = random.choice(
            [
                f'"contains, comma"',
                '""',
                f"  {_rand_str(12)}  ",
                "",
                f'"line1\nline2"',
            ]
        )
        sep = ";" if random.random() < 0.1 else ","
        row = sep.join(
            [
                str(i),
                f"{first} {last}",
                f"{first}.{last}@example.com",
                age,
                score,
                notes,
            ]
        )
        buf.write(row + "\n")
    return buf.getvalue()


# ---------------------------------------------------------------------------
#  JSON generators
# ---------------------------------------------------------------------------


def _generate_json_easy(records: int = 10) -> str:
    """Well-formed JSON array of user records."""
    data = []
    for i in range(1, records + 1):
        first = random.choice(_FIRST_NAMES)
        last = random.choice(_LAST_NAMES)
        data.append(
            {
                "id": i,
                "name": f"{first} {last}",
                "email": f"{first}.{last}@example.com",
                "address": {
                    "street": f"{random.randint(1, 999)} Main St",
                    "city": random.choice(
                        ["Springfield", "Riverside", "Portland", "Salem"]
                    ),
                    "zip": f"{random.randint(10000, 99999)}",
                },
                "tags": random.sample(
                    ["admin", "user", "beta", "premium", "trial"],
                    k=random.randint(1, 3),
                ),
                "active": random.choice([True, False]),
            }
        )
    return json.dumps(data, indent=2)


def _generate_json_hard(records: int = 12) -> str:
    """Noisy JSON: nested objects, mixed types, null values, unicode."""
    data = []
    for i in range(1, records + 1):
        first = random.choice(_FIRST_NAMES)
        last = random.choice(_LAST_NAMES)
        record: dict[str, object] = {
            "id": i if random.random() > 0.1 else str(i),
            "name": f"{first} {last}",
            "email": f"{first}.{last}@example.com" if random.random() > 0.15 else None,
            "metadata": {
                "created": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "source": random.choice(["api", "import", "manual", None]),
                "extra": {
                    "notes": random.choice([f"Note: {_rand_str(20)}", None, ""]),
                    "revision": random.randint(1, 50),
                },
            },
            "scores": [
                round(random.uniform(0, 100), 2) for _ in range(random.randint(0, 5))
            ],
            "label": random.choice(
                ["α-tier", "β-group", "Ωmega", f"cat_{_rand_str(4)}"]
            ),
        }
        data.append(record)
    return json.dumps(data, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
#  Log generators
# ---------------------------------------------------------------------------


def _generate_log_easy(lines: int = 30) -> str:
    """Well-formed server log lines: timestamp level message."""
    buf = io.StringIO()
    base_ts = int(time.time()) - random.randint(0, 86400)
    for i in range(lines):
        ts = base_ts + i * random.randint(1, 5)
        level = random.choice(_LOG_LEVELS)
        method = random.choice(_HTTP_METHODS)
        path = random.choice(_PATHS)
        status = random.choice(_STATUS_CODES)
        ip = _rand_ip()
        msg = f"{method} {path} {status} {ip} {random.randint(1, 500)}ms"
        buf.write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts))}] {level}: {msg}\n"
        )
    return buf.getvalue()


def _generate_log_hard(lines: int = 40) -> str:
    """Noisy logs: mixed formats, multi-line stack traces, missing fields."""
    buf = io.StringIO()
    base_ts = int(time.time()) - random.randint(0, 86400)
    for i in range(lines):
        ts = base_ts + i * random.randint(1, 5)
        r = random.random()
        if r < 0.5:
            # Standard format
            level = random.choice(_LOG_LEVELS)
            method = random.choice(_HTTP_METHODS)
            path = random.choice(_PATHS)
            status = random.choice(_STATUS_CODES)
            buf.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts))}] {level}: "
                f"{method} {path} {status} {_rand_ip()} {random.randint(1, 500)}ms\n"
            )
        elif r < 0.7:
            # Apache-style log
            ip = _rand_ip()
            method = random.choice(_HTTP_METHODS)
            path = random.choice(_PATHS)
            status = random.choice(_STATUS_CODES)
            date_str = time.strftime("%d/%b/%Y:%H:%M:%S +0000", time.gmtime(ts))
            buf.write(
                f'{ip} - - [{date_str}] "{method} {path} HTTP/1.1" {status} {random.randint(100, 5000)}\n'
            )
        elif r < 0.85:
            # Multi-line stack trace
            buf.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts))}] ERROR: Unhandled exception\n"
            )
            buf.write(f"  Traceback (most recent call last):\n")
            buf.write(
                f'    File "/app/{_rand_str(8)}.py", line {random.randint(1, 200)}\n'
            )
            buf.write(
                f"  {random.choice(['ValueError', 'KeyError', 'TypeError'])}: {_rand_str(15)}\n"
            )
        else:
            # Garbled / incomplete line
            buf.write(f"{_rand_str(random.randint(5, 40))}\n")
    return buf.getvalue()


# ---------------------------------------------------------------------------
#  Generator registry
# ---------------------------------------------------------------------------

_GENERATORS: dict[tuple[str, str], tuple[Callable[[], str], str]] = {
    ("csv", "easy"): (_generate_csv_easy, "csv"),
    ("csv", "hard"): (_generate_csv_hard, "csv"),
    ("json", "easy"): (_generate_json_easy, "json"),
    ("json", "hard"): (_generate_json_hard, "json"),
    ("log", "easy"): (_generate_log_easy, "log"),
    ("log", "hard"): (_generate_log_hard, "log"),
}


# ---------------------------------------------------------------------------
#  ResourceEcology
# ---------------------------------------------------------------------------


class ResourceEcology:
    """Generates raw unlabeled files and drops them into agent containers.

    Files are intentionally raw — no instructions, no labels, no hints about
    expected processing. The agent must figure out what to do with each file
    by inspecting its content.

    Args:
        settings: Configuration. Uses defaults from config.yaml if None.
    """

    FAMILIES = ("csv", "json", "log")
    DIFFICULTIES = ("easy", "hard")

    def __init__(self, settings: Settings | None = None) -> None:
        if settings is None:
            settings = Settings.from_yaml()
        self._settings = settings
        self._drop_lambda: float = settings.file_drop_lambda
        self._drop_interval: int = settings.file_drop_interval_turns

    # ------------------------------------------------------------------ #
    #  File generation
    # ------------------------------------------------------------------ #

    def generate_file(self, family: str, difficulty: str) -> tuple[str, str]:
        """Generate a raw unlabeled file.

        Args:
            family: One of 'csv', 'json', 'log'.
            difficulty: One of 'easy', 'hard'.

        Returns:
            (filename, content) tuple. Filename contains no processing hints.

        Raises:
            ValueError: If family or difficulty is invalid.
        """
        key = (family, difficulty)
        if key not in _GENERATORS:
            raise ValueError(
                f"Unknown file spec ({family}, {difficulty}). "
                f"Valid families: {self.FAMILIES}, difficulties: {self.DIFFICULTIES}"
            )

        generator_fn, ext = _GENERATORS[key]
        content: str = generator_fn()
        ts = int(time.time() * 1000)
        filename = f"data_{ts}_{family}_{difficulty}.{ext}"

        logger.info("Generated file: %s (%d bytes)", filename, len(content))
        return filename, content

    # ------------------------------------------------------------------ #
    #  File dropping
    # ------------------------------------------------------------------ #

    def drop_file(
        self,
        sandbox: SandboxManager,
        container_id: str,
        filename: str,
        content: str,
    ) -> str:
        """Write a generated file into the container's /env/incoming/ directory.

        Args:
            sandbox: SandboxManager instance for container file access.
            container_id: Docker container ID.
            filename: Name of the file to write.
            content: File content.

        Returns:
            Confirmation message from sandbox.
        """
        path = f"/env/incoming/{filename}"
        result = sandbox.write_file(container_id, path, content)
        logger.info("Dropped file %s into container %s", filename, container_id[:12])
        return result

    # ------------------------------------------------------------------ #
    #  Scheduling
    # ------------------------------------------------------------------ #

    def schedule_drops(self, current_turn: int) -> list[tuple[str, str]]:
        """Determine which files to drop at the current turn.

        Uses Poisson-like scheduling: files drop every `file_drop_interval_turns`
        turns, with random family/difficulty selection. Between intervals, a
        random check using `file_drop_lambda` determines extra drops.

        Args:
            current_turn: The current simulation turn number.

        Returns:
            List of (filename, content) tuples to drop this turn.
        """
        files_to_drop: list[tuple[str, str]] = []

        if current_turn > 0 and current_turn % self._drop_interval == 0:
            family = random.choice(self.FAMILIES)
            difficulty = random.choice(self.DIFFICULTIES)
            files_to_drop.append(self.generate_file(family, difficulty))
            logger.info("Scheduled periodic drop at turn %d", current_turn)

        if random.random() < self._drop_lambda:
            family = random.choice(self.FAMILIES)
            difficulty = random.choice(self.DIFFICULTIES)
            files_to_drop.append(self.generate_file(family, difficulty))
            logger.info("Stochastic drop triggered at turn %d", current_turn)

        return files_to_drop
