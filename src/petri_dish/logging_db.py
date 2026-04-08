"""
SQLite logging database for Petri Dish MVP.

Implements WAL mode for concurrent reads/writes with:
- PRAGMA journal_mode=WAL
- PRAGMA synchronous=NORMAL
- PRAGMA busy_timeout=5000
- PRAGMA cache_size=-32000 (32KB cache)
"""

import sqlite3
import json
import contextlib
import importlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Generator
from uuid import uuid4


class LoggingDB:
    """SQLite database for logging experiment runs, actions, files, and zod transactions."""

    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize logging database.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory DB.
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def __enter__(self):
        """Context manager entry - opens connection and initializes database."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes connection."""
        self.close()

    def connect(self) -> sqlite3.Connection:
        """Open database connection with WAL mode configuration."""
        if self._conn is None:
            # Use URI mode to support read-only connections for dashboard
            uri = f"file:{self.db_path}?mode=rwc"
            self._conn = sqlite3.connect(uri, uri=True)
            self._conn.row_factory = sqlite3.Row

            # Configure WAL mode and performance settings
            cursor = self._conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA cache_size=-32000")
            cursor.execute("PRAGMA foreign_keys=ON")

            # Initialize schema
            self.init_db()

        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_connection(self) -> sqlite3.Connection:
        """Ensure database connection is established."""
        if self._conn is None:
            raise RuntimeError("Database connection not established")
        return self._conn

    def init_db(self):
        """Initialize database schema with required tables."""
        conn = self._ensure_connection()
        cursor = conn.cursor()

        # Create runs table - tracks experiment runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config_snapshot TEXT,
                end_time TIMESTAMP
            )
        """)

        # Create actions table - logs every tool invocation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                tool_args TEXT,
                result TEXT,
                zod_before REAL NOT NULL,
                zod_after REAL NOT NULL,
                duration_ms INTEGER NOT NULL,
                agent_id TEXT DEFAULT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
        """)

        # Create files table - tracks file drops and processing
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'dropped',
                dropped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                zod_earned REAL DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                UNIQUE(run_id, filename)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zod_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                amount REAL NOT NULL,
                type TEXT NOT NULL,
                reason TEXT,
                balance_after REAL NOT NULL,
                agent_id TEXT DEFAULT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
        """)

        # Create state_transitions table - tracks agent lifecycle changes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                from_state TEXT NOT NULL,
                to_state TEXT NOT NULL,
                reason TEXT,
                balance REAL,
                starvation_counter INTEGER,
                agent_id TEXT DEFAULT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
        """)

        # Create messages table - inter-agent communication
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                recipient_id TEXT NOT NULL,
                content TEXT NOT NULL,
                round_num INTEGER NOT NULL,
                turn INTEGER NOT NULL,
                read INTEGER NOT NULL DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                round_num INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT,
                zod_delta REAL DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scout_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                agent_id TEXT DEFAULT NULL,
                report_id TEXT NOT NULL,
                requesting_agent_id TEXT NOT NULL,
                target_filename TEXT DEFAULT NULL,
                file_family TEXT NOT NULL,
                claimed_pattern TEXT NOT NULL,
                output_summary TEXT NOT NULL,
                confidence REAL NOT NULL,
                verdict TEXT NOT NULL,
                reasoning TEXT,
                suggested_bonus REAL NOT NULL DEFAULT 0,
                report_json TEXT NOT NULL,
                applied INTEGER NOT NULL DEFAULT 0,
                applied_bonus REAL DEFAULT NULL,
                applied_turn INTEGER DEFAULT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS promotion_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id TEXT NOT NULL UNIQUE,
                claimed_pattern TEXT NOT NULL,
                file_family TEXT NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                promoted INTEGER NOT NULL DEFAULT 0,
                promoted_at TIMESTAMP DEFAULT NULL,
                bonus_multiplier REAL NOT NULL DEFAULT 1.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_hit_at TIMESTAMP DEFAULT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trait_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                round_num INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                curiosity REAL DEFAULT 0.0,
                thrift REAL DEFAULT 0.0,
                sociability REAL DEFAULT 0.0,
                persistence REAL DEFAULT 0.0,
                shell_affinity REAL DEFAULT 0.0,
                file_family_affinity_json TEXT DEFAULT '{}',
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_trait_snapshots_run ON trait_snapshots(run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_trait_snapshots_agent ON trait_snapshots(run_id, agent_id)"
        )

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                system_prompt_snippet TEXT DEFAULT '',
                user_prompt_snippet TEXT DEFAULT '',
                response_snippet TEXT DEFAULT '',
                duration_ms REAL DEFAULT 0.0,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_agent ON llm_calls(run_id, agent_id)"
        )

        # Create indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_actions_run_id ON actions(run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON actions(timestamp)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_run_id ON files(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_status ON files(status)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_zod_run_id ON zod_transactions(run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_zod_timestamp ON zod_transactions(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_state_transitions_run_id ON state_transitions(run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_actions_agent_id ON actions(agent_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_zod_agent_id ON zod_transactions(agent_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_state_transitions_agent_id ON state_transitions(agent_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_run_id ON messages(run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_recipient ON messages(recipient_id, read)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_ledger_run_id ON event_ledger(run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_ledger_agent ON event_ledger(agent_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_scout_reports_run_id ON scout_reports(run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_scout_reports_lookup ON scout_reports(run_id, agent_id, target_filename, applied)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_promotion_rules_lookup ON promotion_rules(claimed_pattern, file_family)"
        )

        conn.commit()

    def log_trait_snapshot(
        self,
        run_id: str,
        round_num: int,
        agent_id: str,
        traits_dict: Dict[str, Any],
    ) -> None:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        file_family_affinity = traits_dict.get("file_family_affinity", {})
        if not isinstance(file_family_affinity, dict):
            file_family_affinity = {}
        cursor.execute(
            """
            INSERT INTO trait_snapshots (
                run_id,
                round_num,
                agent_id,
                curiosity,
                thrift,
                sociability,
                persistence,
                shell_affinity,
                file_family_affinity_json,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                int(round_num),
                agent_id,
                float(traits_dict.get("curiosity", 0.0)),
                float(traits_dict.get("thrift", 0.0)),
                float(traits_dict.get("sociability", 0.0)),
                float(traits_dict.get("persistence", 0.0)),
                float(traits_dict.get("shell_affinity", 0.0)),
                json.dumps(file_family_affinity),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()

    def get_trait_snapshots(
        self,
        run_id: str,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        if agent_id is None:
            cursor.execute(
                """
                SELECT *
                FROM trait_snapshots
                WHERE run_id = ?
                ORDER BY round_num ASC, id ASC
                """,
                (run_id,),
            )
        else:
            cursor.execute(
                """
                SELECT *
                FROM trait_snapshots
                WHERE run_id = ? AND agent_id = ?
                ORDER BY round_num ASC, id ASC
                """,
                (run_id, agent_id),
            )

        rows: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            item = dict(row)
            try:
                item["file_family_affinity"] = json.loads(
                    item.get("file_family_affinity_json") or "{}"
                )
            except json.JSONDecodeError:
                item["file_family_affinity"] = {}
            rows.append(item)
        return rows

    def log_llm_call(
        self,
        run_id: str,
        turn: int,
        agent_id: str,
        model_name: str,
        system_prompt_snippet: str,
        user_prompt_snippet: str,
        response_snippet: str,
        duration_ms: float,
    ) -> None:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO llm_calls (
                run_id,
                turn,
                agent_id,
                model_name,
                system_prompt_snippet,
                user_prompt_snippet,
                response_snippet,
                duration_ms,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                int(turn),
                agent_id,
                model_name,
                (system_prompt_snippet or "")[:500],
                (user_prompt_snippet or "")[:500],
                (response_snippet or "")[:500],
                float(duration_ms),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()

    def get_llm_calls(
        self,
        run_id: str,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        if agent_id is None:
            cursor.execute(
                """
                SELECT *
                FROM llm_calls
                WHERE run_id = ?
                ORDER BY turn ASC, id ASC
                """,
                (run_id,),
            )
        else:
            cursor.execute(
                """
                SELECT *
                FROM llm_calls
                WHERE run_id = ? AND agent_id = ?
                ORDER BY turn ASC, id ASC
                """,
                (run_id, agent_id),
            )
        return [dict(row) for row in cursor.fetchall()]

    def log_run_start(self, run_id: str, config_snapshot: Dict[str, Any]) -> None:
        conn = self._ensure_connection()

        _SECRET_PATTERNS = ["key", "secret", "token", "password", "api_key"]

        def _is_credential(val: str) -> bool:
            if len(val) < 16 or " " in val:
                return False
            has_digit = any(c.isdigit() for c in val)
            has_special = any(not c.isalnum() for c in val)
            return has_digit and has_special

        def _redact(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: "***REDACTED***"
                    if any(p in k.lower() for p in _SECRET_PATTERNS)
                    else _redact(v)
                    for k, v in obj.items()
                }
            if isinstance(obj, list):
                return [_redact(v) for v in obj]
            if isinstance(obj, str) and _is_credential(obj):
                return "***REDACTED***"
            return obj

        _safe = _redact(config_snapshot)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO runs (run_id, config_snapshot) VALUES (?, ?)",
            (run_id, json.dumps(_safe)),
        )
        conn.commit()

    def log_action(
        self,
        run_id: str,
        turn: int,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]],
        result: Optional[str],
        zod_before: float,
        zod_after: float,
        duration_ms: int,
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Log a tool action.

        Args:
            run_id: Run identifier
            turn: Turn number in the experiment
            tool_name: Name of the tool invoked
            tool_args: Arguments passed to the tool (JSON serializable)
            result: Result from the tool (truncated if too long)
            zod_before: Zod balance before the action
            zod_after: Zod balance after the action
            duration_ms: Duration of the action in milliseconds
            agent_id: Optional agent identifier for multi-agent runs
        """
        conn = self._ensure_connection()
        cursor = conn.cursor()

        # Truncate result if too long (SQLite TEXT limit is ~1GB, but we'll be reasonable)
        result_str = None
        if result is not None:
            # Keep first 10KB of result for logging
            result_str = result[:10000] if len(result) > 10000 else result

        cursor.execute(
            """
            INSERT INTO actions 
            (run_id, turn, tool_name, tool_args, result, zod_before, zod_after, duration_ms, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                turn,
                tool_name,
                json.dumps(tool_args) if tool_args else None,
                result_str,
                zod_before,
                zod_after,
                duration_ms,
                agent_id,
            ),
        )
        conn.commit()

    def log_file_drop(self, run_id: str, filename: str, file_type: str) -> None:
        """
        Log when a file is dropped into the experiment.

        Args:
            run_id: Run identifier
            filename: Name of the file
            file_type: Type/category of file (e.g., 'python', 'config', 'test')
        """
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO files (run_id, filename, file_type, status, dropped_at)
            VALUES (?, ?, ?, 'dropped', CURRENT_TIMESTAMP)
            ON CONFLICT(run_id, filename) DO UPDATE SET
                file_type = excluded.file_type,
                status = 'dropped',
                dropped_at = CURRENT_TIMESTAMP
            """,
            (run_id, filename, file_type),
        )
        conn.commit()

    def log_file_process(self, run_id: str, filename: str, zod_earned: float) -> None:
        """
        Log when a file is processed and zod is earned.

        Args:
            run_id: Run identifier
            filename: Name of the file
            zod_earned: Zod earned from processing the file
        """
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE files 
            SET status = 'processed',
                processed_at = CURRENT_TIMESTAMP,
                zod_earned = ?
            WHERE run_id = ? AND filename = ?
            """,
            (zod_earned, run_id, filename),
        )
        conn.commit()

    def log_zod_transaction(
        self,
        run_id: str,
        amount: float,
        tx_type: str,
        reason: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Log a zod transaction.

        Args:
            run_id: Run identifier
            amount: Amount of zod (positive for income, negative for expense)
            tx_type: Transaction type (e.g., 'file_processed', 'tool_cost', 'initial_zod')
            reason: Optional description of the transaction
            agent_id: Optional agent identifier for multi-agent runs
        """
        # Get current balance after this transaction, scoped per agent
        conn = self._ensure_connection()
        cursor = conn.cursor()
        if agent_id:
            cursor.execute(
                """
                SELECT balance_after 
                FROM zod_transactions 
                WHERE run_id = ? AND agent_id = ?
                ORDER BY timestamp DESC, id DESC 
                LIMIT 1
                """,
                (run_id, agent_id),
            )
        else:
            cursor.execute(
                """
                SELECT balance_after 
                FROM zod_transactions 
                WHERE run_id = ? AND agent_id IS NULL
                ORDER BY timestamp DESC, id DESC 
                LIMIT 1
                """,
                (run_id,),
            )
        row = cursor.fetchone()
        current_balance = row["balance_after"] if row else 0.0
        balance_after = current_balance + amount

        cursor.execute(
            """
            INSERT INTO zod_transactions (run_id, amount, type, reason, balance_after, agent_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, amount, tx_type, reason, balance_after, agent_id),
        )
        conn.commit()

    def log_state_transition(
        self,
        run_id: str,
        turn: int,
        from_state: str,
        to_state: str,
        reason: Optional[str] = None,
        balance: Optional[float] = None,
        starvation_counter: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO state_transitions (run_id, turn, from_state, to_state, reason, balance, starvation_counter, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                turn,
                from_state,
                to_state,
                reason,
                balance,
                starvation_counter,
                agent_id,
            ),
        )
        conn.commit()

    def get_actions(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Get all actions for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            List of action dictionaries
        """
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                id, run_id, turn, tool_name, tool_args, result,
                zod_before, zod_after, duration_ms, agent_id, timestamp
            FROM actions 
            WHERE run_id = ? 
            ORDER BY turn, timestamp
            """,
            (run_id,),
        )

        actions = []
        for row in cursor.fetchall():
            action = dict(row)
            # Parse JSON fields
            if action["tool_args"]:
                action["tool_args"] = json.loads(action["tool_args"])
            actions.append(action)

        return actions

    def get_balance_history(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Get zod balance history for a run.

        Args:
            run_id: Run identifier

        Returns:
            List of balance snapshots with timestamps
        """
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                timestamp, amount, type, reason, balance_after, agent_id
            FROM zod_transactions 
            WHERE run_id = ? 
            ORDER BY timestamp, id
            """,
            (run_id,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_run_ids(self) -> List[str]:
        """
        Get list of all experiment run IDs.

        Returns:
            List of run IDs
        """
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT run_id FROM runs ORDER BY start_time DESC")
        return [row["run_id"] for row in cursor.fetchall()]

    def get_run_info(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific run.

        Args:
            run_id: Run identifier

        Returns:
            Run information dictionary or None if not found
        """
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT run_id, start_time, config_snapshot, end_time FROM runs WHERE run_id = ?",
            (run_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        info = dict(row)
        if info["config_snapshot"]:
            info["config_snapshot"] = json.loads(info["config_snapshot"])

        return info

    def get_file_stats(self, run_id: str) -> Dict[str, Any]:
        """
        Get file statistics for a run.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary with file statistics
        """
        conn = self._ensure_connection()
        cursor = conn.cursor()

        # Count files by status
        cursor.execute(
            """
            SELECT status, COUNT(*) as count, SUM(zod_earned) as total_zod
            FROM files 
            WHERE run_id = ? 
            GROUP BY status
            """,
            (run_id,),
        )

        stats = {"by_status": {}}
        for row in cursor.fetchall():
            stats["by_status"][row["status"]] = {
                "count": row["count"],
                "total_zod": row["total_zod"] or 0.0,
            }

        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_files,
                SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) as processed_files,
                SUM(zod_earned) as total_zod_earned
            FROM files 
            WHERE run_id = ?
            """,
            (run_id,),
        )

        row = cursor.fetchone()
        stats.update(dict(row))

        return stats

    def log_message(
        self,
        run_id: str,
        sender_id: str,
        recipient_id: str,
        content: str,
        round_num: int,
        turn: int,
    ) -> int:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO messages (run_id, sender_id, recipient_id, content, round_num, turn)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, sender_id, recipient_id, content, round_num, turn),
        )
        conn.commit()
        return cursor.lastrowid or 0

    def get_unread_messages(
        self, run_id: str, recipient_id: str
    ) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, sender_id, content, round_num, turn, timestamp
            FROM messages
            WHERE run_id = ? AND recipient_id = ? AND read = 0
            ORDER BY timestamp ASC
            """,
            (run_id, recipient_id),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def mark_messages_read(self, run_id: str, recipient_id: str) -> int:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE messages SET read = 1
            WHERE run_id = ? AND recipient_id = ? AND read = 0
            """,
            (run_id, recipient_id),
        )
        conn.commit()
        return cursor.rowcount

    def log_event(
        self,
        run_id: str,
        round_num: int,
        agent_id: str,
        event_type: str,
        details: Optional[str] = None,
        zod_delta: float = 0.0,
    ) -> int:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO event_ledger
                (run_id, round_num, agent_id, event_type, details, zod_delta)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, round_num, agent_id, event_type, details, zod_delta),
        )
        conn.commit()
        return cursor.lastrowid or 0

    def get_events(
        self,
        run_id: str,
        agent_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM event_ledger WHERE run_id = ?"
        params: list[Any] = [run_id]
        if agent_id is not None:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY id ASC"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def log_scout_report(
        self,
        run_id: str,
        turn: int,
        report: Dict[str, Any],
        report_json: str,
        *,
        agent_id: Optional[str] = None,
        target_filename: Optional[str] = None,
    ) -> int:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO scout_reports
                (
                    run_id,
                    turn,
                    agent_id,
                    report_id,
                    requesting_agent_id,
                    target_filename,
                    file_family,
                    claimed_pattern,
                    output_summary,
                    confidence,
                    verdict,
                    reasoning,
                    suggested_bonus,
                    report_json
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                turn,
                agent_id,
                str(report.get("report_id", "")),
                str(report.get("requesting_agent_id", "unknown")),
                target_filename,
                str(report.get("file_family", "")),
                str(report.get("claimed_pattern", "")),
                str(report.get("output_summary", "")),
                float(report.get("confidence", 0.0)),
                str(report.get("verdict", "no_sources")),
                str(report.get("reasoning", "")),
                float(report.get("suggested_bonus", 0.0)),
                report_json,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid or 0)

    def get_pending_scout_report_for_file(
        self,
        run_id: str,
        filename: str,
        *,
        agent_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        if agent_id is None:
            cursor.execute(
                """
                SELECT *
                FROM scout_reports
                WHERE run_id = ?
                  AND target_filename = ?
                  AND agent_id IS NULL
                  AND applied = 0
                ORDER BY id DESC
                LIMIT 1
                """,
                (run_id, filename),
            )
        else:
            cursor.execute(
                """
                SELECT *
                FROM scout_reports
                WHERE run_id = ?
                  AND target_filename = ?
                  AND agent_id = ?
                  AND applied = 0
                ORDER BY id DESC
                LIMIT 1
                """,
                (run_id, filename, agent_id),
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def mark_scout_report_applied(
        self,
        scout_report_row_id: int,
        *,
        applied_turn: int,
        applied_bonus: float,
    ) -> None:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE scout_reports
            SET applied = 1,
                applied_turn = ?,
                applied_bonus = ?
            WHERE id = ?
            """,
            (applied_turn, applied_bonus, scout_report_row_id),
        )
        conn.commit()

    def get_scout_reports(
        self,
        run_id: str,
        *,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        if agent_id is None:
            cursor.execute(
                """
                SELECT *
                FROM scout_reports
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            )
        else:
            cursor.execute(
                """
                SELECT *
                FROM scout_reports
                WHERE run_id = ? AND agent_id = ?
                ORDER BY id ASC
                """,
                (run_id, agent_id),
            )
        return [dict(row) for row in cursor.fetchall()]

    def log_overseer_evaluation(
        self,
        run_id: str,
        turn: int,
        agent_id: str,
        bonus: float,
        reasoning: str,
        tags: List[str],
        evaluation_json: str,
    ) -> int:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        report_id = f"overseer-{uuid4()}"
        tag_text = ",".join(tags)
        cursor.execute(
            """
            INSERT INTO scout_reports
                (
                    run_id,
                    turn,
                    agent_id,
                    report_id,
                    requesting_agent_id,
                    target_filename,
                    file_family,
                    claimed_pattern,
                    output_summary,
                    confidence,
                    verdict,
                    reasoning,
                    suggested_bonus,
                    report_json,
                    applied,
                    applied_bonus,
                    applied_turn
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                turn,
                agent_id,
                report_id,
                agent_id,
                None,
                "json",
                tag_text or "overseer_novelty",
                "overseer_evaluation",
                1.0,
                "supported" if bonus > 0 else "unclear",
                reasoning,
                float(bonus),
                evaluation_json,
                0,
                None,
                None,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid or 0)

    def get_agent_history(
        self,
        run_id: str,
        agent_id: str,
        last_n_turns: int = 10,
    ) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        max_turn = 0
        cursor.execute(
            """
            SELECT MAX(turn) as max_turn
            FROM actions
            WHERE run_id = ? AND agent_id = ?
            """,
            (run_id, agent_id),
        )
        row = cursor.fetchone()
        if row and row["max_turn"] is not None:
            max_turn = int(row["max_turn"])
        min_turn = max(0, max_turn - max(1, int(last_n_turns)) + 1)
        cursor.execute(
            """
            SELECT turn, tool_name, tool_args, result, duration_ms, timestamp
            FROM actions
            WHERE run_id = ?
              AND agent_id = ?
              AND turn >= ?
            ORDER BY turn ASC, id ASC
            """,
            (run_id, agent_id, min_turn),
        )
        history: List[Dict[str, Any]] = []
        for action_row in cursor.fetchall():
            action = dict(action_row)
            tool_args = action.get("tool_args")
            if isinstance(tool_args, str) and tool_args:
                try:
                    action["tool_args"] = json.loads(tool_args)
                except json.JSONDecodeError:
                    action["tool_args"] = None
            history.append(action)
        return history

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def _row_to_promotion_rule(self, row: sqlite3.Row) -> Any:
        promotion_module = importlib.import_module("petri_dish.promotion")
        PromotionRule = getattr(promotion_module, "PromotionRule")

        return PromotionRule(
            rule_id=str(row["rule_id"]),
            claimed_pattern=str(row["claimed_pattern"]),
            file_family=str(row["file_family"]),
            hit_count=int(row["hit_count"]),
            promoted=bool(row["promoted"]),
            promoted_at=self._parse_timestamp(row["promoted_at"]),
            bonus_multiplier=float(row["bonus_multiplier"]),
            created_at=self._parse_timestamp(row["created_at"]) or datetime.now(),
            last_hit_at=self._parse_timestamp(row["last_hit_at"]),
        )

    def record_scout_hit(
        self,
        claimed_pattern: str,
        file_family: str,
        bonus_multiplier: float = 1.5,
    ) -> Any:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM promotion_rules
            WHERE claimed_pattern = ? AND file_family = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (claimed_pattern, file_family),
        )
        row = cursor.fetchone()

        if row is None:
            rule_id = str(uuid4())
            cursor.execute(
                """
                INSERT INTO promotion_rules
                    (rule_id, claimed_pattern, file_family, hit_count, bonus_multiplier, last_hit_at)
                VALUES (?, ?, ?, 1, ?, CURRENT_TIMESTAMP)
                """,
                (rule_id, claimed_pattern, file_family, float(bonus_multiplier)),
            )
        else:
            cursor.execute(
                """
                UPDATE promotion_rules
                SET hit_count = hit_count + 1,
                    last_hit_at = CURRENT_TIMESTAMP
                WHERE rule_id = ?
                """,
                (str(row["rule_id"]),),
            )

        conn.commit()

        cursor.execute(
            """
            SELECT *
            FROM promotion_rules
            WHERE claimed_pattern = ? AND file_family = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (claimed_pattern, file_family),
        )
        updated = cursor.fetchone()
        if updated is None:
            raise RuntimeError("Failed to record scout hit for promotion rule")
        return self._row_to_promotion_rule(updated)

    def get_promotion_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM promotion_rules WHERE rule_id = ? LIMIT 1",
            (rule_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        data = dict(row)
        data["promoted"] = bool(data.get("promoted", 0))
        return data

    def get_promotion_rules(self, promoted_only: bool = False) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        if promoted_only:
            cursor.execute(
                "SELECT * FROM promotion_rules WHERE promoted = 1 ORDER BY id ASC"
            )
        else:
            cursor.execute("SELECT * FROM promotion_rules ORDER BY id ASC")
        rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            row["promoted"] = bool(row.get("promoted", 0))
        return rows

    def get_promoted_rules_for_family(self, file_family: str) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM promotion_rules
            WHERE file_family = ? AND promoted = 1
            ORDER BY bonus_multiplier DESC, id ASC
            """,
            (file_family,),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            row["promoted"] = bool(row.get("promoted", 0))
        return rows

    def promote_rule(self, rule_id: str) -> None:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE promotion_rules
            SET promoted = 1,
                promoted_at = CURRENT_TIMESTAMP
            WHERE rule_id = ?
            """,
            (rule_id,),
        )
        conn.commit()

    @contextlib.contextmanager
    def read_only_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for read-only connection (for dashboard).

        Usage:
            with db.read_only_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM runs")
        """
        if self.db_path == ":memory:":
            # In-memory DB can't have separate read-only connection
            yield self._ensure_connection()
        else:
            # Create a separate read-only connection
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()


def create_test_db() -> LoggingDB:
    """Create a test database for verification."""
    import tempfile
    import os

    # Create temporary database file
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_petri_dish.db")

    db = LoggingDB(db_path)
    db.connect()

    return db
