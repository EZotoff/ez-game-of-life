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
from typing import Optional, Dict, Any, List, Iterator


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

        conn.commit()

    def log_run_start(self, run_id: str, config_snapshot: Dict[str, Any]) -> None:
        """
        Log the start of an experiment run.

        Args:
            run_id: Unique identifier for the run (UUID string)
            config_snapshot: Configuration dictionary for the run
        """
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO runs (run_id, config_snapshot) VALUES (?, ?)",
            (run_id, json.dumps(config_snapshot)),
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

    @contextlib.contextmanager
    def read_only_connection(self) -> Iterator[sqlite3.Connection]:
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
