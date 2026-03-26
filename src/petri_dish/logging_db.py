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
from typing import Optional, Dict, Any, List


class LoggingDB:
    """SQLite database for logging experiment runs, actions, files, and credit transactions."""

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
            uri = f"file:{self.db_path}?mode=rw"
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
                credits_before REAL NOT NULL,
                credits_after REAL NOT NULL,
                duration_ms INTEGER NOT NULL,
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
                credits_earned REAL DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                UNIQUE(run_id, filename)
            )
        """)

        # Create credit_transactions table - tracks all credit changes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS credit_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                amount REAL NOT NULL,
                type TEXT NOT NULL,
                reason TEXT,
                balance_after REAL NOT NULL,
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
            "CREATE INDEX IF NOT EXISTS idx_credits_run_id ON credit_transactions(run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_credits_timestamp ON credit_transactions(timestamp)"
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
        credits_before: float,
        credits_after: float,
        duration_ms: int,
    ) -> None:
        """
        Log a tool action.

        Args:
            run_id: Run identifier
            turn: Turn number in the experiment
            tool_name: Name of the tool invoked
            tool_args: Arguments passed to the tool (JSON serializable)
            result: Result from the tool (truncated if too long)
            credits_before: Credit balance before the action
            credits_after: Credit balance after the action
            duration_ms: Duration of the action in milliseconds
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
            (run_id, turn, tool_name, tool_args, result, credits_before, credits_after, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                turn,
                tool_name,
                json.dumps(tool_args) if tool_args else None,
                result_str,
                credits_before,
                credits_after,
                duration_ms,
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

    def log_file_process(
        self, run_id: str, filename: str, credits_earned: float
    ) -> None:
        """
        Log when a file is processed and credits are earned.

        Args:
            run_id: Run identifier
            filename: Name of the file
            credits_earned: Credits earned from processing the file
        """
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE files 
            SET status = 'processed',
                processed_at = CURRENT_TIMESTAMP,
                credits_earned = ?
            WHERE run_id = ? AND filename = ?
            """,
            (credits_earned, run_id, filename),
        )
        conn.commit()

    def log_credit(
        self, run_id: str, amount: float, tx_type: str, reason: Optional[str] = None
    ) -> None:
        """
        Log a credit transaction.

        Args:
            run_id: Run identifier
            amount: Amount of credits (positive for income, negative for expense)
            tx_type: Transaction type (e.g., 'file_processed', 'tool_cost', 'initial_balance')
            reason: Optional description of the transaction
        """
        # Get current balance after this transaction
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT balance_after 
            FROM credit_transactions 
            WHERE run_id = ? 
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
            INSERT INTO credit_transactions (run_id, amount, type, reason, balance_after)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, amount, tx_type, reason, balance_after),
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
                credits_before, credits_after, duration_ms, timestamp
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
        Get credit balance history for a run.

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
                timestamp, amount, type, reason, balance_after
            FROM credit_transactions 
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
            SELECT status, COUNT(*) as count, SUM(credits_earned) as total_credits
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
                "total_credits": row["total_credits"] or 0.0,
            }

        # Total files and credits
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_files,
                SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) as processed_files,
                SUM(credits_earned) as total_credits_earned
            FROM files 
            WHERE run_id = ?
            """,
            (run_id,),
        )

        row = cursor.fetchone()
        stats.update(dict(row))

        return stats

    @contextlib.contextmanager
    def read_only_connection(self):
        """
        Context manager for read-only connection (for dashboard).

        Usage:
            with db.read_only_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM runs")
        """
        if self.db_path == ":memory:":
            # In-memory DB can't have separate read-only connection
            yield self._conn
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
