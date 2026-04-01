#!/usr/bin/env python3
"""
Test script for LoggingDB implementation.
Generates evidence for task-3 verification.
"""

import sys
import os
import tempfile
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from petri_dish.logging_db import LoggingDB


def test_database_initialization():
    """Test 1: Database initializes with correct schema."""
    print("=== Test 1: Database Initialization ===")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Initialize database
        db = LoggingDB(db_path)
        db.connect()

        # Check if tables exist
        cursor = db._conn.cursor()

        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row["name"] for row in cursor.fetchall()]

        print(f"Tables found: {sorted(tables)}")

        # Check for required tables
        required_tables = {"runs", "actions", "files", "zod_transactions"}
        missing_tables = required_tables - set(tables)

        if missing_tables:
            print(f"ERROR: Missing tables: {missing_tables}")
            return False

        print("✓ All required tables exist")

        # Check WAL mode
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()["journal_mode"]
        print(f"Journal mode: {journal_mode}")

        if journal_mode.lower() != "wal":
            print(f"ERROR: Journal mode is not WAL (got: {journal_mode})")
            return False

        print("✓ WAL mode is enabled")

        # Check other pragmas
        cursor.execute("PRAGMA synchronous")
        synchronous = cursor.fetchone()[0]
        print(f"Synchronous mode: {synchronous}")

        cursor.execute("PRAGMA busy_timeout")
        busy_timeout = cursor.fetchone()[0]
        print(f"Busy timeout: {busy_timeout}ms")

        cursor.execute("PRAGMA cache_size")
        cache_size = cursor.fetchone()[0]
        print(
            f"Cache size: {cache_size} pages (approx {abs(cache_size) * 1024 / 1024:.1f}MB)"
        )

        # Verify schema structure
        print("\nTable schemas:")
        for table in required_tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            print(f"\n{table}:")
            for col in columns:
                print(f"  {col['name']} ({col['type']})")

        return True

    finally:
        # Clean up
        db.close()
        os.unlink(db_path)


def test_action_logging():
    """Test 2: Action logging works end-to-end."""
    print("\n=== Test 2: Action Logging ===")

    # Create in-memory database
    db = LoggingDB(":memory:")

    with db:
        # Create a test run
        run_id = "test-run-123"
        config = {"test": True, "version": "1.0"}
        db.log_run_start(run_id, config)

        # Log some actions
        db.log_action(
            run_id=run_id,
            turn=1,
            tool_name="bash",
            tool_args={"command": "ls -la", "description": "List files"},
            result="total 32\ndrwxr-xr-x ...",
            zod_before=100.0,
            zod_after=95.0,
            duration_ms=150,
        )

        db.log_action(
            run_id=run_id,
            turn=2,
            tool_name="read",
            tool_args={"filePath": "/tmp/test.txt"},
            result="File contents...",
            zod_before=95.0,
            zod_after=90.0,
            duration_ms=50,
        )

        # Verify actions were logged
        actions = db.get_actions(run_id)
        print(f"Logged {len(actions)} actions")

        if len(actions) != 2:
            print(f"ERROR: Expected 2 actions, got {len(actions)}")
            return False

        print("✓ Actions logged successfully")

        # Verify action details
        for i, action in enumerate(actions, 1):
            print(f"\nAction {i}:")
            print(f"  Turn: {action['turn']}")
            print(f"  Tool: {action['tool_name']}")
            print(f"  Zod: {action['zod_before']} → {action['zod_after']}")
            print(f"  Duration: {action['duration_ms']}ms")

        # Test file logging
        db.log_file_drop(run_id, "test.py", "python")
        db.log_file_drop(run_id, "config.yaml", "config")
        db.log_file_process(run_id, "test.py", 10.5)

        # Test zod logging
        db.log_zod_transaction(run_id, 100.0, "initial_zod", "Starting zod")
        db.log_zod_transaction(run_id, -5.0, "tool_cost", "bash command")
        db.log_zod_transaction(run_id, 10.5, "file_processed", "test.py processed")

        # Get balance history
        balance_history = db.get_balance_history(run_id)
        print(f"\nBalance history has {len(balance_history)} transactions")

        if len(balance_history) != 3:
            print(f"ERROR: Expected 3 zod transactions, got {len(balance_history)}")
            return False

        print("✓ Zod transactions logged successfully")

        # Verify final balance
        final_balance = balance_history[-1]["balance_after"]
        expected_balance = 100.0 - 5.0 + 10.5  # 105.5
        print(f"Final balance: {final_balance} (expected: {expected_balance})")

        if abs(final_balance - expected_balance) > 0.001:
            print(f"ERROR: Balance mismatch")
            return False

        print("✓ Balance calculation correct")

        # Get run IDs
        run_ids = db.get_run_ids()
        print(f"\nRun IDs in database: {run_ids}")

        if run_id not in run_ids:
            print(f"ERROR: Run ID {run_id} not found in get_run_ids()")
            return False

        print("✓ Run ID retrieval works")

        return True


def generate_evidence():
    """Generate evidence files for task verification."""
    print("\n=== Generating Evidence ===")

    evidence_dir = ".sisyphus/evidence"
    os.makedirs(evidence_dir, exist_ok=True)

    # Test 1: Schema evidence
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db = LoggingDB(db_path)
        db.connect()

        # Get schema information
        cursor = db._conn.cursor()

        # Get all table schemas
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor.fetchall()]

        schema_info = []
        schema_info.append("SQLite Database Schema")
        schema_info.append("=" * 50)
        schema_info.append(f"Generated: {datetime.now().isoformat()}")
        schema_info.append(f"Database: {db_path}")
        schema_info.append("")

        for table in tables:
            schema_info.append(f"Table: {table}")
            schema_info.append("-" * 30)

            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()

            for col in columns:
                pk = " PRIMARY KEY" if col["pk"] else ""
                notnull = " NOT NULL" if col["notnull"] else ""
                default = f" DEFAULT {col['dflt_value']}" if col["dflt_value"] else ""
                schema_info.append(
                    f"  {col['name']:20} {col['type']:15}{pk}{notnull}{default}"
                )

            # Get indexes for this table
            cursor.execute(f"PRAGMA index_list({table})")
            indexes = cursor.fetchall()
            if indexes:
                schema_info.append("")
                schema_info.append("  Indexes:")
                for idx in indexes:
                    schema_info.append(f"    {idx['name']} (unique: {idx['unique']})")

            schema_info.append("")

        # Get pragma settings
        schema_info.append("Database Settings")
        schema_info.append("-" * 30)

        pragmas = [
            ("journal_mode", "Journal mode"),
            ("synchronous", "Synchronous mode"),
            ("busy_timeout", "Busy timeout (ms)"),
            ("cache_size", "Cache size (pages)"),
            ("foreign_keys", "Foreign keys enabled"),
        ]

        for pragma, desc in pragmas:
            cursor.execute(f"PRAGMA {pragma}")
            result = cursor.fetchone()
            value = result[0] if result else "N/A"
            schema_info.append(f"{desc:25} {value}")

        # Write schema evidence
        schema_file = os.path.join(evidence_dir, "task-3-schema.txt")
        with open(schema_file, "w") as f:
            f.write("\n".join(schema_info))

        print(f"✓ Schema evidence written to {schema_file}")

        # Test 2: Action logging evidence
        action_info = []
        action_info.append("Action Logging Test")
        action_info.append("=" * 50)
        action_info.append(f"Generated: {datetime.now().isoformat()}")
        action_info.append("")

        # Create a test run and log actions
        run_id = "evidence-run-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        config = {
            "test": True,
            "task": "T3-SQLite-Logging",
            "timestamp": datetime.now().isoformat(),
        }

        db.log_run_start(run_id, config)

        # Log sample actions
        actions_to_log = [
            {
                "turn": 1,
                "tool": "bash",
                "args": {"command": "pwd", "description": "Get current directory"},
                "result": "/home/ezotoff/projects/petri-dish",
                "zod_before": 100.0,
                "zod_after": 99.5,
                "duration": 100,
            },
            {
                "turn": 2,
                "tool": "read",
                "args": {"filePath": "src/petri_dish/logging_db.py"},
                "result": "class LoggingDB: ...",
                "zod_before": 99.5,
                "zod_after": 99.0,
                "duration": 75,
            },
            {
                "turn": 3,
                "tool": "write",
                "args": {"filePath": "test_output.txt", "content": "Test content"},
                "result": "File written successfully",
                "zod_before": 99.0,
                "zod_after": 98.5,
                "duration": 200,
            },
        ]

        for action in actions_to_log:
            db.log_action(
                run_id=run_id,
                turn=action["turn"],
                tool_name=action["tool"],
                tool_args=action["args"],
                result=action["result"],
                zod_before=action["zod_before"],
                zod_after=action["zod_after"],
                duration_ms=action["duration"],
            )

        db.log_file_drop(run_id, "logging_db.py", "python")
        db.log_file_drop(run_id, "config.yaml", "yaml")
        db.log_file_process(run_id, "logging_db.py", 25.0)

        db.log_zod_transaction(run_id, 100.0, "initial", "Initial balance")
        db.log_zod_transaction(run_id, -0.5, "tool", "bash command")
        db.log_zod_transaction(run_id, -0.5, "tool", "read file")
        db.log_zod_transaction(run_id, -0.5, "tool", "write file")
        db.log_zod_transaction(run_id, 25.0, "file", "logging_db.py processed")

        # Retrieve and display logged data
        action_info.append(f"Run ID: {run_id}")
        action_info.append("")

        # Get actions
        actions = db.get_actions(run_id)
        action_info.append(f"Actions logged: {len(actions)}")
        action_info.append("")

        for i, action in enumerate(actions, 1):
            action_info.append(f"Action {i}:")
            action_info.append(f"  Turn: {action['turn']}")
            action_info.append(f"  Tool: {action['tool_name']}")
            action_info.append(f"  Duration: {action['duration_ms']}ms")
            action_info.append(f"  Zod: {action['zod_before']} → {action['zod_after']}")
            if action["tool_args"]:
                args_str = json.dumps(action["tool_args"], indent=2).replace(
                    "\n", "\n    "
                )
                action_info.append(f"  Args: {args_str}")
            action_info.append("")

        # Get balance history
        balance_history = db.get_balance_history(run_id)
        action_info.append(f"Zod transactions: {len(balance_history)}")
        action_info.append("")

        for i, tx in enumerate(balance_history, 1):
            action_info.append(f"Transaction {i}:")
            action_info.append(f"  Time: {tx['timestamp']}")
            action_info.append(f"  Type: {tx['type']}")
            action_info.append(f"  Amount: {tx['amount']:+.2f}")
            action_info.append(f"  Balance after: {tx['balance_after']:.2f}")
            if tx["reason"]:
                action_info.append(f"  Reason: {tx['reason']}")
            action_info.append("")

        # Get file stats
        file_stats = db.get_file_stats(run_id)
        action_info.append("File Statistics:")
        action_info.append(f"  Total files: {file_stats.get('total_files', 0)}")
        action_info.append(f"  Processed files: {file_stats.get('processed_files', 0)}")
        action_info.append(
            f"  Total zod earned: {file_stats.get('total_zod_earned', 0):.2f}"
        )

        if "by_status" in file_stats:
            action_info.append("  Files by status:")
            for status, stats in file_stats["by_status"].items():
                action_info.append(
                    f"    {status}: {stats['count']} files, {stats['total_zod']:.2f} zod"
                )

        # Write action logging evidence
        action_file = os.path.join(evidence_dir, "task-3-action-log.txt")
        with open(action_file, "w") as f:
            f.write("\n".join(action_info))

        print(f"✓ Action logging evidence written to {action_file}")

        return True

    finally:
        db.close()
        os.unlink(db_path)


def main():
    """Run all tests and generate evidence."""
    print("Testing LoggingDB implementation for Task 3")
    print("=" * 60)

    all_passed = True

    # Run tests
    if not test_database_initialization():
        all_passed = False

    if not test_action_logging():
        all_passed = False

    # Generate evidence
    if all_passed:
        if not generate_evidence():
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Evidence generated.")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
