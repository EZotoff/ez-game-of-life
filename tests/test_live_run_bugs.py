"""Targeted tests for bugs discovered during live 2-agent run."""

from __future__ import annotations

import sys
import tempfile
import os
from pathlib import Path
from importlib import import_module
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LoggingDB = import_module("petri_dish.logging_db").LoggingDB
MultiAgentOrchestrator = import_module("petri_dish.orchestrator").MultiAgentOrchestrator
SandboxManager = import_module("petri_dish.sandbox").SandboxManager
Settings = import_module("petri_dish.config").Settings


class TestLiveRunBugs:
    def test_per_agent_balance_after_in_credit_transactions(self):
        """Test 1: Verify log_credit scopes balance_after by agent_id."""
        db = LoggingDB(":memory:")
        db.connect()

        db.log_run_start("test", "{}")
        db.log_credit("test", 100.0, "initial_balance", agent_id="agent-0")
        db.log_credit("test", 200.0, "initial_balance", agent_id="agent-1")
        db.log_credit("test", -10.0, "tool_cost", agent_id="agent-0")

        history = db.get_balance_history("test")
        agent0_balances = [h for h in history if h.get("agent_id") == "agent-0"]
        agent1_balances = [h for h in history if h.get("agent_id") == "agent-1"]
        agent0_last = agent0_balances[-1] if agent0_balances else None
        agent1_last = agent1_balances[-1] if agent1_balances else None

        assert agent0_last is not None
        assert agent0_last["balance_after"] == 90.0
        assert agent1_last is not None
        assert agent1_last["balance_after"] == 200.0

        db.close()

    def test_event_ledger_populated_from_ecology_drops(self):
        """Test 2: Verify orchestrator._drop_ecology_files() logs events."""

        class FakeSandboxManager:
            def __init__(self):
                self.containers = {}

            def create_container(
                self, name, memory_host_path=None, shared_volume_host_path=None
            ):
                cid = f"fake-{name}"
                self.containers[cid] = {"name": name, "shared": shared_volume_host_path}
                return cid

            def destroy_container(self, container_id):
                self.containers.pop(container_id, None)

            def get_container_stats(self, container_id):
                return {"running": container_id in self.containers}

            def exec_in_container(self, container_id, command):
                return ""

            def read_file(self, container_id, path):
                return ""

            def write_file(self, container_id, path, content):
                pass

            def list_directory(self, container_id, path):
                return []

        class FakeEcology:
            def schedule_drops(self, round_num):
                return [("test.csv", "a,b\n1,2")]

            def drop_file(self, sandbox_manager, container_id, filename, content):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            db = LoggingDB(":memory:")
            db.connect()
            db.log_run_start("test-run", "{}")

            settings = Settings()
            settings.multi_agent_shared_filesystem = True

            orchestrator = MultiAgentOrchestrator(
                settings=settings,
                sandbox_manager=FakeSandboxManager(),
                ecology=FakeEcology(),
                logging_db=db,
            )

            orchestrator._shared_volume_dir = temp_dir
            orchestrator.agent_ids = ["agent-0", "agent-1"]
            orchestrator._drop_ecology_files("test-run")

            events = db.get_events("test-run")
            ecology_drop_events = [
                e for e in events if e["event_type"] == "ecology_drop"
            ]

            assert len(ecology_drop_events) > 0
            drop_event = ecology_drop_events[0]
            assert "file=test.csv" in drop_event["details"]

            db.close()

    def test_sqlite_file_based_db_creation(self):
        """Test 3: Verify LoggingDB works with real file paths (not :memory:)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            db = LoggingDB(db_path)
            db.connect()

            assert os.path.exists(db_path)
            db.log_run_start("test-run-file", "{}")
            run_info = db.get_run_info("test-run-file")

            assert run_info is not None
            assert run_info["run_id"] == "test-run-file"

            db.close()

    def test_container_name_uniqueness_for_multi_agent(self):
        """Test 4: Verify container name uniqueness ([:8] truncation fix)."""
        name0 = SandboxManager._container_name(None, "run-abc123-agent-0")
        name1 = SandboxManager._container_name(None, "run-abc123-agent-1")

        assert name0 != name1
        assert name0 == "petri-run-abc123-agent-0"
        assert name1 == "petri-run-abc123-agent-1"
