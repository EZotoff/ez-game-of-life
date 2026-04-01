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


AgentOrchestrator = import_module("petri_dish.orchestrator").AgentOrchestrator
CreditEconomy = import_module("petri_dish.economy").CreditEconomy
SharedEconomy = import_module("petri_dish.economy").SharedEconomy
FileValidator = import_module("petri_dish.validators").FileValidator
ToolRegistry = import_module("petri_dish.tools.registry").ToolRegistry


class TestRewardNotifications:
    """Tests for deterministic reward notifications injected after FileValidator credits."""

    def _make_fake_sandbox(self, outgoing_files: dict[str, str] | None = None):
        """Create a FakeSandboxManager that returns specified files from /env/outgoing/."""
        files = outgoing_files or {}

        class FakeSandbox:
            def __init__(self):
                self.containers: dict[str, Any] = {}
                self._deleted: list[str] = []

            def create_container(
                self, name, memory_host_path=None, shared_volume_host_path=None
            ):
                cid = f"fake-{name}"
                self.containers[cid] = True
                return cid

            def destroy_container(self, cid):
                self.containers.pop(cid, None)

            def get_container_stats(self, cid):
                return {"running": cid in self.containers}

            def exec_in_container(self, cid, command):
                if "ls /env/outgoing/" in command:
                    return "\n".join(files.keys()) if files else ""
                if command.startswith("rm -f"):
                    self._deleted.append(command)
                return ""

            def read_file(self, cid, path):
                fname = path.split("/")[-1]
                return files.get(fname, "")

            def write_file(self, cid, path, content):
                pass

            def list_directory(self, cid, path):
                return []

        return FakeSandbox()

    def test_single_agent_reward_notification_injected(self):
        """When FileValidator awards credits, a notification message is injected."""
        csv_content = "name,age,email\nAlice,30,a@b.com\nBob,25,c@d.com\n"
        sandbox = self._make_fake_sandbox({"data_001_csv_easy.csv": csv_content})

        settings = Settings()
        db = LoggingDB(":memory:")
        db.connect()
        db.log_run_start("test-reward", "{}")

        economy = CreditEconomy(settings)
        validator = FileValidator(settings)
        registry = ToolRegistry()

        orch = AgentOrchestrator(
            settings=settings,
            llm_client=None,
            tool_parser=None,
            tool_registry=registry,
            credit_economy=economy,
            sandbox_manager=sandbox,
            logging_db=db,
            file_validator=validator,
        )
        orch._container_id = "fake-test"
        orch._messages = []
        orch._turn = 1

        initial_balance = economy.get_balance()
        orch._validate_outputs("test-reward")

        assert economy.get_balance() > initial_balance

        reward_msgs = [m for m in orch._messages if "earned" in m.get("content", "")]
        assert len(reward_msgs) == 1
        assert "📈" in reward_msgs[0]["content"]
        assert "credits" in reward_msgs[0]["content"]
        assert reward_msgs[0]["role"] == "user"

        db.close()

    def test_single_agent_no_notification_when_no_credits(self):
        """No notification when no valid files are in /env/outgoing/."""
        sandbox = self._make_fake_sandbox({})

        settings = Settings()
        db = LoggingDB(":memory:")
        db.connect()
        db.log_run_start("test-no-reward", "{}")

        economy = CreditEconomy(settings)
        validator = FileValidator(settings)
        registry = ToolRegistry()

        orch = AgentOrchestrator(
            settings=settings,
            llm_client=None,
            tool_parser=None,
            tool_registry=registry,
            credit_economy=economy,
            sandbox_manager=sandbox,
            logging_db=db,
            file_validator=validator,
        )
        orch._container_id = "fake-test"
        orch._messages = []
        orch._turn = 1

        orch._validate_outputs("test-no-reward")

        reward_msgs = [m for m in orch._messages if "earned" in m.get("content", "")]
        assert len(reward_msgs) == 0

        db.close()

    def test_multi_agent_reward_notification_injected(self):
        """Multi-agent: reward notification injected into correct agent's messages."""
        csv_content = "name,age,email\nAlice,30,a@b.com\nBob,25,c@d.com\n"
        sandbox = self._make_fake_sandbox({"data_001_csv_easy.csv": csv_content})

        settings = Settings()
        settings.multi_agent_enabled = True
        settings.multi_agent_count = 2

        db = LoggingDB(":memory:")
        db.connect()
        db.log_run_start("test-multi-reward", "{}")

        shared_econ = SharedEconomy(settings, ["agent-0", "agent-1"])
        validator = FileValidator(settings)

        orch = MultiAgentOrchestrator(
            settings=settings,
            shared_economy=shared_econ,
            sandbox_manager=sandbox,
            logging_db=db,
            file_validator=validator,
            agent_names=["agent-0", "agent-1"],
        )
        orch._agent_containers = {"agent-0": "fake-agent-0", "agent-1": "fake-agent-1"}

        initial = shared_econ.get_agent_economy("agent-0").get_balance()
        orch._validate_agent_outputs("test-multi-reward", "agent-0", 1, "fake-agent-0")

        assert shared_econ.get_agent_economy("agent-0").get_balance() > initial

        a0_reward = [
            m
            for m in orch._agent_messages["agent-0"]
            if "earned" in m.get("content", "")
        ]
        a1_reward = [
            m
            for m in orch._agent_messages["agent-1"]
            if "earned" in m.get("content", "")
        ]
        assert len(a0_reward) == 1
        assert len(a1_reward) == 0
        assert "📈" in a0_reward[0]["content"]

        db.close()


PromptManager = import_module("petri_dish.prompt").PromptManager


class TestEnvDiscoverability:
    """Tests for the /env/ discoverability hint in prompts."""

    def test_single_agent_prompt_has_env_hint(self):
        """Single-agent prompt mentions /env/ structure."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mod_path = f.name

        try:
            pm = PromptManager(modifications_path=mod_path)
            prompt = pm.build_system_prompt(
                tools=[{"name": "file_list", "description": "List files"}],
                tool_costs={"file_list": 0.0},
                balance=100.0,
                state_summary="Turn: 1",
                has_persistent_memory=False,
                agent_state="active",
                starvation_remaining=7,
            )
            assert "/env/" in prompt
            assert "discover" in prompt.lower()
        finally:
            os.unlink(mod_path)

    def test_multi_agent_prompt_has_env_hint(self):
        """Multi-agent prompt mentions /env/ structure."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mod_path = f.name

        try:
            pm = PromptManager(modifications_path=mod_path)
            prompt = pm.build_multi_agent_system_prompt(
                agent_id="agent-0",
                tools=[{"name": "file_list", "description": "List files"}],
                tool_costs={"file_list": 0.0},
                balance=100.0,
                agent_state="active",
                starvation_remaining=7,
                agent_summaries=[],
                actions_per_turn=4,
            )
            assert "/env/" in prompt
            assert "discover" in prompt.lower()
        finally:
            os.unlink(mod_path)
