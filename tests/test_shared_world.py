"""Comprehensive tests for Wave C shared-world features."""

from __future__ import annotations

import asyncio
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Settings = import_module("petri_dish.config").Settings
AgentState = import_module("petri_dish.economy").AgentState
SharedReserve = import_module("petri_dish.economy").SharedReserve
LoggingDB = import_module("petri_dish.logging_db").LoggingDB
MultiAgentOrchestrator = import_module("petri_dish.orchestrator").MultiAgentOrchestrator
PromptManager = import_module("petri_dish.prompt").PromptManager
FileValidator = import_module("petri_dish.validators").FileValidator


class FakeSandboxManager:
    def __init__(self) -> None:
        self.created_container_ids: list[str] = []
        self.destroyed_container_ids: list[str] = []
        self.create_calls: list[dict[str, str | None]] = []

    def create_container(
        self,
        run_id: str,
        memory_host_path: str | None = None,
        shared_volume_host_path: str | None = None,
    ) -> str:
        cid = f"fake-{run_id}"
        self.created_container_ids.append(cid)
        self.create_calls.append(
            {
                "run_id": run_id,
                "memory_host_path": memory_host_path,
                "shared_volume_host_path": shared_volume_host_path,
            }
        )
        return cid

    def destroy_container(self, cid: str) -> None:
        self.destroyed_container_ids.append(cid)

    def exec_in_container(self, cid: str, cmd: str) -> str:
        _ = (cid, cmd)
        return ""

    def get_container_stats(self, cid: str) -> dict[str, float]:
        _ = cid
        return {
            "cpu_percent": 0.0,
            "memory_usage_mb": 1.0,
            "memory_limit_mb": 512.0,
        }

    def read_file(self, cid: str, path: str) -> str:
        _ = (cid, path)
        return ""

    def list_directory(self, cid: str, path: str) -> str:
        _ = (cid, path)
        return ""


class FakeOllamaClient:
    def __init__(self, responses: list[tuple[str, list[Any]]] | None = None) -> None:
        self._responses = responses or []
        self._idx = 0

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[Any]] | None:
        _ = (system_prompt, messages, tools)
        if self._idx >= len(self._responses):
            return "", []
        out = self._responses[self._idx]
        self._idx += 1
        return out


class FakeEcology:
    def __init__(self, drops: dict[int, list[tuple[str, str]]] | None = None) -> None:
        self._drops = drops or {}
        self.drop_calls: list[tuple[str, str, str]] = []
        self.schedule_calls: list[int] = []

    def schedule_drops(self, round_num: int) -> list[tuple[str, str]]:
        self.schedule_calls.append(round_num)
        return list(self._drops.get(round_num, []))

    def drop_file(
        self,
        sandbox_manager: Any,
        container_id: str,
        filename: str,
        content: str,
    ) -> None:
        _ = sandbox_manager
        self.drop_calls.append((container_id, filename, content))


def _db_with_run(run_id: str = "run-1") -> Any:
    db = LoggingDB(":memory:")
    db.connect()
    db.log_run_start(run_id, {"test": True})
    return db


def _build_multi_agent(
    settings: Any,
    *,
    llm_responses: dict[str, list[tuple[str, list[Any]]]] | None = None,
    agent_names: list[str] | None = None,
    ecology: Any | None = None,
) -> tuple[Any, Any, FakeSandboxManager, Any]:
    names = agent_names or ["agent-a", "agent-b"]
    sandbox = FakeSandboxManager()
    logging_db = LoggingDB(":memory:")
    shared_economy = SharedReserve(settings=settings, agent_ids=names)
    llm_clients = {
        aid: FakeOllamaClient((llm_responses or {}).get(aid, [])) for aid in names
    }
    orchestrator = MultiAgentOrchestrator(
        settings=settings,
        shared_economy=shared_economy,
        sandbox_manager=sandbox,
        logging_db=logging_db,
        file_validator=FileValidator(settings=settings),
        llm_clients=llm_clients,
        agent_names=names,
        ecology=ecology,
    )
    return orchestrator, logging_db, sandbox, shared_economy


class TestEventLedger:
    def test_log_event_returns_integer_id(self):
        db = _db_with_run("r1")

        event_id = db.log_event("r1", 1, "agent-a", "stripped")

        assert isinstance(event_id, int)
        assert event_id > 0

    def test_get_events_returns_empty_for_wrong_run_id(self):
        db = _db_with_run("r1")
        db.log_run_start("r2", {"test": True})
        _ = db.log_event("r1", 1, "agent-a", "stripped")

        rows = db.get_events("unknown-run")

        assert rows == []

    def test_get_events_returns_all_for_run_in_insert_order(self):
        db = _db_with_run("r1")
        _ = db.log_event("r1", 1, "agent-a", "stripped")
        _ = db.log_event("r1", 2, "agent-a", "death")
        _ = db.log_event("r1", 3, "agent-a", "reentry")

        rows = db.get_events("r1")

        assert [r["event_type"] for r in rows] == ["stripped", "death", "reentry"]

    def test_get_events_filters_by_agent_id(self):
        db = _db_with_run("r1")
        _ = db.log_event("r1", 1, "agent-a", "stripped")
        _ = db.log_event("r1", 1, "agent-b", "stripped")

        rows = db.get_events("r1", agent_id="agent-b")

        assert len(rows) == 1
        assert rows[0]["agent_id"] == "agent-b"

    def test_get_events_filters_by_event_type(self):
        db = _db_with_run("r1")
        _ = db.log_event("r1", 1, "agent-a", "stripped")
        _ = db.log_event("r1", 2, "agent-a", "death")

        rows = db.get_events("r1", event_type="death")

        assert len(rows) == 1
        assert rows[0]["event_type"] == "death"

    def test_get_events_filters_by_agent_id_and_event_type(self):
        db = _db_with_run("r1")
        _ = db.log_event("r1", 1, "agent-a", "stripped")
        _ = db.log_event("r1", 2, "agent-a", "death")
        _ = db.log_event("r1", 3, "agent-b", "death")

        rows = db.get_events("r1", agent_id="agent-a", event_type="death")

        assert len(rows) == 1
        assert rows[0]["agent_id"] == "agent-a"
        assert rows[0]["event_type"] == "death"

    def test_get_events_isolated_across_runs(self):
        db = _db_with_run("r1")
        db.log_run_start("r2", {"test": True})
        _ = db.log_event("r1", 1, "agent-a", "stripped")
        _ = db.log_event("r2", 1, "agent-a", "death")

        run1_rows = db.get_events("r1")
        run2_rows = db.get_events("r2")

        assert len(run1_rows) == 1
        assert run1_rows[0]["event_type"] == "stripped"
        assert len(run2_rows) == 1
        assert run2_rows[0]["event_type"] == "death"

    def test_log_event_persists_details_text(self):
        db = _db_with_run("r1")

        _ = db.log_event("r1", 5, "agent-a", "reentry", details="balance=20.0")
        rows = db.get_events("r1")

        assert rows[0]["details"] == "balance=20.0"

    def test_log_event_accepts_none_details(self):
        db = _db_with_run("r1")

        _ = db.log_event("r1", 5, "agent-a", "reentry", details=None)
        rows = db.get_events("r1")

        assert rows[0]["details"] is None

    def test_log_event_stores_zod_delta(self):
        db = _db_with_run("r1")

        _ = db.log_event("r1", 1, "agent-a", "reentry", zod_delta=12.5)
        rows = db.get_events("r1")

        assert rows[0]["zod_delta"] == pytest.approx(12.5)

    def test_log_event_default_zod_delta_is_zero(self):
        db = _db_with_run("r1")

        _ = db.log_event("r1", 1, "agent-a", "stripped")
        rows = db.get_events("r1")

        assert rows[0]["zod_delta"] == pytest.approx(0.0)

    def test_get_events_returns_empty_when_run_has_no_events(self):
        db = _db_with_run("r1")

        rows = db.get_events("r1")

        assert rows == []


class TestSharedFilesystemWiring:
    def test_run_with_shared_filesystem_passes_shared_path_to_create_container(self):
        settings = Settings(
            max_turns=0,
            decay_rate_per_turn=0.0,
            multi_agent_shared_filesystem=True,
        )
        orchestrator, _, sandbox, _ = _build_multi_agent(
            settings,
            agent_names=["agent-a", "agent-b", "agent-c"],
        )

        _ = asyncio.run(orchestrator.run("shared-fs-enabled"))

        assert len(sandbox.create_calls) == 3
        shared_paths = {
            call["shared_volume_host_path"] for call in sandbox.create_calls
        }
        assert len(shared_paths) == 1
        shared_path = next(iter(shared_paths))
        assert isinstance(shared_path, str)
        assert shared_path

    def test_run_without_shared_filesystem_passes_none_shared_path(self):
        settings = Settings(
            max_turns=0,
            decay_rate_per_turn=0.0,
            multi_agent_shared_filesystem=False,
        )
        orchestrator, _, sandbox, _ = _build_multi_agent(
            settings,
            agent_names=["agent-a", "agent-b"],
        )

        _ = asyncio.run(orchestrator.run("shared-fs-disabled"))

        assert len(sandbox.create_calls) == 2
        assert all(
            call["shared_volume_host_path"] is None for call in sandbox.create_calls
        )

    def test_shared_volume_temp_dir_is_cleaned_after_run(self):
        settings = Settings(
            max_turns=0,
            decay_rate_per_turn=0.0,
            multi_agent_shared_filesystem=True,
        )
        orchestrator, _, _, _ = _build_multi_agent(settings, agent_names=["agent-a"])

        _ = asyncio.run(orchestrator.run("shared-fs-cleanup"))

        assert orchestrator._shared_volume_dir is not None
        assert not Path(orchestrator._shared_volume_dir).exists()

    def test_shared_filesystem_uses_same_shared_path_for_all_agents(self):
        settings = Settings(
            max_turns=0,
            decay_rate_per_turn=0.0,
            multi_agent_shared_filesystem=True,
        )
        orchestrator, _, sandbox, _ = _build_multi_agent(
            settings,
            agent_names=["a1", "a2"],
        )

        _ = asyncio.run(orchestrator.run("shared-fs-same-path"))

        p1 = sandbox.create_calls[0]["shared_volume_host_path"]
        p2 = sandbox.create_calls[1]["shared_volume_host_path"]
        assert p1 == p2


class TestEcologySharedWorldDrops:
    def test_drop_ecology_files_writes_files_into_shared_volume_dir(
        self, tmp_path: Path
    ):
        settings = Settings(max_turns=0, decay_rate_per_turn=0.0)
        ecology = FakeEcology({2: [("alpha.txt", "hello"), ("beta.json", "{}")]})
        orchestrator, db, _, _ = _build_multi_agent(
            settings,
            ecology=ecology,
            agent_names=["agent-a", "agent-b"],
        )
        db.connect()
        db.log_run_start("eco-run", {"test": True})
        orchestrator._round = 2
        orchestrator._shared_volume_dir = str(tmp_path)

        orchestrator._drop_ecology_files("eco-run")

        assert (tmp_path / "alpha.txt").read_text() == "hello"
        assert (tmp_path / "beta.json").read_text() == "{}"

    def test_drop_ecology_files_logs_file_drop_rows_with_shared_volume(
        self, tmp_path: Path
    ):
        settings = Settings(max_turns=0, decay_rate_per_turn=0.0)
        ecology = FakeEcology({1: [("drop.csv", "a,b\n1,2")]})
        orchestrator, db, _, _ = _build_multi_agent(
            settings,
            ecology=ecology,
            agent_names=["agent-a"],
        )
        db.connect()
        db.log_run_start("eco-log", {"test": True})
        orchestrator._round = 1
        orchestrator._shared_volume_dir = str(tmp_path)

        orchestrator._drop_ecology_files("eco-log")
        stats = db.get_file_stats("eco-log")

        assert stats["total_files"] == 1
        assert stats["by_status"]["dropped"]["count"] == 1

    def test_drop_ecology_files_noop_when_schedule_returns_empty(self, tmp_path: Path):
        settings = Settings(max_turns=0, decay_rate_per_turn=0.0)
        ecology = FakeEcology({})
        orchestrator, db, _, _ = _build_multi_agent(settings, ecology=ecology)
        db.connect()
        db.log_run_start("eco-empty", {"test": True})
        orchestrator._round = 4
        orchestrator._shared_volume_dir = str(tmp_path)

        orchestrator._drop_ecology_files("eco-empty")

        assert list(tmp_path.iterdir()) == []
        assert db.get_file_stats("eco-empty")["total_files"] == 0

    def test_drop_ecology_files_without_shared_volume_uses_first_agent_container(self):
        settings = Settings(max_turns=0, decay_rate_per_turn=0.0)
        ecology = FakeEcology({3: [("x.log", "line1")]})
        orchestrator, db, _, _ = _build_multi_agent(
            settings,
            ecology=ecology,
            agent_names=["agent-a", "agent-b"],
        )
        db.connect()
        db.log_run_start("eco-fallback", {"test": True})
        orchestrator._round = 3
        orchestrator._shared_volume_dir = None
        orchestrator._agent_containers = {"agent-a": "cid-a", "agent-b": "cid-b"}

        orchestrator._drop_ecology_files("eco-fallback")

        assert ecology.drop_calls == [("cid-a", "x.log", "line1")]

    def test_drop_ecology_files_without_agents_does_not_call_drop_file(self):
        settings = Settings(max_turns=0, decay_rate_per_turn=0.0)
        ecology = FakeEcology({1: [("lonely.txt", "solo")]})
        orchestrator, db, _, _ = _build_multi_agent(
            settings,
            ecology=ecology,
            agent_names=["agent-a"],
        )
        db.connect()
        db.log_run_start("eco-no-agents", {"test": True})
        orchestrator._round = 1
        orchestrator._shared_volume_dir = None
        orchestrator.agent_ids = []
        orchestrator._agent_containers = {}

        orchestrator._drop_ecology_files("eco-no-agents")

        assert ecology.drop_calls == []
        assert db.get_file_stats("eco-no-agents")["total_files"] == 0


class TestSharedFilesystemPromptAwareness:
    def test_build_multi_agent_system_prompt_shared_true_mentions_shared_path(
        self, tmp_path: Path
    ):
        manager = PromptManager(str(tmp_path / "mods.json"))

        prompt = manager.build_multi_agent_system_prompt(
            agent_id="agent-a",
            tools=[],
            tool_costs={},
            balance=100.0,
            agent_state="active",
            starvation_remaining=7,
            agent_summaries=[],
            shared_filesystem=True,
        )

        assert "Shared filesystem: /env/shared/" in prompt
        assert "/env/shared/" in prompt

    def test_build_multi_agent_system_prompt_shared_false_omits_shared_path(
        self, tmp_path: Path
    ):
        manager = PromptManager(str(tmp_path / "mods.json"))

        prompt = manager.build_multi_agent_system_prompt(
            agent_id="agent-a",
            tools=[],
            tool_costs={},
            balance=100.0,
            agent_state="active",
            starvation_remaining=7,
            agent_summaries=[],
            shared_filesystem=False,
        )

        assert "Shared filesystem: /env/shared/" not in prompt
        assert "Files placed there are contested" not in prompt


class TestOrchestratorEventLogging:
    def test_run_logs_stripped_event(self):
        settings = Settings(
            max_turns=1,
            decay_rate_per_turn=0.0,
            initial_zod=0.0,
            starvation_turns=5,
        )
        orchestrator, db, _, _ = _build_multi_agent(
            settings,
            llm_responses={"agent-a": [("idle", [])]},
            agent_names=["agent-a"],
        )

        _ = asyncio.run(orchestrator.run("evt-stripped"))
        events = db.get_events("evt-stripped", event_type="stripped")

        assert len(events) == 1
        assert events[0]["agent_id"] == "agent-a"
        assert events[0]["event_type"] == "stripped"

    def test_run_logs_death_event(self):
        settings = Settings(
            max_turns=1,
            decay_rate_per_turn=0.0,
            initial_zod=0.0,
            starvation_turns=1,
        )
        orchestrator, db, _, _ = _build_multi_agent(
            settings,
            llm_responses={"agent-a": [("idle", [])]},
            agent_names=["agent-a"],
        )

        _ = asyncio.run(orchestrator.run("evt-death"))
        events = db.get_events("evt-death", event_type="death")

        assert len(events) == 1
        assert events[0]["agent_id"] == "agent-a"
        assert events[0]["event_type"] == "death"

    def test_run_logs_reentry_event_for_dead_agent(self):
        settings = Settings(
            max_turns=1,
            decay_rate_per_turn=0.0,
            multi_agent_spectator_rounds=1,
            multi_agent_reentry_fee=20.0,
        )
        orchestrator, db, _, shared = _build_multi_agent(
            settings,
            agent_names=["agent-a", "agent-b"],
        )
        econ = shared.get_agent_economy("agent-a")
        econ.state = AgentState.DEAD
        shared.common_pool = 100.0

        _ = asyncio.run(orchestrator.run("evt-reentry"))
        events = db.get_events("evt-reentry", event_type="reentry")

        assert len(events) == 1
        assert events[0]["agent_id"] == "agent-a"
        assert events[0]["event_type"] == "reentry"

    def test_run_logs_stripped_for_each_agent_that_depletes(self):
        settings = Settings(
            max_turns=1,
            decay_rate_per_turn=0.0,
            initial_zod=0.0,
            starvation_turns=5,
        )
        orchestrator, db, _, _ = _build_multi_agent(
            settings,
            llm_responses={
                "agent-a": [("a", [])],
                "agent-b": [("b", [])],
            },
            agent_names=["agent-a", "agent-b"],
        )

        _ = asyncio.run(orchestrator.run("evt-multi-stripped"))
        events = db.get_events("evt-multi-stripped", event_type="stripped")

        assert len(events) == 2
        assert {e["agent_id"] for e in events} == {"agent-a", "agent-b"}

    def test_lifecycle_event_types_are_reentry_stripped_and_death(self):
        settings = Settings(
            max_turns=1,
            decay_rate_per_turn=0.0,
            initial_zod=0.0,
            starvation_turns=1,
            multi_agent_spectator_rounds=1,
            multi_agent_reentry_fee=20.0,
        )
        orchestrator, db, _, shared = _build_multi_agent(
            settings,
            llm_responses={
                "agent-a": [("a", [])],
                "agent-b": [("b", [])],
            },
            agent_names=["agent-a", "agent-b", "agent-c"],
        )
        shared.get_agent_economy("agent-c").state = AgentState.DEAD
        shared.common_pool = 100.0

        _ = asyncio.run(orchestrator.run("evt-types"))
        event_types = {e["event_type"] for e in db.get_events("evt-types")}

        assert "reentry" in event_types
        assert "stripped" in event_types
        assert "death" in event_types
