"""Comprehensive tests for multi-agent Petri Dish components."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Settings = import_module("petri_dish.config").Settings
AgentState = import_module("petri_dish.economy").AgentState
CreditEconomy = import_module("petri_dish.economy").CreditEconomy
SharedEconomy = import_module("petri_dish.economy").SharedEconomy
LoggingDB = import_module("petri_dish.logging_db").LoggingDB
MultiAgentOrchestrator = import_module("petri_dish.orchestrator").MultiAgentOrchestrator
PromptManager = import_module("petri_dish.prompt").PromptManager
FileValidator = import_module("petri_dish.validators").FileValidator


class FakeSandboxManager:
    def __init__(self) -> None:
        self.created_container_ids: list[str] = []
        self.destroyed_container_ids: list[str] = []

    def create_container(
        self,
        run_id: str,
        memory_host_path: str | None = None,
        shared_volume_host_path: str | None = None,
    ) -> str:
        _ = (memory_host_path, shared_volume_host_path)
        cid = f"fake-{run_id}"
        self.created_container_ids.append(cid)
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


def _build_multi_agent(
    settings: Any,
    *,
    llm_responses: dict[str, list[tuple[str, list[Any]]]] | None = None,
    agent_names: list[str] | None = None,
) -> tuple[Any, Any, FakeSandboxManager, Any]:
    names = agent_names or ["agent-a", "agent-b"]
    sandbox = FakeSandboxManager()
    logging_db = LoggingDB(":memory:")
    shared_economy = SharedEconomy(settings=settings, agent_ids=names)
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
    )
    return orchestrator, logging_db, sandbox, shared_economy


class TestSharedEconomy:
    def test_init_creates_per_agent_economies(self):
        settings = Settings(initial_balance=123.0, multi_agent_count=2)
        economy = SharedEconomy(settings=settings, agent_ids=["a1", "a2"])

        assert set(economy.get_agent_ids()) == {"a1", "a2"}
        assert isinstance(economy.get_agent_economy("a1"), CreditEconomy)
        assert isinstance(economy.get_agent_economy("a2"), CreditEconomy)
        assert economy.get_agent_balance("a1") == pytest.approx(123.0)
        assert economy.get_agent_balance("a2") == pytest.approx(123.0)

    def test_debit_credits_per_agent(self):
        settings = Settings(initial_balance=10.0, burn_rate_per_turn=1.5)
        economy = SharedEconomy(settings=settings, agent_ids=["a1", "a2"])

        economy.debit("a1", turns=2)
        economy.debit("a2", turns=1)

        assert economy.get_agent_balance("a1") == pytest.approx(7.0)
        assert economy.get_agent_balance("a2") == pytest.approx(8.5)

    def test_credit_with_debt_garnishing(self):
        settings = Settings(multi_agent_debt_garnish_pct=0.5)
        economy = SharedEconomy(settings=settings, agent_ids=["a1"])
        economy.agent_debt["a1"] = 20.0

        before = economy.get_agent_balance("a1")
        economy.credit("a1", 50.0)

        assert economy.get_agent_balance("a1") == pytest.approx(before + 30.0)
        assert economy.get_agent_debt("a1") == pytest.approx(0.0)
        assert economy.get_common_pool() == pytest.approx(20.0)

    def test_handle_death_burns_and_salvages(self):
        settings = Settings(
            initial_balance=100.0,
            multi_agent_burn_pct=0.3,
            multi_agent_salvage_pct=0.3,
        )
        economy = SharedEconomy(settings=settings, agent_ids=["a1"])

        salvage = economy.handle_death("a1")

        assert salvage == pytest.approx(30.0)
        assert economy.get_common_pool() == pytest.approx(30.0)
        assert economy.get_agent_balance("a1") == pytest.approx(40.0)

    def test_handle_death_at_zero_balance_creates_debt(self):
        settings = Settings(
            initial_balance=100.0,
            multi_agent_burn_pct=0.3,
            multi_agent_salvage_pct=0.3,
        )
        economy = SharedEconomy(settings=settings, agent_ids=["a1"])

        economy.debit("a1", turns=1000)
        assert economy.get_agent_balance("a1") <= 0

        salvage = economy.handle_death("a1")

        assert salvage == pytest.approx(30.0)
        assert economy.get_common_pool() == pytest.approx(30.0)
        assert economy.get_agent_debt("a1") == pytest.approx(60.0)

    def test_handle_death_uses_lifetime_earned_as_penalty_base(self):
        settings = Settings(
            initial_balance=100.0,
            multi_agent_burn_pct=0.3,
            multi_agent_salvage_pct=0.3,
        )
        economy = SharedEconomy(settings=settings, agent_ids=["a1"])

        economy.credit("a1", 200.0)
        economy.debit("a1", turns=3000)
        assert economy.get_agent_balance("a1") <= 0
        assert economy.get_agent_economy("a1").lifetime_credits_earned == 300.0

        salvage = economy.handle_death("a1")

        assert salvage == pytest.approx(90.0)
        assert economy.get_common_pool() == pytest.approx(90.0)

    def test_handle_death_sets_state_to_dead(self):
        economy = SharedEconomy(settings=Settings(), agent_ids=["a1"])

        economy.handle_death("a1")

        assert economy.get_agent_state("a1") == AgentState.DEAD

    def test_tick_spectator_increments_counter(self):
        economy = SharedEconomy(
            settings=Settings(multi_agent_spectator_rounds=3), agent_ids=["a1"]
        )
        economy.get_agent_economy("a1").state = AgentState.DEAD

        economy.tick_spectator("a1")

        assert economy.spectator_counters["a1"] == 1

    def test_tick_spectator_returns_true_when_cooldown_reached(self):
        economy = SharedEconomy(
            settings=Settings(multi_agent_spectator_rounds=2), agent_ids=["a1"]
        )
        economy.get_agent_economy("a1").state = AgentState.DEAD

        assert economy.tick_spectator("a1") is False
        assert economy.tick_spectator("a1") is True

    def test_reentry_fails_before_cooldown(self):
        settings = Settings(
            multi_agent_spectator_rounds=2, multi_agent_reentry_fee=20.0
        )
        economy = SharedEconomy(settings=settings, agent_ids=["a1"])
        economy.get_agent_economy("a1").state = AgentState.DEAD
        economy.common_pool = 100.0
        economy.spectator_counters["a1"] = 1

        assert economy.reentry("a1") is False

    def test_reentry_succeeds_after_cooldown(self):
        settings = Settings(
            multi_agent_spectator_rounds=2, multi_agent_reentry_fee=20.0
        )
        economy = SharedEconomy(settings=settings, agent_ids=["a1"])
        economy.get_agent_economy("a1").state = AgentState.DEAD
        economy.common_pool = 50.0
        economy.spectator_counters["a1"] = 2

        ok = economy.reentry("a1")

        assert ok is True
        assert economy.get_agent_state("a1") == AgentState.ACTIVE
        assert economy.get_agent_debt("a1") == pytest.approx(20.0)
        assert economy.get_common_pool() == pytest.approx(30.0)

    def test_reentry_fails_if_pool_insufficient(self):
        settings = Settings(
            multi_agent_spectator_rounds=1, multi_agent_reentry_fee=20.0
        )
        economy = SharedEconomy(settings=settings, agent_ids=["a1"])
        economy.get_agent_economy("a1").state = AgentState.DEAD
        economy.common_pool = 10.0
        economy.spectator_counters["a1"] = 1

        assert economy.reentry("a1") is False

    def test_get_living_agents_excludes_dead(self):
        economy = SharedEconomy(settings=Settings(), agent_ids=["a1", "a2", "a3"])
        economy.get_agent_economy("a2").state = AgentState.DEAD

        assert set(economy.get_living_agents()) == {"a1", "a3"}

    def test_get_agent_summaries_structure(self):
        economy = SharedEconomy(settings=Settings(), agent_ids=["a1", "a2"])

        summaries = economy.get_agent_summaries()

        assert len(summaries) == 2
        for summary in summaries:
            assert set(summary.keys()) == {
                "agent_id",
                "balance",
                "state",
                "degradation",
                "starvation_remaining",
            }


class TestMultiAgentPrompt:
    def test_multi_agent_prompt_contains_agent_id(self, tmp_path: Path):
        manager = PromptManager(str(tmp_path / "modifications.json"))
        prompt = manager.build_multi_agent_system_prompt(
            agent_id="agent-a",
            tools=[],
            tool_costs={},
            balance=100.0,
            agent_state="active",
            starvation_remaining=7,
            agent_summaries=[],
        )

        assert "You are agent 'agent-a'" in prompt
        assert "Your identity: agent-a" in prompt

    def test_multi_agent_prompt_shows_other_agents(self, tmp_path: Path):
        manager = PromptManager(str(tmp_path / "modifications.json"))
        prompt = manager.build_multi_agent_system_prompt(
            agent_id="agent-a",
            tools=[],
            tool_costs={},
            balance=100.0,
            agent_state="active",
            starvation_remaining=7,
            agent_summaries=[
                {"agent_id": "agent-a", "state": "active", "balance": 100.0},
                {"agent_id": "agent-b", "state": "active", "balance": 98.5},
                {"agent_id": "agent-c", "state": "dead", "balance": 0.0},
            ],
        )

        assert "Other agents in this environment:" in prompt
        assert "- agent-b: state=active, balance=98.5" in prompt
        assert "- agent-c: DEAD" in prompt
        assert "agent-a: state=" not in prompt

    def test_multi_agent_prompt_actions_per_turn(self, tmp_path: Path):
        manager = PromptManager(str(tmp_path / "modifications.json"))
        prompt = manager.build_multi_agent_system_prompt(
            agent_id="agent-a",
            tools=[],
            tool_costs={},
            balance=100.0,
            agent_state="active",
            starvation_remaining=7,
            agent_summaries=[],
            actions_per_turn=4,
        )

        assert "Action budget: 4 tool calls per turn" in prompt

    def test_multi_agent_prompt_stripped_warning(self, tmp_path: Path):
        manager = PromptManager(str(tmp_path / "modifications.json"))
        prompt = manager.build_multi_agent_system_prompt(
            agent_id="agent-a",
            tools=[{"name": "check_balance", "description": ""}],
            tool_costs={"check_balance": 0.0},
            balance=0.0,
            agent_state="stripped",
            starvation_remaining=2,
            agent_summaries=[],
        )

        assert "STRIPPED STATE" in prompt
        assert "You have 2 turns before starvation death" in prompt
        assert "Only observational tools available: check_balance." in prompt

    def test_single_agent_prompt_no_other_agents_section(self, tmp_path: Path):
        manager = PromptManager(str(tmp_path / "modifications.json"))
        prompt = manager.build_system_prompt(
            tools=[],
            tool_costs={},
            balance=100.0,
            state_summary="state=active",
        )

        assert "Other agents in this environment:" not in prompt
        assert "You are an autonomous agent in an isolated environment." in prompt


class TestMultiAgentOrchestrator:
    def test_round_robin_both_agents_get_turns(self):
        settings = Settings(
            max_turns=2,
            burn_rate_per_turn=0.0,
            multi_agent_actions_per_turn=4,
            initial_balance=10.0,
        )
        responses = {
            "agent-a": [("a-turn-1", []), ("a-turn-2", [])],
            "agent-b": [("b-turn-1", []), ("b-turn-2", [])],
        }
        orchestrator, _, _, _ = _build_multi_agent(
            settings,
            llm_responses=responses,
            agent_names=["agent-a", "agent-b"],
        )

        result = asyncio.run(orchestrator.run("ma-round-robin"))

        assert result.agent_results["agent-a"].total_turns == 2
        assert result.agent_results["agent-b"].total_turns == 2
        assert result.termination_reason == "max_rounds_reached"

    def test_actions_per_turn_cap(self):
        settings = Settings(
            max_turns=1,
            burn_rate_per_turn=0.0,
            multi_agent_actions_per_turn=4,
            initial_balance=10.0,
        )
        calls = [{"name": "check_balance", "arguments": {}} for _ in range(6)]
        responses = {"agent-a": [("do many actions", calls)]}
        orchestrator, db, _, _ = _build_multi_agent(
            settings,
            llm_responses=responses,
            agent_names=["agent-a"],
        )

        _ = asyncio.run(orchestrator.run("ma-actions-cap"))
        actions = db.get_actions("ma-actions-cap")
        check_balance_calls = [a for a in actions if a["tool_name"] == "check_balance"]

        assert len(check_balance_calls) == 4

    def test_dead_agent_skipped(self):
        settings = Settings(max_turns=2, burn_rate_per_turn=0.0, initial_balance=10.0)
        responses = {
            "agent-a": [("a", []), ("a", [])],
            "agent-b": [("b", []), ("b", [])],
        }
        orchestrator, _, _, shared = _build_multi_agent(
            settings,
            llm_responses=responses,
            agent_names=["agent-a", "agent-b"],
        )
        shared.get_agent_economy("agent-a").state = AgentState.DEAD

        result = asyncio.run(orchestrator.run("ma-dead-skipped"))

        assert result.agent_results["agent-a"].total_turns == 0
        assert result.agent_results["agent-b"].total_turns == 2

    def test_death_salvage_to_common_pool(self):
        settings = Settings(
            max_turns=1,
            burn_rate_per_turn=0.0,
            initial_balance=10.0,
            starvation_turns=1,
            multi_agent_burn_pct=0.3,
            multi_agent_salvage_pct=0.3,
        )
        responses = {
            "agent-a": [("stripped turn", [])],
            "agent-b": [("normal turn", [])],
        }
        orchestrator, _, _, shared = _build_multi_agent(
            settings,
            llm_responses=responses,
            agent_names=["agent-a", "agent-b"],
        )
        shared.get_agent_economy("agent-a").state = AgentState.STRIPPED

        result = asyncio.run(orchestrator.run("ma-salvage"))

        assert shared.get_agent_state("agent-a") == AgentState.DEAD
        assert result.common_pool == pytest.approx(3.0)

    def test_all_agents_dead_terminates(self):
        settings = Settings(max_turns=5, burn_rate_per_turn=0.0, initial_balance=10.0)
        orchestrator, _, _, shared = _build_multi_agent(
            settings,
            llm_responses={},
            agent_names=["agent-a", "agent-b"],
        )
        shared.get_agent_economy("agent-a").state = AgentState.DEAD
        shared.get_agent_economy("agent-b").state = AgentState.DEAD

        result = asyncio.run(orchestrator.run("ma-all-dead"))

        assert result.termination_reason == "all_agents_dead"

    def test_max_rounds_terminates(self):
        settings = Settings(max_turns=1, burn_rate_per_turn=0.0, initial_balance=10.0)
        responses = {"agent-a": [("a", [])]}
        orchestrator, _, _, _ = _build_multi_agent(
            settings,
            llm_responses=responses,
            agent_names=["agent-a"],
        )

        result = asyncio.run(orchestrator.run("ma-max-rounds"))

        assert result.agent_results["agent-a"].total_turns == 1
        assert result.termination_reason == "max_rounds_reached"

    def test_logging_has_agent_id(self):
        settings = Settings(max_turns=1, burn_rate_per_turn=0.0, initial_balance=10.0)
        responses = {
            "agent-a": [("a", [{"name": "check_balance", "arguments": {}}])],
            "agent-b": [("b", [{"name": "check_balance", "arguments": {}}])],
        }
        orchestrator, db, _, _ = _build_multi_agent(
            settings,
            llm_responses=responses,
            agent_names=["agent-a", "agent-b"],
        )

        _ = asyncio.run(orchestrator.run("ma-agent-id-logs"))
        actions = db.get_actions("ma-agent-id-logs")

        assert actions
        assert {a["agent_id"] for a in actions} == {"agent-a", "agent-b"}
        assert all(a["agent_id"] is not None for a in actions)

    def test_stripped_transition_logged_per_agent(self):
        settings = Settings(
            max_turns=1,
            initial_balance=0.0,
            burn_rate_per_turn=0.0,
            starvation_turns=3,
        )
        responses = {
            "agent-a": [("a", [])],
            "agent-b": [("b", [])],
        }
        orchestrator, db, _, _ = _build_multi_agent(
            settings,
            llm_responses=responses,
            agent_names=["agent-a", "agent-b"],
        )

        _ = asyncio.run(orchestrator.run("ma-stripped-transition"))
        conn: sqlite3.Connection = db._ensure_connection()
        rows = conn.execute(
            """
            SELECT from_state, to_state, agent_id
            FROM state_transitions
            WHERE run_id = ?
            ORDER BY id
            """,
            ("ma-stripped-transition",),
        ).fetchall()

        stripped_rows = [
            r
            for r in rows
            if r["from_state"] == "active" and r["to_state"] == "stripped"
        ]
        assert len(stripped_rows) == 2
        assert {r["agent_id"] for r in stripped_rows} == {"agent-a", "agent-b"}
