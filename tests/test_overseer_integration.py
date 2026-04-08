from __future__ import annotations

import asyncio
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

AgentOrchestrator = import_module("petri_dish.orchestrator").AgentOrchestrator
MultiAgentOrchestrator = import_module("petri_dish.orchestrator").MultiAgentOrchestrator
AgentReserve = import_module("petri_dish.economy").AgentReserve
SharedReserve = import_module("petri_dish.economy").SharedReserve
ToolRegistry = import_module("petri_dish.tools.registry").ToolRegistry
LoggingDB = import_module("petri_dish.logging_db").LoggingDB
Settings = import_module("petri_dish.config").Settings


class _FakeSandbox:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def exec_in_container(self, container_id: str, command: str) -> str:
        _ = container_id
        if command.startswith("rm -f /env/outgoing/"):
            self.deleted.append(command)
        return ""


class _FakeValidator:
    def __init__(
        self, outputs: list[tuple[str, str]], rewards: dict[str, tuple[bool, float]]
    ):
        self._outputs = outputs
        self._rewards = rewards

    def collect_outputs(self, sandbox_manager, container_id):
        _ = (sandbox_manager, container_id)
        return list(self._outputs)

    def validate(self, filename: str, content: str) -> tuple[bool, float]:
        _ = content
        return self._rewards.get(filename, (False, 0.0))


class _FakeOverseer:
    def __init__(self, evaluations):
        self.evaluations = evaluations

    async def maybe_evaluate(self, run_id: str, turn: int, **kwargs):
        _ = (run_id, turn)
        return list(self.evaluations)


class TestOverseerIntegration:
    def test_single_agent_overseer_bonus_with_validation(self):
        """Overseer bonus + validation reward both apply."""
        settings = Settings(overseer_enabled=False)
        db = LoggingDB(":memory:")
        db.connect()
        run_id = "single-overseer-with-validation"
        db.log_run_start(run_id, {})

        validator = _FakeValidator(
            outputs=[("data_001_csv_easy.csv", "content")],
            rewards={"data_001_csv_easy.csv": (True, 2.0)},
        )
        reserve = AgentReserve(settings)
        orchestrator = AgentOrchestrator(
            settings=settings,
            llm_client=None,
            tool_parser=None,
            tool_registry=ToolRegistry(),
            agent_reserve=reserve,
            sandbox_manager=_FakeSandbox(),
            logging_db=db,
            file_validator=validator,
        )
        orchestrator._container_id = "c-single"
        orchestrator._turn = 1
        orchestrator._overseer = _FakeOverseer(
            [
                {
                    "agent_id": "agent-0",
                    "bonus": 0.12,
                    "reasoning": "bounded novelty",
                    "tags": ["rows look realistic::csv"],
                }
            ]
        )

        start_balance = reserve.get_balance()
        orchestrator._validate_outputs(run_id)
        asyncio.run(orchestrator._apply_overseer_bonuses_single(run_id, 1))

        assert reserve.get_balance() == start_balance + 2.12
        txs = db.get_balance_history(run_id)
        assert any(t["type"] == "validation_reward" and t["amount"] == 2.0 for t in txs)
        assert any(t["type"] == "overseer_bonus" and t["amount"] == 0.12 for t in txs)

    def test_single_agent_overseer_bonus_without_validation(self):
        """Overseer bonus applies even when validation reward is 0 (gate removed)."""
        settings = Settings(overseer_enabled=False)
        db = LoggingDB(":memory:")
        db.connect()
        run_id = "single-overseer-no-validation"
        db.log_run_start(run_id, {})

        validator = _FakeValidator(
            outputs=[("data_001_csv_easy.csv", "content")],
            rewards={"data_001_csv_easy.csv": (True, 0.0)},
        )
        reserve = AgentReserve(settings)
        orchestrator = AgentOrchestrator(
            settings=settings,
            llm_client=None,
            tool_parser=None,
            tool_registry=ToolRegistry(),
            agent_reserve=reserve,
            sandbox_manager=_FakeSandbox(),
            logging_db=db,
            file_validator=validator,
        )
        orchestrator._container_id = "c-single"
        orchestrator._turn = 1
        orchestrator._overseer = _FakeOverseer(
            [{"agent_id": "agent-0", "bonus": 0.12, "reasoning": "x", "tags": []}]
        )

        start_balance = reserve.get_balance()
        orchestrator._validate_outputs(run_id)
        asyncio.run(orchestrator._apply_overseer_bonuses_single(run_id, 1))

        assert reserve.get_balance() == start_balance + 0.12
        txs = db.get_balance_history(run_id)
        assert any(t["type"] == "overseer_bonus" and t["amount"] == 0.12 for t in txs)

    def test_multi_agent_overseer_bonus_applies_to_all(self):
        """Overseer bonus applies to all agents regardless of validation pass (gate removed)."""
        settings = Settings(
            multi_agent_enabled=True, multi_agent_count=2, overseer_enabled=False
        )
        db = LoggingDB(":memory:")
        db.connect()
        run_id = "multi-overseer-ungated"
        db.log_run_start(run_id, {})

        validator = _FakeValidator(
            outputs=[("data_001_csv_easy.csv", "content")],
            rewards={"data_001_csv_easy.csv": (True, 1.5)},
        )
        shared = SharedReserve(settings=settings, agent_ids=["agent-0", "agent-1"])
        orchestrator = MultiAgentOrchestrator(
            settings=settings,
            shared_economy=shared,
            sandbox_manager=_FakeSandbox(),
            logging_db=db,
            file_validator=validator,
            agent_names=["agent-0", "agent-1"],
        )
        orchestrator._overseer = _FakeOverseer(
            [
                {"agent_id": "agent-0", "bonus": 0.1, "reasoning": "novel", "tags": []},
                {"agent_id": "agent-1", "bonus": 0.1, "reasoning": "novel", "tags": []},
            ]
        )

        start_a0 = shared.get_agent_balance("agent-0")
        start_a1 = shared.get_agent_balance("agent-1")

        asyncio.run(
            orchestrator._apply_overseer_bonuses_multi(
                run_id,
                1,
            )
        )

        assert shared.get_agent_balance("agent-0") == start_a0 + 0.1
        assert shared.get_agent_balance("agent-1") == start_a1 + 0.1

        txs = db.get_balance_history(run_id)
        a0 = [
            t
            for t in txs
            if t.get("agent_id") == "agent-0" and t["type"] == "overseer_bonus"
        ]
        a1 = [
            t
            for t in txs
            if t.get("agent_id") == "agent-1" and t["type"] == "overseer_bonus"
        ]
        assert len(a0) == 1
        assert len(a1) == 1
