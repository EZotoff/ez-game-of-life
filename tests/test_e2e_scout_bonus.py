from __future__ import annotations

import asyncio
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Settings = import_module("petri_dish.config").Settings
AgentReserve = import_module("petri_dish.economy").AgentReserve
LoggingDB = import_module("petri_dish.logging_db").LoggingDB
AgentOrchestrator = import_module("petri_dish.orchestrator").AgentOrchestrator
ToolRegistry = import_module("petri_dish.tools.registry").ToolRegistry


class _FakeSandbox:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def exec_in_container(self, container_id: str, command: str) -> str:
        _ = container_id
        if command.startswith("rm -f /env/outgoing/"):
            self.deleted.append(command)
        return ""


class _FakeValidator:
    def __init__(self, passed: bool, reward: float) -> None:
        self.passed = passed
        self.reward = reward

    def collect_outputs(self, sandbox_manager, container_id):
        _ = (sandbox_manager, container_id)
        return [("data_001_json_easy.json", "{}")]

    def validate(self, filename: str, content: str):
        _ = (filename, content)
        return self.passed, self.reward


class _FakeOverseer:
    def __init__(self, evaluations):
        self.evaluations = evaluations

    async def maybe_evaluate(self, run_id: str, turn: int, **kwargs):
        _ = (run_id, turn)
        return list(self.evaluations)


def test_single_agent_overseer_bonus_applies_with_validation() -> None:
    """Overseer bonus + validation reward both apply."""
    settings = Settings(overseer_enabled=False)
    db = LoggingDB(":memory:")
    db.connect()
    run_id = "e2e-overseer-bonus-with-validation"
    db.log_run_start(run_id, {})

    reserve = AgentReserve(settings)
    orch = AgentOrchestrator(
        settings=settings,
        llm_client=None,
        tool_registry=ToolRegistry(),
        agent_reserve=reserve,
        sandbox_manager=_FakeSandbox(),
        logging_db=db,
        file_validator=_FakeValidator(True, 2.0),
    )
    orch._container_id = "c1"
    orch._turn = 1
    orch._overseer = _FakeOverseer(
        [
            {
                "agent_id": "agent-0",
                "bonus": 0.1,
                "reasoning": "novel behavior",
                "tags": ["pattern::json"],
            }
        ]
    )

    start_balance = reserve.get_balance()
    orch._validate_outputs(run_id)
    asyncio.run(orch._apply_overseer_bonuses_single(run_id, 1))

    assert reserve.get_balance() == start_balance + 2.1
    txs = db.get_balance_history(run_id)
    assert any(t["type"] == "validation_reward" and t["amount"] == 2.0 for t in txs)
    assert any(t["type"] == "overseer_bonus" and t["amount"] == 0.1 for t in txs)


def test_single_agent_overseer_bonus_applies_without_validation() -> None:
    """Overseer bonus applies even when validation reward is 0 (gate removed)."""
    settings = Settings(overseer_enabled=False)
    db = LoggingDB(":memory:")
    db.connect()
    run_id = "e2e-overseer-bonus-no-validation"
    db.log_run_start(run_id, {})

    reserve = AgentReserve(settings)
    orch = AgentOrchestrator(
        settings=settings,
        llm_client=None,
        tool_registry=ToolRegistry(),
        agent_reserve=reserve,
        sandbox_manager=_FakeSandbox(),
        logging_db=db,
        file_validator=_FakeValidator(True, 0.0),
    )
    orch._container_id = "c2"
    orch._turn = 1
    orch._overseer = _FakeOverseer(
        [{"agent_id": "agent-0", "bonus": 0.1, "reasoning": "x", "tags": []}]
    )

    start_balance = reserve.get_balance()
    orch._validate_outputs(run_id)
    asyncio.run(orch._apply_overseer_bonuses_single(run_id, 1))

    txs = db.get_balance_history(run_id)
    assert any(t["type"] == "overseer_bonus" and t["amount"] == 0.1 for t in txs)
    assert reserve.get_balance() == start_balance + 0.1
