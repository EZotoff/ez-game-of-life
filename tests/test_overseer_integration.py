"""Integration tests for overseer scout logging and bonus gating."""

from __future__ import annotations

import json
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


def _scout_result(target_filename: str, suggested_bonus: float = 0.1) -> str:
    return json.dumps(
        {
            "report_id": "rep-1",
            "requesting_agent_id": "agent-0",
            "file_family": "csv",
            "target_filename": target_filename,
            "claimed_pattern": "rows look realistic",
            "output_summary": "2 valid rows",
            "confidence": 0.8,
            "verdict": "supports",
            "reasoning": "bounded lookup",
            "suggested_bonus": suggested_bonus,
        }
    )


class TestOverseerIntegration:
    def test_single_agent_applies_scout_bonus_only_with_positive_validator_reward(self):
        settings = Settings()
        db = LoggingDB(":memory:")
        db.connect()
        run_id = "single-scout-positive"
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

        start_balance = reserve.get_balance()
        orchestrator._capture_scout_report(
            run_id,
            1,
            tool_name="overseer_scout",
            tool_args={"target_filename": "data_001_csv_easy.csv"},
            result=_scout_result("data_001_csv_easy.csv", suggested_bonus=0.12),
        )
        orchestrator._validate_outputs(run_id)

        assert reserve.get_balance() == start_balance + 2.12
        txs = db.get_balance_history(run_id)
        assert any(t["type"] == "validation_reward" and t["amount"] == 2.0 for t in txs)
        assert any(t["type"] == "scout_bonus" and t["amount"] == 0.12 for t in txs)

        reports = db.get_scout_reports(run_id)
        assert len(reports) == 1
        assert reports[0]["applied"] == 1
        assert reports[0]["applied_bonus"] == 0.12

    def test_single_agent_no_scout_bonus_when_validator_reward_is_zero(self):
        settings = Settings()
        db = LoggingDB(":memory:")
        db.connect()
        run_id = "single-scout-zero"
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

        orchestrator._capture_scout_report(
            run_id,
            1,
            tool_name="overseer_scout",
            tool_args={"target_filename": "data_001_csv_easy.csv"},
            result=_scout_result("data_001_csv_easy.csv", suggested_bonus=0.12),
        )
        orchestrator._validate_outputs(run_id)

        txs = db.get_balance_history(run_id)
        assert not any(t["type"] == "scout_bonus" for t in txs)
        reports = db.get_scout_reports(run_id)
        assert len(reports) == 1
        assert reports[0]["applied"] == 0

    def test_multi_agent_bonus_is_gated_by_validator_reward_and_agent_scope(self):
        settings = Settings(multi_agent_enabled=True, multi_agent_count=2)
        db = LoggingDB(":memory:")
        db.connect()
        run_id = "multi-scout-gated"
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

        start_a0 = shared.get_agent_balance("agent-0")
        start_a1 = shared.get_agent_balance("agent-1")
        orchestrator._capture_scout_report(
            run_id,
            1,
            tool_name="overseer_scout",
            tool_args={"target_filename": "data_001_csv_easy.csv"},
            result=_scout_result("data_001_csv_easy.csv", suggested_bonus=0.1),
            agent_id="agent-0",
        )

        orchestrator._validate_agent_outputs(run_id, "agent-1", 1, "c-agent-1")
        orchestrator._validate_agent_outputs(run_id, "agent-0", 2, "c-agent-0")

        assert shared.get_agent_balance("agent-1") == start_a1 + 1.5
        assert shared.get_agent_balance("agent-0") == start_a0 + 1.6

        txs = db.get_balance_history(run_id)
        a1_scout_bonus = [
            t for t in txs if t["agent_id"] == "agent-1" and t["type"] == "scout_bonus"
        ]
        a0_scout_bonus = [
            t for t in txs if t["agent_id"] == "agent-0" and t["type"] == "scout_bonus"
        ]
        assert len(a1_scout_bonus) == 0
        assert len(a0_scout_bonus) == 1
        assert a0_scout_bonus[0]["amount"] == 0.1

        reports = db.get_scout_reports(run_id, agent_id="agent-0")
        assert len(reports) == 1
        assert reports[0]["applied"] == 1
