"""Tests for the request_task tool, TaskBroker, and orchestrator integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from petri_dish.config import Settings
from petri_dish.economy import AgentReserve
from petri_dish.logging_db import LoggingDB
from petri_dish.orchestrator import AgentOrchestrator
from petri_dish.tools import get_all_tools
from petri_dish.tools.registry import ToolRegistry
from petri_dish.tools.task_broker import TaskBroker, TaskComplexity, TaskQuote


# ---------------------------------------------------------------------------
# Unit tests: TaskBroker
# ---------------------------------------------------------------------------


class TestTaskComplexity:
    def test_complexity_values(self) -> None:
        assert TaskComplexity.SIMPLE == 1
        assert TaskComplexity.MODERATE == 2
        assert TaskComplexity.COMPLEX == 3
        assert TaskComplexity.VERY_COMPLEX == 4

    def test_complexity_ordering(self) -> None:
        assert (
            TaskComplexity.SIMPLE
            < TaskComplexity.MODERATE
            < TaskComplexity.COMPLEX
            < TaskComplexity.VERY_COMPLEX
        )


class TestTaskQuote:
    def test_quote_creation(self) -> None:
        q = TaskQuote(complexity=TaskComplexity.MODERATE, cost_zod=2.0, summary="test")
        assert q.complexity == TaskComplexity.MODERATE
        assert q.cost_zod == 2.0
        assert q.summary == "test"


class TestTaskBrokerInit:
    def test_default_rates(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)
        assert broker._complexity_rates == {
            "SIMPLE": 0.5,
            "MODERATE": 2.0,
            "COMPLEX": 5.0,
            "VERY_COMPLEX": 10.0,
        }
        assert broker._max_cost == 15.0

    def test_custom_rates(self) -> None:
        settings = Settings(
            request_task_enabled=True,
            request_task_complexity_rates={
                "SIMPLE": 1.0,
                "MODERATE": 3.0,
                "COMPLEX": 7.0,
                "VERY_COMPLEX": 15.0,
            },
            request_task_max_cost=20.0,
        )
        broker = TaskBroker(settings)
        assert broker._complexity_rates["SIMPLE"] == 1.0
        assert broker._max_cost == 20.0


class TestParseEstimatorResponse:
    def test_valid_json(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)
        result = broker._parse_estimator_response(
            '{"complexity": "COMPLEX", "summary": "multi-step analysis"}'
        )
        assert result is not None
        complexity, summary = result
        assert complexity == TaskComplexity.COMPLEX
        assert summary == "multi-step analysis"

    def test_lowercase_complexity(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)
        result = broker._parse_estimator_response(
            '{"complexity": "simple", "summary": "trivial lookup"}'
        )
        assert result is not None
        complexity, _ = result
        assert complexity == TaskComplexity.SIMPLE

    def test_code_fence_wrapped(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)
        raw = '```json\n{"complexity": "MODERATE", "summary": "test"}\n```'
        result = broker._parse_estimator_response(raw)
        assert result is not None
        complexity, _ = result
        assert complexity == TaskComplexity.MODERATE

    def test_invalid_json_returns_none(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)
        result = broker._parse_estimator_response("not json at all")
        assert result is None

    def test_unknown_complexity_returns_moderate(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)
        result = broker._parse_estimator_response(
            '{"complexity": "ULTRA_HARD", "summary": "????"}'
        )
        assert result is not None
        complexity, _ = result
        assert complexity == TaskComplexity.MODERATE

    def test_non_dict_returns_none(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)
        result = broker._parse_estimator_response('["complexity", "SIMPLE"]')
        assert result is None


class TestEstimateCostFallback:
    """estimate_cost falls back to MODERATE when LLM call fails."""

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)

        with patch.object(
            broker, "_call_llm", new_callable=AsyncMock, return_value=None
        ):
            quote = await broker.estimate_cost("do something")

        assert quote.complexity == TaskComplexity.MODERATE
        assert quote.cost_zod == 2.0
        assert "fallback" in quote.summary.lower()

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)

        with patch.object(
            broker,
            "_call_llm",
            new_callable=AsyncMock,
            side_effect=ConnectionError("no network"),
        ):
            quote = await broker.estimate_cost("do something")

        assert quote.complexity == TaskComplexity.MODERATE
        assert quote.cost_zod == 2.0

    @pytest.mark.asyncio
    async def test_successful_estimation(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)

        with patch.object(
            broker,
            "_call_llm",
            new_callable=AsyncMock,
            return_value='{"complexity": "COMPLEX", "summary": "multi-file analysis"}',
        ):
            quote = await broker.estimate_cost(
                "analyze and refactor the entire codebase"
            )

        assert quote.complexity == TaskComplexity.COMPLEX
        assert quote.cost_zod == 5.0
        assert quote.summary == "multi-file analysis"

    @pytest.mark.asyncio
    async def test_cost_capped_by_max(self) -> None:
        settings = Settings(
            request_task_enabled=True,
            request_task_max_cost=3.0,
        )
        broker = TaskBroker(settings)

        with patch.object(
            broker,
            "_call_llm",
            new_callable=AsyncMock,
            return_value='{"complexity": "VERY_COMPLEX", "summary": "huge task"}',
        ):
            quote = await broker.estimate_cost("design an entire system")

        assert quote.complexity == TaskComplexity.VERY_COMPLEX
        assert quote.cost_zod == 3.0  # capped from 10.0


class TestExecuteTask:
    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)

        with patch.object(
            broker, "_call_llm", new_callable=AsyncMock, return_value="Task result here"
        ):
            result = await broker.execute_task("do the thing", TaskComplexity.SIMPLE)

        assert result == "Task result here"

    @pytest.mark.asyncio
    async def test_execute_failure_returns_error(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)

        with patch.object(
            broker, "_call_llm", new_callable=AsyncMock, return_value=None
        ):
            result = await broker.execute_task("do the thing", TaskComplexity.SIMPLE)

        assert "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_truncates_long_output(self) -> None:
        settings = Settings(request_task_enabled=True)
        broker = TaskBroker(settings)
        long_output = "x" * 10000

        with patch.object(
            broker, "_call_llm", new_callable=AsyncMock, return_value=long_output
        ):
            result = await broker.execute_task("do the thing", TaskComplexity.SIMPLE)

        assert len(result) <= 4000


# ---------------------------------------------------------------------------
# Tool registry tests
# ---------------------------------------------------------------------------


class TestRequestTaskRegistration:
    def test_request_task_in_registry(self) -> None:
        settings = Settings(request_task_enabled=True)
        registry = get_all_tools(settings=settings)
        tool = registry.get_tool("request_task")
        assert tool is not None
        assert tool.name == "request_task"

    def test_request_task_is_host_side(self) -> None:
        settings = Settings(request_task_enabled=True)
        registry = get_all_tools(settings=settings)
        tool = registry.get_tool("request_task")
        assert tool is not None
        assert tool.host_side is True

    def test_request_task_schema(self) -> None:
        settings = Settings(request_task_enabled=True)
        registry = get_all_tools(settings=settings)
        tool = registry.get_tool("request_task")
        assert tool is not None
        param_names = [p.name for p in tool.parameters]
        assert "task_description" in param_names


# ---------------------------------------------------------------------------
# Orchestrator integration tests
# ---------------------------------------------------------------------------


class _FakeSandbox:
    def exec_in_container(self, container_id: str, command: str) -> str:
        return ""


class _FakeValidator:
    def collect_outputs(self, sandbox_manager, container_id):
        return []

    def validate(self, filename: str, content: str):
        return False, 0.0


class TestOrchestratorRequestTask:
    """Test that the orchestrator properly initializes TaskBroker and handles request_task."""

    def test_task_broker_initialized_when_enabled(self) -> None:
        settings = Settings(request_task_enabled=True)
        reserve = AgentReserve(settings)
        db = LoggingDB(":memory:")
        db.connect()

        orch = AgentOrchestrator(
            settings=settings,
            llm_client=None,
            tool_registry=ToolRegistry(),
            agent_reserve=reserve,
            sandbox_manager=_FakeSandbox(),
            logging_db=db,
            file_validator=_FakeValidator(),
        )

        assert orch._task_broker is not None

    def test_task_broker_not_initialized_when_disabled(self) -> None:
        settings = Settings(request_task_enabled=False)
        reserve = AgentReserve(settings)
        db = LoggingDB(":memory:")
        db.connect()

        orch = AgentOrchestrator(
            settings=settings,
            llm_client=None,
            tool_registry=ToolRegistry(),
            agent_reserve=reserve,
            sandbox_manager=_FakeSandbox(),
            logging_db=db,
            file_validator=_FakeValidator(),
        )

        assert orch._task_broker is None

    @pytest.mark.asyncio
    async def test_handle_request_task_debits_zod(self) -> None:
        """Verify that _handle_request_task debits the quoted cost from agent balance."""
        settings = Settings(
            request_task_enabled=True,
            initial_zod=100.0,
        )
        reserve = AgentReserve(settings)
        db = LoggingDB(":memory:")
        db.connect()
        run_id = "test-request-task-debit"
        db.log_run_start(run_id, {})

        orch = AgentOrchestrator(
            settings=settings,
            llm_client=None,
            tool_registry=ToolRegistry(),
            agent_reserve=reserve,
            sandbox_manager=_FakeSandbox(),
            logging_db=db,
            file_validator=_FakeValidator(),
        )

        assert orch._task_broker is not None

        # Mock the broker methods
        quote = TaskQuote(
            complexity=TaskComplexity.MODERATE,
            cost_zod=2.0,
            summary="Test task",
        )

        call = type(
            "Call",
            (),
            {
                "name": "request_task",
                "id": "call_123",
                "arguments": {"task_description": "test task"},
            },
        )()

        with (
            patch.object(
                orch._task_broker,
                "estimate_cost",
                new_callable=AsyncMock,
                return_value=quote,
            ),
            patch.object(
                orch._task_broker,
                "execute_task",
                new_callable=AsyncMock,
                return_value="Task done",
            ),
        ):
            balance_before = reserve.get_balance()
            result = await orch._handle_request_task(run_id, call)

        # Verify zod was debited
        assert reserve.get_balance() == balance_before - 2.0
        # Verify result contains cost info
        assert "MODERATE" in result
        assert "2.00" in result
        assert "Task done" in result

        # Verify DB logged the transaction
        txs = db.get_balance_history(run_id)
        request_task_txs = [t for t in txs if t.get("type") == "request_task"]
        assert len(request_task_txs) == 1
        assert request_task_txs[0]["amount"] == -2.0

    @pytest.mark.asyncio
    async def test_handle_request_task_insufficient_balance(self) -> None:
        """Verify that _handle_request_task rejects when balance is too low."""
        settings = Settings(
            request_task_enabled=True,
            initial_zod=1.0,  # less than MODERATE cost (2.0)
        )
        reserve = AgentReserve(settings)
        db = LoggingDB(":memory:")
        db.connect()
        run_id = "test-request-task-insufficient"
        db.log_run_start(run_id, {})

        orch = AgentOrchestrator(
            settings=settings,
            llm_client=None,
            tool_registry=ToolRegistry(),
            agent_reserve=reserve,
            sandbox_manager=_FakeSandbox(),
            logging_db=db,
            file_validator=_FakeValidator(),
        )

        quote = TaskQuote(
            complexity=TaskComplexity.MODERATE,
            cost_zod=2.0,
            summary="Test task",
        )

        call = type(
            "Call",
            (),
            {
                "name": "request_task",
                "id": "call_456",
                "arguments": {"task_description": "test task"},
            },
        )()

        with patch.object(
            orch._task_broker,
            "estimate_cost",
            new_callable=AsyncMock,
            return_value=quote,
        ):
            result = await orch._handle_request_task(run_id, call)

        # Verify balance unchanged
        assert reserve.get_balance() == 1.0
        # Verify result indicates insufficient balance
        assert "Insufficient" in result
        assert "2.00" in result

    @pytest.mark.asyncio
    async def test_handle_request_task_empty_description(self) -> None:
        """Verify that empty task_description is rejected."""
        settings = Settings(request_task_enabled=True, initial_zod=100.0)
        reserve = AgentReserve(settings)
        db = LoggingDB(":memory:")
        db.connect()

        orch = AgentOrchestrator(
            settings=settings,
            llm_client=None,
            tool_registry=ToolRegistry(),
            agent_reserve=reserve,
            sandbox_manager=_FakeSandbox(),
            logging_db=db,
            file_validator=_FakeValidator(),
        )

        call = type(
            "Call",
            (),
            {
                "name": "request_task",
                "id": "call_789",
                "arguments": {"task_description": ""},
            },
        )()

        result = await orch._handle_request_task("test-empty-desc", call)
        assert "Error" in result
        assert "non-empty" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_request_task_with_llm_failure(self) -> None:
        """Verify graceful handling when broker LLM fails (fallback to MODERATE, execution fails)."""
        settings = Settings(
            request_task_enabled=True,
            initial_zod=100.0,
        )
        reserve = AgentReserve(settings)
        db = LoggingDB(":memory:")
        db.connect()
        run_id = "test-request-task-llm-fail"
        db.log_run_start(run_id, {})

        orch = AgentOrchestrator(
            settings=settings,
            llm_client=None,
            tool_registry=ToolRegistry(),
            agent_reserve=reserve,
            sandbox_manager=_FakeSandbox(),
            logging_db=db,
            file_validator=_FakeValidator(),
        )

        # estimate_cost will use fallback (MODERATE), execute_task will fail
        call = type(
            "Call",
            (),
            {
                "name": "request_task",
                "id": "call_fail",
                "arguments": {"task_description": "do something"},
            },
        )()

        # Patch _call_llm to simulate failure (returns None)
        with patch.object(
            orch._task_broker, "_call_llm", new_callable=AsyncMock, return_value=None
        ):
            balance_before = reserve.get_balance()
            result = await orch._task_broker.estimate_cost("do something")
            # Should fallback to MODERATE
            assert result.complexity == TaskComplexity.MODERATE

            # Now test full flow through orchestrator
            result = await orch._handle_request_task(run_id, call)

        # Should have debited MODERATE cost (2.0)
        assert reserve.get_balance() == balance_before - 2.0
        # Result should indicate completion even though LLM failed
        assert "MODERATE" in result


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestRequestTaskConfig:
    def test_config_defaults(self) -> None:
        settings = Settings()
        assert settings.request_task_enabled is False
        assert settings.request_task_model == ""
        assert settings.request_task_max_cost == 15.0

    def test_config_enabled(self) -> None:
        settings = Settings(request_task_enabled=True)
        assert settings.request_task_enabled is True

    def test_custom_complexity_rates(self) -> None:
        rates = {"SIMPLE": 1.0, "MODERATE": 4.0, "COMPLEX": 10.0, "VERY_COMPLEX": 20.0}
        settings = Settings(
            request_task_enabled=True,
            request_task_complexity_rates=rates,
        )
        assert settings.request_task_complexity_rates["SIMPLE"] == 1.0
        assert settings.request_task_complexity_rates["VERY_COMPLEX"] == 20.0

    def test_config_yaml_round_trip(self) -> None:
        """Verify config can be serialized and loaded back."""
        settings = Settings(
            request_task_enabled=True,
            request_task_model="test-model",
            request_task_max_cost=25.0,
            request_task_complexity_rates={
                "SIMPLE": 1.0,
                "MODERATE": 3.0,
                "COMPLEX": 8.0,
                "VERY_COMPLEX": 20.0,
            },
        )
        dump = settings.model_dump()
        assert dump["request_task_enabled"] is True
        assert dump["request_task_max_cost"] == 25.0
        assert dump["request_task_complexity_rates"]["MODERATE"] == 3.0
