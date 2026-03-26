"""Central agent orchestrator state machine for Petri Dish MVP."""

from __future__ import annotations

import json
import signal
import sqlite3
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import Any

from petri_dish.config import Settings
from petri_dish.economy import CreditEconomy
from petri_dish.llm_client import OllamaClient
from petri_dish.logging_db import LoggingDB
from petri_dish.prompt import PromptManager
from petri_dish.sandbox import ContainerNotRunningError, SandboxError, SandboxManager
from petri_dish.tool_parser import ToolCall, ToolCallParser
from petri_dish.tools import get_all_tools
from petri_dish.tools.agent_tools import get_prompt_overrides
from petri_dish.tools.registry import ToolRegistry


class OrchestratorState(str, Enum):
    """Orchestrator lifecycle states."""

    IDLE = "IDLE"
    WAITING_FOR_LLM = "WAITING_FOR_LLM"
    EXECUTING_TOOL = "EXECUTING_TOOL"
    LOGGING = "LOGGING"
    TERMINATED = "TERMINATED"


@dataclass(slots=True)
class RunResult:
    """Terminal summary for a run."""

    total_turns: int
    final_balance: float
    tiers_reached: list[str]
    termination_reason: str


@dataclass(slots=True)
class _ToolCallPayload:
    name: str
    arguments: dict[str, Any]


class AgentOrchestrator:
    """Coordinates LLM inference, tool execution, economy, logging, and shutdown."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm_client: OllamaClient | None = None,
        tool_parser: ToolCallParser | None = None,
        tool_registry: ToolRegistry | None = None,
        credit_economy: CreditEconomy | None = None,
        sandbox_manager: SandboxManager | None = None,
        logging_db: LoggingDB | None = None,
        snapshot_interval_turns: int | None = None,
    ) -> None:
        self.settings = settings or Settings.from_yaml()
        self.tool_registry = tool_registry or get_all_tools(settings=self.settings)
        self.tool_parser = tool_parser or ToolCallParser()
        self.llm_client = llm_client or OllamaClient(settings=self.settings)
        self.credit_economy = credit_economy or CreditEconomy(settings=self.settings)
        self.sandbox_manager = sandbox_manager or SandboxManager()
        self.logging_db = logging_db or LoggingDB(":memory:")
        self.snapshot_interval_turns = (
            snapshot_interval_turns
            if snapshot_interval_turns is not None
            else max(1, int(self.settings.context_summary_interval_turns))
        )

        self.state = OrchestratorState.IDLE
        self._shutdown_requested = False
        self._container_crashed = False
        self._termination_reason = "unknown"
        self._turn = 0
        self._consecutive_empty_turns = 0
        self._messages: list[dict[str, Any]] = []
        self._tiers_reached: set[str] = {self.credit_economy.get_degradation_level()}
        self._container_id = ""
        self._active_run_id = ""

    async def run(self, run_id: str) -> RunResult:
        """Execute orchestrator loop until termination conditions are met."""
        self._active_run_id = run_id
        self._messages = []
        self._shutdown_requested = False
        self._container_crashed = False
        self._termination_reason = "unknown"
        self._turn = 0
        self._consecutive_empty_turns = 0
        self._tiers_reached = {self.credit_economy.get_degradation_level()}
        self.state = OrchestratorState.IDLE

        self._connect_logging_db()
        self.logging_db.log_run_start(
            run_id,
            {
                "settings": self.settings.model_dump(),
                "snapshot_interval_turns": self.snapshot_interval_turns,
            },
        )
        self.logging_db.log_credit(
            run_id,
            self.credit_economy.get_balance(),
            "initial_balance",
            "Run started",
        )

        previous_handlers = self._install_signal_handlers()
        self._container_id = self.sandbox_manager.create_container(run_id)

        try:
            while True:
                if self._shutdown_requested:
                    self._termination_reason = "graceful_shutdown"
                    break

                if self.credit_economy.is_depleted():
                    self._termination_reason = "credits_depleted"
                    break

                if self._turn >= self.settings.max_turns:
                    self._termination_reason = "max_turns_reached"
                    break

                if (
                    self._consecutive_empty_turns
                    >= self.settings.max_consecutive_empty_turns
                ):
                    self._termination_reason = "max_consecutive_empty_turns"
                    break

                self._turn += 1
                self.state = OrchestratorState.WAITING_FOR_LLM
                credits_before_turn = self.credit_economy.get_balance()
                self.credit_economy.debit(turns=1)
                self.logging_db.log_credit(
                    run_id,
                    -self.settings.burn_rate_per_turn,
                    "turn_cost",
                    f"Turn {self._turn} inference",
                )

                llm_result = await self.llm_client.chat(
                    system_prompt=self._build_system_prompt(),
                    messages=self._messages,
                    tools=self.tool_registry.get_all_schemas(),
                )

                if llm_result is None:
                    assistant_text = ""
                    parsed_calls_raw: list[Any] = []
                else:
                    assistant_text, parsed_calls_raw = llm_result

                tool_calls = self._normalize_tool_calls(parsed_calls_raw)
                if not tool_calls and assistant_text:
                    reparsed = self.tool_parser.parse(assistant_text)
                    tool_calls = self._normalize_tool_calls(reparsed)

                if assistant_text:
                    self._messages.append(
                        {"role": "assistant", "content": assistant_text}
                    )

                if not tool_calls:
                    self._consecutive_empty_turns += 1
                    self.state = OrchestratorState.LOGGING
                    self.logging_db.log_action(
                        run_id=run_id,
                        turn=self._turn,
                        tool_name="__empty_turn__",
                        tool_args=None,
                        result=assistant_text or "",
                        credits_before=credits_before_turn,
                        credits_after=self.credit_economy.get_balance(),
                        duration_ms=0,
                    )
                else:
                    self._consecutive_empty_turns = 0
                    allowed_calls = tool_calls[
                        : max(1, int(self.settings.max_turns_per_tool))
                    ]
                    for call in allowed_calls:
                        call_started = time.perf_counter()
                        self.state = OrchestratorState.EXECUTING_TOOL
                        before_tool = self.credit_economy.get_balance()

                        result = ""
                        failed = False
                        try:
                            result = self._execute_tool_call(call)
                        except ContainerNotRunningError as exc:
                            failed = True
                            self._container_crashed = True
                            result = f"Container crash detected: {exc}"
                        except SandboxError as exc:
                            failed = True
                            result = f"Sandbox error: {exc}"
                        except Exception as exc:
                            failed = True
                            result = (
                                f"Tool execution failed: {type(exc).__name__}: {exc}"
                            )

                        self._debit_tool_cost(run_id, call.name)
                        after_tool = self.credit_economy.get_balance()
                        elapsed_ms = int((time.perf_counter() - call_started) * 1000)

                        self.state = OrchestratorState.LOGGING
                        suffix = " [FAILED]" if failed else ""
                        self.logging_db.log_action(
                            run_id=run_id,
                            turn=self._turn,
                            tool_name=call.name,
                            tool_args=call.arguments,
                            result=f"{result}{suffix}",
                            credits_before=before_tool,
                            credits_after=after_tool,
                            duration_ms=elapsed_ms,
                        )
                        self._messages.append(
                            {
                                "role": "tool",
                                "name": call.name,
                                "content": result,
                            }
                        )

                        if self._container_crashed:
                            self._termination_reason = "container_crash"
                            break

                self._tiers_reached.add(self.credit_economy.get_degradation_level())

                if self._turn % self.snapshot_interval_turns == 0:
                    self._save_state_snapshot(run_id)

                if self._container_crashed:
                    break

            return RunResult(
                total_turns=self._turn,
                final_balance=self.credit_economy.get_balance(),
                tiers_reached=sorted(self._tiers_reached),
                termination_reason=self._termination_reason,
            )
        finally:
            self.state = OrchestratorState.TERMINATED
            self._save_state_snapshot(run_id)
            self._restore_signal_handlers(previous_handlers)
            if self._container_id:
                self.sandbox_manager.destroy_container(self._container_id)
            self._mark_run_end(run_id)

    def _build_system_prompt(self) -> str:
        tool_costs = self.settings.tool_costs

        tool_list = []
        for tool_name in self.tool_registry.get_tool_names():
            tool_def = self.tool_registry.get_tool(tool_name)
            if tool_def:
                tool_list.append(
                    {"name": tool_name, "description": tool_def.description or ""}
                )

        state_summary = (
            f"Turn: {self._turn}, "
            f"State: {self.state.value}, "
            f"Degradation: {self.credit_economy.get_degradation_level()}, "
            f"Consecutive empty turns: {self._consecutive_empty_turns}"
        )

        prompt_manager = PromptManager(
            modifications_path=str(
                Path(self.logging_db.db_path).parent / "modifications.json"
            )
            if self.logging_db.db_path != ":memory:"
            else "/tmp/petri_dish_modifications.json"
        )
        return prompt_manager.build_system_prompt(
            tools=tool_list,
            tool_costs=tool_costs,
            balance=self.credit_economy.get_balance(),
            state_summary=state_summary,
        )

    def _normalize_tool_calls(self, raw_calls: list[Any]) -> list[_ToolCallPayload]:
        normalized: list[_ToolCallPayload] = []
        for call in raw_calls:
            if isinstance(call, ToolCall):
                normalized.append(
                    _ToolCallPayload(name=call.name, arguments=dict(call.arguments))
                )
                continue

            if isinstance(call, dict):
                if "function" in call and isinstance(call.get("function"), dict):
                    function = call["function"]
                    name = function.get("name", "")
                    arguments = function.get("arguments", {})
                else:
                    name = call.get("name", "")
                    arguments = call.get("arguments", {})

                if isinstance(name, str) and name.strip():
                    if isinstance(arguments, str):
                        try:
                            parsed = json.loads(arguments)
                            arguments = parsed if isinstance(parsed, dict) else {}
                        except json.JSONDecodeError:
                            arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    normalized.append(
                        _ToolCallPayload(name=name.strip(), arguments=arguments)
                    )
                continue

            if hasattr(call, "name") and hasattr(call, "arguments"):
                name = getattr(call, "name", "")
                arguments = getattr(call, "arguments", {})
                if (
                    isinstance(name, str)
                    and name.strip()
                    and isinstance(arguments, dict)
                ):
                    normalized.append(
                        _ToolCallPayload(name=name.strip(), arguments=arguments)
                    )
        return normalized

    def _execute_tool_call(self, call: _ToolCallPayload) -> str:
        tool_def = self.tool_registry.get_tool(call.name)
        if tool_def is None:
            return f"Unknown tool: {call.name}"

        if not tool_def.host_side:
            _ = self.sandbox_manager.get_container_stats(self._container_id)

        return self.tool_registry.execute_tool(
            call.name, call.arguments, self._container_id
        )

    def _debit_tool_cost(self, run_id: str, tool_name: str) -> None:
        cost = float(self.tool_registry.get_tool_cost(tool_name))
        if cost <= 0:
            return
        _ = self.credit_economy.credit(-cost)
        self.logging_db.log_credit(
            run_id,
            -cost,
            "tool_cost",
            f"Tool invocation: {tool_name}",
        )

    def _connect_logging_db(self) -> None:
        try:
            self.logging_db.connect()
        except sqlite3.OperationalError:
            if self.logging_db.db_path != ":memory:":
                with open(self.logging_db.db_path, "a", encoding="utf-8"):
                    pass
                self.logging_db.connect()
            else:
                raise

    def _save_state_snapshot(self, run_id: str) -> None:
        conn = self.logging_db._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS state_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                state TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
            """
        )
        snapshot = {
            "state": self.state.value,
            "turn": self._turn,
            "consecutive_empty_turns": self._consecutive_empty_turns,
            "balance": self.credit_economy.get_balance(),
            "degradation_level": self.credit_economy.get_degradation_level(),
            "tiers_reached": sorted(self._tiers_reached),
            "container_id": self._container_id,
            "shutdown_requested": self._shutdown_requested,
            "container_crashed": self._container_crashed,
            "termination_reason": self._termination_reason,
            "messages": self._messages,
            "prompt_overrides": get_prompt_overrides(),
        }
        cursor.execute(
            """
            INSERT INTO state_snapshots (run_id, turn, state, snapshot_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, self._turn, self.state.value, json.dumps(snapshot)),
        )
        conn.commit()

    def _mark_run_end(self, run_id: str) -> None:
        conn = self.logging_db._ensure_connection()
        conn.execute(
            "UPDATE runs SET end_time = CURRENT_TIMESTAMP WHERE run_id = ?",
            (run_id,),
        )
        conn.commit()

    def _install_signal_handlers(self) -> dict[signal.Signals, Any]:
        previous = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }

        def _handler(signum: int, _frame: FrameType | None) -> None:
            self._shutdown_requested = True
            if self._termination_reason == "unknown":
                self._termination_reason = (
                    "graceful_shutdown"
                    if signum in {signal.SIGINT, signal.SIGTERM}
                    else "signal"
                )

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        return previous

    def _restore_signal_handlers(self, previous: dict[signal.Signals, Any]) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    def debug_state(self) -> dict[str, Any]:
        """Return state snapshot for QA tooling/tests."""
        return {
            "state": self.state.value,
            "turn": self._turn,
            "consecutive_empty_turns": self._consecutive_empty_turns,
            "shutdown_requested": self._shutdown_requested,
            "container_crashed": self._container_crashed,
            "termination_reason": self._termination_reason,
            "result_shape": asdict(
                RunResult(
                    total_turns=self._turn,
                    final_balance=self.credit_economy.get_balance(),
                    tiers_reached=sorted(self._tiers_reached),
                    termination_reason=self._termination_reason,
                )
            ),
        }
