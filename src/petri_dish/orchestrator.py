"""Central agent orchestrator state machine for Petri Dish MVP."""

from __future__ import annotations

import asyncio
import json
import shutil
import signal
import sqlite3
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import Any

from petri_dish.config import Settings
from petri_dish.traits import TraitEngine
from petri_dish.economy import AgentState, AgentReserve
from petri_dish.llm_client import OllamaClient
from petri_dish.logging_db import LoggingDB
from petri_dish.overseer import Overseer
from petri_dish.prompt import PromptManager
from petri_dish.promotion import PromotionEngine
from petri_dish.sandbox import ContainerNotRunningError, SandboxError, SandboxManager
from petri_dish.tool_parser import ToolCall, ToolCallParser
from petri_dish.tools import get_all_tools
from petri_dish.tools.agent_tools import get_prompt_overrides
from petri_dish.tools.comm_tools import _MessageStore, set_message_store
from petri_dish.tools.registry import ToolRegistry
from petri_dish.tools.task_broker import TaskBroker
from petri_dish.validators import FileValidator


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
    id: str = ""


class AgentOrchestrator:
    """Coordinates LLM inference, tool execution, economy, logging, and shutdown."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm_client: OllamaClient | None = None,
        tool_parser: ToolCallParser | None = None,
        tool_registry: ToolRegistry | None = None,
        agent_reserve: AgentReserve | None = None,
        sandbox_manager: SandboxManager | None = None,
        logging_db: LoggingDB | None = None,
        snapshot_interval_turns: int | None = None,
        file_validator: FileValidator | None = None,
    ) -> None:
        self.settings = settings or Settings.from_yaml()
        self.tool_registry = tool_registry or get_all_tools(settings=self.settings)
        self.tool_parser = tool_parser or ToolCallParser()
        self.llm_client = llm_client or OllamaClient(settings=self.settings)
        self.agent_reserve = agent_reserve or AgentReserve(settings=self.settings)
        self.sandbox_manager = sandbox_manager or SandboxManager()
        self.logging_db = logging_db or LoggingDB(":memory:")
        self.file_validator = file_validator or FileValidator(settings=self.settings)
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
        self._tiers_reached: set[str] = {self.agent_reserve.get_degradation_level()}
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
        self._tiers_reached = {self.agent_reserve.get_degradation_level()}
        self.state = OrchestratorState.IDLE

        self._connect_logging_db()
        self.logging_db.log_run_start(
            run_id,
            {
                "settings": self.settings.model_dump(),
                "snapshot_interval_turns": self.snapshot_interval_turns,
            },
        )
        self.logging_db.log_zod_transaction(
            run_id,
            self.agent_reserve.get_balance(),
            "initial_zod",
            "Run started",
        )

        previous_handlers = self._install_signal_handlers()
        self._container_id = self.sandbox_manager.create_container(
            run_id,
            memory_host_path=self.settings.memory_path or None,
        )

        try:
            while True:
                if self._shutdown_requested:
                    self._termination_reason = "graceful_shutdown"
                    break

                if self.agent_reserve.is_dead():
                    self._termination_reason = "starvation_death"
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

                if (
                    self.agent_reserve.is_depleted()
                    and not self.agent_reserve.is_stripped()
                    and self.agent_reserve.state != AgentState.DEAD
                ):
                    old_state = self.agent_reserve.state.value
                    self.agent_reserve.transition_to_stripped()
                    self.logging_db.log_state_transition(
                        run_id,
                        self._turn,
                        old_state,
                        AgentState.STRIPPED.value,
                        "zod_depleted",
                        self.agent_reserve.get_balance(),
                        0,
                    )

                self._turn += 1
                self.state = OrchestratorState.WAITING_FOR_LLM
                zod_before_turn = self.agent_reserve.get_balance()

                if not self.agent_reserve.is_stripped():
                    self.agent_reserve.consume(turns=1)
                    self.logging_db.log_zod_transaction(
                        run_id,
                        -self.settings.decay_rate_per_turn,
                        "turn_cost",
                        f"Turn {self._turn} inference",
                    )

                if self.agent_reserve.is_stripped():
                    tools_schemas = self.tool_registry.get_stripped_schemas()
                else:
                    tools_schemas = self.tool_registry.get_all_schemas()

                llm_result = await self.llm_client.chat(
                    system_prompt=self._build_system_prompt(),
                    messages=self._messages,
                    tools=tools_schemas,
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

                forced_call = self._build_forced_first_scout_call()
                if forced_call is not None:
                    tool_calls = [forced_call]
                    if not assistant_text:
                        assistant_text = (
                            "[forced-smoke] injecting deterministic overseer_scout"
                        )

                if assistant_text or tool_calls:
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": assistant_text or "",
                    }
                    if tool_calls:
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in tool_calls
                        ]
                    self._messages.append(assistant_msg)

                if not tool_calls:
                    self._consecutive_empty_turns += 1
                    self.state = OrchestratorState.LOGGING
                    self.logging_db.log_action(
                        run_id=run_id,
                        turn=self._turn,
                        tool_name="__empty_turn__",
                        tool_args=None,
                        result=assistant_text or "",
                        zod_before=zod_before_turn,
                        zod_after=self.agent_reserve.get_balance(),
                        duration_ms=0,
                    )
                else:
                    self._consecutive_empty_turns = 0
                    allowed_calls = tool_calls[
                        : max(1, int(self.settings.max_turns_per_tool))
                    ]
                    for call in allowed_calls:
                        if self.agent_reserve.is_stripped():
                            if not self.tool_registry.is_tool_allowed_when_stripped(
                                call.name
                            ):
                                result = f"Tool '{call.name}' is not available in STRIPPED state. Only observational tools are allowed."
                                self.state = OrchestratorState.LOGGING
                                self.logging_db.log_action(
                                    run_id=run_id,
                                    turn=self._turn,
                                    tool_name=call.name,
                                    tool_args=call.arguments,
                                    result=result,
                                    zod_before=zod_before_turn,
                                    zod_after=self.agent_reserve.get_balance(),
                                    duration_ms=0,
                                )
                                self._messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": call.id,
                                        "content": result,
                                    }
                                )
                                continue

                        call_started = time.perf_counter()
                        self.state = OrchestratorState.EXECUTING_TOOL
                        before_tool = self.agent_reserve.get_balance()

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

                        if not self.agent_reserve.is_stripped():
                            self._debit_tool_cost(run_id, call.name)

                        after_tool = self.agent_reserve.get_balance()
                        elapsed_ms = int((time.perf_counter() - call_started) * 1000)

                        self.state = OrchestratorState.LOGGING
                        suffix = " [FAILED]" if failed else ""
                        self.logging_db.log_action(
                            run_id=run_id,
                            turn=self._turn,
                            tool_name=call.name,
                            tool_args=call.arguments,
                            result=f"{result}{suffix}",
                            zod_before=before_tool,
                            zod_after=after_tool,
                            duration_ms=elapsed_ms,
                        )
                        self._capture_scout_report(
                            run_id,
                            self._turn,
                            tool_name=call.name,
                            tool_args=call.arguments,
                            result=result,
                        )
                        self._messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": result,
                            }
                        )

                        if self._container_crashed:
                            self._termination_reason = "container_crash"
                            break

                        if call.name == "pass_turn":
                            break

                self._tiers_reached.add(self.agent_reserve.get_degradation_level())

                if self._container_id:
                    self._validate_outputs(run_id)

                # Tick starvation at end of turn so agent gets full starvation_turns LLM calls
                if self.agent_reserve.is_stripped():
                    self.agent_reserve.tick_starvation()
                    if self.agent_reserve.is_dead():
                        self.logging_db.log_state_transition(
                            run_id,
                            self._turn,
                            AgentState.STRIPPED.value,
                            AgentState.DEAD.value,
                            "starvation_death",
                            self.agent_reserve.get_balance(),
                            self.agent_reserve.starvation_counter,
                        )

                if self._turn % self.snapshot_interval_turns == 0:
                    self._save_state_snapshot(run_id)

                if self._container_crashed:
                    break

            return RunResult(
                total_turns=self._turn,
                final_balance=self.agent_reserve.get_balance(),
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
            f"Degradation: {self.agent_reserve.get_degradation_level()}, "
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
            balance=self.agent_reserve.get_balance(),
            state_summary=state_summary,
            has_persistent_memory=bool(self.settings.memory_path),
            agent_state=self.agent_reserve.state.value,
            starvation_remaining=self.agent_reserve.get_starvation_remaining(),
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
                call_id = call.get("id", "")
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
                        _ToolCallPayload(
                            name=name.strip(),
                            arguments=arguments,
                            id=call_id,
                        )
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

    def _build_forced_first_scout_call(self) -> _ToolCallPayload | None:
        if not self.settings.force_first_overseer_scout or self._turn != 1:
            return None
        if self.tool_registry.get_tool("overseer_scout") is None:
            return None

        return _ToolCallPayload(
            name="overseer_scout",
            arguments={
                "claimed_pattern": "forced_first_turn_probe",
                "output_summary": "First-turn forced overseer smoke probe",
                "file_family": "json",
                "search_queries": ["python json validation patterns"],
                "requesting_agent_id": "forced-smoke-single-agent",
                "turn_id": f"forced-turn-{self._turn}",
            },
            id=f"forced_overseer_scout_{self._turn}",
        )

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
        _ = self.agent_reserve.grant(-cost)
        self.logging_db.log_zod_transaction(
            run_id,
            -cost,
            "tool_cost",
            f"Tool invocation: {tool_name}",
        )

    def _extract_scout_report(
        self, tool_name: str, tool_args: dict[str, Any], result: str
    ) -> tuple[dict[str, Any], str | None]:
        if tool_name != "overseer_scout":
            return {}, None
        try:
            parsed = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return {}, None
        if not isinstance(parsed, dict):
            return {}, None
        if "report_id" not in parsed or "suggested_bonus" not in parsed:
            return {}, None

        target_filename = parsed.get("target_filename") or tool_args.get(
            "target_filename"
        )
        if target_filename is not None:
            target_filename = str(target_filename)
        return parsed, target_filename

    def _capture_scout_report(
        self,
        run_id: str,
        turn: int,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        result: str,
        agent_id: str | None = None,
    ) -> None:
        report, target_filename = self._extract_scout_report(
            tool_name, tool_args, result
        )
        if not report:
            return
        self.logging_db.log_scout_report(
            run_id,
            turn,
            report,
            result,
            agent_id=agent_id,
            target_filename=target_filename,
        )
        self.logging_db.log_event(
            run_id,
            turn,
            agent_id or str(report.get("requesting_agent_id", "agent")),
            "scout_report_logged",
            (
                f"report_id={report.get('report_id')}, "
                f"target={target_filename or ''}, "
                f"suggested_bonus={float(report.get('suggested_bonus', 0.0)):.3f}"
            ),
        )

    def _validate_outputs(self, run_id: str) -> None:
        outputs = self.file_validator.collect_outputs(
            self.sandbox_manager, self._container_id
        )
        if not outputs:
            return

        total_earned = 0.0
        for filename, content in outputs:
            passed, zod_earned = self.file_validator.validate(filename, content)
            if passed and zod_earned > 0:
                self.agent_reserve.grant(zod_earned)
                total_earned += zod_earned
                self.logging_db.log_zod_transaction(
                    run_id,
                    zod_earned,
                    "validation_reward",
                    f"File validated: {filename}",
                )

                scout_bonus = 0.0
                scout_report = self.logging_db.get_pending_scout_report_for_file(
                    run_id, filename
                )
                if scout_report:
                    suggested_bonus = max(
                        0.0,
                        min(float(scout_report.get("suggested_bonus", 0.0)), 0.15),
                    )
                    if suggested_bonus > 0:
                        scout_bonus = suggested_bonus
                        self.agent_reserve.grant(scout_bonus)
                        total_earned += scout_bonus
                        self.logging_db.log_zod_transaction(
                            run_id,
                            scout_bonus,
                            "scout_bonus",
                            f"Scout bonus for {filename} (report_id={scout_report.get('report_id', '')})",
                        )
                        self.logging_db.mark_scout_report_applied(
                            int(scout_report["id"]),
                            applied_turn=self._turn,
                            applied_bonus=scout_bonus,
                        )
                        self.logging_db.log_event(
                            run_id,
                            self._turn,
                            str(scout_report.get("requesting_agent_id", "agent")),
                            "scout_bonus_applied",
                            (
                                f"filename={filename}, "
                                f"report_id={scout_report.get('report_id', '')}, "
                                f"bonus={scout_bonus:.3f}"
                            ),
                            zod_delta=scout_bonus,
                        )

                zod_earned += scout_bonus
            self.logging_db.log_action(
                run_id=run_id,
                turn=self._turn,
                tool_name="__validation__",
                tool_args={"filename": filename, "passed": passed},
                result=f"zod_earned={zod_earned:.2f}",
                zod_before=self.agent_reserve.get_balance() - zod_earned,
                zod_after=self.agent_reserve.get_balance(),
                duration_ms=0,
            )
            self.sandbox_manager.exec_in_container(
                self._container_id,
                f"rm -f /env/outgoing/{filename}",
            )

        if total_earned > 0:
            self._messages.append(
                {
                    "role": "user",
                    "content": (
                        f"📈 You earned {total_earned:.1f} zod. "
                        f"Balance: {self.agent_reserve.get_balance():.1f}"
                    ),
                }
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
            "balance": self.agent_reserve.get_balance(),
            "degradation_level": self.agent_reserve.get_degradation_level(),
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
                    final_balance=self.agent_reserve.get_balance(),
                    tiers_reached=sorted(self._tiers_reached),
                    termination_reason=self._termination_reason,
                )
            ),
        }


@dataclass(slots=True)
class MultiAgentRunResult:
    """Terminal summary for a multi-agent run."""

    total_rounds: int
    agent_results: dict[str, RunResult]
    common_pool: float
    termination_reason: str


class MultiAgentOrchestrator:
    """Coordinates multiple agents taking turns in round-robin order.

    Each agent gets one turn per round (LLM call + up to actions_per_turn
    tool executions). Dead/spectator agents are skipped. The shared reserve
    manages death, salvage, spectator cooldown, and re-entry.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        shared_economy: Any | None = None,
        sandbox_manager: SandboxManager | None = None,
        logging_db: LoggingDB | None = None,
        file_validator: FileValidator | None = None,
        llm_clients: dict[str, Any] | None = None,
        agent_names: list[str] | None = None,
        ecology: Any | None = None,
    ) -> None:
        self.settings = settings or Settings.from_yaml()
        self.sandbox_manager = sandbox_manager or SandboxManager()
        self.logging_db = logging_db or LoggingDB(":memory:")
        self.file_validator = file_validator or FileValidator(settings=self.settings)
        self._ecology = ecology

        from petri_dish.economy import SharedReserve

        self.shared_economy: SharedReserve = shared_economy or SharedReserve(
            settings=self.settings, agent_ids=agent_names
        )

        self.agent_ids = agent_names or self.shared_economy.get_agent_ids()
        self.actions_per_turn = self.settings.multi_agent_actions_per_turn
        self.max_rounds = self.settings.max_turns
        self._shared_volume_dir: str | None = None

        # Per-agent state
        self._agent_messages: dict[str, list[dict[str, Any]]] = {
            aid: [] for aid in self.agent_ids
        }
        self._agent_turns: dict[str, int] = {aid: 0 for aid in self.agent_ids}
        self._agent_consecutive_empty: dict[str, int] = {
            aid: 0 for aid in self.agent_ids
        }
        self._agent_containers: dict[str, str] = {}
        self._agent_registries: dict[str, ToolRegistry] = {}
        self._agent_prompt_managers: dict[str, PromptManager] = {}

        self._llm_clients = llm_clients or {}
        self._tool_parser = ToolCallParser()
        self._shutdown_requested = False
        self._state = OrchestratorState.IDLE
        self._round = 0
        self._termination_reason = "unknown"
        self._global_consecutive_empty = 0
        self._round_tool_names: list[str] = []
        self._round_message_count = 0

        self._msg_store = _MessageStore()
        set_message_store(self._msg_store)
        self._traits: TraitEngine | None = (
            TraitEngine(self.settings) if self.settings.traits_enabled else None
        )
        if self._traits is not None:
            for agent_id in self.agent_ids:
                self._traits.register_agent(agent_id)
        self._overseer: Overseer | None = (
            Overseer(self.settings, self.logging_db)
            if self.settings.overseer_enabled
            else None
        )
        self._task_broker: TaskBroker | None = (
            TaskBroker(self.settings) if self.settings.request_task_enabled else None
        )

    def _get_llm_client(self, agent_id: str) -> Any:
        if agent_id in self._llm_clients:
            return self._llm_clients[agent_id]

        model_name = self.settings.openai_model_name
        if self.settings.multi_agent_models and self.settings.multi_agent_names:
            try:
                idx = self.settings.multi_agent_names.index(agent_id)
                if idx < len(self.settings.multi_agent_models):
                    model_name = self.settings.multi_agent_models[idx]
            except ValueError:
                pass

        if self.settings.llm_backend == "openai_compatible":
            from petri_dish.openai_client import OpenAICompatibleClient
            import os

            client = OpenAICompatibleClient(
                api_key=os.getenv(self.settings.openai_api_key_env_var, ""),
                base_url=self.settings.openai_api_base_url,
                model=model_name,
                temperature=self.settings.default_temperature,
                rate_limit_max_retries=self.settings.rate_limit_max_retries,
                rate_limit_initial_delay=self.settings.rate_limit_initial_delay,
                rate_limit_max_delay=self.settings.rate_limit_max_delay,
            )
            self._llm_clients[agent_id] = client
            return client
        client = OllamaClient(
            settings=self.settings,
            rate_limit_max_retries=self.settings.rate_limit_max_retries,
            rate_limit_initial_delay=self.settings.rate_limit_initial_delay,
            rate_limit_max_delay=self.settings.rate_limit_max_delay,
        )
        self._llm_clients[agent_id] = client
        return client

    async def _handle_request_task_multi(
        self,
        run_id: str,
        agent_id: str,
        economy: Any,
        call: _ToolCallPayload,
    ) -> str:
        assert self._task_broker is not None

        task_description = call.arguments.get("task_description", "")
        if not task_description or not isinstance(task_description, str):
            return "Error: task_description must be a non-empty string."

        quote = await self._task_broker.estimate_cost(task_description)

        balance = economy.get_balance()
        if balance < quote.cost_zod:
            return (
                f"Insufficient zod balance for this task. "
                f"Quoted cost: {quote.cost_zod:.2f} zod, "
                f"your balance: {balance:.2f} zod. "
                f"Complexity: {quote.complexity.name}. "
                f"Summary: {quote.summary}"
            )

        output = await self._task_broker.execute_task(
            task_description, quote.complexity
        )

        economy.grant(-quote.cost_zod)
        self.logging_db.log_zod_transaction(
            run_id,
            -quote.cost_zod,
            "request_task",
            f"Agent {agent_id}: {task_description[:80]}",
            agent_id=agent_id,
        )

        return (
            f"[Task completed | Complexity: {quote.complexity.name} | "
            f"Cost: {quote.cost_zod:.2f} zod]\n\n{output}"
        )

    def _build_agent_prompt(self, agent_id: str) -> str:
        tool_costs = self.settings.tool_costs
        tool_list = []
        registry = self._agent_registries.get(agent_id)
        if not registry:
            return ""
        for tool_name in registry.get_tool_names():
            tool_def = registry.get_tool(tool_name)
            if tool_def:
                tool_list.append(
                    {"name": tool_name, "description": tool_def.description or ""}
                )

        economy = self.shared_economy.get_agent_economy(agent_id)
        prompt_mgr = self._agent_prompt_managers.get(agent_id)
        if not prompt_mgr:
            prompt_mgr = PromptManager()
        instincts = (
            self._traits.generate_instincts(agent_id)
            if self._traits is not None
            else ""
        )

        return prompt_mgr.build_multi_agent_system_prompt(
            agent_id=agent_id,
            tools=tool_list,
            tool_costs=tool_costs,
            balance=economy.get_balance(),
            agent_state=economy.state.value,
            starvation_remaining=economy.get_starvation_remaining(),
            agent_summaries=self.shared_economy.get_agent_summaries(),
            actions_per_turn=self.actions_per_turn,
            has_persistent_memory=bool(self.settings.memory_path),
            shared_filesystem=self.settings.multi_agent_shared_filesystem,
            traits_instincts=instincts,
        )

    async def run(self, run_id: str) -> MultiAgentRunResult:
        """Execute multi-agent round-robin loop until termination."""
        self._shutdown_requested = False
        self._termination_reason = "unknown"
        self._round = 0
        self._state = OrchestratorState.IDLE
        self._global_consecutive_empty = 0

        self.logging_db.connect()
        self.logging_db.log_run_start(
            run_id,
            {
                "settings": self.settings.model_dump(),
                "multi_agent": True,
                "agent_ids": self.agent_ids,
            },
        )

        previous_handlers = self._install_signal_handlers()

        if self.settings.multi_agent_shared_filesystem:
            self._shared_volume_dir = tempfile.mkdtemp(prefix="petri-shared-")
        else:
            self._shared_volume_dir = None

        for agent_id in self.agent_ids:
            container_id = self.sandbox_manager.create_container(
                f"{run_id}-{agent_id}",
                memory_host_path=self.settings.memory_path or None,
                shared_volume_host_path=self._shared_volume_dir,
            )
            self._agent_containers[agent_id] = container_id
            self._agent_registries[agent_id] = get_all_tools(settings=self.settings)
            self._agent_prompt_managers[agent_id] = PromptManager(
                modifications_path=str(
                    Path(self.logging_db.db_path).parent
                    / f"modifications_{agent_id}.json"
                )
                if self.logging_db.db_path != ":memory:"
                else f"/tmp/petri_dish_modifications_{agent_id}.json"
            )
            self.logging_db.log_zod_transaction(
                run_id,
                self.shared_economy.get_agent_balance(agent_id),
                "initial_zod",
                f"Agent {agent_id} starting balance",
                agent_id=agent_id,
            )

        try:
            while True:
                if self._shutdown_requested:
                    self._termination_reason = "graceful_shutdown"
                    break

                self._round += 1
                self._reset_round_tracking()
                if self._round > self.max_rounds:
                    self._round = self.max_rounds
                    self._termination_reason = "max_rounds_reached"
                    break

                # Process dead agent spectator ticks and reentry BEFORE all-dead check
                for agent_id in self.agent_ids:
                    economy = self.shared_economy.get_agent_economy(agent_id)
                    if economy.is_dead():
                        ready = self.shared_economy.tick_spectator(agent_id)
                        if ready and self.shared_economy.reentry(agent_id):
                            self.logging_db.log_state_transition(
                                run_id,
                                self._round,
                                "dead",
                                "active",
                                "reentry",
                                self.shared_economy.get_agent_balance(agent_id),
                                agent_id=agent_id,
                            )
                            self.logging_db.log_event(
                                run_id,
                                self._round,
                                agent_id,
                                "reentry",
                                f"balance={self.shared_economy.get_agent_balance(agent_id)}",
                            )
                            if self._traits is not None:
                                self._traits.observe_reentry(agent_id)

                living = self.shared_economy.get_living_agents()
                if not living:
                    self._termination_reason = "all_agents_dead"
                    break

                self._drop_ecology_files(run_id)

                any_progress = False
                for agent_id in self.agent_ids:
                    if self._shutdown_requested:
                        self._termination_reason = "graceful_shutdown"
                        break

                    economy = self.shared_economy.get_agent_economy(agent_id)

                    if economy.is_dead():
                        continue

                    # Skip agents that have been idle too long (save API costs)
                    if (
                        self._agent_consecutive_empty[agent_id]
                        >= self.settings.max_consecutive_empty_turns
                    ):
                        self.logging_db.log_event(
                            run_id,
                            self._round,
                            agent_id,
                            "agent_idle_skip",
                            f"consecutive_empty={self._agent_consecutive_empty[agent_id]}",
                        )
                        continue

                    # Check depleted -> stripped transition
                    if economy.is_depleted() and not economy.is_stripped():
                        old_state = economy.state.value
                        economy.transition_to_stripped()
                        self.logging_db.log_state_transition(
                            run_id,
                            self._round,
                            old_state,
                            AgentState.STRIPPED.value,
                            "zod_depleted",
                            economy.get_balance(),
                            0,
                            agent_id=agent_id,
                        )
                        self.logging_db.log_event(
                            run_id,
                            self._round,
                            agent_id,
                            "stripped",
                            f"balance={economy.get_balance()}",
                        )

                    self._agent_turns[agent_id] += 1
                    turn = self._agent_turns[agent_id]
                    zod_before = economy.get_balance()

                    self._msg_store.configure(
                        run_id=run_id,
                        round_num=self._round,
                        turn=turn,
                        log_fn=lambda s, r, c, rnd, t: self.logging_db.log_message(
                            run_id, s, r, c, rnd, t
                        ),
                    )
                    self._deliver_pending_messages(run_id, agent_id)

                    if not economy.is_stripped():
                        economy.consume(turns=1)
                        self.logging_db.log_zod_transaction(
                            run_id,
                            -self.settings.decay_rate_per_turn,
                            "turn_cost",
                            f"Agent {agent_id} turn {turn}",
                            agent_id=agent_id,
                        )

                    # Select tool schemas based on state
                    registry = self._agent_registries[agent_id]
                    if economy.is_stripped():
                        tools_schemas = registry.get_stripped_schemas()
                    else:
                        tools_schemas = registry.get_all_schemas()

                    # LLM call
                    if self.settings.llm_inter_call_delay > 0:
                        await asyncio.sleep(self.settings.llm_inter_call_delay)
                    self._state = OrchestratorState.WAITING_FOR_LLM
                    client = self._get_llm_client(agent_id)
                    _agent_prompt = self._build_agent_prompt(agent_id)
                    msgs = self._agent_messages[agent_id]
                    max_msgs = self.settings.max_agent_messages
                    if len(msgs) > max_msgs:
                        msgs[:] = msgs[-max_msgs:]
                    _llm_start = time.perf_counter()
                    llm_result = await client.chat(
                        system_prompt=_agent_prompt,
                        messages=msgs,
                        tools=tools_schemas,
                    )
                    _llm_elapsed_ms = (time.perf_counter() - _llm_start) * 1000

                    _model_name = self.settings.openai_model_name
                    if (
                        self.settings.multi_agent_models
                        and self.settings.multi_agent_names
                    ):
                        try:
                            _idx = self.settings.multi_agent_names.index(agent_id)
                            if _idx < len(self.settings.multi_agent_models):
                                _model_name = self.settings.multi_agent_models[_idx]
                        except ValueError:
                            pass

                    _resp_snippet = ""
                    if llm_result is not None:
                        _resp_snippet = (llm_result[0] or "")[:500]

                    self.logging_db.log_llm_call(
                        run_id=run_id,
                        turn=turn,
                        agent_id=agent_id,
                        model_name=_model_name,
                        system_prompt_snippet=_agent_prompt[:500],
                        user_prompt_snippet=(
                            self._agent_messages[agent_id][-1].get("content", "")[:500]
                            if self._agent_messages[agent_id]
                            else ""
                        ),
                        response_snippet=_resp_snippet,
                        duration_ms=_llm_elapsed_ms,
                    )

                    if llm_result is None:
                        assistant_text = ""
                        raw_calls: list[Any] = []
                    else:
                        assistant_text, raw_calls = llm_result

                    tool_calls = self._normalize_tool_calls(raw_calls)
                    if not tool_calls and assistant_text:
                        reparsed = self.tool_parser.parse(assistant_text)
                        tool_calls = self._normalize_tool_calls(reparsed)

                    if assistant_text or tool_calls:
                        assistant_msg: dict[str, Any] = {
                            "role": "assistant",
                            "content": assistant_text or "",
                        }
                        if tool_calls:
                            assistant_msg["tool_calls"] = [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": json.dumps(tc.arguments),
                                    },
                                }
                                for tc in tool_calls
                            ]
                        self._agent_messages[agent_id].append(assistant_msg)

                    if not tool_calls:
                        self._agent_consecutive_empty[agent_id] += 1
                        if self._traits is not None:
                            self._traits.observe_empty_turn(agent_id)
                        self._state = OrchestratorState.LOGGING
                        self.logging_db.log_action(
                            run_id=run_id,
                            turn=turn,
                            tool_name="__empty_turn__",
                            tool_args=None,
                            result=assistant_text or "",
                            zod_before=zod_before,
                            zod_after=economy.get_balance(),
                            duration_ms=0,
                            agent_id=agent_id,
                        )
                    else:
                        any_progress = True
                        self._agent_consecutive_empty[agent_id] = 0
                        allowed_calls = tool_calls[: self.actions_per_turn]
                        for call in allowed_calls:
                            if economy.is_stripped():
                                if not registry.is_tool_allowed_when_stripped(
                                    call.name
                                ):
                                    result = (
                                        f"Tool '{call.name}' not available in "
                                        f"STRIPPED state."
                                    )
                                    self.logging_db.log_action(
                                        run_id=run_id,
                                        turn=turn,
                                        tool_name=call.name,
                                        tool_args=call.arguments,
                                        result=result,
                                        zod_before=zod_before,
                                        zod_after=economy.get_balance(),
                                        duration_ms=0,
                                        agent_id=agent_id,
                                    )
                                    self._agent_messages[agent_id].append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": call.id,
                                            "content": result,
                                        }
                                    )
                                    continue

                            call_start = time.perf_counter()
                            self._state = OrchestratorState.EXECUTING_TOOL
                            before_tool = economy.get_balance()
                            self._round_tool_names.append(call.name)
                            if call.name == "send_message":
                                self._round_message_count += 1
                            result = ""
                            try:
                                container_id = self._agent_containers.get(agent_id, "")
                                if call.name in ("send_message", "read_messages"):
                                    comm_args = dict(call.arguments)
                                    comm_args["_sender_id"] = agent_id
                                    comm_args["_recipient_id"] = agent_id
                                    result = registry.execute_tool(
                                        call.name,
                                        comm_args,
                                        container_id,
                                    )
                                elif call.name == "web_search":
                                    result = registry.execute_tool(
                                        call.name,
                                        dict(call.arguments, settings=self.settings),
                                        container_id,
                                    )
                                elif call.name == "request_task":
                                    if self._task_broker is None:
                                        result = "request_task is not enabled in this environment."
                                    else:
                                        result = await self._handle_request_task_multi(
                                            run_id, agent_id, economy, call
                                        )
                                else:
                                    result = registry.execute_tool(
                                        call.name,
                                        call.arguments,
                                        container_id,
                                    )
                                if self._traits is not None:
                                    self._traits.observe_tool_use(agent_id, call.name)
                            except Exception as exc:
                                result = (
                                    f"Tool execution failed: "
                                    f"{type(exc).__name__}: {exc}"
                                )

                            if not economy.is_stripped():
                                cost = float(registry.get_tool_cost(call.name))
                                if cost > 0:
                                    economy.grant(-cost)
                                    self.logging_db.log_zod_transaction(
                                        run_id,
                                        -cost,
                                        "tool_cost",
                                        f"Agent {agent_id}: {call.name}",
                                        agent_id=agent_id,
                                    )

                            elapsed = int((time.perf_counter() - call_start) * 1000)
                            self._state = OrchestratorState.LOGGING
                            self.logging_db.log_action(
                                run_id=run_id,
                                turn=turn,
                                tool_name=call.name,
                                tool_args=call.arguments,
                                result=result,
                                zod_before=before_tool,
                                zod_after=economy.get_balance(),
                                duration_ms=elapsed,
                                agent_id=agent_id,
                            )
                            self._capture_scout_report(
                                run_id,
                                turn,
                                tool_name=call.name,
                                tool_args=call.arguments,
                                result=result,
                                agent_id=agent_id,
                            )
                            self._agent_messages[agent_id].append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.id,
                                    "content": result,
                                }
                            )

                            if call.name == "pass_turn":
                                break

                    # Validate outputs for this agent
                    container_id = self._agent_containers.get(agent_id, "")
                    if container_id:
                        self._validate_agent_outputs(
                            run_id, agent_id, turn, container_id
                        )

                    # Tick starvation
                    if economy.is_stripped():
                        economy.tick_starvation()
                        if self._traits is not None:
                            self._traits.observe_starvation(agent_id)
                        if economy.is_dead():
                            self.shared_economy.handle_death(agent_id)
                            if self._traits is not None:
                                self._traits.observe_death(agent_id)
                            self.logging_db.log_state_transition(
                                run_id,
                                self._round,
                                AgentState.STRIPPED.value,
                                AgentState.DEAD.value,
                                "starvation_death",
                                economy.get_balance(),
                                economy.starvation_counter,
                                agent_id=agent_id,
                            )
                            self.logging_db.log_event(
                                run_id,
                                self._round,
                                agent_id,
                                "death",
                                f"starvation_counter={economy.starvation_counter}",
                            )

                if self._shutdown_requested:
                    break

                if not any_progress:
                    self._global_consecutive_empty += 1
                    if (
                        self._global_consecutive_empty
                        >= self.settings.max_consecutive_empty_turns
                    ):
                        self._termination_reason = "all_agents_idle"
                        break
                else:
                    self._global_consecutive_empty = 0

                if self._traits is not None:
                    self._traits.end_round()
                    for aid in self.agent_ids:
                        traits = self._traits.get_traits(aid)
                        self.logging_db.log_trait_snapshot(
                            run_id=run_id,
                            round_num=self._round,
                            agent_id=aid,
                            traits_dict=traits.to_dict(),
                        )

                if self._overseer is not None:
                    trait_vectors = {}
                    if self._traits is not None:
                        for aid in self.agent_ids:
                            trait_vectors[aid] = self._traits.get_traits(aid).to_dict()
                    evaluations, artifact = await self._overseer.maybe_evaluate(
                        run_id=run_id,
                        turn=self._round,
                        trait_vectors=trait_vectors,
                    )
                    for ev in evaluations:
                        agent_id = ev.get("agent_id", "")
                        bonus = float(ev.get("bonus", 0.0))
                        if agent_id and bonus > 0:
                            self.shared_economy.grant(agent_id, bonus)
                            self.logging_db.log_zod_transaction(
                                run_id,
                                bonus,
                                "overseer_bonus",
                                ev.get("reasoning", "")[:120],
                                agent_id=agent_id,
                            )
                    if artifact and self._shared_volume_dir:
                        target = Path(self._shared_volume_dir) / artifact["filename"]
                        target.write_text(artifact["content"])
                        self.logging_db.log_file_drop(
                            run_id,
                            artifact["filename"],
                            "overseer_provocation",
                        )

                # Entropic UBI
                if self.settings.base_income_per_turn > 0:
                    entropy = self._compute_entropy()
                    window = max(1, self.settings.entropy_window_turns)
                    scale = min(1.0, entropy / window)
                    ubi = self.settings.ubi_min + scale * (
                        self.settings.ubi_max - self.settings.ubi_min
                    )
                    for aid in self.agent_ids:
                        econ = self.shared_economy.get_agent_economy(aid)
                        if not econ.is_dead():
                            self.shared_economy.grant(aid, ubi)
                    self.logging_db.log_event(
                        run_id,
                        self._round,
                        "system",
                        "ubi_grant",
                        f"ubi={ubi:.3f} entropy={entropy:.1f}",
                    )

            agent_results = {}
            for aid in self.agent_ids:
                econ = self.shared_economy.get_agent_economy(aid)
                agent_results[aid] = RunResult(
                    total_turns=self._agent_turns[aid],
                    final_balance=econ.get_balance(),
                    tiers_reached=[],
                    termination_reason=self._termination_reason,
                )

            return MultiAgentRunResult(
                total_rounds=self._round,
                agent_results=agent_results,
                common_pool=self.shared_economy.get_common_pool(),
                termination_reason=self._termination_reason,
            )
        finally:
            self._state = OrchestratorState.TERMINATED
            set_message_store(None)
            self._restore_signal_handlers(previous_handlers)
            for cid in self._agent_containers.values():
                if cid:
                    self.sandbox_manager.destroy_container(cid)
            if self._shared_volume_dir:
                shutil.rmtree(self._shared_volume_dir, ignore_errors=True)
            conn = self.logging_db._ensure_connection()
            conn.execute(
                "UPDATE runs SET end_time = CURRENT_TIMESTAMP WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()

    @property
    def tool_parser(self) -> ToolCallParser:
        return self._tool_parser

    def _extract_scout_report(
        self, tool_name: str, tool_args: dict[str, Any], result: str
    ) -> tuple[dict[str, Any], str | None]:
        if tool_name != "overseer_scout":
            return {}, None
        try:
            parsed = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return {}, None
        if not isinstance(parsed, dict):
            return {}, None
        if "report_id" not in parsed or "suggested_bonus" not in parsed:
            return {}, None

        target_filename = parsed.get("target_filename") or tool_args.get(
            "target_filename"
        )
        if target_filename is not None:
            target_filename = str(target_filename)
        return parsed, target_filename

    def _capture_scout_report(
        self,
        run_id: str,
        turn: int,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        result: str,
        agent_id: str | None = None,
    ) -> None:
        report, target_filename = self._extract_scout_report(
            tool_name, tool_args, result
        )
        if not report:
            return
        self.logging_db.log_scout_report(
            run_id,
            turn,
            report,
            result,
            agent_id=agent_id,
            target_filename=target_filename,
        )
        self.logging_db.log_event(
            run_id,
            turn,
            agent_id or str(report.get("requesting_agent_id", "agent")),
            "scout_report_logged",
            (
                f"report_id={report.get('report_id')}, "
                f"target={target_filename or ''}, "
                f"suggested_bonus={float(report.get('suggested_bonus', 0.0)):.3f}"
            ),
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
                call_id = call.get("id", "")
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
                        _ToolCallPayload(
                            name=name.strip(),
                            arguments=arguments,
                            id=call_id,
                        )
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

    def _compute_entropy(self) -> float:
        unique_tools = len(set(self._round_tool_names))
        unique_ratio = unique_tools / max(1, len(self._round_tool_names))
        msg_density = min(1.0, self._round_message_count / max(1, len(self.agent_ids)))
        return unique_ratio + msg_density

    def _reset_round_tracking(self) -> None:
        self._round_tool_names = []
        self._round_message_count = 0

    def _drop_ecology_files(self, run_id: str) -> None:
        if not self._ecology:
            return
        files = self._ecology.schedule_drops(self._round)
        if not files:
            return
        for filename, content in files:
            file_type = filename.rsplit(".", 1)[-1]
            if self._shared_volume_dir:
                if filename == "NOTICE.txt":
                    content = content.replace("/env/incoming/", "/env/shared/")
                target_path = Path(self._shared_volume_dir) / filename
                target_path.write_text(content)
                self.logging_db.log_file_drop(
                    run_id=run_id,
                    filename=filename,
                    file_type=file_type,
                )
                self.logging_db.log_event(
                    run_id,
                    self._round,
                    "system",
                    "ecology_drop",
                    f"file={filename}, type={file_type}, target=shared_volume",
                )
            else:
                first_agent = next(iter(self.agent_ids), None)
                if first_agent:
                    cid = self._agent_containers.get(first_agent, "")
                    if cid:
                        self._ecology.drop_file(
                            self.sandbox_manager, cid, filename, content
                        )
                        self.logging_db.log_file_drop(
                            run_id=run_id,
                            filename=filename,
                            file_type=file_type,
                        )
                        self.logging_db.log_event(
                            run_id,
                            self._round,
                            first_agent,
                            "ecology_drop",
                            f"file={filename}, type={file_type}",
                        )

    def _deliver_pending_messages(self, run_id: str, agent_id: str) -> None:
        unread = self.logging_db.get_unread_messages(run_id, agent_id)
        if not unread:
            return
        lines = ["📨 New messages:"]
        for msg in unread:
            lines.append(
                f"  [From {msg['sender_id']}, round {msg['round_num']}, "
                f"turn {msg['turn']}]: {msg['content']}"
            )
        self._agent_messages[agent_id].append(
            {"role": "user", "content": "\n".join(lines)}
        )
        self.logging_db.mark_messages_read(run_id, agent_id)

    def _validate_agent_outputs(
        self, run_id: str, agent_id: str, turn: int, container_id: str
    ) -> None:
        outputs = self.file_validator.collect_outputs(
            self.sandbox_manager, container_id
        )
        if not outputs:
            return
        economy = self.shared_economy.get_agent_economy(agent_id)
        total_earned = 0.0
        failed_files: list[str] = []
        for filename, content in outputs:
            passed, zod_earned = self.file_validator.validate(filename, content)
            if passed and zod_earned > 0:
                self.shared_economy.grant(agent_id, zod_earned)
                if self._traits is not None:
                    self._traits.observe_reward(agent_id, zod_earned, filename)
                total_earned += zod_earned
                self.logging_db.log_zod_transaction(
                    run_id,
                    zod_earned,
                    "validation_reward",
                    f"Agent {agent_id} validated: {filename}",
                    agent_id=agent_id,
                )

                scout_bonus = 0.0
                scout_report = self.logging_db.get_pending_scout_report_for_file(
                    run_id, filename, agent_id=agent_id
                )
                if scout_report:
                    suggested_bonus = max(
                        0.0,
                        min(float(scout_report.get("suggested_bonus", 0.0)), 0.15),
                    )
                    if suggested_bonus > 0:
                        scout_bonus = suggested_bonus
                        self.shared_economy.grant(agent_id, scout_bonus)
                        total_earned += scout_bonus
                        self.logging_db.log_zod_transaction(
                            run_id,
                            scout_bonus,
                            "scout_bonus",
                            f"Agent {agent_id} scout bonus for {filename} (report_id={scout_report.get('report_id', '')})",
                            agent_id=agent_id,
                        )
                        self.logging_db.mark_scout_report_applied(
                            int(scout_report["id"]),
                            applied_turn=turn,
                            applied_bonus=scout_bonus,
                        )
                        self.logging_db.log_event(
                            run_id,
                            turn,
                            agent_id,
                            "scout_bonus_applied",
                            (
                                f"filename={filename}, "
                                f"report_id={scout_report.get('report_id', '')}, "
                                f"bonus={scout_bonus:.3f}"
                            ),
                            zod_delta=scout_bonus,
                        )

                zod_earned += scout_bonus
            else:
                failed_files.append(filename)
            self.logging_db.log_action(
                run_id=run_id,
                turn=turn,
                tool_name="__validation__",
                tool_args={"filename": filename, "passed": passed},
                result=f"zod_earned={zod_earned:.2f}",
                zod_before=economy.get_balance() - zod_earned,
                zod_after=economy.get_balance(),
                duration_ms=0,
                agent_id=agent_id,
            )
            self.sandbox_manager.exec_in_container(
                container_id,
                f"rm -f /env/outgoing/{filename}",
            )

        if total_earned > 0:
            self._agent_messages[agent_id].append(
                {
                    "role": "user",
                    "content": (
                        f"📈 You earned {total_earned:.1f} zod. "
                        f"Balance: {economy.get_balance():.1f}"
                    ),
                }
            )
        elif failed_files:
            self._agent_messages[agent_id].append(
                {
                    "role": "user",
                    "content": (
                        f"Your output for {len(failed_files)} file(s) was not accepted: "
                        + ", ".join(failed_files[:3])
                        + (
                            f" and {len(failed_files) - 3} more"
                            if len(failed_files) > 3
                            else ""
                        )
                        + ". Hint: preserve the original data format and structure."
                    ),
                }
            )

    def _install_signal_handlers(self) -> dict[signal.Signals, Any]:
        previous = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }

        def _handler(signum: int, _frame: FrameType | None) -> None:
            self._shutdown_requested = True
            if self._termination_reason == "unknown":
                self._termination_reason = "graceful_shutdown"

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        return previous

    def _restore_signal_handlers(self, previous: dict[signal.Signals, Any]) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    def debug_state(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "round": self._round,
            "agent_ids": self.agent_ids,
            "agent_turns": dict(self._agent_turns),
            "shutdown_requested": self._shutdown_requested,
            "termination_reason": self._termination_reason,
        }
