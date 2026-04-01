"""Central agent orchestrator state machine for Petri Dish MVP."""

from __future__ import annotations

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
from petri_dish.economy import AgentState, AgentReserve
from petri_dish.llm_client import OllamaClient
from petri_dish.logging_db import LoggingDB
from petri_dish.prompt import PromptManager
from petri_dish.sandbox import ContainerNotRunningError, SandboxError, SandboxManager
from petri_dish.tool_parser import ToolCall, ToolCallParser
from petri_dish.tools import get_all_tools
from petri_dish.tools.agent_tools import get_prompt_overrides
from petri_dish.tools.comm_tools import _MessageStore, set_message_store
from petri_dish.tools.registry import ToolRegistry
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
                                        "name": call.name,
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
        _ = self.agent_reserve.grant(-cost)
        self.logging_db.log_zod_transaction(
            run_id,
            -cost,
            "tool_cost",
            f"Tool invocation: {tool_name}",
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

        self._msg_store = _MessageStore()
        set_message_store(self._msg_store)

    def _get_llm_client(self, agent_id: str) -> Any:
        if agent_id in self._llm_clients:
            return self._llm_clients[agent_id]
        return OllamaClient(settings=self.settings)

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
        )

    async def run(self, run_id: str) -> MultiAgentRunResult:
        """Execute multi-agent round-robin loop until termination."""
        self._shutdown_requested = False
        self._termination_reason = "unknown"
        self._round = 0
        self._state = OrchestratorState.IDLE

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
                if self._round > self.max_rounds:
                    self._termination_reason = "max_rounds_reached"
                    break

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

                    # Handle dead agents: tick spectator cooldown, attempt re-entry
                    if economy.is_dead():
                        ready = self.shared_economy.tick_spectator(agent_id)
                        if ready:
                            self.shared_economy.reentry(agent_id)
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
                    self._state = OrchestratorState.WAITING_FOR_LLM
                    client = self._get_llm_client(agent_id)
                    llm_result = await client.chat(
                        system_prompt=self._build_agent_prompt(agent_id),
                        messages=self._agent_messages[agent_id],
                        tools=tools_schemas,
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

                    if assistant_text:
                        self._agent_messages[agent_id].append(
                            {"role": "assistant", "content": assistant_text}
                        )

                    if not tool_calls:
                        self._agent_consecutive_empty[agent_id] += 1
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
                                            "name": call.name,
                                            "content": result,
                                        }
                                    )
                                    continue

                            call_start = time.perf_counter()
                            self._state = OrchestratorState.EXECUTING_TOOL
                            before_tool = economy.get_balance()
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
                                else:
                                    result = registry.execute_tool(
                                        call.name,
                                        call.arguments,
                                        container_id,
                                    )
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
                            self._agent_messages[agent_id].append(
                                {
                                    "role": "tool",
                                    "name": call.name,
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
                        if economy.is_dead():
                            self.shared_economy.handle_death(agent_id)
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

    def _drop_ecology_files(self, run_id: str) -> None:
        if not self._ecology:
            return
        files = self._ecology.schedule_drops(self._round)
        if not files:
            return
        for filename, content in files:
            file_type = filename.rsplit(".", 1)[-1]
            if self._shared_volume_dir:
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
        for filename, content in outputs:
            passed, zod_earned = self.file_validator.validate(filename, content)
            if passed and zod_earned > 0:
                self.shared_economy.grant(agent_id, zod_earned)
                total_earned += zod_earned
                self.logging_db.log_zod_transaction(
                    run_id,
                    zod_earned,
                    "validation_reward",
                    f"Agent {agent_id} validated: {filename}",
                    agent_id=agent_id,
                )
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
