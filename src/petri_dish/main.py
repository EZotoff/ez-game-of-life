"""Petri Dish experiment entrypoint and end-to-end wiring."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Protocol, cast
from uuid import uuid4

from petri_dish.config import Settings
from petri_dish.context_manager import ContextManager
from petri_dish.degradation import DegradationManager
from petri_dish.ecology import ResourceEcology
from petri_dish.economy import AgentReserve, SharedReserve
from petri_dish.llm_client import OllamaClient
from petri_dish.logging_db import LoggingDB
from petri_dish.null_model import NullModel
from petri_dish.openai_client import OpenAICompatibleClient
from petri_dish.orchestrator import (
    AgentOrchestrator,
    MultiAgentOrchestrator,
    MultiAgentRunResult,
    RunResult,
)
from petri_dish.sandbox import SandboxManager
from petri_dish.tool_parser import ToolCallParser
from petri_dish.tools import get_all_tools
from petri_dish.validators import FileValidator


class _ChatClient(Protocol):
    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]] | None: ...


class _EcologyChatAdapter:
    """Injects file ecology drops on each orchestrator turn."""

    def __init__(
        self,
        *,
        base_client: _ChatClient,
        ecology: ResourceEcology,
        sandbox_manager: SandboxManager,
        logging_db: LoggingDB,
        run_id: str,
        get_container_id: Callable[[], str],
    ) -> None:
        self._base_client = base_client
        self._ecology = ecology
        self._sandbox_manager = sandbox_manager
        self._logging_db = logging_db
        self._run_id = run_id
        self._get_container_id = get_container_id
        self._turn = 0

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]] | None:
        self._turn += 1
        container_id = self._get_container_id()
        if container_id:
            for filename, content in self._ecology.schedule_drops(self._turn):
                self._ecology.drop_file(
                    self._sandbox_manager,
                    container_id,
                    filename,
                    content,
                )
                self._logging_db.log_file_drop(
                    run_id=self._run_id,
                    filename=filename,
                    file_type=filename.rsplit(".", 1)[-1],
                )

        return await self._base_client.chat(system_prompt, messages, tools)


def _resolve_run_id(explicit_run_id: str | None = None) -> str:
    if explicit_run_id:
        return explicit_run_id
    env_run_id = os.getenv("PETRI_DISH_RUN_ID")
    if env_run_id:
        return env_run_id
    return f"run-{uuid4().hex[:12]}"


def _resolve_db_path(run_id: str) -> str:
    db_dir = Path(".sisyphus") / "runs"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / f"{run_id}.sqlite")


def _create_llm_client(settings: Settings, null_model: bool) -> _ChatClient:
    if null_model:
        return NullModel(null_model_type=settings.null_model_type)

    if settings.llm_backend == "openai_compatible":
        api_key = os.getenv(settings.openai_api_key_env_var, "")
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=settings.openai_api_base_url,
            model=settings.openai_model_name,
            temperature=settings.default_temperature,
        )

    return OllamaClient(settings=settings)


async def _run_experiment_async(
    *,
    config_path: str,
    null_model: bool,
    run_id: str | None,
) -> RunResult:
    settings = Settings.from_yaml(config_path)
    resolved_run_id = _resolve_run_id(run_id)

    if settings.multi_agent_enabled:
        result = await _run_multi_agent_async(
            settings=settings,
            config_path=config_path,
            null_model=null_model,
            run_id=resolved_run_id,
        )
        return (
            list(result.agent_results.values())[0]
            if result.agent_results
            else RunResult(
                total_turns=0,
                final_balance=0,
                tiers_reached=[],
                termination_reason="no_agents",
            )
        )

    agent_reserve = AgentReserve(settings=settings)
    logging_db = LoggingDB(db_path=_resolve_db_path(resolved_run_id))
    tool_parser = ToolCallParser()
    tool_registry = get_all_tools(settings=settings)
    sandbox_manager = SandboxManager()

    base_chat_client = _create_llm_client(settings, null_model)

    context_manager = ContextManager(settings=settings)
    degradation_manager = DegradationManager(settings=settings)
    resource_ecology = ResourceEcology(settings=settings)
    file_validator = FileValidator(settings=settings)

    orchestrator = AgentOrchestrator(
        settings=settings,
        tool_parser=tool_parser,
        tool_registry=tool_registry,
        agent_reserve=agent_reserve,
        sandbox_manager=sandbox_manager,
        logging_db=logging_db,
        snapshot_interval_turns=settings.context_summary_interval_turns,
        file_validator=file_validator,
    )

    cast(Any, orchestrator).llm_client = _EcologyChatAdapter(
        base_client=base_chat_client,
        ecology=resource_ecology,
        sandbox_manager=sandbox_manager,
        logging_db=logging_db,
        run_id=resolved_run_id,
        get_container_id=lambda: getattr(orchestrator, "_container_id", ""),
    )

    print(
        json.dumps(
            {
                "event": "run_start",
                "run_id": resolved_run_id,
                "config_path": config_path,
                "model": "null" if null_model else settings.model_name,
                "db_path": logging_db.db_path,
                "context_window_tokens": context_manager.context_window_tokens,
                "economy_mode": degradation_manager.economy_mode,
            }
        )
    )

    result = await orchestrator.run(resolved_run_id)

    logging_db.log_action(
        run_id=resolved_run_id,
        turn=result.total_turns,
        tool_name="__run_summary__",
        tool_args={"null_model": null_model},
        result=json.dumps(
            {
                "termination_reason": result.termination_reason,
                "tiers_reached": result.tiers_reached,
                "total_turns": result.total_turns,
                "final_balance": result.final_balance,
            }
        ),
        zod_before=result.final_balance,
        zod_after=result.final_balance,
        duration_ms=0,
    )

    print(
        json.dumps(
            {
                "event": "run_complete",
                "run_id": resolved_run_id,
                "result": {
                    "total_turns": result.total_turns,
                    "final_balance": result.final_balance,
                    "tiers_reached": result.tiers_reached,
                    "termination_reason": result.termination_reason,
                },
            }
        )
    )

    return result


async def _run_multi_agent_async(
    *,
    settings: Settings,
    config_path: str,
    null_model: bool,
    run_id: str,
) -> MultiAgentRunResult:
    agent_names = settings.multi_agent_names or [
        f"agent-{i}" for i in range(settings.multi_agent_count)
    ]
    logging_db = LoggingDB(db_path=_resolve_db_path(run_id))
    sandbox_manager = SandboxManager()
    shared_economy = SharedReserve(settings=settings, agent_ids=agent_names)
    file_validator = FileValidator(settings=settings)
    resource_ecology = ResourceEcology(settings=settings)

    llm_clients: dict[str, _ChatClient] = {}

    orchestrator = MultiAgentOrchestrator(
        settings=settings,
        shared_economy=shared_economy,
        sandbox_manager=sandbox_manager,
        logging_db=logging_db,
        file_validator=file_validator,
        llm_clients=llm_clients,
        agent_names=agent_names,
        ecology=resource_ecology,
    )

    print(
        json.dumps(
            {
                "event": "multi_agent_run_start",
                "run_id": run_id,
                "config_path": config_path,
                "agent_count": len(agent_names),
                "agent_names": agent_names,
                "model": "null" if null_model else settings.model_name,
            }
        )
    )

    result = await orchestrator.run(run_id)

    print(
        json.dumps(
            {
                "event": "multi_agent_run_complete",
                "run_id": run_id,
                "total_rounds": result.total_rounds,
                "common_pool": result.common_pool,
                "termination_reason": result.termination_reason,
                "agent_results": {
                    aid: {
                        "total_turns": ar.total_turns,
                        "final_balance": ar.final_balance,
                        "termination_reason": ar.termination_reason,
                    }
                    for aid, ar in result.agent_results.items()
                },
            }
        )
    )

    return result


def run_experiment(
    config_path: str = "config.yaml", null_model: bool = False
) -> RunResult:
    """Run one experiment and return terminal metrics."""
    return asyncio.run(
        _run_experiment_async(
            config_path=config_path,
            null_model=null_model,
            run_id=None,
        )
    )


def run_experiment_with_id(
    config_path: str = "config.yaml",
    null_model: bool = False,
    run_id: str | None = None,
) -> RunResult:
    """Run one experiment with optional explicit run_id."""
    return asyncio.run(
        _run_experiment_async(
            config_path=config_path,
            null_model=null_model,
            run_id=run_id,
        )
    )
