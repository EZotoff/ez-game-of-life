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
from petri_dish.economy import CreditEconomy
from petri_dish.llm_client import OllamaClient
from petri_dish.logging_db import LoggingDB
from petri_dish.null_model import NullModel
from petri_dish.orchestrator import AgentOrchestrator, RunResult
from petri_dish.sandbox import SandboxManager
from petri_dish.tool_parser import ToolCallParser
from petri_dish.tools import get_all_tools


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


async def _run_experiment_async(
    *,
    config_path: str,
    null_model: bool,
    run_id: str | None,
) -> RunResult:
    settings = Settings.from_yaml(config_path)
    resolved_run_id = _resolve_run_id(run_id)

    credit_economy = CreditEconomy(settings=settings)
    logging_db = LoggingDB(db_path=_resolve_db_path(resolved_run_id))
    tool_parser = ToolCallParser()
    tool_registry = get_all_tools(settings=settings)
    sandbox_manager = SandboxManager()

    base_chat_client: _ChatClient
    if null_model:
        base_chat_client = NullModel()
    else:
        base_chat_client = OllamaClient(settings=settings)

    context_manager = ContextManager(settings=settings)
    degradation_manager = DegradationManager(settings=settings)
    resource_ecology = ResourceEcology(settings=settings)

    orchestrator = AgentOrchestrator(
        settings=settings,
        tool_parser=tool_parser,
        tool_registry=tool_registry,
        credit_economy=credit_economy,
        sandbox_manager=sandbox_manager,
        logging_db=logging_db,
        snapshot_interval_turns=settings.context_summary_interval_turns,
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
        credits_before=result.final_balance,
        credits_after=result.final_balance,
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
