from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from petri_dish.config import Settings
from petri_dish.null_model import JsonValue
from petri_dish.null_model import NullModel
from petri_dish.orchestrator import AgentOrchestrator
from petri_dish.tool_parser import ToolCallParser
from petri_dish.tools import get_all_tools


def _tool_schema(name: str, required: list[str] | None = None) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"tool {name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "claimed_pattern": {"type": "string"},
                    "output_summary": {"type": "string"},
                    "file_family": {"type": "string"},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": required or [],
            },
        },
    }


def test_null_model_overseer_smoke_forces_scout_first() -> None:
    model = NullModel(seed=7, null_model_type="overseer_smoke")
    tools = [
        _tool_schema(
            "overseer_scout",
            required=["claimed_pattern", "output_summary", "file_family"],
        ),
        _tool_schema("self_modify", required=["key", "value"]),
        _tool_schema("file_read", required=[]),
    ]

    typed_tools = cast(list[dict[str, JsonValue]], tools)
    _, calls = __import__("asyncio").run(model.chat("", [], typed_tools))

    assert len(calls) == 1
    function = calls[0]["function"]
    assert function["name"] == "overseer_scout"
    arguments = function["arguments"]
    assert arguments["claimed_pattern"]
    assert arguments["output_summary"]
    assert arguments["file_family"]


def test_null_model_overseer_smoke_excludes_self_modify_after_first_turn() -> None:
    model = NullModel(seed=11, null_model_type="overseer_smoke")
    tools = [
        _tool_schema(
            "overseer_scout",
            required=["claimed_pattern", "output_summary", "file_family"],
        ),
        _tool_schema("self_modify", required=["key", "value"]),
        _tool_schema("file_read", required=[]),
    ]

    seen: list[str] = []
    for _ in range(12):
        typed_tools = cast(list[dict[str, JsonValue]], tools)
        _, calls = __import__("asyncio").run(model.chat("", [], typed_tools))
        seen.append(str(calls[0]["function"]["name"]))

    assert seen[0] == "overseer_scout"
    assert "self_modify" not in seen


def test_forced_smoke_config_loads_expected_mode() -> None:
    settings = Settings.from_yaml("config_overseer_forced_smoke.yaml")

    assert settings.null_model_type == "overseer_smoke"
    assert settings.max_turns == 5
    assert settings.overseer_scout_daily_budget == 3
    assert settings.overseer_search_max_queries_per_call == 1
    assert settings.force_first_overseer_scout is True


def test_forced_smoke_config_file_exists_and_is_json_safe() -> None:
    path = Path("config_overseer_forced_smoke.yaml")
    assert path.exists()

    settings = Settings.from_yaml(str(path))
    payload = {
        "null_model_type": settings.null_model_type,
        "budget": settings.overseer_scout_daily_budget,
        "max_turns": settings.max_turns,
    }
    assert json.loads(json.dumps(payload)) == {
        "null_model_type": "overseer_smoke",
        "budget": 3,
        "max_turns": 5,
    }


def test_agent_orchestrator_builds_forced_first_scout_call() -> None:
    settings = Settings.from_yaml("config_overseer_forced_smoke.yaml")
    orchestrator = AgentOrchestrator(
        settings=settings,
        tool_parser=ToolCallParser(),
        tool_registry=get_all_tools(settings=settings),
    )
    orchestrator._turn = 1

    call = orchestrator._build_forced_first_scout_call()

    assert call is not None
    assert call.name == "overseer_scout"
    assert call.arguments["claimed_pattern"] == "forced_first_turn_probe"
    assert call.arguments["requesting_agent_id"] == "forced-smoke-single-agent"
    assert call.arguments["turn_id"] == "forced-turn-1"


def test_agent_orchestrator_does_not_force_after_first_turn() -> None:
    settings = Settings.from_yaml("config_overseer_forced_smoke.yaml")
    orchestrator = AgentOrchestrator(
        settings=settings,
        tool_parser=ToolCallParser(),
        tool_registry=get_all_tools(settings=settings),
    )
    orchestrator._turn = 2

    assert orchestrator._build_forced_first_scout_call() is None
