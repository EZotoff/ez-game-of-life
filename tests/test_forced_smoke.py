from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from petri_dish.config import Settings
from petri_dish.null_model import JsonValue
from petri_dish.null_model import NullModel
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
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": required or [],
            },
        },
    }


def test_null_model_random_mode_generates_tool_calls() -> None:
    model = NullModel(seed=7, null_model_type="random")
    tools = [
        _tool_schema("web_search", required=["query"]),
        _tool_schema("self_modify", required=["key", "value"]),
        _tool_schema("file_read", required=[]),
    ]

    typed_tools = cast(list[dict[str, JsonValue]], tools)
    _, calls = __import__("asyncio").run(model.chat("", [], typed_tools))
    assert len(calls) == 1
    assert calls[0]["function"]["name"] in {"web_search", "self_modify", "file_read"}


def test_null_model_random_mode_can_choose_multiple_tools() -> None:
    model = NullModel(seed=11, null_model_type="random")
    tools = [
        _tool_schema("web_search", required=["query"]),
        _tool_schema("self_modify", required=["key", "value"]),
        _tool_schema("file_read", required=[]),
    ]

    seen: set[str] = set()
    for _ in range(16):
        typed_tools = cast(list[dict[str, JsonValue]], tools)
        _, calls = __import__("asyncio").run(model.chat("", [], typed_tools))
        seen.add(str(calls[0]["function"]["name"]))

    assert len(seen) >= 2


def test_forced_smoke_config_loads_expected_mode() -> None:
    settings = Settings.from_yaml("config_overseer_forced_smoke.yaml")

    assert settings.null_model_type == "random"
    assert settings.max_turns == 8
    assert settings.web_search_daily_budget == 5
    assert settings.web_search_max_queries_per_call == 1
    assert settings.web_search_provider == "stackexchange_advanced"
    assert settings.overseer_enabled is False


def test_forced_smoke_config_file_exists_and_is_json_safe() -> None:
    path = Path("config_overseer_forced_smoke.yaml")
    assert path.exists()

    settings = Settings.from_yaml(str(path))
    payload = {
        "null_model_type": settings.null_model_type,
        "budget": settings.web_search_daily_budget,
        "max_turns": settings.max_turns,
    }
    assert json.loads(json.dumps(payload)) == {
        "null_model_type": "random",
        "budget": 5,
        "max_turns": 8,
    }


def test_forced_smoke_tavily_config_loads_expected_mode() -> None:
    settings = Settings.from_yaml("config_overseer_forced_smoke_tavily.yaml")

    assert settings.null_model_type == "random"
    assert settings.max_turns == 8
    assert settings.web_search_daily_budget == 5
    assert settings.web_search_provider == "tavily"
    assert settings.web_search_base_url == "https://api.tavily.com/search"
    assert settings.tavily_api_key == ""


def test_tool_registry_contains_web_search_for_forced_smoke_config() -> None:
    settings = Settings.from_yaml("config_overseer_forced_smoke.yaml")
    registry = get_all_tools(settings=settings)
    assert registry.get_tool("web_search") is not None
