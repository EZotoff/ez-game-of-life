from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Settings = import_module("petri_dish.config").Settings
get_all_tools = import_module("petri_dish.tools").get_all_tools
web_search_mod = import_module("petri_dish.tools.web_search")
web_search = web_search_mod.web_search
reset_web_search_state = web_search_mod.reset_web_search_state


def _default_settings(**overrides):
    data = {
        "web_search_provider": "duckduckgo_instant_answer",
        "web_search_base_url": "https://api.duckduckgo.com/",
        "web_search_user_agent": "PetriDish-Test/1.0",
        "web_search_timeout_seconds": 5,
        "web_search_max_queries_per_call": 3,
        "web_search_max_results_per_query": 5,
        "web_search_chars_per_result": 500,
        "web_search_allow_redirects": False,
        "web_search_blocked_domains": ["localhost", "internal.local"],
        "web_search_daily_budget": 10,
        "web_search_calls_per_turn": 2,
        "tavily_api_key": "",
        "tool_costs": {"web_search": 0.15},
    }
    data.update(overrides)
    return Settings(**data)


def test_web_search_returns_raw_results_without_evaluation(monkeypatch) -> None:
    reset_web_search_state()
    settings = _default_settings()

    monkeypatch.setattr(
        web_search_mod,
        "_execute_duckduckgo_query",
        lambda query, cfg: [
            {
                "title": "Result A",
                "url": "https://example.com/a",
                "snippet": "alpha",
            }
        ],
    )

    payload = json.loads(web_search("json schema validation", settings=settings))
    assert payload["provider"] == "duckduckgo_instant_answer"
    assert payload["queries_executed"] == ["json schema validation"]
    assert payload["results"] == [
        {
            "title": "Result A",
            "url": "https://example.com/a",
            "snippet": "alpha",
        }
    ]
    assert "verdict" not in payload
    assert "confidence" not in payload


def test_web_search_daily_budget_enforced(monkeypatch) -> None:
    reset_web_search_state()
    settings = _default_settings(web_search_daily_budget=1, web_search_calls_per_turn=5)

    monkeypatch.setattr(
        web_search_mod,
        "_execute_duckduckgo_query",
        lambda query, cfg: [{"title": "ok", "url": "https://x", "snippet": "y"}],
    )

    first = json.loads(web_search("first", settings=settings))
    second = json.loads(web_search("second", settings=settings))

    assert len(first["results"]) == 1
    assert second["results"] == []
    assert second["queries_executed"] == []


def test_web_search_per_turn_limit_enforced(monkeypatch) -> None:
    reset_web_search_state()
    settings = _default_settings(
        web_search_daily_budget=10, web_search_calls_per_turn=1
    )

    monkeypatch.setattr(
        web_search_mod,
        "_execute_duckduckgo_query",
        lambda query, cfg: [{"title": "ok", "url": "https://x", "snippet": "y"}],
    )

    first = json.loads(web_search("first", settings=settings))
    second = json.loads(web_search("second", settings=settings))

    assert len(first["results"]) == 1
    assert second["results"] == []
    assert second["queries_executed"] == []


def test_provider_fallback_to_ddg_when_primary_fails(monkeypatch) -> None:
    reset_web_search_state()
    settings = _default_settings(web_search_provider="stackexchange_advanced")

    monkeypatch.setattr(web_search_mod, "_execute_stackexchange_query", lambda q, c: [])
    monkeypatch.setattr(
        web_search_mod,
        "_execute_duckduckgo_query",
        lambda q, c: [{"title": "fallback", "url": "https://ddg", "snippet": "ok"}],
    )

    payload = json.loads(web_search("fallback please", settings=settings))
    assert payload["provider"] == "duckduckgo_instant_answer"
    assert payload["results"][0]["title"] == "fallback"


def test_web_search_uses_web_search_settings(monkeypatch) -> None:
    reset_web_search_state()
    settings = _default_settings(web_search_base_url="https://example.com/custom")
    observed = {"base": ""}

    def _capture(query, cfg):
        observed["base"] = cfg.web_search_base_url
        return []

    monkeypatch.setattr(web_search_mod, "_execute_duckduckgo_query", _capture)
    json.loads(web_search("observe config", settings=settings))

    assert observed["base"] == "https://example.com/custom"


def test_registry_has_web_search_host_side_with_expected_cost() -> None:
    settings = _default_settings()
    registry = get_all_tools(settings=settings)
    tool = registry.get_tool("web_search")
    assert tool is not None
    assert tool.host_side is True
    assert tool.free_when_stripped is False
    assert registry.get_tool_cost("web_search") == 0.15
