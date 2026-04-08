from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Settings = import_module("petri_dish.config").Settings
web_search_mod = import_module("petri_dish.tools.web_search")
web_search = web_search_mod.web_search
reset_web_search_state = web_search_mod.reset_web_search_state


def _settings(**overrides):
    data = {
        "web_search_provider": "duckduckgo_instant_answer",
        "web_search_base_url": "https://api.duckduckgo.com/",
        "web_search_user_agent": "PetriDish-Test/1.0",
        "web_search_timeout_seconds": 5,
        "web_search_max_queries_per_call": 3,
        "web_search_max_results_per_query": 5,
        "web_search_chars_per_result": 500,
        "web_search_allow_redirects": False,
        "web_search_blocked_domains": ["localhost", "127.0.0.1", "internal.local"],
        "web_search_daily_budget": 10,
        "web_search_calls_per_turn": 5,
    }
    data.update(overrides)
    return Settings(**data)


def test_blocked_domain_query_is_rejected() -> None:
    reset_web_search_state()
    payload = json.loads(
        web_search(
            "inspect https://localhost/admin",
            settings=_settings(),
        )
    )
    assert payload["results"] == []
    assert payload["queries_executed"] == []


def test_private_ip_query_is_rejected() -> None:
    reset_web_search_state()
    payload = json.loads(
        web_search(
            "check 192.168.1.10 service",
            settings=_settings(),
        )
    )
    assert payload["results"] == []
    assert payload["queries_executed"] == []


def test_unsafe_base_url_is_rejected(monkeypatch) -> None:
    reset_web_search_state()
    called = {"count": 0}

    def _provider(query, cfg):
        called["count"] += 1
        return [{"title": "x", "url": "https://example.com", "snippet": "y"}]

    monkeypatch.setattr(web_search_mod, "_execute_duckduckgo_query", _provider)

    payload = json.loads(
        web_search(
            "safe question",
            settings=_settings(web_search_base_url="http://localhost:8080/search"),
        )
    )
    assert called["count"] == 0
    assert payload["results"] == []


def test_blocked_hostname_helper_handles_subdomains() -> None:
    assert web_search_mod._is_blocked_hostname("api.internal.local", ["internal.local"])
    assert web_search_mod._is_blocked_hostname("localhost", ["example.com"])
    assert not web_search_mod._is_blocked_hostname(
        "docs.python.org", ["internal.local"]
    )
