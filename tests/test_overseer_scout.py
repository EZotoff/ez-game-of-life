from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Settings = import_module("petri_dish.config").Settings
ScoutReport = import_module("petri_dish.scout_report").ScoutReport
build_scout_report = import_module("petri_dish.scout_report").build_scout_report
overseer_scout_module = import_module("petri_dish.tools.overseer_scout")
overseer_scout = overseer_scout_module.overseer_scout
reset_overseer_scout_state = overseer_scout_module.reset_overseer_scout_state
_build_ddg_url = overseer_scout_module._build_ddg_url
_query_contains_blocked_target = overseer_scout_module._query_contains_blocked_target


@pytest.fixture(autouse=True)
def reset_scout_limits() -> None:
    reset_overseer_scout_state()


def _default_settings(**overrides: Any) -> Any:
    base = {
        "overseer_search_provider": "duckduckgo_instant_answer",
        "overseer_search_base_url": "https://api.duckduckgo.com/",
        "overseer_search_timeout_seconds": 10,
        "overseer_search_max_queries_per_call": 3,
        "overseer_search_max_related_topics": 5,
        "overseer_search_chars_per_result": 2000,
        "overseer_scout_calls_per_turn": 1,
        "overseer_scout_daily_budget": 50,
        "overseer_search_allow_redirects": False,
        "overseer_search_blocked_domains": [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "local",
        ],
    }
    base.update(overrides)
    return Settings(**base)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = (exc_type, exc, tb)


def test_scout_report_json_roundtrip() -> None:
    report = build_scout_report(
        requesting_agent_id="agent-7",
        claimed_pattern="sorted by timestamp",
        output_summary="generated JSON logs",
        file_family="json",
        queries_executed=["json timestamp ordering"],
        result_snippets=["JSON logs are typically sorted by timestamp"],
        confidence=0.9,
        verdict="supported",
        reasoning="Matched multiple snippets",
        suggested_bonus=0.15,
    )
    restored = ScoutReport.from_json(report.to_json())
    assert restored.claimed_pattern == "sorted by timestamp"
    assert restored.file_family == "json"
    assert restored.verdict == "supported"
    assert restored.results_found == 1


def test_scout_report_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ScoutReport.from_dict(
            {
                "report_id": "r1",
                "requesting_agent_id": "a1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "claimed_pattern": "x",
                "output_summary": "y",
                "file_family": "xml",
                "queries_executed": [],
                "results_found": 0,
                "result_snippets": [],
                "confidence": 0.0,
                "verdict": "supported",
                "reasoning": "",
                "suggested_bonus": 0.0,
            }
        )


def test_build_ddg_url_uses_v1_contract_params() -> None:
    url = _build_ddg_url("https://api.duckduckgo.com/", "json schema")
    assert "https://api.duckduckgo.com/?" in url
    assert "q=json+schema" in url
    assert "format=json" in url
    assert "no_html=1" in url
    assert "skip_disambig=1" in url
    assert "no_redirect=1" in url


def test_query_blocking_private_targets() -> None:
    blocked = ["localhost", "127.0.0.1", "0.0.0.0", "local"]
    assert _query_contains_blocked_target("debug localhost status", blocked)
    assert _query_contains_blocked_target("check 10.1.2.3 endpoint", blocked)
    assert _query_contains_blocked_target("query app.internal.local", blocked)
    assert not _query_contains_blocked_target("python csv dialect", blocked)


def test_overseer_scout_limits_queries_and_related_topics(monkeypatch: Any) -> None:
    payload = {
        "AbstractText": "alpha pattern mention",
        "AbstractSource": "DDG",
        "AbstractURL": "https://example.com/a",
        "RelatedTopics": [
            {"Text": "alpha topic 1", "FirstURL": "https://example.com/1"},
            {"Text": "alpha topic 2", "FirstURL": "https://example.com/2"},
            {"Text": "alpha topic 3", "FirstURL": "https://example.com/3"},
        ],
    }

    calls: list[str] = []

    def fake_open_url(
        url: str, user_agent: str, timeout_seconds: int, allow_redirects: bool
    ) -> _FakeResponse:
        _ = (user_agent, timeout_seconds, allow_redirects)
        calls.append(url)
        return _FakeResponse(payload)

    monkeypatch.setattr(overseer_scout_module, "_open_url", fake_open_url)

    settings = _default_settings(
        overseer_search_max_queries_per_call=2,
        overseer_search_max_related_topics=2,
    )
    raw = overseer_scout(
        claimed_pattern="alpha",
        output_summary="summary",
        file_family="json",
        search_queries=["q1", "q2", "q3"],
        requesting_agent_id="agent-a",
        settings=settings,
    )
    report = ScoutReport.from_json(raw)

    assert len(calls) == 2
    assert report.queries_executed == ["q1", "q2"]
    assert report.results_found == 2
    assert len(report.result_snippets) == 2


def test_overseer_scout_daily_budget_enforced(monkeypatch: Any) -> None:
    def fake_open_url(
        url: str, user_agent: str, timeout_seconds: int, allow_redirects: bool
    ) -> _FakeResponse:
        _ = (url, user_agent, timeout_seconds, allow_redirects)
        return _FakeResponse({"AbstractText": "", "RelatedTopics": []})

    monkeypatch.setattr(overseer_scout_module, "_open_url", fake_open_url)

    settings = _default_settings(overseer_scout_daily_budget=1)
    first = ScoutReport.from_json(
        overseer_scout(
            claimed_pattern="p",
            output_summary="o",
            file_family="csv",
            settings=settings,
        )
    )
    second = ScoutReport.from_json(
        overseer_scout(
            claimed_pattern="p",
            output_summary="o",
            file_family="csv",
            settings=settings,
        )
    )

    assert first.reasoning != "Daily scout budget exhausted"
    assert second.reasoning == "Daily scout budget exhausted"
    assert second.verdict == "no_sources"


def test_overseer_scout_per_turn_limit_enforced(monkeypatch: Any) -> None:
    def fake_open_url(
        url: str, user_agent: str, timeout_seconds: int, allow_redirects: bool
    ) -> _FakeResponse:
        _ = (url, user_agent, timeout_seconds, allow_redirects)
        return _FakeResponse({"AbstractText": "", "RelatedTopics": []})

    monkeypatch.setattr(overseer_scout_module, "_open_url", fake_open_url)

    settings = _default_settings(overseer_scout_calls_per_turn=1)
    _ = overseer_scout(
        claimed_pattern="p",
        output_summary="o",
        file_family="log",
        turn_id="t-1",
        settings=settings,
    )
    blocked = ScoutReport.from_json(
        overseer_scout(
            claimed_pattern="p",
            output_summary="o",
            file_family="log",
            turn_id="t-1",
            settings=settings,
        )
    )
    assert blocked.reasoning == "Per-turn scout call limit reached"


def test_overseer_scout_blocks_unsafe_base_url() -> None:
    settings = _default_settings(overseer_search_base_url="http://localhost:8080")
    report = ScoutReport.from_json(
        overseer_scout(
            claimed_pattern="p",
            output_summary="o",
            file_family="json",
            settings=settings,
        )
    )
    assert report.reasoning == "Unsafe overseer_search_base_url configuration"
    assert report.verdict == "no_sources"


def test_overseer_scout_registered_host_side_with_cost() -> None:
    get_all_tools = import_module("petri_dish.tools").get_all_tools
    settings = _default_settings(
        tool_costs={
            "file_read": 0.01,
            "file_write": 0.01,
            "file_list": 0.01,
            "shell_exec": 0.05,
            "check_balance": 0.0,
            "http_request": 0.1,
            "overseer_scout": 0.15,
            "self_modify": 0.02,
            "get_env_info": 0.0,
        }
    )
    registry = get_all_tools(settings=settings)
    tool = registry.get_tool("overseer_scout")
    assert tool is not None
    assert tool.host_side is True
    assert tool.free_when_stripped is False
    assert registry.get_tool_cost("overseer_scout") == 0.15
