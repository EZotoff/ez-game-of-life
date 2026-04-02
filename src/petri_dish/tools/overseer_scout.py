from __future__ import annotations

import ipaddress
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from types import TracebackType
from typing import Protocol, cast
from typing_extensions import override

from petri_dish.config import Settings
from petri_dish.scout_report import FileFamily, ScoutVerdict, build_scout_report

_daily_call_count = 0
_turn_calls: dict[str, int] = {}
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HOST_RE = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,63}\b")


class _HTTPResponseProtocol(Protocol):
    status: int

    def read(self, amt: int = -1) -> bytes: ...

    def __enter__(self) -> "_HTTPResponseProtocol": ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...


def _is_private_ip(candidate: str) -> bool:
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _is_blocked_hostname(hostname: str, blocked_domains: list[str]) -> bool:
    host = hostname.lower().strip().rstrip(".")
    if host == "localhost" or host.endswith(".local"):
        return True

    normalized = [x.lower().strip().lstrip("*.") for x in blocked_domains if x.strip()]
    for blocked in normalized:
        if not blocked:
            continue
        if host == blocked or host.endswith(f".{blocked}"):
            return True
    return False


def _query_contains_blocked_target(query: str, blocked_domains: list[str]) -> bool:
    lowered = query.lower()
    if "localhost" in lowered:
        return True

    for match_obj in _IPV4_RE.finditer(query):
        match = match_obj.group(0)
        if _is_private_ip(match):
            return True

    for host_match in _HOST_RE.finditer(query):
        host = host_match.group(0)
        if _is_blocked_hostname(host, blocked_domains):
            return True

    return False


def _flatten_related_topics(
    topics: Iterable[dict[str, object]],
) -> list[dict[str, str]]:
    flat: list[dict[str, str]] = []
    for item in topics:
        text = item.get("Text")
        url = item.get("FirstURL")
        if isinstance(text, str):
            flat.append({"text": text, "url": str(url) if isinstance(url, str) else ""})
        nested = item.get("Topics")
        if isinstance(nested, list):
            nested_dicts: list[dict[str, object]] = []
            nested_items = cast(list[object], nested)
            for nested_item in nested_items:
                if isinstance(nested_item, dict):
                    nested_dicts.append(
                        {
                            str(k): v
                            for k, v in cast(dict[object, object], nested_item).items()
                        }
                    )
            flat.extend(_flatten_related_topics(nested_dicts))
    return flat


def _build_ddg_url(base_url: str, query: str) -> str:
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
        "no_redirect": "1",
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def _is_safe_search_base_url(base_url: str, blocked_domains: list[str]) -> bool:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme.lower() != "https":
        return False

    host = parsed.hostname or ""
    if not host:
        return False

    if _is_blocked_hostname(host, blocked_domains):
        return False

    if _is_private_ip(host):
        return False

    return True


def _open_url(
    url: str, user_agent: str, timeout_seconds: int, allow_redirects: bool
) -> _HTTPResponseProtocol:
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": user_agent})
    if allow_redirects:
        response = cast(
            object,
            urllib.request.urlopen(req, timeout=timeout_seconds),
        )
        return cast(_HTTPResponseProtocol, response)

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        @override
        def redirect_request(
            self,
            req: object,
            fp: object,
            code: int,
            msg: str,
            headers: object,
            newurl: str,
        ) -> None:
            _ = (headers,)
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    return cast(_HTTPResponseProtocol, opener.open(req, timeout=timeout_seconds))


def _execute_duckduckgo_query(query: str, settings: Settings) -> list[str]:
    url = _build_ddg_url(settings.overseer_search_base_url, query)
    try:
        with _open_url(
            url=url,
            user_agent=settings.overseer_search_user_agent,
            timeout_seconds=settings.overseer_search_timeout_seconds,
            allow_redirects=settings.overseer_search_allow_redirects,
        ) as response:
            if int(getattr(response, "status", 200)) != 200:
                return []
            raw = response.read()
    except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
        return []

    try:
        raw_text = raw.decode("utf-8", errors="replace")
        payload_obj = cast(object, json.loads(raw_text))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload_obj, dict):
        return []
    payload_raw = cast(dict[object, object], payload_obj)
    payload: dict[str, object] = {}
    for key, value in payload_raw.items():
        payload[str(key)] = value

    snippets: list[str] = []
    abstract = payload.get("AbstractText")
    abstract_source = payload.get("AbstractSource")
    abstract_url = payload.get("AbstractURL")
    if isinstance(abstract, str) and abstract.strip():
        source = (
            f" ({abstract_source})"
            if isinstance(abstract_source, str) and abstract_source
            else ""
        )
        url_part = (
            f" [{abstract_url}]"
            if isinstance(abstract_url, str) and abstract_url
            else ""
        )
        snippets.append(f"{abstract}{source}{url_part}".strip())

    topics = payload.get("RelatedTopics")
    if isinstance(topics, list):
        topic_dicts: list[dict[str, object]] = []
        topics_list = cast(list[object], topics)
        for topic in topics_list:
            if isinstance(topic, dict):
                topic_dicts.append(
                    {str(k): v for k, v in cast(dict[object, object], topic).items()}
                )
        flat_topics = _flatten_related_topics(topic_dicts)
        for item in flat_topics[: settings.overseer_search_max_related_topics]:
            text = item.get("text", "")
            if text:
                url_part = f" [{item['url']}]" if item.get("url") else ""
                snippets.append(f"{text}{url_part}".strip())

    snippets = snippets[: settings.overseer_search_max_related_topics]
    return [x[: settings.overseer_search_chars_per_result] for x in snippets]


def _evaluate_support(
    claimed_pattern: str, snippets: list[str]
) -> tuple[float, ScoutVerdict, str, float]:
    if not snippets:
        return 0.0, "no_sources", "No sources returned from provider", 0.0

    pattern = claimed_pattern.lower().strip()
    hits = sum(1 for s in snippets if pattern and pattern in s.lower())

    if hits >= 2:
        return 0.9, "supported", "Pattern appears in multiple external sources", 0.15
    if hits == 1:
        return 0.7, "supported", "Pattern appears in one external source", 0.05
    return 0.4, "unclear", "Sources found but pattern is not explicit", 0.0


def overseer_scout(
    claimed_pattern: str,
    output_summary: str,
    file_family: FileFamily,
    search_queries: list[str] | None = None,
    requesting_agent_id: str = "unknown",
    turn_id: str | None = None,
    settings: Settings | None = None,
) -> str:
    global _daily_call_count

    cfg = settings or Settings.from_yaml()

    if cfg.overseer_search_provider != "duckduckgo_instant_answer":
        report = build_scout_report(
            requesting_agent_id=requesting_agent_id,
            claimed_pattern=claimed_pattern,
            output_summary=output_summary,
            file_family=file_family,
            queries_executed=[],
            result_snippets=[],
            confidence=0.0,
            verdict="no_sources",
            reasoning="Unsupported overseer_search_provider for v1",
            suggested_bonus=0.0,
        )
        return report.to_json()

    if not _is_safe_search_base_url(
        cfg.overseer_search_base_url, cfg.overseer_search_blocked_domains
    ):
        report = build_scout_report(
            requesting_agent_id=requesting_agent_id,
            claimed_pattern=claimed_pattern,
            output_summary=output_summary,
            file_family=file_family,
            queries_executed=[],
            result_snippets=[],
            confidence=0.0,
            verdict="no_sources",
            reasoning="Unsafe overseer_search_base_url configuration",
            suggested_bonus=0.0,
        )
        return report.to_json()

    if _daily_call_count >= cfg.overseer_scout_daily_budget:
        report = build_scout_report(
            requesting_agent_id=requesting_agent_id,
            claimed_pattern=claimed_pattern,
            output_summary=output_summary,
            file_family=file_family,
            queries_executed=[],
            result_snippets=[],
            confidence=0.0,
            verdict="no_sources",
            reasoning="Daily scout budget exhausted",
            suggested_bonus=0.0,
        )
        return report.to_json()

    if turn_id is not None:
        current_turn_calls = _turn_calls.get(turn_id, 0)
        if current_turn_calls >= cfg.overseer_scout_calls_per_turn:
            report = build_scout_report(
                requesting_agent_id=requesting_agent_id,
                claimed_pattern=claimed_pattern,
                output_summary=output_summary,
                file_family=file_family,
                queries_executed=[],
                result_snippets=[],
                confidence=0.0,
                verdict="no_sources",
                reasoning="Per-turn scout call limit reached",
                suggested_bonus=0.0,
            )
            return report.to_json()
        _turn_calls[turn_id] = current_turn_calls + 1

    _daily_call_count += 1

    raw_queries = search_queries or [f"{file_family} {claimed_pattern}"]
    queries = [q.strip() for q in raw_queries if q.strip()]
    queries = queries[: cfg.overseer_search_max_queries_per_call]

    safe_queries: list[str] = []
    for query in queries:
        if not _query_contains_blocked_target(
            query, cfg.overseer_search_blocked_domains
        ):
            safe_queries.append(query)

    snippets: list[str] = []
    for query in safe_queries:
        snippets.extend(_execute_duckduckgo_query(query, cfg))
    snippets = snippets[: cfg.overseer_search_max_related_topics]

    confidence, verdict, reasoning, bonus = _evaluate_support(claimed_pattern, snippets)
    report = build_scout_report(
        requesting_agent_id=requesting_agent_id,
        claimed_pattern=claimed_pattern,
        output_summary=output_summary,
        file_family=file_family,
        queries_executed=safe_queries,
        result_snippets=snippets,
        confidence=confidence,
        verdict=verdict,
        reasoning=reasoning,
        suggested_bonus=bonus,
    )
    return report.to_json()


def reset_overseer_scout_state() -> None:
    global _daily_call_count
    _daily_call_count = 0
    _turn_calls.clear()
