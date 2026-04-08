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


def _build_stackexchange_url(base_url: str, query: str) -> str:
    params = {
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": "stackoverflow",
        "filter": "withbody",
        "pagesize": str(3),
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def _build_tavily_request(query: str, settings: Settings) -> dict[str, object]:
    return {
        "query": query,
        "max_results": min(settings.web_search_max_results_per_query, 5),
        "search_depth": "basic",
        "include_answer": False,
    }


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
            _ = (headers, req, fp, code, msg, newurl)
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    return cast(_HTTPResponseProtocol, opener.open(req, timeout=timeout_seconds))


def _execute_duckduckgo_query(query: str, settings: Settings) -> list[dict[str, str]]:
    url = _build_ddg_url(settings.web_search_base_url, query)
    try:
        with _open_url(
            url=url,
            user_agent=settings.web_search_user_agent,
            timeout_seconds=settings.web_search_timeout_seconds,
            allow_redirects=settings.web_search_allow_redirects,
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

    results: list[dict[str, str]] = []
    abstract = payload.get("AbstractText")
    abstract_source = payload.get("AbstractSource")
    abstract_url = payload.get("AbstractURL")
    if isinstance(abstract, str) and abstract.strip():
        results.append(
            {
                "title": str(abstract_source)
                if isinstance(abstract_source, str)
                else "Abstract",
                "url": str(abstract_url) if isinstance(abstract_url, str) else "",
                "snippet": abstract.strip()[: settings.web_search_chars_per_result],
            }
        )

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
        for item in flat_topics[: settings.web_search_max_results_per_query]:
            text = item.get("text", "")
            if text:
                results.append(
                    {
                        "title": "DuckDuckGo Related",
                        "url": item.get("url", ""),
                        "snippet": text[: settings.web_search_chars_per_result],
                    }
                )

    return results[: settings.web_search_max_results_per_query]


def _execute_stackexchange_query(
    query: str, settings: Settings
) -> list[dict[str, str]]:
    url = _build_stackexchange_url(settings.web_search_base_url, query)
    try:
        with _open_url(
            url=url,
            user_agent=settings.web_search_user_agent,
            timeout_seconds=settings.web_search_timeout_seconds,
            allow_redirects=settings.web_search_allow_redirects,
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
    items_obj = payload_raw.get("items")
    if not isinstance(items_obj, list):
        return []

    results: list[dict[str, str]] = []
    for item_obj in items_obj[: settings.web_search_max_results_per_query]:
        if not isinstance(item_obj, dict):
            continue
        item = {str(k): v for k, v in cast(dict[object, object], item_obj).items()}
        title = item.get("title")
        link = item.get("link")
        body = item.get("body_markdown") or item.get("body")
        snippet = ""
        if isinstance(body, str) and body.strip():
            cleaned = re.sub(r"<[^>]+>", " ", body)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            snippet = cleaned[: settings.web_search_chars_per_result]
        results.append(
            {
                "title": title.strip()
                if isinstance(title, str) and title.strip()
                else "StackOverflow Result",
                "url": link.strip() if isinstance(link, str) and link.strip() else "",
                "snippet": snippet,
            }
        )
    return results


def _execute_tavily_query(query: str, settings: Settings) -> list[dict[str, str]]:
    if not settings.tavily_api_key:
        return []

    payload = _build_tavily_request(query, settings)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        settings.web_search_base_url,
        data=body,
        method="POST",
        headers={
            "User-Agent": settings.web_search_user_agent,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.tavily_api_key}",
        },
    )

    try:
        with cast(
            _HTTPResponseProtocol,
            urllib.request.urlopen(req, timeout=settings.web_search_timeout_seconds),
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
    results_obj = payload_raw.get("results")
    if not isinstance(results_obj, list):
        return []

    results: list[dict[str, str]] = []
    for item_obj in results_obj[: settings.web_search_max_results_per_query]:
        if not isinstance(item_obj, dict):
            continue
        item = {str(k): v for k, v in cast(dict[object, object], item_obj).items()}
        title = item.get("title")
        raw_url = item.get("url")
        content = item.get("content")
        cleaned = ""
        if isinstance(content, str) and content.strip():
            cleaned = re.sub(r"<[^>]+>", " ", content)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
        results.append(
            {
                "title": title.strip()
                if isinstance(title, str) and title.strip()
                else "Tavily Result",
                "url": raw_url.strip()
                if isinstance(raw_url, str) and raw_url.strip()
                else "",
                "snippet": cleaned[: settings.web_search_chars_per_result],
            }
        )
    return results


def web_search(
    query: str,
    max_results: int = 3,
    settings: Settings | None = None,
) -> str:
    global _daily_call_count

    cfg = settings or Settings.from_yaml()
    blocked_domains = cfg.web_search_blocked_domains
    provider_name = cfg.web_search_provider

    provider_exec = {
        "duckduckgo_instant_answer": _execute_duckduckgo_query,
        "stackexchange_advanced": _execute_stackexchange_query,
        "tavily": _execute_tavily_query,
    }.get(provider_name)

    if provider_exec is None:
        return json.dumps(
            {
                "results": [],
                "provider": provider_name,
                "queries_executed": [],
            }
        )

    if not _is_safe_search_base_url(cfg.web_search_base_url, blocked_domains):
        return json.dumps(
            {
                "results": [],
                "provider": provider_name,
                "queries_executed": [],
            }
        )

    if _daily_call_count >= cfg.web_search_daily_budget:
        return json.dumps(
            {
                "results": [],
                "provider": provider_name,
                "queries_executed": [],
            }
        )

    if _turn_calls.get("global", 0) >= cfg.web_search_calls_per_turn:
        return json.dumps(
            {
                "results": [],
                "provider": provider_name,
                "queries_executed": [],
            }
        )

    clean_query = query.strip()
    if not clean_query or _query_contains_blocked_target(clean_query, blocked_domains):
        return json.dumps(
            {
                "results": [],
                "provider": provider_name,
                "queries_executed": [],
            }
        )

    _daily_call_count += 1
    _turn_calls["global"] = _turn_calls.get("global", 0) + 1

    provider_used = provider_name
    results = provider_exec(clean_query, cfg)
    if not results and provider_name != "duckduckgo_instant_answer":
        fallback = _execute_duckduckgo_query(clean_query, cfg)
        if fallback:
            results = fallback
            provider_used = "duckduckgo_instant_answer"
    bounded = results[
        : max(1, min(int(max_results), cfg.web_search_max_results_per_query))
    ]
    return json.dumps(
        {
            "results": bounded,
            "provider": provider_used,
            "queries_executed": [clean_query],
        }
    )


def reset_web_search_state() -> None:
    global _daily_call_count
    _daily_call_count = 0
    _turn_calls.clear()
