"""Async Ollama LLM client with retry and tool-call parsing."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Protocol, cast

import httpx

from petri_dish.config import Settings

logger = logging.getLogger(__name__)

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


def _is_json_object(value: object) -> bool:
    return isinstance(value, dict)


def _as_json_object(value: object) -> JsonObject:
    return cast(JsonObject, value)


class ToolParserProtocol(Protocol):
    """Protocol for parser integration (implemented in T4)."""

    def parse(self, response_text: str) -> list[dict[str, JsonValue]]: ...


class ToolRegistryProtocol(Protocol):
    """Protocol for registry integration (implemented in T5)."""

    def get_tool_schemas(self) -> list[dict[str, JsonValue]]: ...


@dataclass(slots=True)
class _ClientConfig:
    model_name: str
    num_ctx: int
    ollama_base_url: str


class _FallbackToolCallParser:
    """Minimal parser fallback when ToolCallParser is unavailable."""

    def parse(self, response_text: str) -> list[dict[str, JsonValue]]:
        try:
            payload = cast(object, json.loads(response_text))
        except json.JSONDecodeError:
            return []

        if not _is_json_object(payload):
            return []
        payload_obj = _as_json_object(payload)

        message = payload_obj.get("message", {})
        if not _is_json_object(message):
            return []
        message_obj = _as_json_object(message)
        calls = message_obj.get("tool_calls", [])
        if not isinstance(calls, list):
            return []

        normalized: list[dict[str, JsonValue]] = []
        for call in calls:
            if _is_json_object(call):
                normalized.append(_as_json_object(call))
        return normalized


class OllamaClient:
    """Raw HTTP Ollama chat client for deterministic tool-calling behavior."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        retry_count: int = 3,
        max_retry_delay_seconds: float = 30.0,
        tool_parser: ToolParserProtocol | None = None,
        tool_registry: ToolRegistryProtocol | None = None,
        rate_limit_max_retries: int = 0,
        rate_limit_initial_delay: float = 2.0,
        rate_limit_max_delay: float = 120.0,
    ) -> None:
        resolved_settings = settings or Settings.from_yaml()
        resolved_base_url = (base_url or resolved_settings.ollama_base_url).rstrip("/")
        num_ctx_raw = getattr(
            resolved_settings, "num_ctx", resolved_settings.context_window_tokens
        )

        self.settings: Settings = resolved_settings
        self.client_config: _ClientConfig = _ClientConfig(
            model_name=resolved_settings.model_name,
            num_ctx=int(num_ctx_raw),
            ollama_base_url=resolved_base_url,
        )
        self.chat_url: str = f"{self.client_config.ollama_base_url}/api/chat"
        self.timeout_seconds: float = timeout_seconds
        self.retry_count: int = retry_count
        self.max_retry_delay_seconds: float = max_retry_delay_seconds
        self.rate_limit_max_retries: int = rate_limit_max_retries
        self.rate_limit_initial_delay: float = rate_limit_initial_delay
        self.rate_limit_max_delay: float = rate_limit_max_delay

        if tool_parser is not None:
            self.tool_parser: ToolParserProtocol = tool_parser
        else:
            self.tool_parser = _FallbackToolCallParser()

        self.tool_registry: ToolRegistryProtocol | None = tool_registry

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, JsonValue]],
        tools: list[dict[str, JsonValue]],
    ) -> tuple[str, list[dict[str, JsonValue]]] | None:
        """Send a non-streaming chat request to Ollama.

        Returns:
            Tuple of (assistant_text, parsed_tool_calls), or None when retries are
            exhausted for connection/timeout failures.
        """
        tool_schemas = self._resolve_tool_schemas(tools)
        payload = {
            "model": self.client_config.model_name,
            "messages": self._build_messages(system_prompt, messages),
            "tools": tool_schemas,
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": self.client_config.num_ctx,
            },
        }

        delay_seconds = 1.0
        for attempt in range(self.retry_count + 1):
            try:
                timeout = httpx.Timeout(self.timeout_seconds)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(self.chat_url, json=payload)

                if response.status_code == 404:
                    logger.error("Ollama model not found (404): %s", self.model_name)
                    return None

                if response.status_code == 429:
                    rl_delay = self.rate_limit_initial_delay
                    rl_attempt = 0
                    while True:
                        wait = min(rl_delay, self.rate_limit_max_delay)
                        logger.warning(
                            "Ollama rate limited (429), pausing %.1fs before retry "
                            "(attempt %d, max_retries=%s)",
                            wait,
                            rl_attempt + 1,
                            "∞"
                            if self.rate_limit_max_retries == 0
                            else str(self.rate_limit_max_retries),
                        )
                        await asyncio.sleep(wait)

                        async with httpx.AsyncClient(
                            timeout=httpx.Timeout(self.timeout_seconds)
                        ) as rl_client:
                            response = await rl_client.post(self.chat_url, json=payload)

                        if response.status_code != 429:
                            break

                        rl_attempt += 1
                        if (
                            self.rate_limit_max_retries > 0
                            and rl_attempt >= self.rate_limit_max_retries
                        ):
                            logger.error(
                                "Ollama rate limited after %d retries, giving up",
                                rl_attempt,
                            )
                            return None

                        rl_delay = min(rl_delay * 2, self.rate_limit_max_delay)

                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Ollama server error: {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                _ = response.raise_for_status()
                response_payload = cast(object, response.json())
                if not _is_json_object(response_payload):
                    logger.error("Unexpected Ollama response payload type")
                    return "", []
                response_obj = _as_json_object(response_payload)
                response_text = self._extract_response_text(response_obj)
                parsed_calls = self._parse_tool_calls(response_obj, response_text)
                return response_text, parsed_calls

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt >= self.retry_count:
                    logger.error("Ollama connection failed after retries: %s", exc)
                    return None

                logger.warning(
                    "Ollama connect/timeout attempt %d/%d failed: %s",
                    attempt + 1,
                    self.retry_count + 1,
                    exc,
                )
                await asyncio.sleep(min(delay_seconds, self.max_retry_delay_seconds))
                delay_seconds = min(delay_seconds * 2, self.max_retry_delay_seconds)

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response else None
                retryable = status_code in {500, 502, 503, 504}
                if not retryable or attempt >= self.retry_count:
                    logger.exception("Fatal HTTP error from Ollama: %s", exc)
                    raise

                logger.warning(
                    "Retryable HTTP error attempt %d/%d (status=%s)",
                    attempt + 1,
                    self.retry_count + 1,
                    status_code,
                )
                await asyncio.sleep(min(delay_seconds, self.max_retry_delay_seconds))
                delay_seconds = min(delay_seconds * 2, self.max_retry_delay_seconds)

            except httpx.HTTPError as exc:
                if attempt >= self.retry_count:
                    logger.error("HTTP error after retries exhausted: %s", exc)
                    return None

                logger.warning(
                    "Transient HTTP error attempt %d/%d: %s",
                    attempt + 1,
                    self.retry_count + 1,
                    exc,
                )
                await asyncio.sleep(min(delay_seconds, self.max_retry_delay_seconds))
                delay_seconds = min(delay_seconds * 2, self.max_retry_delay_seconds)

        return None

    def _build_messages(
        self,
        system_prompt: str,
        messages: list[dict[str, JsonValue]],
    ) -> list[dict[str, JsonValue]]:
        return [{"role": "system", "content": system_prompt}, *messages]

    def _resolve_tool_schemas(
        self, tools: list[dict[str, JsonValue]]
    ) -> list[dict[str, JsonValue]]:
        if tools:
            return tools

        registry = self.tool_registry
        if registry is None:
            return []

        return registry.get_tool_schemas()

    def _extract_response_text(self, response_payload: JsonObject) -> str:
        message = response_payload.get("message", {})
        if not _is_json_object(message):
            return ""
        message_obj = _as_json_object(message)
        content = message_obj.get("content", "")
        return content if isinstance(content, str) else ""

    def _parse_tool_calls(
        self,
        response_payload: JsonObject,
        response_text: str,
    ) -> list[dict[str, JsonValue]]:
        parser_input = json.dumps(response_payload, ensure_ascii=False)
        try:
            parsed = self.tool_parser.parse(parser_input)
            return parsed
        except Exception:
            logger.exception("Tool parser failed on payload JSON")

        try:
            parsed = self.tool_parser.parse(response_text)
            return parsed
        except Exception:
            logger.exception("Tool parser failed on content text")

        message = response_payload.get("message", {})
        if _is_json_object(message):
            message_obj = _as_json_object(message)
            tool_calls = message_obj.get("tool_calls", [])
            if isinstance(tool_calls, list):
                normalized: list[dict[str, JsonValue]] = []
                for call in tool_calls:
                    if _is_json_object(call):
                        normalized.append(_as_json_object(call))
                return normalized
        return []
