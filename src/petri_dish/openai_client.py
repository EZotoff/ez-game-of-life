"""Async client for OpenAI-compatible APIs (Z.AI, OpenAI, etc.)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.z.ai/api/paas/v4",
        model: str = "glm-5",
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        max_retry_delay_seconds: float = 30.0,
        temperature: float = 0.8,
    ) -> None:
        self.api_key = api_key or os.getenv("ZAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self.temperature = temperature

        if not self.api_key:
            logger.warning(
                "No API key provided. Set ZAI_API_KEY env var or pass api_key parameter."
            )

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]] | None:
        all_messages = [{"role": "system", "content": system_prompt}, *messages]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": all_messages,
            "temperature": self.temperature,
            "stream": False,
        }

        if tools:
            payload["tools"] = tools

        delay = 1.0

        for attempt in range(self.max_retries + 1):
            try:
                timeout = httpx.Timeout(self.timeout_seconds)
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )

                if response.status_code == 401:
                    logger.error("API authentication failed (401). Check your API key.")
                    return None

                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    wait = float(retry_after) if retry_after else delay
                    if attempt < self.max_retries:
                        logger.warning("Rate limited (429), retrying in %.1fs", wait)
                        await asyncio.sleep(min(wait, self.max_retry_delay_seconds))
                        delay = min(delay * 2, self.max_retry_delay_seconds)
                        continue
                    logger.error("Rate limited after %d retries", attempt + 1)
                    return None

                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Server error: {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                response.raise_for_status()
                data = response.json()

                return self._parse_response(data)

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt >= self.max_retries:
                    logger.error("Connection failed after retries: %s", exc)
                    return None

                logger.warning(
                    "Connect/timeout attempt %d/%d: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                await asyncio.sleep(min(delay, self.max_retry_delay_seconds))
                delay = min(delay * 2, self.max_retry_delay_seconds)

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response else None
                retryable = status in {500, 502, 503, 504}
                if not retryable or attempt >= self.max_retries:
                    logger.exception("Fatal HTTP error: %s", exc)
                    return None

                logger.warning("Retryable HTTP error (status=%s)", status)
                await asyncio.sleep(min(delay, self.max_retry_delay_seconds))
                delay = min(delay * 2, self.max_retry_delay_seconds)

            except httpx.HTTPError as exc:
                if attempt >= self.max_retries:
                    logger.error("HTTP error after retries: %s", exc)
                    return None

                await asyncio.sleep(min(delay, self.max_retry_delay_seconds))
                delay = min(delay * 2, self.max_retry_delay_seconds)

        return None

    def _parse_response(self, data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        choices = data.get("choices", [])
        if not choices:
            return "", []

        message = choices[0].get("message", {})
        content = message.get("content", "") or ""

        raw_tool_calls = message.get("tool_calls", [])
        tool_calls: list[dict[str, Any]] = []

        for tc in raw_tool_calls:
            function = tc.get("function", {})
            name = function.get("name", "")
            arguments_str = function.get("arguments", "{}")

            if isinstance(arguments_str, str):
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(arguments_str, dict):
                arguments = arguments_str
            else:
                arguments = {}

            tool_calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                }
            )

        return content, tool_calls
