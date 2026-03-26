# pyright: reportDeprecated=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnnecessaryIsInstance=false, reportUnreachable=false, reportUnannotatedClassAttribute=false, reportReturnType=false
"""Tool call parser with multi-strategy fallbacks for Ollama responses."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool call payload."""

    name: str
    arguments: dict[str, Any]


def validate_arguments(
    tool_name: str,
    args: dict[str, Any],
    tool_registry: dict[str, Any],
) -> dict[str, Any]:
    """Normalize argument payload using tool schema when available.

    Behavior:
    - Accepts dict arguments as-is and type-normalizes known fields.
    - If arguments are JSON strings, decode into dict.
    - Never raises on malformed data; returns best-effort dict.
    """

    if isinstance(args, str):
        parsed = _safe_json_loads(args)
        args = parsed if isinstance(parsed, dict) else {}
    elif not isinstance(args, dict):
        return {}

    schema = tool_registry.get(tool_name, {}) if isinstance(tool_registry, dict) else {}
    properties = _extract_schema_properties(schema)
    if not properties:
        return dict(args)

    normalized: dict[str, Any] = {}
    for key, value in args.items():
        expected_type = properties.get(key, {}).get("type")
        normalized[key] = _coerce_type(value, expected_type)

    return normalized


class ToolCallParser:
    """Multi-strategy parser for extracting tool calls from LLM responses."""

    _NAME_ARGS_PATTERN = re.compile(
        r'"name"\s*:\s*"(?P<name>[^"\\]+)"[\s\S]{0,240}?"arguments"\s*:\s*',
        flags=re.IGNORECASE,
    )

    def __init__(self, tool_registry: Mapping[str, Any] | None = None) -> None:
        self.tool_registry: dict[str, Any] = dict(tool_registry or {})

    def parse(self, response: Any) -> list[ToolCall]:
        """Parse tool calls with three ordered fallback strategies."""

        if response is None:
            return []

        strategies = (
            ("native_ollama", self._strategy_native_ollama),
            ("tag_strip_retry", self._strategy_tag_strip_retry),
            ("regex_extraction", self._strategy_regex_extraction),
        )

        for strategy_name, strategy in strategies:
            calls = strategy(response)
            if calls:
                logger.info(
                    "ToolCallParser succeeded via strategy=%s tool_calls=%d",
                    strategy_name,
                    len(calls),
                )
                return calls

        return []

    def _strategy_native_ollama(self, response: Any) -> list[ToolCall]:
        """Strategy 1: parse native message.tool_calls from JSON."""

        payload = self._coerce_payload(response)
        if not payload:
            return []
        return self._extract_from_payload(payload)

    def _strategy_tag_strip_retry(self, response: Any) -> list[ToolCall]:
        """Strategy 2: strip thinking tags/comments and retry JSON parsing."""

        text = self._response_to_text(response)
        if not text:
            return []

        cleaned = self._strip_thinking_contamination(text)
        if not cleaned.strip():
            return []

        payload = self._coerce_payload(cleaned)
        if payload:
            calls = self._extract_from_payload(payload)
            if calls:
                return calls

        for candidate in self._extract_json_objects(cleaned):
            calls = self._extract_from_payload(candidate)
            if calls:
                return calls

        return []

    def _strategy_regex_extraction(self, response: Any) -> list[ToolCall]:
        """Strategy 3: regex extraction from full response text."""

        text = self._response_to_text(response)
        if not text:
            return []

        calls: list[ToolCall] = []
        for match in self._NAME_ARGS_PATTERN.finditer(text):
            name = (match.group("name") or "").strip()
            if not name:
                continue

            value = self._extract_json_value(text, match.end())
            if value is None:
                continue

            args = self._decode_arguments(value)
            if not isinstance(args, dict):
                continue

            calls.append(
                ToolCall(
                    name=name,
                    arguments=validate_arguments(name, args, self.tool_registry),
                )
            )

        return calls

    def _extract_from_payload(self, payload: Mapping[str, Any]) -> list[ToolCall]:
        message = payload.get("message")
        if isinstance(message, Mapping):
            tool_calls = message.get("tool_calls")
            calls = self._normalize_tool_calls(tool_calls)
            if calls:
                return calls

        tool_calls = payload.get("tool_calls")
        return self._normalize_tool_calls(tool_calls)

    def _normalize_tool_calls(self, tool_calls: Any) -> list[ToolCall]:
        if not isinstance(tool_calls, list):
            return []

        normalized: list[ToolCall] = []
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue

            function = call.get("function")
            if isinstance(function, Mapping):
                name = function.get("name")
                arguments = function.get("arguments", {})
            else:
                name = call.get("name")
                arguments = call.get("arguments", {})

            if not isinstance(name, str) or not name.strip():
                continue

            decoded_args = self._decode_arguments(arguments)
            if not isinstance(decoded_args, dict):
                decoded_args = {}

            normalized.append(
                ToolCall(
                    name=name.strip(),
                    arguments=validate_arguments(
                        name.strip(),
                        decoded_args,
                        self.tool_registry,
                    ),
                )
            )

        return normalized

    @staticmethod
    def _coerce_payload(response: Any) -> dict[str, Any]:
        if isinstance(response, Mapping):
            return dict(response)
        if isinstance(response, str):
            parsed = _safe_json_loads(response)
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _response_to_text(response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, Mapping):
            message = response.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
            try:
                return json.dumps(response)
            except (TypeError, ValueError):
                return str(response)
        return str(response)

    @staticmethod
    def _strip_thinking_contamination(text: str) -> str:
        cleaned = re.sub(
            r"<!--\s*thinking\s*-->|<!--\s*/thinking\s*-->",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL
        )
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _extract_json_objects(self, text: str) -> list[dict[str, Any]]:
        payload = self._coerce_payload(text)
        if payload:
            return [payload]

        objects: list[dict[str, Any]] = []
        for idx, char in enumerate(text):
            if char != "{":
                continue
            candidate = self._extract_balanced_braces(text, idx)
            if not candidate:
                continue
            parsed = _safe_json_loads(candidate)
            if isinstance(parsed, dict):
                objects.append(parsed)
        return objects

    @staticmethod
    def _extract_balanced_braces(text: str, start: int) -> str:
        depth = 0
        in_string = False
        escaped = False

        for idx in range(start, len(text)):
            ch = text[idx]

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

        return ""

    @staticmethod
    def _extract_json_value(text: str, start: int) -> str | None:
        idx = start
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            return None

        ch = text[idx]
        if ch == "{":
            value = ToolCallParser._extract_balanced_braces(text, idx)
            return value or None

        if ch == '"':
            end = idx + 1
            escaped = False
            while end < len(text):
                cur = text[end]
                if escaped:
                    escaped = False
                elif cur == "\\":
                    escaped = True
                elif cur == '"':
                    return text[idx : end + 1]
                end += 1
            return None

        end = idx
        while end < len(text) and text[end] not in ",}]":
            end += 1
        token = text[idx:end].strip()
        return token or None

    @staticmethod
    def _decode_arguments(value: Any) -> dict[str, Any] | Any:
        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            parsed = _safe_json_loads(value)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                nested = _safe_json_loads(parsed)
                if isinstance(nested, dict):
                    return nested
            return {}

        return {}


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _extract_schema_properties(schema: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(schema, Mapping):
        return {}

    function = schema.get("function")
    if isinstance(function, Mapping):
        parameters = function.get("parameters")
        if isinstance(parameters, Mapping):
            properties = parameters.get("properties")
            if isinstance(properties, Mapping):
                return dict(properties)

    parameters = schema.get("parameters")
    if isinstance(parameters, Mapping):
        properties = parameters.get("properties")
        if isinstance(properties, Mapping):
            return dict(properties)

    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        return dict(properties)

    return {}


def _coerce_type(value: Any, expected_type: Any) -> Any:
    if expected_type == "string":
        if value is None:
            return ""
        return str(value)

    if expected_type == "integer":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value.strip()))
            except ValueError:
                return 0
        return 0

    if expected_type == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return 0.0
        return 0.0

    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return False

    if expected_type == "object":
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = _safe_json_loads(value)
            return parsed if isinstance(parsed, dict) else {}
        return {}

    if expected_type == "array":
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            parsed = _safe_json_loads(value)
            return parsed if isinstance(parsed, list) else []
        return []

    return value
