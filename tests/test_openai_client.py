"""Tests for OpenAI-compatible API client."""

from __future__ import annotations

import json
import pytest

from petri_dish.openai_client import OpenAICompatibleClient


class TestOpenAIClientParseResponse:
    def test_text_only_response(self):
        client = OpenAICompatibleClient(api_key="test-key")
        data = {
            "choices": [{"message": {"content": "Hello, world!", "role": "assistant"}}]
        }
        text, tool_calls = client._parse_response(data)
        assert text == "Hello, world!"
        assert tool_calls == []

    def test_tool_call_response(self):
        client = OpenAICompatibleClient(api_key="test-key")
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'll read that file.",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "file_read",
                                    "arguments": json.dumps({"path": "/env/data.csv"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        text, tool_calls = client._parse_response(data)
        assert text == "I'll read that file."
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "file_read"
        assert tool_calls[0]["arguments"] == {"path": "/env/data.csv"}

    def test_multiple_tool_calls(self):
        client = OpenAICompatibleClient(api_key="test-key")
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "file_read",
                                    "arguments": json.dumps({"path": "/env/a.txt"}),
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "file_write",
                                    "arguments": json.dumps(
                                        {"path": "/env/b.txt", "content": "hello"}
                                    ),
                                },
                            },
                        ],
                    }
                }
            ]
        }
        text, tool_calls = client._parse_response(data)
        assert text == ""
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "file_read"
        assert tool_calls[1]["name"] == "file_write"
        assert tool_calls[1]["arguments"]["content"] == "hello"

    def test_empty_choices(self):
        client = OpenAICompatibleClient(api_key="test-key")
        text, tool_calls = client._parse_response({"choices": []})
        assert text == ""
        assert tool_calls == []

    def test_invalid_json_arguments(self):
        client = OpenAICompatibleClient(api_key="test-key")
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "shell_exec",
                                    "arguments": "not valid json {{{",
                                },
                            }
                        ],
                    }
                }
            ]
        }
        text, tool_calls = client._parse_response(data)
        assert tool_calls[0]["name"] == "shell_exec"
        assert tool_calls[0]["arguments"] == {}

    def test_null_content_treated_as_empty_string(self):
        client = OpenAICompatibleClient(api_key="test-key")
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "check_balance",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
        text, tool_calls = client._parse_response(data)
        assert text == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "check_balance"


class TestOpenAIClientInit:
    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "env-key-123")
        client = OpenAICompatibleClient()
        assert client.api_key == "env-key-123"

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "env-key-123")
        client = OpenAICompatibleClient(api_key="explicit-key")
        assert client.api_key == "explicit-key"

    def test_base_url_stripped(self):
        client = OpenAICompatibleClient(
            api_key="test", base_url="https://api.z.ai/api/paas/v4/"
        )
        assert client.base_url == "https://api.z.ai/api/paas/v4"

    def test_custom_model_and_temperature(self):
        client = OpenAICompatibleClient(
            api_key="test", model="glm-4.7", temperature=0.5
        )
        assert client.model == "glm-4.7"
        assert client.temperature == 0.5
