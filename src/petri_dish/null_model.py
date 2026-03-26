"""Null Model (Random-Action Baseline) — Experimental control.

Implements a random-action baseline that matches the OllamaClient interface
but generates random tool calls instead of using an LLM.
"""

import random
from typing import Any, Optional

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class NullModel:
    """Random-action baseline that matches OllamaClient interface.

    Generates random but valid tool calls for experimental control.
    No LLM calls — pure random baseline with configurable seed.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize null model with optional random seed.

        Args:
            seed: Random seed for reproducibility. If None, uses system time.
        """
        self._rng: random.Random = random.Random(seed)
        self._seed: Optional[int] = seed

    async def chat(
        self,
        system_prompt: str,  # noqa: ARG002 (unused parameter for interface compatibility)
        messages: list[dict[str, JsonValue]],  # noqa: ARG002 (unused parameter for interface compatibility)
        tools: list[dict[str, JsonValue]],
    ) -> tuple[str, list[dict[str, JsonValue]]]:
        """Generate random tool call matching OllamaClient signature.

        Randomly selects one tool from available tools and generates
        valid random arguments for it.

        Args:
            system_prompt: Ignored (for interface compatibility).
            messages: Ignored (for interface compatibility).
            tools: List of tool schemas in OpenAI function calling format.

        Returns:
            Tuple of (empty_text, [tool_call]) where tool_call is a randomly
            generated valid tool call.
        """
        # If no tools available, return empty response
        if not tools:
            return "", []

        # Extract tool names and schemas
        tool_schemas: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "function":
                func = tool.get("function")
                if isinstance(func, dict):
                    tool_schemas.append(func)

        if not tool_schemas:
            return "", []

        # Randomly select a tool
        selected_schema: dict[str, Any] = self._rng.choice(tool_schemas)
        tool_name: Any = selected_schema.get("name", "")
        if not isinstance(tool_name, str) or not tool_name:
            return "", []

        # Generate random arguments for the selected tool
        parameters: Any = selected_schema.get("parameters", {})
        if not isinstance(parameters, dict):
            return "", []

        properties: Any = parameters.get("properties", {})
        if not isinstance(properties, dict):
            return "", []

        required: Any = parameters.get("required", [])
        if not isinstance(required, list):
            required = []

        # Generate random arguments
        args: dict[str, JsonValue] = {}
        for param_name, param_schema in properties.items():
            if not isinstance(param_schema, dict):
                continue

            # Only generate required parameters (simplifies random generation)
            if param_name in required:
                param_type: Any = param_schema.get("type", "string")
                if not isinstance(param_type, str):
                    param_type = "string"

                args[param_name] = self._generate_random_value(
                    param_name, param_type, param_schema
                )

        # Create tool call in Ollama format
        tool_call: dict[str, JsonValue] = {
            "function": {
                "name": tool_name,
                "arguments": args,
            },
            "id": f"call_{self._rng.randint(1000, 9999)}",
            "type": "function",
        }

        # Return empty text response with the random tool call
        return "", [tool_call]

    def _generate_random_value(
        self, param_name: str, param_type: str, param_schema: dict[str, Any]
    ) -> JsonValue:
        """Generate random but valid value for a tool parameter.

        Args:
            param_name: Parameter name (used for context-aware generation).
            param_type: JSON Schema type.
            param_schema: Full parameter schema.

        Returns:
            Random value appropriate for the parameter.
        """
        # Context-aware generation based on parameter name
        param_name_lower = param_name.lower()

        # File paths
        if "path" in param_name_lower or "file" in param_name_lower:
            # Generate random file paths
            directories = ["/env/incoming/", "/agent/", "/tmp/", "/home/agent/"]
            files = ["config.yaml", "data.txt", "log.txt", "script.py", "readme.md"]
            dir_choice = self._rng.choice(directories)
            file_choice = self._rng.choice(files)
            return f"{dir_choice}{file_choice}"

        # Directory paths
        if "directory" in param_name_lower:
            directories = ["/env/incoming", "/agent", "/tmp", "/home/agent", "."]
            return self._rng.choice(directories)

        # Commands for shell_exec
        if param_name_lower == "command":
            commands = [
                "ls -la",
                "pwd",
                "cat /etc/os-release",
                "echo 'test'",
                "whoami",
                "date",
                "uname -a",
                "df -h",
            ]
            return self._rng.choice(commands)

        # URLs for http_request
        if param_name_lower == "url":
            domains = ["example.com", "localhost", "127.0.0.1"]
            paths = ["/api/status", "/health", "/", "/test"]
            protocol = self._rng.choice(["http://", "https://"])
            domain = self._rng.choice(domains)
            path = self._rng.choice(paths)
            return f"{protocol}{domain}{path}"

        # HTTP methods
        if param_name_lower == "method":
            return self._rng.choice(["GET", "POST"])

        # Timeout values
        if param_name_lower == "timeout":
            return self._rng.randint(5, 60)

        # Key-value pairs for self_modify
        if param_name_lower == "key":
            keys = ["strategy", "focus", "priority", "approach", "method"]
            return self._rng.choice(keys)

        if param_name_lower == "value":
            values = [
                "exploratory",
                "focused",
                "aggressive",
                "conservative",
                "systematic",
                "random",
                "efficient",
            ]
            return self._rng.choice(values)

        # Content for file_write
        if param_name_lower == "content":
            contents = [
                "Test content",
                "Hello world",
                "# Configuration\nkey: value",
                "Lorem ipsum dolor sit amet",
                "Random data: " + str(self._rng.random()),
            ]
            return self._rng.choice(contents)

        # Default type-based generation
        if param_type == "string":
            # Check for enum constraints
            enum: Any = param_schema.get("enum")
            if isinstance(enum, list) and enum:
                return self._rng.choice(enum)
            return f"random_{param_name}_{self._rng.randint(1, 100)}"

        elif param_type == "integer":
            return self._rng.randint(1, 100)

        elif param_type == "number":
            return round(self._rng.uniform(0.0, 100.0), 2)

        elif param_type == "boolean":
            return self._rng.choice([True, False])

        # Fallback to empty string
        return ""

    def get_seed(self) -> Optional[int]:
        """Get the random seed used by this null model.

        Returns:
            The seed value or None if using system time.
        """
        return self._seed
