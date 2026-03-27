"""Tool registry for Petri Dish agent simulation.

Manages tool definitions, generates Ollama-compatible schemas (OpenAI function
calling format), and dispatches tool execution.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from petri_dish.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Single parameter in a tool's function signature."""

    name: str
    type: str  # JSON Schema type: "string", "integer", "number", "boolean"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class ToolDefinition:
    """Complete definition of an agent tool.

    Attributes:
        name: Tool function name (e.g. "file_read").
        description: Human-readable description for the LLM.
        parameters: List of ToolParameter definitions.
        handler: Callable that implements the tool logic.
        host_side: If True, runs on host (not in container).
        cost: Credit cost per invocation (loaded from config).
        free_when_stripped: If True, tool is available even when agent wallet is ≤ 0.
    """

    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable[..., str]] = None
    host_side: bool = False
    cost: float = 0.0
    free_when_stripped: bool = False

    def to_ollama_schema(self) -> Dict[str, Any]:
        """Generate OpenAI function calling format schema for Ollama.

        Returns:
            Dict matching the Ollama tools API format:
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": {...},
                        "required": [...]
                    }
                }
            }
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in self.parameters:
            prop: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum is not None:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        schema: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return schema


class ToolRegistry:
    """Registry that manages tool definitions and dispatches execution.

    Loads tool costs from config. Generates Ollama-compatible schemas.
    Routes execute_tool calls to the correct handler.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize registry with tool cost config.

        Args:
            settings: Settings instance. If None, loads from config.yaml.
        """
        if settings is None:
            settings = Settings.from_yaml()

        self._settings = settings
        self._tools: Dict[str, ToolDefinition] = {}
        self._tool_costs = settings.tool_costs

        logger.info(
            "ToolRegistry initialized with %d tool costs", len(self._tool_costs)
        )

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition.

        Applies cost from config if available.

        Args:
            tool: ToolDefinition to register.
        """
        if tool.name in self._tool_costs:
            tool.cost = self._tool_costs[tool.name]
        self._tools[tool.name] = tool
        logger.info(
            "Registered tool: %s (cost=%.4f, host_side=%s)",
            tool.name,
            tool.cost,
            tool.host_side,
        )

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Generate Ollama tool schemas for all registered tools.

        Returns:
            List of OpenAI function calling format dicts.
        """
        return [tool.to_ollama_schema() for tool in self._tools.values()]

    def get_tool_names(self) -> List[str]:
        """Return list of all registered tool names."""
        return list(self._tools.keys())

    def get_tool_cost(self, name: str) -> float:
        """Get the credit cost for a tool invocation.

        Args:
            name: Tool name.

        Returns:
            Cost in credits, or 0.0 if unknown.
        """
        tool = self._tools.get(name)
        if tool is not None:
            return tool.cost
        return self._tool_costs.get(name, 0.0)

    def execute_tool(self, name: str, args: Dict[str, Any], container_id: str) -> str:
        """Dispatch tool execution to the registered handler.

        Args:
            name: Tool name to execute.
            args: Arguments dict parsed from LLM tool call.
            container_id: Docker container ID for container-side tools.

        Returns:
            String result from tool execution.

        Raises:
            ValueError: If tool is not registered or has no handler.
        """
        tool = self._tools.get(name)
        if tool is None:
            msg = f"Unknown tool: {name}"
            logger.error(msg)
            raise ValueError(msg)

        if tool.handler is None:
            msg = f"Tool '{name}' has no handler"
            logger.error(msg)
            raise ValueError(msg)

        logger.info(
            "Executing tool: %s (args=%s, container=%s)", name, args, container_id
        )

        try:
            if tool.host_side:
                result = tool.handler(**args)
            else:
                result = tool.handler(container_id=container_id, **args)
        except Exception as e:
            error_msg = f"Tool '{name}' failed: {type(e).__name__}: {e}"
            logger.error(error_msg)
            return error_msg

        return str(result)

    def get_stripped_schemas(self) -> List[Dict[str, Any]]:
        """Generate Ollama schemas for tools available when agent is stripped.

        Returns schemas for tools where free_when_stripped is True.
        """
        return [
            tool.to_ollama_schema()
            for tool in self._tools.values()
            if tool.free_when_stripped
        ]

    def is_tool_allowed_when_stripped(self, name: str) -> bool:
        """Check if a tool is available in stripped state.

        Args:
            name: Tool name to check.

        Returns:
            True if the tool has free_when_stripped=True.
        """
        tool = self._tools.get(name)
        return tool is not None and tool.free_when_stripped

    def get_stripped_tool_names(self) -> List[str]:
        """Return names of tools available in stripped state."""
        return [name for name, tool in self._tools.items() if tool.free_when_stripped]

    def __len__(self) -> int:
        return len(self._tools)
