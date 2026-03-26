"""Petri Dish tool system.

Provides get_all_tools() to build a fully-wired ToolRegistry
with all 8 agent tools registered.
"""

from typing import Optional

from petri_dish.config import Settings
from petri_dish.tools.registry import ToolDefinition, ToolParameter, ToolRegistry
from petri_dish.tools import container_tools, host_tools, agent_tools


def _build_tool_definitions() -> list[ToolDefinition]:
    """Build the 8 tool definitions with their schemas and handlers."""
    return [
        ToolDefinition(
            name="file_read",
            description="Read a file from the container filesystem. Returns file contents as string.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file to read",
                ),
            ],
            handler=container_tools.file_read,
        ),
        ToolDefinition(
            name="file_write",
            description="Write content to a file in the container filesystem.",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file to write",
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write to the file",
                ),
            ],
            handler=container_tools.file_write,
        ),
        ToolDefinition(
            name="file_list",
            description="List directory contents in the container.",
            parameters=[
                ToolParameter(
                    name="directory",
                    type="string",
                    description="Directory path to list (defaults to current directory)",
                    required=False,
                    default=".",
                ),
            ],
            handler=container_tools.file_list,
        ),
        ToolDefinition(
            name="shell_exec",
            description="Execute a shell command inside the container. Output truncated at 10KB.",
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="Shell command to execute",
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds (default 30)",
                    required=False,
                    default=30,
                ),
            ],
            handler=container_tools.shell_exec,
        ),
        ToolDefinition(
            name="check_balance",
            description="Query your current credit balance.",
            parameters=[],
            handler=host_tools.check_balance,
            host_side=True,
        ),
        ToolDefinition(
            name="http_request",
            description="Make an HTTP request from the host. Returns response body.",
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="Target URL to request",
                ),
                ToolParameter(
                    name="method",
                    type="string",
                    description="HTTP method (GET or POST)",
                    required=False,
                    default="GET",
                    enum=["GET", "POST"],
                ),
            ],
            handler=host_tools.http_request,
            host_side=True,
        ),
        ToolDefinition(
            name="self_modify",
            description="Modify your own system prompt by setting a key-value override.",
            parameters=[
                ToolParameter(
                    name="key",
                    type="string",
                    description="Aspect of prompt to modify (e.g. 'strategy', 'focus')",
                ),
                ToolParameter(
                    name="value",
                    type="string",
                    description="New value for that aspect",
                ),
            ],
            handler=agent_tools.self_modify,
            host_side=True,
        ),
        ToolDefinition(
            name="get_env_info",
            description="Get information about the runtime environment and available tools.",
            parameters=[],
            handler=agent_tools.get_env_info,
            host_side=True,
        ),
    ]


def get_all_tools(settings: Optional[Settings] = None) -> ToolRegistry:
    """Build and return a ToolRegistry with all 8 tools registered.

    Args:
        settings: Optional Settings instance. If None, loads from config.yaml.

    Returns:
        Fully-wired ToolRegistry ready for use.
    """
    registry = ToolRegistry(settings=settings)
    for tool_def in _build_tool_definitions():
        registry.register(tool_def)
    return registry
