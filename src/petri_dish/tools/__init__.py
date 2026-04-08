"""Petri Dish tool system.

Provides get_all_tools() to build a fully-wired ToolRegistry
with all 12 agent tools registered (8 base + 2 communication + 2 new).
"""

from petri_dish.config import Settings
from petri_dish.tools.registry import ToolDefinition, ToolParameter, ToolRegistry
from petri_dish.tools import (
    container_tools,
    host_tools,
    agent_tools,
    comm_tools,
    task_broker_tools,
)


def _build_tool_definitions() -> list[ToolDefinition]:
    """Build the 12 tool definitions with their schemas and handlers."""
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
            free_when_stripped=True,
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
            free_when_stripped=True,
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
            name="python_exec",
            description="Execute a Python script inside the container. Better than shell_exec for multi-line code. Output truncated at 10KB.",
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Python code to execute",
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds (default 60)",
                    required=False,
                    default=60,
                ),
            ],
            handler=container_tools.python_exec,
        ),
        ToolDefinition(
            name="pass_turn",
            description="End your turn early without using all remaining actions. Costs minimal zod. Does NOT count as an empty turn.",
            parameters=[
                ToolParameter(
                    name="reason",
                    type="string",
                    description="Optional reason for passing (logged for analysis)",
                    required=False,
                    default="",
                ),
            ],
            handler=container_tools.pass_turn,
            free_when_stripped=True,
        ),
        ToolDefinition(
            name="check_balance",
            description="Query your current zod balance.",
            parameters=[],
            handler=host_tools.check_balance,
            host_side=True,
            free_when_stripped=True,
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
            name="web_search",
            description=(
                "Search the web using configured provider and return raw "
                "read-only results (title/url/snippet)."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query to execute",
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum results to return (default 3)",
                    required=False,
                    default=3,
                ),
            ],
            handler=host_tools.web_search,
            host_side=True,
            free_when_stripped=False,
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
            free_when_stripped=True,
        ),
        ToolDefinition(
            name="send_message",
            description="Send a text message to another agent in the environment.",
            parameters=[
                ToolParameter(
                    name="recipient",
                    type="string",
                    description="Agent ID of the recipient",
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Message content to send",
                ),
            ],
            handler=comm_tools.send_message,
            host_side=True,
            free_when_stripped=True,
        ),
        ToolDefinition(
            name="read_messages",
            description="Read all unread messages addressed to you. Messages are consumed on read.",
            parameters=[],
            handler=comm_tools.read_messages,
            host_side=True,
            free_when_stripped=True,
        ),
        ToolDefinition(
            name="request_task",
            description=(
                "Delegate a task to a specialist sub-agent. The broker estimates complexity, "
                "quotes a zod cost, and if you can afford it, runs the task. Returns the result. "
                "Use for analysis, code generation, research, or any reasoning-heavy work "
                "beyond simple tool calls."
            ),
            parameters=[
                ToolParameter(
                    name="task_description",
                    type="string",
                    description="Clear, specific description of the task to delegate.",
                ),
            ],
            handler=task_broker_tools.request_task_stub,
            host_side=True,
        ),
    ]


def get_all_tools(settings: Settings | None = None) -> ToolRegistry:
    """Build and return a ToolRegistry with all 10 tools registered.

    Args:
        settings: Optional Settings instance. If None, loads from config.yaml.

    Returns:
        Fully-wired ToolRegistry ready for use.
    """
    registry = ToolRegistry(settings=settings)
    for tool_def in _build_tool_definitions():
        registry.register(tool_def)
    return registry
