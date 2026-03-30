"""Agent-side tools for self-modification and environment introspection.

Tools: self_modify, get_env_info.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

_agent_prompt_overrides: Dict[str, str] = {}


def self_modify(key: str, value: str, **kwargs: object) -> str:
    """Modify agent's system prompt by setting a key-value override.

    Stores modifications to /agent/modifications.json with max 10 limit.
    The orchestrator reads these modifications when constructing the next
    system prompt and includes them as [Agent modification - key]: value.
    """
    if not key or not value:
        return "Error: key and value must not be empty"

    modifications_path = Path("/agent/modifications.json")

    try:
        modifications_path.parent.mkdir(parents=True, exist_ok=True)

        modifications = {}
        if modifications_path.exists():
            try:
                with open(modifications_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        modifications = data
            except (json.JSONDecodeError, IOError):
                modifications = {}

        timestamp = datetime.now().isoformat()
        modifications[key] = {
            "value": value,
            "timestamp": timestamp,
            "applied_at": timestamp,
        }

        if len(modifications) > 10:
            oldest_key = min(
                modifications.keys(), key=lambda k: modifications[k]["timestamp"]
            )
            del modifications[oldest_key]

        with open(modifications_path, "w") as f:
            json.dump(modifications, f, indent=2)

        logger.info("self_modify: set %s = %s", key, value[:100])
        return f"Updated prompt modification: {key} = {value}"
    except Exception as e:
        logger.error("Failed to save modification: %s", e)
        return f"Error saving modification: {e}"


def clear_prompt_overrides() -> None:
    """Reset all prompt overrides by removing the modifications file."""
    modifications_path = Path("/agent/modifications.json")
    _agent_prompt_overrides.clear()
    if modifications_path.exists():
        try:
            modifications_path.unlink()
        except OSError:
            pass


def get_prompt_overrides() -> Dict[str, str]:
    """Return current prompt overrides from filesystem (used by orchestrator)."""
    modifications_path = Path("/agent/modifications.json")

    if not modifications_path.exists():
        return {}

    try:
        with open(modifications_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {}

        overrides = {}
        for key, mod_data in data.items():
            if isinstance(mod_data, dict) and "value" in mod_data:
                overrides[key] = mod_data["value"]
            else:
                overrides[key] = str(mod_data)

        return overrides
    except (json.JSONDecodeError, IOError):
        return {}


def get_env_info(**kwargs: object) -> str:
    """Return available environment information.

    Provides the agent with context about its runtime environment,
    available tools, and basic system info.
    """
    import platform

    modifications_path = Path("/agent/modifications.json")
    active_overrides = len(get_prompt_overrides())

    info_lines = [
        "=== Environment Info ===",
        f"Platform: {platform.system()} {platform.machine()}",
        f"Python: {platform.python_version()}",
        "Runtime: Docker container (sandboxed)",
        "Available tools: file_read, file_write, file_list, shell_exec, "
        "check_balance, http_request, self_modify, get_env_info",
        "Network: disabled inside container",
        "Storage: limited (see config)",
        f"Prompt overrides active: {active_overrides}",
    ]
    return "\n".join(info_lines)
