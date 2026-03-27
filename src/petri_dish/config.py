"""Configuration management for Petri Dish using Pydantic Settings."""

from typing import Dict, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path


class Settings(BaseSettings):
    """Main configuration class for Petri Dish."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Economy and Game Parameters
    initial_balance: float = 1000.0
    burn_rate_per_turn: float = 0.1
    file_drop_lambda: float = 0.02
    file_drop_interval_turns: int = 50
    max_turns: int = 1000
    max_turns_per_tool: int = 1
    turn_timeout_seconds: int = 60
    max_consecutive_empty_turns: int = 10

    # Model and Context Parameters
    context_window_tokens: int = 8192
    context_summary_interval_turns: int = 40
    model_name: str = "qwen3.5:9b-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    default_temperature: float = 0.8

    # Docker Container Parameters
    docker_image: str = "python:3.12-slim"
    docker_mem_limit: str = "512m"
    docker_cpu_quota: int = 50000
    docker_storage_limit: str = "1g"

    # Economy Mode and Degradation
    economy_mode: Literal["visible", "degraded"] = "visible"
    degradation_tiers: Dict[str, float] = {"premium": 0.6, "balanced": 0.3, "eco": 0.0}

    # File System Parameters
    num_file_families: int = 3
    validator_scoring_weights: Dict[str, float] = {"csv": 1.0, "json": 1.0, "log": 1.0}
    credit_rewards: Dict[str, float] = {"easy": 0.3, "hard": 2.0}

    # Null Model Configuration
    null_model_type: Literal["random", "constant", "none"] = "random"

    # Death System
    starvation_turns: int = 7

    # Persistent Memory
    memory_path: str = ".sisyphus/memory"

    # Tool Costs
    tool_costs: Dict[str, float] = {
        "file_read": 0.01,
        "file_write": 0.01,
        "file_list": 0.01,
        "shell_exec": 0.05,
        "check_balance": 0.0,
        "http_request": 0.1,
        "self_modify": 0.02,
        "get_env_info": 0.0,
    }

    @classmethod
    def from_yaml(cls, yaml_path: str = "config.yaml") -> "Settings":
        """Load settings from a YAML file."""
        yaml_path_obj = Path(yaml_path)
        if not yaml_path_obj.exists():
            return cls()

        with open(yaml_path_obj, "r") as f:
            yaml_data = yaml.safe_load(f)

        # Handle nested structures properly
        if yaml_data:
            # Ensure degradation_tiers is a dict
            if "degradation_tiers" in yaml_data:
                if not isinstance(yaml_data["degradation_tiers"], dict):
                    yaml_data["degradation_tiers"] = {}

            # Ensure validator_scoring_weights is a dict
            if "validator_scoring_weights" in yaml_data:
                if not isinstance(yaml_data["validator_scoring_weights"], dict):
                    yaml_data["validator_scoring_weights"] = {}

            # Ensure tool_costs is a dict
            if "tool_costs" in yaml_data:
                if not isinstance(yaml_data["tool_costs"], dict):
                    yaml_data["tool_costs"] = {}

        return cls(**yaml_data) if yaml_data else cls()


# Global settings instance
settings = Settings.from_yaml()
