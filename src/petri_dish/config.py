"""Configuration management for Petri Dish using Pydantic Settings."""

from typing import Dict, List, Literal, Optional
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
    initial_zod: float = 1000.0
    decay_rate_per_turn: float = 0.1
    file_drop_lambda: float = 0.02
    file_drop_interval_turns: int = 50
    max_turns: int = 1000
    max_turns_per_tool: int = 1
    turn_timeout_seconds: int = 60
    max_consecutive_empty_turns: int = 10

    # Model and Context Parameters
    context_window_tokens: int = 8192
    context_summary_interval_turns: int = 40
    max_agent_messages: int = 60
    model_name: str = "qwen3.5:9b-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    default_temperature: float = 0.8

    # LLM Backend Selection
    llm_backend: Literal["ollama", "openai_compatible"] = "ollama"
    openai_api_base_url: str = "https://api.z.ai/api/paas/v4"
    openai_api_key_env_var: str = "ZAI_API_KEY"
    openai_model_name: str = "glm-5"

    # Rate-Limit Resilience (429 handling)
    # 0 = retry indefinitely (the world waits), N = give up after N retries
    rate_limit_max_retries: int = 0
    rate_limit_initial_delay: float = 2.0
    rate_limit_max_delay: float = 600.0
    llm_inter_call_delay: float = 2.0

    # Entropic UBI (artificial resourcefulness mode)
    base_income_per_turn: float = 0.3
    entropy_window_turns: int = 5
    ubi_min: float = 0.1
    ubi_max: float = 0.8

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
    zod_rewards: Dict[str, float] = {"easy": 0.3, "hard": 2.0}

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
        "web_search": 0.15,
        "self_modify": 0.02,
        "get_env_info": 0.0,
        "request_task": 0.0,
    }

    web_search_cost: float = 0.15
    web_search_provider: str = "duckduckgo_instant_answer"
    web_search_base_url: str = "https://api.duckduckgo.com/"
    web_search_user_agent: str = "PetriDish-Scout/1.0 (+read-only)"
    web_search_timeout_seconds: int = 10
    web_search_max_queries_per_call: int = 3
    web_search_max_results_per_query: int = 5
    web_search_chars_per_result: int = 2000
    web_search_allow_redirects: bool = False
    web_search_blocked_domains: List[str] = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "local",
    ]
    web_search_daily_budget: int = 50
    web_search_calls_per_turn: int = 1
    tavily_api_key: str = ""

    overseer_enabled: bool = False
    overseer_evaluation_interval: int = 5
    overseer_bonus_cap: float = 0.15
    overseer_max_bonus_per_evaluation: float = 0.5

    # Multi-Agent Configuration
    multi_agent_enabled: bool = False
    multi_agent_count: int = 2
    multi_agent_models: List[str] = []
    multi_agent_names: List[str] = []
    multi_agent_shared_filesystem: bool = True
    multi_agent_actions_per_turn: int = 4
    multi_agent_death_forfeit_pct: float = 0.6
    multi_agent_decay_pct: float = 0.3
    multi_agent_salvage_pct: float = 0.3
    multi_agent_reentry_fee: float = 20.0
    multi_agent_spectator_rounds: int = 2
    multi_agent_debt_garnish_pct: float = 0.5

    traits_enabled: bool = True
    traits_ema_alpha: float = 0.3
    traits_decay_factor: float = 0.95
    traits_rebirth_factor: float = 0.85
    traits_instinct_threshold: float = 0.3
    trait_snapshot_interval: int = 1
    llm_call_log_rate: float = 0.2

    # Task Broker Configuration
    request_task_enabled: bool = False
    request_task_model: str = ""
    request_task_max_cost: float = 15.0
    request_task_complexity_rates: Dict[str, float] = {
        "SIMPLE": 0.5,
        "MODERATE": 2.0,
        "COMPLEX": 5.0,
        "VERY_COMPLEX": 10.0,
    }

    promotion_enabled: bool = False
    promotion_threshold: int = 3
    promotion_bonus_multiplier: float = 1.5
    max_promoted_rules: int = 10

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

            # Ensure request_task_complexity_rates is a dict
            if "request_task_complexity_rates" in yaml_data:
                if not isinstance(yaml_data["request_task_complexity_rates"], dict):
                    yaml_data["request_task_complexity_rates"] = {}

            # Ensure list fields are lists
            for key in ("multi_agent_models", "multi_agent_names"):
                if key in yaml_data and not isinstance(yaml_data[key], list):
                    yaml_data[key] = []

        instance = cls(**yaml_data) if yaml_data else cls()

        # Allow TAVILY_API_KEY env var to override empty YAML value
        if not instance.tavily_api_key:
            import os

            env_key = os.environ.get("TAVILY_API_KEY", "").strip()
            if env_key:
                instance.tavily_api_key = env_key

        return instance

    # Global settings instance


settings = Settings.from_yaml()
