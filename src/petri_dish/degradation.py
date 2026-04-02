"""Graduated degradation toggle for Petri Dish agent simulation.

Manages model parameter degradation based on zod balance and economy mode.
Two modes:
  - visible: agent sees balance via check_balance tool, no model changes
  - degraded: at balance thresholds, model parameters change

Degradation tiers (configurable thresholds):
  - premium (balance > 60%): default model params
  - balanced (30-60%): num_ctx reduced to 4096, temperature +0.2
  - eco (< 30%): num_ctx reduced to 2048, temperature +0.5

Default mode: visible (Analyst dissent — degradation confounds results)
"""

from typing import Any, Dict, Literal
from petri_dish.config import Settings


class DegradationManager:
    """Manages model parameter degradation based on zod balance and economy mode.

    Parameters are loaded from Settings (config.yaml). When economy_mode is 'visible',
    always returns default parameters regardless of balance.

    Attributes:
        economy_mode: Either 'visible' or 'degraded'
        degradation_tiers: Dictionary mapping tier names to threshold ratios
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize degradation manager from config settings.

        Args:
            settings: Settings instance. If None, loads from config.yaml.
        """
        if settings is None:
            settings = Settings.from_yaml()

        self.economy_mode: Literal["visible", "degraded"] = settings.economy_mode
        self.degradation_tiers: Dict[str, float] = settings.degradation_tiers
        self.default_temperature: float = settings.default_temperature

        # Extract thresholds for easy access
        self.premium_threshold: float = self.degradation_tiers.get("premium", 0.6)
        self.balanced_threshold: float = self.degradation_tiers.get("balanced", 0.3)
        self.eco_threshold: float = self.degradation_tiers.get("eco", 0.0)

    def get_tier(self, balance: float, initial_zod: float) -> str:
        """Return current degradation tier based on balance ratio.

        Args:
            balance: Current zod balance
            initial_zod: Initial zod balance

        Returns:
            "premium" if balance > premium_threshold * initial_zod,
            "balanced" if balance >= balanced_threshold * initial_zod,
            "eco" if balance < balanced_threshold * initial_zod.
        """
        if initial_zod <= 0:
            return "eco"

        ratio = balance / initial_zod

        if ratio > self.premium_threshold:
            return "premium"
        elif ratio >= self.balanced_threshold:
            return "balanced"
        else:
            return "eco"

    def get_model_params(self, balance: float, initial_zod: float) -> Dict[str, Any]:
        """Return Ollama-compatible model parameters based on balance and mode.

        When economy_mode is 'visible', always returns empty dict (default params).
        When economy_mode is 'degraded', returns tier-specific parameters.

        Args:
            balance: Current zod balance
            initial_zod: Initial zod balance

        Returns:
            Dictionary of Ollama options. Empty dict for default parameters.
        """
        # In visible mode, always return default params regardless of balance
        if self.economy_mode == "visible":
            return {}

        # In degraded mode, apply tier-specific parameters
        tier = self.get_tier(balance, initial_zod)

        if tier == "premium":
            # Default parameters
            return {}
        elif tier == "balanced":
            # Reduced context, slightly higher temperature
            return {
                "num_ctx": 4096,
                "temperature": self.default_temperature + 0.2,
            }
        else:  # eco tier
            # Further reduced context, higher temperature
            return {
                "num_ctx": 2048,
                "temperature": self.default_temperature + 0.5,
            }
