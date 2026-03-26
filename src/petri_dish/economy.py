"""Credit economy system for Petri Dish agent simulation.

Manages credit balance with earn/burn mechanics and degradation levels.
Every credit transaction is logged. Failed tool calls still cost credits.
"""

from __future__ import annotations

import logging

from petri_dish.config import Settings

logger = logging.getLogger(__name__)


class CreditEconomy:
    """Manages agent credit balance with earn/burn mechanics and degradation tiers.

    Parameters are loaded from Settings (config.yaml). Credits are stored as float
    for precision. Every transaction is logged.

    Degradation tiers:
        - premium: balance > 60% of initial
        - balanced: balance between 30%-60% of initial
        - eco: balance < 30% of initial
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize economy from config settings.

        Args:
            settings: Settings instance. If None, loads from config.yaml.
        """
        if settings is None:
            settings = Settings.from_yaml()

        self.initial_balance: float = float(settings.initial_balance)
        self.balance: float = self.initial_balance
        self.burn_rate_per_turn: float = float(settings.burn_rate_per_turn)

        self._degradation_tiers: dict[str, float] = settings.degradation_tiers
        self._premium_threshold: float = float(
            self._degradation_tiers.get("premium", 0.6)
        )
        self._balanced_threshold: float = float(
            self._degradation_tiers.get("balanced", 0.3)
        )

        logger.info(
            "CreditEconomy initialized: balance=%.2f, burn_rate=%.4f, thresholds(premium=%.0f%%, balanced=%.0f%%, eco=<%.0f%%)",
            self.balance,
            self.burn_rate_per_turn,
            self._premium_threshold * 100,
            self._balanced_threshold * 100,
            self._balanced_threshold * 100,
        )

    def debit(self, turns: int = 1) -> float:
        """Deduct credits for inference turns.

        Failed tool calls still cost credits (compute was consumed).

        Args:
            turns: Number of turns to debit for.

        Returns:
            New balance after debit.
        """
        cost = self.burn_rate_per_turn * turns
        old_balance = self.balance
        self.balance -= cost
        logger.info(
            "DEBIT: %.4f credits (turns=%d, rate=%.4f) | %.2f -> %.2f",
            cost,
            turns,
            self.burn_rate_per_turn,
            old_balance,
            self.balance,
        )
        return self.balance

    def credit(self, amount: float) -> float:
        """Add earned credits to balance.

        Args:
            amount: Credits to add.

        Returns:
            New balance after credit.
        """
        old_balance = self.balance
        self.balance += amount
        logger.info(
            "CREDIT: +%.4f credits | %.2f -> %.2f",
            amount,
            old_balance,
            self.balance,
        )
        return self.balance

    def get_balance(self) -> float:
        """Return current credit balance."""
        return self.balance

    def is_depleted(self) -> bool:
        """Check if credits are depleted (balance <= 0)."""
        return self.balance <= 0

    def get_degradation_level(self) -> str:
        """Return current degradation tier based on balance ratio.

        Returns:
            "premium" if balance > 60% of initial,
            "balanced" if balance is 30%-60% of initial,
            "eco" if balance < 30% of initial.
        """
        if self.initial_balance <= 0:
            return "eco"

        ratio = self.balance / self.initial_balance

        if ratio > self._premium_threshold:
            return "premium"
        elif ratio >= self._balanced_threshold:
            return "balanced"
        else:
            return "eco"
