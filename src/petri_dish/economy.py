"""Credit economy system for Petri Dish agent simulation.

Manages credit balance with earn/burn mechanics and degradation levels.
Every credit transaction is logged. Failed tool calls still cost credits.
"""

from __future__ import annotations

import logging
from enum import Enum

from petri_dish.config import Settings

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    ACTIVE = "active"
    STRIPPED = "stripped"
    DEAD = "dead"


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
        self.starvation_turns: int = int(settings.starvation_turns)

        self._degradation_tiers: dict[str, float] = settings.degradation_tiers
        self._premium_threshold: float = float(
            self._degradation_tiers.get("premium", 0.6)
        )
        self._balanced_threshold: float = float(
            self._degradation_tiers.get("balanced", 0.3)
        )

        self.state: AgentState = AgentState.ACTIVE
        self.starvation_counter: int = 0

        logger.info(
            "CreditEconomy initialized: balance=%.2f, burn_rate=%.4f, thresholds(premium=%.0f%%, balanced=%.0f%%, eco=<%.0f%%), starvation_turns=%d",
            self.balance,
            self.burn_rate_per_turn,
            self._premium_threshold * 100,
            self._balanced_threshold * 100,
            self._balanced_threshold * 100,
            self.starvation_turns,
        )

    def debit(self, turns: int = 1) -> float:
        """Deduct credits for inference turns.

        No-op when agent is STRIPPED — no burn rate charged in stripped state.
        """
        if self.state == AgentState.STRIPPED:
            return self.balance

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
        """Return current degradation tier based on balance ratio."""
        if self.initial_balance <= 0:
            return "eco"

        ratio = self.balance / self.initial_balance

        if ratio > self._premium_threshold:
            return "premium"
        elif ratio >= self._balanced_threshold:
            return "balanced"
        else:
            return "eco"

    def transition_to_stripped(self) -> bool:
        if self.state != AgentState.ACTIVE:
            logger.warning("transition_to_stripped called from %s, ignored", self.state)
            return False
        self.state = AgentState.STRIPPED
        self.starvation_counter = 0
        logger.info("STATE: ACTIVE -> STRIPPED (balance=%.2f)", self.balance)
        return True

    def tick_starvation(self) -> AgentState:
        self.starvation_counter += 1
        remaining = self.starvation_turns - self.starvation_counter
        logger.info(
            "STARVATION: tick %d/%d (remaining=%d, balance=%.2f)",
            self.starvation_counter,
            self.starvation_turns,
            remaining,
            self.balance,
        )
        if self.starvation_counter >= self.starvation_turns:
            self.state = AgentState.DEAD
            logger.info("STATE: STRIPPED -> DEAD (starvation)")
        return self.state

    def rescue(self, amount: float) -> bool:
        if self.state != AgentState.STRIPPED:
            logger.warning("rescue called from %s, ignored", self.state)
            return False
        if amount <= 0:
            return False
        self.credit(amount)
        self.state = AgentState.ACTIVE
        self.starvation_counter = 0
        logger.info(
            "STATE: STRIPPED -> ACTIVE (rescue +%.2f, balance=%.2f)",
            amount,
            self.balance,
        )
        return True

    def is_stripped(self) -> bool:
        return self.state == AgentState.STRIPPED

    def is_dead(self) -> bool:
        return self.state == AgentState.DEAD

    def get_starvation_remaining(self) -> int:
        return max(0, self.starvation_turns - self.starvation_counter)


class SharedEconomy:
    """Multi-agent economy with common pool and per-agent wallets.

    Manages N CreditEconomy instances, a shared common pool for
    salvage/re-entry, debt garnishing on earnings, and spectator
    cooldown after death.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        agent_ids: list[str] | None = None,
    ) -> None:
        if settings is None:
            settings = Settings.from_yaml()

        self.settings = settings
        self.common_pool: float = 0.0
        self.agent_economies: dict[str, CreditEconomy] = {}
        self.agent_debt: dict[str, float] = {}
        self.spectator_counters: dict[str, int] = {}

        ids = agent_ids or [f"agent-{i}" for i in range(settings.multi_agent_count)]
        for agent_id in ids:
            self.agent_economies[agent_id] = CreditEconomy(settings)
            self.agent_debt[agent_id] = 0.0
            self.spectator_counters[agent_id] = 0

    def debit(self, agent_id: str, turns: int = 1) -> float:
        return self.agent_economies[agent_id].debit(turns)

    def credit(self, agent_id: str, amount: float) -> float:
        economy = self.agent_economies[agent_id]
        if amount > 0 and self.agent_debt.get(agent_id, 0) > 0:
            garnish = min(
                self.agent_debt[agent_id],
                amount * self.settings.multi_agent_debt_garnish_pct,
            )
            self.agent_debt[agent_id] -= garnish
            self.common_pool += garnish
            amount -= garnish
        return economy.credit(amount)

    def handle_death(self, agent_id: str) -> float:
        economy = self.agent_economies[agent_id]
        balance = economy.get_balance()
        burn = balance * self.settings.multi_agent_burn_pct
        salvage = balance * self.settings.multi_agent_salvage_pct

        economy.credit(-(burn + salvage))
        self.common_pool += salvage

        economy.state = AgentState.DEAD
        economy.starvation_counter = 0
        self.spectator_counters[agent_id] = 0
        logger.info(
            "DEATH: %s forfeited %.2f (burned=%.2f, salvage=%.2f)",
            agent_id,
            burn + salvage,
            burn,
            salvage,
        )
        return salvage

    def tick_spectator(self, agent_id: str) -> bool:
        if not self.agent_economies[agent_id].is_dead():
            return False
        self.spectator_counters[agent_id] += 1
        cooldown = self.settings.multi_agent_spectator_rounds
        logger.info(
            "SPECTATOR: %s tick %d/%d",
            agent_id,
            self.spectator_counters[agent_id],
            cooldown,
        )
        return self.spectator_counters[agent_id] >= cooldown

    def reentry(self, agent_id: str) -> bool:
        economy = self.agent_economies[agent_id]
        if not economy.is_dead():
            return False
        cooldown = self.settings.multi_agent_spectator_rounds
        if self.spectator_counters.get(agent_id, 0) < cooldown:
            return False
        fee = self.settings.multi_agent_reentry_fee
        if self.common_pool < fee:
            return False
        self.common_pool -= fee
        economy.credit(fee)
        economy.state = AgentState.ACTIVE
        economy.starvation_counter = 0
        self.agent_debt[agent_id] = fee
        self.spectator_counters[agent_id] = 0
        logger.info(
            "REENTRY: %s paid %.2f from common pool, debt=%.2f",
            agent_id,
            fee,
            self.agent_debt[agent_id],
        )
        return True

    def get_agent_balance(self, agent_id: str) -> float:
        return self.agent_economies[agent_id].get_balance()

    def get_agent_state(self, agent_id: str) -> AgentState:
        return self.agent_economies[agent_id].state

    def get_common_pool(self) -> float:
        return self.common_pool

    def get_agent_ids(self) -> list[str]:
        return list(self.agent_economies.keys())

    def get_agent_economy(self, agent_id: str) -> CreditEconomy:
        return self.agent_economies[agent_id]

    def get_agent_debt(self, agent_id: str) -> float:
        return self.agent_debt.get(agent_id, 0.0)

    def is_agent_stripped(self, agent_id: str) -> bool:
        return self.agent_economies[agent_id].is_stripped()

    def is_agent_dead(self, agent_id: str) -> bool:
        return self.agent_economies[agent_id].is_dead()

    def transition_agent_to_stripped(self, agent_id: str) -> bool:
        return self.agent_economies[agent_id].transition_to_stripped()

    def tick_agent_starvation(self, agent_id: str) -> AgentState:
        return self.agent_economies[agent_id].tick_starvation()

    def get_agent_starvation_remaining(self, agent_id: str) -> int:
        return self.agent_economies[agent_id].get_starvation_remaining()

    def get_living_agents(self) -> list[str]:
        return [aid for aid, econ in self.agent_economies.items() if not econ.is_dead()]

    def get_agent_summaries(self) -> list[dict]:
        return [
            {
                "agent_id": aid,
                "balance": econ.get_balance(),
                "state": econ.state.value,
                "degradation": econ.get_degradation_level(),
                "starvation_remaining": econ.get_starvation_remaining(),
            }
            for aid, econ in self.agent_economies.items()
        ]
