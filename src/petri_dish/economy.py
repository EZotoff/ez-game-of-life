"""Agent reserve system for Petri Dish agent simulation.

Manages zod balance with grant/decay mechanics and degradation levels.
Every zod transaction is logged. Failed tool calls still consume zod.
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


class AgentReserve:
    """Manages agent zod balance with grant/decay mechanics and degradation tiers.

    Parameters are loaded from Settings (config.yaml). Zod is stored as float
    for precision. Every transaction is logged.

    Degradation tiers:
        - premium: balance > 60% of initial
        - balanced: balance between 30%-60% of initial
        - eco: balance < 30% of initial
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize reserve from config settings.

        Args:
            settings: Settings instance. If None, loads from config.yaml.
        """
        if settings is None:
            settings = Settings.from_yaml()

        self.initial_zod: float = float(settings.initial_zod)
        self.balance: float = self.initial_zod
        self.decay_rate_per_turn: float = float(settings.decay_rate_per_turn)
        self.starvation_turns: int = int(settings.starvation_turns)
        self.lifetime_zod_earned: float = self.initial_zod

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
            "AgentReserve initialized: balance=%.2f, decay_rate=%.4f, thresholds(premium=%.0f%%, balanced=%.0f%%, eco=<%.0f%%), starvation_turns=%d",
            self.balance,
            self.decay_rate_per_turn,
            self._premium_threshold * 100,
            self._balanced_threshold * 100,
            self._balanced_threshold * 100,
            self.starvation_turns,
        )

    def consume(self, turns: int = 1) -> float:
        """Consume zod for inference turns.

        No-op when agent is STRIPPED — no decay rate charged in stripped state.
        """
        if self.state == AgentState.STRIPPED:
            return self.balance

        cost = self.decay_rate_per_turn * turns
        old_balance = self.balance
        self.balance = max(0.0, self.balance - cost)
        logger.info(
            "CONSUME: %.4f zod (turns=%d, rate=%.4f) | %.2f -> %.2f",
            cost,
            turns,
            self.decay_rate_per_turn,
            old_balance,
            self.balance,
        )
        return self.balance

    def grant(self, amount: float) -> float:
        """Grant earned zod to balance.

        Args:
            amount: Zod to add.

        Returns:
            New balance after grant.
        """
        old_balance = self.balance
        self.balance = max(0.0, self.balance + amount)
        if amount > 0:
            self.lifetime_zod_earned += amount
        logger.info(
            "GRANT: +%.4f zod | %.2f -> %.2f",
            amount,
            old_balance,
            self.balance,
        )
        return self.balance

    def get_balance(self) -> float:
        """Return current zod balance."""
        return self.balance

    def is_depleted(self) -> bool:
        """Check if zod is depleted (balance <= 0)."""
        return self.balance <= 0

    def get_degradation_level(self) -> str:
        """Return current degradation tier based on balance ratio."""
        if self.initial_zod <= 0:
            return "eco"

        ratio = self.balance / self.initial_zod

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
        self.grant(amount)
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


class SharedReserve:
    """Multi-agent reserve with common pool and per-agent reserves.

    Manages N AgentReserve instances, a shared common pool for
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
        self.agent_reserves: dict[str, AgentReserve] = {}
        self.agent_debt: dict[str, float] = {}
        self.spectator_counters: dict[str, int] = {}

        ids = agent_ids or [f"agent-{i}" for i in range(settings.multi_agent_count)]
        for agent_id in ids:
            self.agent_reserves[agent_id] = AgentReserve(settings)
            self.agent_debt[agent_id] = 0.0
            self.spectator_counters[agent_id] = 0

    def consume(self, agent_id: str, turns: int = 1) -> float:
        return self.agent_reserves[agent_id].consume(turns)

    def grant(self, agent_id: str, amount: float) -> float:
        economy = self.agent_reserves[agent_id]
        if amount > 0 and self.agent_debt.get(agent_id, 0) > 0:
            garnish = min(
                self.agent_debt[agent_id],
                amount * self.settings.multi_agent_debt_garnish_pct,
            )
            self.agent_debt[agent_id] -= garnish
            self.common_pool += garnish
            amount -= garnish
        return economy.grant(amount)

    def handle_death(self, agent_id: str) -> float:
        economy = self.agent_reserves[agent_id]
        balance = economy.get_balance()

        # Penalty base uses lifetime earnings (not just current balance) so death
        # is costly even when the agent has spent down to 0 before dying.
        penalty_base = max(
            balance,
            economy.lifetime_zod_earned,
            self.settings.initial_zod,
        )
        decay = penalty_base * self.settings.multi_agent_decay_pct
        salvage = penalty_base * self.settings.multi_agent_salvage_pct

        total_penalty = decay + salvage
        if balance >= total_penalty:
            economy.grant(-total_penalty)
        else:
            economy.grant(-balance)
            self.agent_debt[agent_id] += total_penalty - balance

        self.common_pool += salvage

        economy.state = AgentState.DEAD
        economy.starvation_counter = 0
        self.spectator_counters[agent_id] = 0
        logger.info(
            "DEATH: %s forfeited %.2f of %.2f lifetime (decayed=%.2f, salvage=%.2f, debt=%.2f)",
            agent_id,
            total_penalty,
            penalty_base,
            decay,
            salvage,
            self.agent_debt[agent_id],
        )
        return salvage

    def tick_spectator(self, agent_id: str) -> bool:
        if not self.agent_reserves[agent_id].is_dead():
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
        economy = self.agent_reserves[agent_id]
        if not economy.is_dead():
            return False
        cooldown = self.settings.multi_agent_spectator_rounds
        if self.spectator_counters.get(agent_id, 0) < cooldown:
            return False
        fee = self.settings.multi_agent_reentry_fee
        if self.common_pool < fee:
            return False
        self.common_pool -= fee
        economy.grant(fee)
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
        return self.agent_reserves[agent_id].get_balance()

    def get_agent_state(self, agent_id: str) -> AgentState:
        return self.agent_reserves[agent_id].state

    def get_common_pool(self) -> float:
        return self.common_pool

    def get_agent_ids(self) -> list[str]:
        return list(self.agent_reserves.keys())

    def get_agent_economy(self, agent_id: str) -> AgentReserve:
        return self.agent_reserves[agent_id]

    def get_agent_debt(self, agent_id: str) -> float:
        return self.agent_debt.get(agent_id, 0.0)

    def is_agent_stripped(self, agent_id: str) -> bool:
        return self.agent_reserves[agent_id].is_stripped()

    def is_agent_dead(self, agent_id: str) -> bool:
        return self.agent_reserves[agent_id].is_dead()

    def transition_agent_to_stripped(self, agent_id: str) -> bool:
        return self.agent_reserves[agent_id].transition_to_stripped()

    def tick_agent_starvation(self, agent_id: str) -> AgentState:
        return self.agent_reserves[agent_id].tick_starvation()

    def get_agent_starvation_remaining(self, agent_id: str) -> int:
        return self.agent_reserves[agent_id].get_starvation_remaining()

    def get_living_agents(self) -> list[str]:
        return [aid for aid, econ in self.agent_reserves.items() if not econ.is_dead()]

    def get_agent_summaries(self) -> list[dict]:
        return [
            {
                "agent_id": aid,
                "balance": econ.get_balance(),
                "state": econ.state.value,
                "degradation": econ.get_degradation_level(),
                "starvation_remaining": econ.get_starvation_remaining(),
            }
            for aid, econ in self.agent_reserves.items()
        ]
