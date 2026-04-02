"""EndorphinEngine — mechanistic behavioral trait tracking for Petri Dish agents.

Maintains per-agent trait vectors updated from observed outcomes.
Expresses traits as natural language 'instincts' injected into the system prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from petri_dish.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class TraitVector:
    """Bounded trait vector for a single agent.

    Each trait is clamped to [-1.0, 1.0]. Updated via EMA.
    file_family_affinity is a dict of file-type → float (also bounded).
    """

    curiosity: float = 0.0
    thrift: float = 0.0
    sociability: float = 0.0
    persistence: float = 0.0
    shell_affinity: float = 0.0
    file_family_affinity: dict[str, float] = field(
        default_factory=lambda: {"csv": 0.0, "json": 0.0, "log": 0.0}
    )

    @staticmethod
    def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _is_scalar_trait(name: str) -> bool:
        return name in {
            "curiosity",
            "thrift",
            "sociability",
            "persistence",
            "shell_affinity",
        }

    def _get_scalar_trait(self, name: str) -> float:
        if name == "curiosity":
            return self.curiosity
        if name == "thrift":
            return self.thrift
        if name == "sociability":
            return self.sociability
        if name == "persistence":
            return self.persistence
        if name == "shell_affinity":
            return self.shell_affinity
        return 0.0

    def _set_scalar_trait(self, name: str, value: float) -> None:
        clamped = self._clamp(value)
        if name == "curiosity":
            self.curiosity = clamped
        elif name == "thrift":
            self.thrift = clamped
        elif name == "sociability":
            self.sociability = clamped
        elif name == "persistence":
            self.persistence = clamped
        elif name == "shell_affinity":
            self.shell_affinity = clamped

    def update_trait(self, name: str, delta: float, ema_alpha: float = 0.3) -> None:
        """Update a single trait via EMA."""
        if not self._is_scalar_trait(name):
            return
        current = self._get_scalar_trait(name)
        new_val = current + ema_alpha * (delta - current)
        self._set_scalar_trait(name, new_val)

    def update_file_family(
        self, family: str, delta: float, ema_alpha: float = 0.3
    ) -> None:
        """Update a file family affinity."""
        current = self.file_family_affinity.get(family, 0.0)
        new_val = current + ema_alpha * (delta - current)
        self.file_family_affinity[family] = self._clamp(new_val)

    def decay(self, factor: float = 0.95) -> None:
        """Decay all traits toward 0."""
        for name in (
            "curiosity",
            "thrift",
            "sociability",
            "persistence",
            "shell_affinity",
        ):
            self._set_scalar_trait(name, self._get_scalar_trait(name) * factor)
        for family in self.file_family_affinity:
            self.file_family_affinity[family] = self._clamp(
                self.file_family_affinity[family] * factor
            )

    def rebirth_carryover(self, factor: float = 0.85) -> None:
        """Apply rebirth damping — traits persist but weaker."""
        self.decay(factor=factor)


_TRAIT_INSTINCTS: list[tuple[str, float, str, str]] = [
    (
        "curiosity",
        0.3,
        "You feel drawn to explore new files and areas.",
        "You feel cautious about venturing into unknown areas.",
    ),
    (
        "thrift",
        0.3,
        "Expensive operations feel risky — you instinctively prefer cheap tools.",
        "You feel bold about spending zod on powerful operations.",
    ),
    (
        "sociability",
        0.3,
        "You feel a strong urge to coordinate with other agents.",
        "You feel self-reliant — others seem less relevant.",
    ),
    (
        "persistence",
        0.3,
        "When something doesn't work, you feel compelled to try again differently.",
        "When stuck, you feel drawn to switch to a completely different approach.",
    ),
    (
        "shell_affinity",
        0.3,
        "Code execution feels natural — you lean toward python_exec and shell_exec.",
        "Direct file operations feel more reliable than shell commands.",
    ),
]


class EndorphinEngine:
    """Manages per-agent trait vectors and generates instinct prompts.

    Observes agent actions and outcomes, updates traits mechanically,
    and produces natural language instincts for the system prompt.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        if settings is None:
            settings = Settings.from_yaml()
        self._traits: dict[str, TraitVector] = {}
        self._settings: Settings = settings
        self._ema_alpha: float = float(getattr(settings, "endorphin_ema_alpha", 0.3))
        self._decay_factor: float = float(
            getattr(settings, "endorphin_decay_factor", 0.95)
        )
        self._rebirth_factor: float = float(
            getattr(settings, "endorphin_rebirth_factor", 0.85)
        )
        self._instinct_threshold: float = float(
            getattr(settings, "endorphin_instinct_threshold", 0.3)
        )
        self._recent_tools: dict[str, list[str]] = {}
        self._round_rewards: dict[str, float] = {}

    def register_agent(self, agent_id: str) -> None:
        """Register a new agent with default trait vector."""
        self._traits[agent_id] = TraitVector()
        self._recent_tools[agent_id] = []
        self._round_rewards[agent_id] = 0.0

    def get_traits(self, agent_id: str) -> TraitVector:
        """Get trait vector for an agent."""
        if agent_id not in self._traits:
            self.register_agent(agent_id)
        return self._traits[agent_id]

    def observe_tool_use(self, agent_id: str, tool_name: str) -> None:
        """Observe a tool usage event."""
        if agent_id not in self._recent_tools:
            self._recent_tools[agent_id] = []
        self._recent_tools[agent_id].append(tool_name)

        traits = self.get_traits(agent_id)

        if tool_name in ("shell_exec", "python_exec"):
            traits.update_trait(
                "shell_affinity",
                min(1.0, traits.shell_affinity + 0.1),
                self._ema_alpha,
            )
        elif tool_name in ("file_read", "file_write", "file_list"):
            traits.update_trait(
                "shell_affinity",
                max(-1.0, traits.shell_affinity - 0.05),
                self._ema_alpha,
            )

        if tool_name == "send_message":
            traits.update_trait(
                "sociability", min(1.0, traits.sociability + 0.15), self._ema_alpha
            )

        cost = float(self._settings.tool_costs.get(tool_name, 0.0))
        if cost >= 1.0:
            traits.update_trait(
                "thrift", max(-1.0, traits.thrift - 0.1), self._ema_alpha
            )
        elif cost == 0.0:
            traits.update_trait(
                "thrift", min(1.0, traits.thrift + 0.05), self._ema_alpha
            )

    def observe_reward(self, agent_id: str, amount: float, filename: str = "") -> None:
        """Observe a reward earned by an agent."""
        if agent_id not in self._round_rewards:
            self._round_rewards[agent_id] = 0.0
        self._round_rewards[agent_id] += amount

        traits = self.get_traits(agent_id)

        traits.update_trait(
            "curiosity",
            min(1.0, traits.curiosity + 0.1 * (amount / 15.0)),
            self._ema_alpha,
        )

        traits.update_trait(
            "persistence", min(1.0, traits.persistence + 0.1), self._ema_alpha
        )

        if filename:
            for family in ("csv", "json", "log"):
                if f"_{family}_" in filename or filename.endswith(f".{family}"):
                    traits.update_file_family(
                        family,
                        min(
                            1.0,
                            traits.file_family_affinity.get(family, 0.0) + 0.15,
                        ),
                        self._ema_alpha,
                    )

    def observe_starvation(self, agent_id: str) -> None:
        """Observe that an agent is starving — weaken positive traits."""
        traits = self.get_traits(agent_id)
        for name in ("curiosity", "sociability", "persistence"):
            current = traits._get_scalar_trait(name)
            traits._set_scalar_trait(name, current * 0.8)

    def observe_death(self, agent_id: str) -> None:
        """Observe agent death — apply rebirth carryover."""
        if agent_id in self._traits:
            self._traits[agent_id].rebirth_carryover(self._rebirth_factor)

    def observe_reentry(self, agent_id: str) -> None:
        """Observe agent reentry — traits already dampened by rebirth_carryover."""
        logger.info(
            "ENDORPHIN: %s re-entered, traits: %s", agent_id, self._traits.get(agent_id)
        )

    def observe_empty_turn(self, agent_id: str) -> None:
        """Observe that agent did nothing useful this turn."""
        traits = self.get_traits(agent_id)
        traits.update_trait("persistence", traits.persistence - 0.05, self._ema_alpha)

    def end_round(self) -> None:
        """End-of-round processing: decay all traits, reset per-round counters."""
        for _, traits in self._traits.items():
            traits.decay(self._decay_factor)
        for agent_id in self._recent_tools:
            self._recent_tools[agent_id] = []
        for agent_id in self._round_rewards:
            self._round_rewards[agent_id] = 0.0

    def generate_instincts(self, agent_id: str) -> str:
        """Generate natural language instincts section for the system prompt.

        Returns empty string if no traits are strong enough to express.
        """
        traits = self.get_traits(agent_id)
        instinct_lines = []

        threshold = self._instinct_threshold

        for trait_name, _pos_thresh, pos_text, neg_text in _TRAIT_INSTINCTS:
            value = traits._get_scalar_trait(trait_name)
            if value > threshold:
                instinct_lines.append(pos_text)
            elif value < -threshold:
                instinct_lines.append(neg_text)

        if traits.file_family_affinity:
            max_family = max(
                traits.file_family_affinity.items(), key=lambda item: item[1]
            )[0]
            max_val = traits.file_family_affinity[max_family]
            if max_val > threshold:
                instinct_lines.append(f"You feel drawn toward {max_family} files.")

        if not instinct_lines:
            return ""

        return (
            "\nInstincts (subtle behavioral tendencies):\n"
            + "\n".join(f"- {line}" for line in instinct_lines)
            + "\n"
        )

    def get_trait_snapshot(self, agent_id: str) -> dict[str, object]:
        """Get a serializable snapshot of agent traits for logging."""
        traits = self.get_traits(agent_id)
        return {
            "agent_id": agent_id,
            "curiosity": round(traits.curiosity, 3),
            "thrift": round(traits.thrift, 3),
            "sociability": round(traits.sociability, 3),
            "persistence": round(traits.persistence, 3),
            "shell_affinity": round(traits.shell_affinity, 3),
            "file_family_affinity": {
                key: round(value, 3)
                for key, value in traits.file_family_affinity.items()
            },
        }
