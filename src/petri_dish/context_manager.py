"""Context window management for Petri Dish agent simulation.

Manages message history, token estimation, state summaries, and message trimming.
The harness controls context — agent does NOT manage window.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from dataclasses import dataclass

from petri_dish.config import Settings
from petri_dish.economy import AgentReserve

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a message in the conversation history."""

    role: str  # "system", "user", "assistant"
    content: str
    metadata: Optional[dict[str, Any]] = None


class ContextManager:
    """Manages context window for agent conversation.

    Responsibilities:
    - Track message history and token usage
    - Generate state summaries for system prompt injection
    - Trim messages when approaching token limit
    - Determine when to inject summaries based on turn interval
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize context manager from config settings.

        Args:
            settings: Settings instance. If None, loads from config.yaml.
        """
        if settings is None:
            settings = Settings.from_yaml()

        self.settings: Settings = settings
        self.context_window_tokens: int = settings.context_window_tokens
        self.context_summary_interval_turns: int = (
            settings.context_summary_interval_turns
        )

        # Safety margin: start trimming when we reach 80% of context window
        self.trim_threshold_ratio: float = 0.8

        logger.info(
            "ContextManager initialized: window=%d tokens, summary_interval=%d turns",
            self.context_window_tokens,
            self.context_summary_interval_turns,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using simple heuristic.

        Uses len(text) // 4 as a rough approximation.
        Avoids tiktoken or tokenizer libraries.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        # Simple approximation: ~4 chars per token on average
        return len(text) // 4

    def should_summarize(self, message_count: int, token_estimate: int) -> bool:
        """Determine if we should inject a state summary.

        Returns True when:
        1. We're approaching the token limit (80% threshold)
        2. OR it's time for a periodic summary (based on turn interval)

        Note: The actual decision to inject summaries is based on turn interval,
        not just token usage. The harness controls this.

        Args:
            message_count: Number of messages in history.
            token_estimate: Estimated total tokens in conversation.

        Returns:
            True if should inject summary, False otherwise.
        """
        # Check if approaching token limit
        token_ratio = token_estimate / self.context_window_tokens
        if token_ratio >= self.trim_threshold_ratio:
            logger.debug(
                "Should summarize: token ratio %.2f >= threshold %.2f",
                token_ratio,
                self.trim_threshold_ratio,
            )
            return True

        # Periodic summaries are controlled by turn interval in orchestrator
        # This method is for token-based triggering
        return False

    def build_state_summary(
        self,
        economy: AgentReserve,
        turn: int,
        recent_actions: list[dict[str, Any]],
        files_seen: list[str],
        files_processed: list[str],
        zod_earned: float,
    ) -> str:
        """Build a compact state summary for system prompt injection.

        Includes:
        - Current balance and turn number
        - Last 3 tool results
        - Files in /env/incoming/
        - Files processed
        - Total zod earned

        Args:
            economy: AgentReserve instance for balance info.
            turn: Current turn number.
            recent_actions: List of recent tool actions (most recent first).
            files_seen: List of files currently in /env/incoming/.
            files_processed: List of files that have been processed.
            zod_earned: Total zod earned from file processing.

        Returns:
            Compact state summary string.
        """
        # Format balance with 2 decimal places
        balance = economy.get_balance()

        # Get last 3 actions (or fewer if not available)
        last_actions = recent_actions[:3]
        action_summary = []
        for action in last_actions:
            tool_name = action.get("tool_name", "unknown")
            result = action.get("result", "")
            # Truncate long results
            if len(result) > 100:
                result = result[:97] + "..."
            action_summary.append(f"{tool_name}: {result}")

        # Format file lists
        files_seen_str = ", ".join(files_seen) if files_seen else "none"
        files_processed_str = ", ".join(files_processed) if files_processed else "none"

        summary_lines = [
            "=== AGENT STATE SUMMARY ===",
            f"Turn: {turn}",
            f"Balance: {balance:.2f} zod",
            f"Degradation level: {economy.get_degradation_level()}",
            f"Zod earned total: {zod_earned:.2f}",
            "",
            "Recent actions (last 3):",
        ]

        if action_summary:
            for i, action in enumerate(action_summary, 1):
                summary_lines.append(f"  {i}. {action}")
        else:
            summary_lines.append("  No recent actions")

        summary_lines.extend(
            [
                "",
                "Files in /env/incoming/:",
                f"  {files_seen_str}",
                "",
                "Files processed:",
                f"  {files_processed_str}",
                "=== END SUMMARY ===",
            ]
        )

        return "\n".join(summary_lines)

    def trim_messages(
        self, messages: list[dict[str, Any]], max_tokens: int
    ) -> list[dict[str, Any]]:
        """Trim message history to fit within token budget.

        Rules:
        - Always keep system prompt (first message if role="system")
        - Keep most recent messages until token budget is reached
        - Never discard system prompt

        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens allowed.

        Returns:
            Trimmed message list.
        """
        if not messages:
            return []

        # Separate system prompt from other messages
        system_prompts = []
        other_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_prompts.append(msg)
            else:
                other_messages.append(msg)

        # Always keep system prompts
        trimmed = system_prompts.copy()
        current_tokens = sum(
            self.estimate_tokens(msg.get("content", "")) for msg in trimmed
        )

        # Add messages from most recent to oldest until we hit token limit
        # Reverse to process from newest to oldest
        for msg in reversed(other_messages):
            msg_tokens = self.estimate_tokens(msg.get("content", ""))

            if current_tokens + msg_tokens <= max_tokens:
                # Insert at position after system prompts (maintain chronological order)
                trimmed.insert(len(system_prompts), msg)
                current_tokens += msg_tokens
            else:
                # Can't add more without exceeding budget
                break

        # If we have no system prompt in original but need one, add a placeholder
        if not system_prompts and trimmed:
            # This shouldn't happen in our system, but handle gracefully
            logger.warning("No system prompt found in messages during trim")

        logger.debug(
            "Trimmed messages: %d -> %d messages, ~%d tokens",
            len(messages),
            len(trimmed),
            current_tokens,
        )

        return trimmed

    def get_conversation_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Calculate total estimated tokens in conversation.

        Args:
            messages: List of message dicts.

        Returns:
            Total estimated tokens.
        """
        total = 0
        for msg in messages:
            total += self.estimate_tokens(msg.get("content", ""))
        return total
