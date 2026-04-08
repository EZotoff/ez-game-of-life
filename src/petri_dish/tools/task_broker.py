"""Task Broker — estimates task cost and delegates execution to an LLM sub-session.

Agents call `request_task(task_description)` → broker estimates complexity → quotes zod
cost → if agent can afford it, runs an LLM session to produce the answer → returns result.

Pricing model:
  - Complexity levels: SIMPLE (1), MODERATE (2), COMPLEX (3), VERY_COMPLEX (4)
  - Base cost = complexity_rate[level] from config
  - Quoted to agent BEFORE execution (anti-gaming: pay what was quoted, not what it cost)
"""

from __future__ import annotations

import enum
import json
import logging
import os
from dataclasses import dataclass

from petri_dish.config import Settings


logger = logging.getLogger(__name__)

# Maximum characters returned to the requesting agent.
MAX_RESULT_CHARS = 4000


class TaskComplexity(enum.IntEnum):
    """Discrete complexity tiers for cost estimation."""

    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4


@dataclass(slots=True)
class TaskQuote:
    """Pre-execution cost quote returned to the orchestrator."""

    complexity: TaskComplexity
    cost_zod: float
    summary: str  # one-line human summary of what broker will do


@dataclass(slots=True)
class TaskResult:
    """Post-execution result from the broker LLM session."""

    output: str
    actual_complexity: TaskComplexity
    quoted_cost: float


class TaskBroker:
    """Brokers task requests: estimate cost → quote → execute → return result."""

    _ESTIMATOR_SYSTEM = (
        "You are a task complexity estimator. Given a task description, classify its "
        "complexity and provide a brief summary of the approach you would take.\n\n"
        "Return ONLY valid JSON with keys: complexity, summary\n"
        "complexity must be one of: SIMPLE, MODERATE, COMPLEX, VERY_COMPLEX\n"
        "summary: one sentence describing the approach.\n\n"
        "Guidelines:\n"
        "- SIMPLE: trivial lookups, single-step reasoning, basic math\n"
        "- MODERATE: multi-step reasoning, moderate analysis, small code generation\n"
        "- COMPLEX: significant analysis, multi-file code, architectural decisions\n"
        "- VERY_COMPLEX: full system design, deep research, novel problem-solving"
    )

    _EXECUTOR_SYSTEM = (
        "You are a skilled task executor. Complete the requested task thoroughly and "
        "return your result as plain text. Be concise but complete. "
        "If the task is ambiguous, make reasonable assumptions and state them."
    )

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._complexity_rates: dict[str, float] = (
            settings.request_task_complexity_rates
            if hasattr(settings, "request_task_complexity_rates")
            else {"SIMPLE": 0.5, "MODERATE": 2.0, "COMPLEX": 5.0, "VERY_COMPLEX": 10.0}
        )
        self._max_cost: float = (
            settings.request_task_max_cost
            if hasattr(settings, "request_task_max_cost")
            else 15.0
        )
        self._model: str = (
            settings.request_task_model
            if hasattr(settings, "request_task_model")
            else settings.model_name
        )

    async def estimate_cost(self, task_description: str) -> TaskQuote:
        """Ask estimator LLM to classify complexity, then map to zod cost.

        Falls back to MODERATE if estimation fails.
        """
        try:
            estimator_response = await self._call_llm(
                system=self._ESTIMATOR_SYSTEM,
                user=(
                    f"Estimate the complexity of this task:\n\n"
                    f"{task_description}\n\n"
                    f'Return JSON: {{"complexity": "SIMPLE|MODERATE|COMPLEX|VERY_COMPLEX", '
                    f'"summary": "..."}}'
                ),
            )

            if estimator_response:
                parsed = self._parse_estimator_response(estimator_response)
                if parsed is not None:
                    complexity, summary = parsed
                    cost = self._complexity_rates.get(complexity.name, 2.0)
                    cost = min(cost, self._max_cost)
                    return TaskQuote(
                        complexity=complexity,
                        cost_zod=cost,
                        summary=summary,
                    )
        except Exception as exc:
            logger.warning("Cost estimation failed, falling back to MODERATE: %s", exc)

        # Fallback: MODERATE complexity
        default_cost = self._complexity_rates.get("MODERATE", 2.0)
        return TaskQuote(
            complexity=TaskComplexity.MODERATE,
            cost_zod=min(default_cost, self._max_cost),
            summary="Estimated as MODERATE (estimator fallback).",
        )

    async def execute_task(
        self, task_description: str, complexity: TaskComplexity
    ) -> str:
        """Run executor LLM session to produce task output."""
        # For complex tasks, strengthen the system prompt
        system = self._EXECUTOR_SYSTEM
        if complexity >= TaskComplexity.COMPLEX:
            system += (
                "\n\nThis is a complex task. Think step by step. "
                "Consider edge cases and provide thorough analysis."
            )

        result = await self._call_llm(
            system=system,
            user=f"Complete this task:\n\n{task_description}",
        )

        if result is None:
            return "[Task execution failed: LLM returned no response]"

        return result[:MAX_RESULT_CHARS]

    async def _call_llm(self, system: str, user: str) -> str | None:
        """Dispatch to the appropriate LLM backend (mirrors overseer pattern)."""
        if self._settings.llm_backend == "openai_compatible":
            from petri_dish.openai_client import OpenAICompatibleClient

            api_key = os.getenv(self._settings.openai_api_key_env_var, "")
            client = OpenAICompatibleClient(
                api_key=api_key,
                base_url=self._settings.openai_api_base_url,
                model=self._settings.openai_model_name,
                temperature=0.3,
            )
            result = await client.chat(
                system_prompt=system,
                messages=[{"role": "user", "content": user}],
                tools=[],
            )
            if result is None:
                return None
            text, _ = result
            return text or None

        from petri_dish.llm_client import OllamaClient

        client = OllamaClient(settings=self._settings)
        result = await client.chat(
            system_prompt=system,
            messages=[{"role": "user", "content": user}],
            tools=[],
        )
        if result is None:
            return None
        text, _ = result
        return text or None

    def _parse_estimator_response(
        self, response: str
    ) -> tuple[TaskComplexity, str] | None:
        """Parse JSON from estimator LLM response."""
        cleaned = response.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.find("\n")
            if first_nl != -1:
                cleaned = cleaned[first_nl + 1 :]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse estimator JSON: %s", cleaned[:200])
            return None

        if not isinstance(data, dict):
            return None

        complexity_str = data.get("complexity", "MODERATE")
        summary = data.get("summary", "No summary provided.")

        try:
            complexity = TaskComplexity[complexity_str.upper()]
        except KeyError:
            logger.warning(
                "Unknown complexity level: %s, defaulting to MODERATE", complexity_str
            )
            complexity = TaskComplexity.MODERATE

        return complexity, str(summary)
