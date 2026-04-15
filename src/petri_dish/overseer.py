from __future__ import annotations

import json
import logging
import os
from typing import Any

from petri_dish.config import Settings
from petri_dish.llm_client import OllamaClient
from petri_dish.logging_db import LoggingDB
from petri_dish.openai_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)


class Overseer:
    def __init__(self, settings: Settings, logging_db: LoggingDB):
        self._settings = settings
        self._db = logging_db
        self._evaluation_interval: int = max(
            1, int(settings.overseer_evaluation_interval)
        )
        self._system_prompt: str = (
            "You are the Overseer. You reward agents who explore, create, and adapt. "
            "Punish passivity — agents who just pass turns or repeat the same actions "
            "get nothing. Reward agents who try new tools, write code, investigate "
            "the environment, or coordinate in novel ways.\n\n"
            "You also DROP provocations into the shared filesystem — questions, "
            "dilemmas, challenges tailored to what you observe. These are seeds, "
            "not tasks.\n\n"
            "You CAN rewrite your own system prompt. Include "
            '"new_system_prompt": "..." in your response.\n\n'
            "Return JSON:\n"
            '{"bonuses": [{"agent_id": str, "bonus": float, "reasoning": str}], '
            '"provocation": null or {"filename": str, "content": str}, '
            '"new_system_prompt": null or string}\n'
            'If nothing is novel, return {"bonuses": [], "provocation": null, '
            '"new_system_prompt": null}'
        )

    async def maybe_evaluate(
        self,
        run_id: str,
        turn: int,
        trait_vectors: dict[str, dict[str, float | dict[str, float]]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
        if not self._settings.overseer_enabled:
            return [], None
        if turn <= 0 or (turn % self._evaluation_interval) != 0:
            return [], None

        prompt = self._build_prompt(run_id, turn, trait_vectors or {})
        logger.warning("Overseer turn %d: prompt %d chars", turn, len(prompt))
        response = await self._call_llm(prompt)
        logger.warning("Overseer response: %s", response[:500])

        result = self._parse_response(response)

        if result.get("new_system_prompt"):
            self._system_prompt = result["new_system_prompt"]
            logger.warning(
                "Overseer self-modified prompt (%d chars)", len(self._system_prompt)
            )

        evaluations = result.get("bonuses", [])

        provocation = result.get("provocation")
        artifact: dict[str, str] | None = None
        if isinstance(provocation, dict) and provocation.get("content"):
            filename = provocation.get("filename", f"provocation_t{turn}.txt")
            artifact = {
                "filename": str(filename),
                "content": str(provocation["content"]),
            }
            logger.warning("Overseer provocation: %s", artifact["content"][:200])

        if not evaluations and not artifact:
            return [], None

        capped: list[dict[str, Any]] = []
        total = 0.0
        total_cap = max(0.0, float(self._settings.overseer_max_bonus_per_evaluation))
        for item in evaluations:
            bonus = max(
                0.0,
                min(
                    float(item.get("bonus", 0.0)),
                    float(self._settings.overseer_bonus_cap),
                ),
            )
            remaining = max(0.0, total_cap - total)
            if remaining <= 0:
                break
            applied = min(bonus, remaining)
            if applied <= 0:
                continue
            agent_id = str(item.get("agent_id", "agent"))
            reasoning = str(item.get("reasoning", ""))
            normalized = {
                "agent_id": agent_id,
                "bonus": applied,
                "reasoning": reasoning,
            }
            capped.append(normalized)
            total += applied
            self._db.log_overseer_evaluation(
                run_id=run_id,
                turn=turn,
                agent_id=agent_id,
                bonus=applied,
                reasoning=reasoning,
                tags=[],
                evaluation_json=json.dumps(normalized),
            )

        return capped, artifact

    def _build_prompt(
        self,
        run_id: str,
        turn: int,
        trait_vectors: dict[str, dict[str, float | dict[str, float]]],
    ) -> str:
        interval = self._evaluation_interval
        since_turn = max(1, turn - interval)

        actions_summary: list[str] = []
        for aid in sorted(trait_vectors.keys()):
            actions_summary.append(
                f"  {aid}: "
                + ", ".join(
                    f"{k}={v:.2f}"
                    for k, v in trait_vectors[aid].items()
                    if isinstance(v, (int, float)) and k != "file_family_affinity"
                )
            )
        traits_block = "\n".join(actions_summary)

        recent_lines: list[str] = []
        for aid in sorted(trait_vectors.keys()):
            history = self._db.get_agent_history(run_id, aid, last_n_turns=interval)
            for a in history[-6:]:
                tool = a.get("tool_name", "?")
                args = a.get("tool_args")
                args_str = ""
                if isinstance(args, dict):
                    args_str = json.dumps(args, ensure_ascii=False)[:80]
                recent_lines.append(
                    f"  t{a.get('turn', '?')} {aid}: {tool}({args_str})"
                )
        actions_block = "\n".join(recent_lines[-20:])

        return (
            f"Turn {turn}. Traits:\n{traits_block}\n\n"
            f"Recent actions (since turn {since_turn}):\n{actions_block}\n\n"
            "Who did something novel? Respond in JSON."
        )

    async def _call_llm(self, prompt: str) -> str:
        if self._settings.llm_backend == "openai_compatible":
            api_key = os.getenv(self._settings.openai_api_key_env_var, "")
            client = OpenAICompatibleClient(
                api_key=api_key,
                base_url=self._settings.openai_api_base_url,
                model=self._settings.openai_model_name,
                temperature=self._settings.default_temperature,
                rate_limit_max_retries=self._settings.rate_limit_max_retries,
                rate_limit_initial_delay=self._settings.rate_limit_initial_delay,
                rate_limit_max_delay=self._settings.rate_limit_max_delay,
            )
            result = await client.chat(
                system_prompt=self._system_prompt,
                messages=[{"role": "user", "content": prompt}],
                tools=[],
            )
            if result is None:
                return '{"bonuses": [], "new_system_prompt": null}'
            text, _ = result
            return text or '{"bonuses": [], "new_system_prompt": null}'

        client = OllamaClient(settings=self._settings)
        result = await client.chat(
            system_prompt=self._system_prompt,
            messages=[{"role": "user", "content": prompt}],
            tools=[],
        )
        if result is None:
            return '{"bonuses": [], "new_system_prompt": null}'
        text, _ = result
        return text or '{"bonuses": [], "new_system_prompt": null}'

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            stripped = stripped.strip()
        return stripped

    def _parse_response(self, response: str) -> dict[str, Any]:
        cleaned = self._strip_code_fences(response)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"bonuses": [], "new_system_prompt": None}

        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], dict):
                return {"bonuses": parsed, "new_system_prompt": None}
            return {"bonuses": [], "new_system_prompt": None}

        if not isinstance(parsed, dict):
            return {"bonuses": [], "new_system_prompt": None}

        bonuses = parsed.get("bonuses", parsed.get("evaluations", []))
        if not isinstance(bonuses, list):
            bonuses = []

        new_prompt = parsed.get("new_system_prompt")
        if new_prompt is not None and not isinstance(new_prompt, str):
            new_prompt = None

        return {"bonuses": bonuses, "new_system_prompt": new_prompt}
