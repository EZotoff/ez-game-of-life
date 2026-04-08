from __future__ import annotations

import json
import os
from typing import Any

from petri_dish.config import Settings
from petri_dish.llm_client import OllamaClient
from petri_dish.logging_db import LoggingDB
from petri_dish.openai_client import OpenAICompatibleClient


class Overseer:
    def __init__(self, settings: Settings, logging_db: LoggingDB):
        self._settings = settings
        self._db = logging_db
        self._archive: list[dict[str, Any]] = []
        self._evaluation_interval: int = max(
            1, int(settings.overseer_evaluation_interval)
        )

    async def maybe_evaluate(
        self,
        run_id: str,
        turn: int,
        trait_vectors: dict[str, dict[str, float | dict[str, float]]] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._settings.overseer_enabled:
            return []
        if turn <= 0 or (turn % self._evaluation_interval) != 0:
            return []

        prompt = self._build_evaluation_prompt(
            run_id, turn, trait_vectors=trait_vectors or {}
        )
        response = await self._call_llm(prompt)
        evaluations = self._parse_response(response)
        if not evaluations:
            return []

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
            tags = [str(x) for x in item.get("tags", []) if isinstance(x, str)]
            normalized = {
                "agent_id": agent_id,
                "bonus": applied,
                "reasoning": reasoning,
                "tags": tags,
            }
            capped.append(normalized)
            total += applied
            self._db.log_overseer_evaluation(
                run_id=run_id,
                turn=turn,
                agent_id=agent_id,
                bonus=applied,
                reasoning=reasoning,
                tags=tags,
                evaluation_json=json.dumps(normalized),
            )

        self._update_archive(capped)
        return capped

    def _build_evaluation_prompt(
        self,
        run_id: str,
        turn: int,
        trait_vectors: dict[str, dict[str, float | dict[str, float]]] | None = None,
    ) -> str:
        all_actions = self._db.get_actions(run_id)
        recent = [
            a
            for a in all_actions
            if int(a.get("turn", 0)) > max(0, turn - self._evaluation_interval)
        ]
        agent_ids = sorted(
            {str(a.get("agent_id")) for a in recent if a.get("agent_id")}
        )
        if not agent_ids:
            agent_ids = ["agent"]

        history_by_agent: dict[str, list[dict[str, Any]]] = {}
        for agent_id in agent_ids:
            history_by_agent[agent_id] = self._db.get_agent_history(
                run_id, agent_id, last_n_turns=10
            )

        recent_without_agent = [
            {
                "turn": int(a.get("turn", 0)),
                "tool_name": str(a.get("tool_name", "")),
                "tool_args": a.get("tool_args"),
                "result": str(a.get("result", ""))[:500],
            }
            for a in recent
            if not a.get("agent_id")
        ]

        payload = {
            "run_id": run_id,
            "turn": turn,
            "evaluation_interval": self._evaluation_interval,
            "agent_ids": agent_ids,
            "recent_actions": recent_without_agent,
            "agent_histories": history_by_agent,
            "archive_summary": self._archive[-30:],
            "trait_vectors": trait_vectors or {},
        }

        return (
            "You are Overseer. You observe agents in a shared sandbox and reward "
            "genuinely novel behavior. Be conservative — most turns deserve no bonus.\n\n"
            "THREE INVARIANTS — ask these every evaluation window:\n"
            "1. NOVEL: Is this behavior genuinely new relative to the archive and "
            "trait vectors, or just a minor variation of something already seen?\n"
            "2. SHAPING: Does it alter the interaction structure — how agents "
            "coordinate, compete, or use the environment — or is it cosmetic?\n"
            "3. PERSISTENT: Does the behavior recur, propagate, or create "
            "downstream consequences, or is it a one-turn gimmick?\n\n"
            "If the answer to any invariant is clearly NO, award no bonus.\n\n"
            "OPTIONAL LENSES (not scorecards — just ways to notice novelty):\n"
            "- Coordination patterns (alliances, signaling, resource sharing)\n"
            "- Adaptation to economic pressure (zod conservation, risk strategies)\n"
            "- Environmental exploitation (tool combos, file strategies)\n\n"
            "REJECT: self-labeled cleverness, verbose justifications, theatrical "
            "roleplay shifts with no behavioral consequence, random chaos, and "
            "one-turn stunts that leave no trace.\n\n"
            "Describe novelty in your own words. Do not name dimensions.\n\n"
            "Return strict JSON array only:\n"
            '[{"agent_id": str, "bonus": float, "reasoning": str, "tags": [str]}]\n'
            "If no agent deserves a bonus this window, return [].\n"
            "Bonuses are capped in code — you cannot exceed limits.\n\n"
            f"Context:\n{json.dumps(payload, ensure_ascii=False)}"
        )

    async def _call_llm(self, prompt: str) -> str:
        if self._settings.llm_backend == "openai_compatible":
            api_key = os.getenv(self._settings.openai_api_key_env_var, "")
            client = OpenAICompatibleClient(
                api_key=api_key,
                base_url=self._settings.openai_api_base_url,
                model=self._settings.openai_model_name,
                temperature=self._settings.default_temperature,
            )
            result = await client.chat(
                system_prompt="You observe multi-agent simulations and identify genuinely novel behavior. Be skeptical and conservative. Return only valid JSON.",
                messages=[{"role": "user", "content": prompt}],
                tools=[],
            )
            if result is None:
                return "[]"
            text, _ = result
            return text or "[]"

        client = OllamaClient(settings=self._settings)
        result = await client.chat(
            system_prompt="You observe multi-agent simulations and identify genuinely novel behavior. Be skeptical and conservative. Return only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
            tools=[],
        )
        if result is None:
            return "[]"
        text, _ = result
        return text or "[]"

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences (```json ... ```) from LLM response."""
        stripped = text.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            stripped = stripped.strip()
        return stripped

    def _parse_response(self, response: str) -> list[dict[str, Any]]:
        cleaned = self._strip_code_fences(response)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed, dict) and isinstance(parsed.get("evaluations"), list):
            parsed = parsed["evaluations"]

        if not isinstance(parsed, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            agent_id = str(item.get("agent_id", "")).strip()
            if not agent_id:
                continue
            tags_obj = item.get("tags", [])
            tags = [str(x) for x in tags_obj] if isinstance(tags_obj, list) else []
            try:
                bonus = float(item.get("bonus", 0.0))
            except (TypeError, ValueError):
                bonus = 0.0
            normalized.append(
                {
                    "agent_id": agent_id,
                    "bonus": bonus,
                    "reasoning": str(item.get("reasoning", "")),
                    "tags": tags,
                }
            )
        return normalized

    def _update_archive(self, evaluations: list[dict[str, Any]]) -> None:
        for item in evaluations:
            self._archive.append(
                {
                    "agent_id": item.get("agent_id", ""),
                    "reasoning": item.get("reasoning", ""),
                    "tags": list(item.get("tags", [])),
                }
            )
