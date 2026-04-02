from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, cast
from uuid import uuid4

ScoutVerdict = Literal["supported", "contradicted", "unclear", "no_sources"]
FileFamily = Literal["csv", "json", "log"]


def _to_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast(list[object], value)
    return [str(item) for item in items]


def _to_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _decode_json_object(raw: str) -> dict[str, object]:
    payload_obj = cast(object, json.loads(raw))
    if not isinstance(payload_obj, dict):
        raise ValueError("ScoutReport JSON must decode to object")

    payload_raw = cast(dict[object, object], payload_obj)
    payload: dict[str, object] = {}
    for key, value in payload_raw.items():
        payload[str(key)] = value
    return payload


@dataclass(frozen=True)
class ScoutReport:
    report_id: str
    requesting_agent_id: str
    timestamp: datetime
    claimed_pattern: str
    output_summary: str
    file_family: FileFamily
    queries_executed: list[str]
    results_found: int
    result_snippets: list[str]
    confidence: float
    verdict: ScoutVerdict
    reasoning: str
    suggested_bonus: float

    def to_dict(self) -> dict[str, object]:
        return {
            "report_id": self.report_id,
            "requesting_agent_id": self.requesting_agent_id,
            "timestamp": self.timestamp.isoformat(),
            "claimed_pattern": self.claimed_pattern,
            "output_summary": self.output_summary,
            "file_family": self.file_family,
            "queries_executed": self.queries_executed,
            "results_found": self.results_found,
            "result_snippets": self.result_snippets,
            "confidence": self.confidence,
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "suggested_bonus": self.suggested_bonus,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ScoutReport":
        file_family_raw = str(data["file_family"])
        if file_family_raw not in {"csv", "json", "log"}:
            raise ValueError(f"Invalid file_family: {file_family_raw}")
        verdict_raw = str(data.get("verdict", "no_sources"))
        if verdict_raw not in {"supported", "contradicted", "unclear", "no_sources"}:
            raise ValueError(f"Invalid verdict: {verdict_raw}")
        return cls(
            report_id=str(data["report_id"]),
            requesting_agent_id=str(data["requesting_agent_id"]),
            timestamp=datetime.fromisoformat(str(data["timestamp"])),
            claimed_pattern=str(data["claimed_pattern"]),
            output_summary=str(data["output_summary"]),
            file_family=cast(FileFamily, file_family_raw),
            queries_executed=_to_str_list(data.get("queries_executed", [])),
            results_found=_to_int(data.get("results_found", 0)),
            result_snippets=_to_str_list(data.get("result_snippets", [])),
            confidence=_to_float(data.get("confidence", 0.0)),
            verdict=cast(ScoutVerdict, verdict_raw),
            reasoning=str(data.get("reasoning", "")),
            suggested_bonus=_to_float(data.get("suggested_bonus", 0.0)),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ScoutReport":
        return cls.from_dict(_decode_json_object(raw))


def build_scout_report(
    requesting_agent_id: str,
    claimed_pattern: str,
    output_summary: str,
    file_family: FileFamily,
    queries_executed: list[str],
    result_snippets: list[str],
    confidence: float,
    verdict: ScoutVerdict,
    reasoning: str,
    suggested_bonus: float = 0.0,
) -> ScoutReport:
    bounded_confidence = max(0.0, min(1.0, float(confidence)))
    bounded_bonus = max(0.0, min(0.15, float(suggested_bonus)))
    return ScoutReport(
        report_id=str(uuid4()),
        requesting_agent_id=requesting_agent_id,
        timestamp=datetime.now(timezone.utc),
        claimed_pattern=claimed_pattern,
        output_summary=output_summary,
        file_family=file_family,
        queries_executed=queries_executed,
        results_found=len(result_snippets),
        result_snippets=result_snippets,
        confidence=bounded_confidence,
        verdict=verdict,
        reasoning=reasoning,
        suggested_bonus=bounded_bonus,
    )
