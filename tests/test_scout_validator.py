from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LoggingDB = import_module("petri_dish.logging_db").LoggingDB
build_scout_report = import_module("petri_dish.scout_report").build_scout_report
ScoutReplayValidator = import_module("petri_dish.scout_validator").ScoutReplayValidator
scout_reports_from_db = import_module(
    "petri_dish.scout_validator"
).scout_reports_from_db


def test_validate_replay_returns_consistent_result_structure() -> None:
    original = build_scout_report(
        requesting_agent_id="agent-1",
        claimed_pattern="timestamps are ISO-8601",
        output_summary="normalized log fields",
        file_family="log",
        queries_executed=["log iso8601 timestamp format"],
        result_snippets=["ISO timestamps are common in logs"],
        confidence=0.9,
        verdict="supported",
        reasoning="seed",
        suggested_bonus=0.1,
    )

    replay_conf = {
        "log iso8601 timestamp format": ("supported", 0.90),
        "best practice log iso8601 timestamp format": ("supported", 0.88),
        "official guidance log iso8601 timestamp format": ("supported", 0.86),
    }

    def fake_execute(query: str, pattern: str, file_family: str):
        verdict, confidence = replay_conf[query]
        return build_scout_report(
            requesting_agent_id="agent-1",
            claimed_pattern=pattern,
            output_summary=f"replay for {query}",
            file_family=file_family,
            queries_executed=[query],
            result_snippets=[f"source says {pattern}"],
            confidence=confidence,
            verdict=verdict,
            reasoning="replay",
            suggested_bonus=0.1,
        )

    validator = ScoutReplayValidator(fake_execute)
    replay = validator.validate_replay(original, num_replay_queries=3)

    assert replay.consistent is True
    assert replay.replay_count == 3
    assert replay.verdicts == ["supported", "supported", "supported"]
    assert replay.promotion_recommended is True
    assert replay.avg_confidence == (0.90 + 0.88 + 0.86) / 3
    assert 0.0 <= replay.confidence_variance <= 0.05


def test_generalization_and_candidate_generation_are_deterministic() -> None:
    def fake_execute(query: str, pattern: str, file_family: str):
        if "best practices" in query:
            verdict, confidence = "unclear", 0.55
        else:
            verdict, confidence = "supported", 0.8
        return build_scout_report(
            requesting_agent_id="agent-9",
            claimed_pattern=pattern,
            output_summary=query,
            file_family=file_family,
            queries_executed=[query],
            result_snippets=[f"{query} evidence"],
            confidence=confidence,
            verdict=verdict,
            reasoning="gen-check",
            suggested_bonus=0.0,
        )

    validator = ScoutReplayValidator(fake_execute)
    generalization = validator.check_generalization("schema required fields", "json")
    assert generalization.pattern == "schema required fields"
    assert generalization.file_family == "json"
    assert len(generalization.queries_used) == 3
    assert generalization.verdicts == ["supported", "supported", "unclear"]
    assert generalization.generalized is True

    source_report = build_scout_report(
        requesting_agent_id="agent-9",
        claimed_pattern="schema required fields",
        output_summary="output summary",
        file_family="json",
        queries_executed=["json schema required fields"],
        result_snippets=["required field list"],
        confidence=0.9,
        verdict="supported",
        reasoning="seed",
        suggested_bonus=0.15,
    )
    replay = validator.validate_replay(source_report, num_replay_queries=1)
    candidate = validator.generate_promotion_candidate(
        source_report,
        replay,
        generalization,
    )
    assert candidate.report_id == source_report.report_id
    assert candidate.promotion_candidate is False
    assert candidate.risk_assessment in {"medium", "high"}
    assert "generalized=True" in candidate.evidence_summary


def test_scout_reports_from_db_reuses_scout_report_model() -> None:
    db = LoggingDB(":memory:")
    db.connect()
    run_id = "validator-db-integration"
    db.log_run_start(run_id, {})

    report = build_scout_report(
        requesting_agent_id="agent-2",
        claimed_pattern="csv has stable columns",
        output_summary="3 rows processed",
        file_family="csv",
        queries_executed=["csv column consistency"],
        result_snippets=["columns should be consistent"],
        confidence=0.8,
        verdict="supported",
        reasoning="evidence",
        suggested_bonus=0.1,
    )

    db.log_scout_report(
        run_id, 1, report.to_dict(), report.to_json(), agent_id="agent-2"
    )

    loaded = scout_reports_from_db(db, run_id, agent_id="agent-2")
    assert len(loaded) == 1
    assert loaded[0].report_id == report.report_id
    assert loaded[0].claimed_pattern == "csv has stable columns"
    assert loaded[0].verdict == "supported"
