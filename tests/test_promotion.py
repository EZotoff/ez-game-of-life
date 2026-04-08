from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petri_dish.config import Settings
from petri_dish.logging_db import LoggingDB
from petri_dish.orchestrator import AgentOrchestrator
from petri_dish.promotion import PromotionEngine
from petri_dish.validators import FileValidator


def _connect_db(tmp_path: Path) -> LoggingDB:
    db = LoggingDB(str(tmp_path / "promotion_test.db"))
    db.connect()
    return db


class _FakeSandbox:
    def __init__(self) -> None:
        self._outgoing: dict[str, str] = {}

    def create_container(self, run_id: str, **kwargs: Any) -> str:
        return f"container-{run_id}"

    def destroy_container(self, container_id: str) -> None:
        return None

    def exec_in_container(self, container_id: str, command: str) -> str:
        if "ls /env/outgoing/" in command:
            return "\n".join(self._outgoing.keys())
        if command.startswith("rm -f /env/outgoing/"):
            fname = command.rsplit("/", 1)[-1].strip()
            self._outgoing.pop(fname, None)
        return ""

    def read_file(self, container_id: str, path: str) -> str:
        fname = path.rsplit("/", 1)[-1]
        return self._outgoing.get(fname, "")

    def get_container_stats(self, container_id: str) -> str:
        return ""

    def set_outgoing(self, filename: str, content: str) -> None:
        self._outgoing[filename] = content


def _make_mock_sandbox() -> Any:
    state: dict[str, str] = {}

    sandbox = Mock()

    def _exec_in_container(container_id: str, command: str) -> str:
        if "ls /env/outgoing/" in command:
            return "\n".join(state.keys())
        if command.startswith("rm -f /env/outgoing/"):
            fname = command.rsplit("/", 1)[-1].strip()
            state.pop(fname, None)
        return ""

    def _read_file(container_id: str, path: str) -> str:
        fname = path.rsplit("/", 1)[-1]
        return state.get(fname, "")

    def _set_outgoing(filename: str, content: str) -> None:
        state[filename] = content

    sandbox.create_container.return_value = "container-test"
    sandbox.destroy_container.return_value = None
    sandbox.get_container_stats.return_value = ""
    sandbox.exec_in_container.side_effect = _exec_in_container
    sandbox.read_file.side_effect = _read_file
    sandbox.set_outgoing = _set_outgoing
    return sandbox


def test_promotion_rule_creation(tmp_path: Path):
    db = _connect_db(tmp_path)
    engine = PromotionEngine(
        db, Settings(promotion_enabled=True, promotion_threshold=3)
    )

    rule = engine.record_hit("valid_csv_with_header", "csv")

    assert rule.claimed_pattern == "valid_csv_with_header"
    assert rule.file_family == "csv"
    assert rule.hit_count == 1
    assert rule.promoted is False


def test_promotion_hit_increments(tmp_path: Path):
    db = _connect_db(tmp_path)
    engine = PromotionEngine(
        db, Settings(promotion_enabled=True, promotion_threshold=10)
    )

    engine.record_hit("valid_json_records", "json")
    engine.record_hit("valid_json_records", "json")
    rule = engine.record_hit("valid_json_records", "json")

    assert rule.hit_count == 3
    assert rule.promoted is False


def test_auto_promote_after_threshold(tmp_path: Path):
    db = _connect_db(tmp_path)
    engine = PromotionEngine(
        db,
        Settings(
            promotion_enabled=True,
            promotion_threshold=3,
            promotion_bonus_multiplier=1.5,
        ),
    )

    engine.record_hit("log_pattern", "log")
    engine.record_hit("log_pattern", "log")
    rule = engine.record_hit("log_pattern", "log")

    assert rule.promoted is True
    assert rule.hit_count == 3
    assert rule.promoted_at is not None


def test_no_promote_below_threshold(tmp_path: Path):
    db = _connect_db(tmp_path)
    engine = PromotionEngine(
        db, Settings(promotion_enabled=True, promotion_threshold=3)
    )

    engine.record_hit("csv_pattern", "csv")
    rule = engine.record_hit("csv_pattern", "csv")

    assert rule.hit_count == 2
    assert rule.promoted is False


def test_max_promoted_rules_enforced(tmp_path: Path):
    db = _connect_db(tmp_path)
    engine = PromotionEngine(
        db,
        Settings(
            promotion_enabled=True,
            promotion_threshold=1,
            max_promoted_rules=1,
        ),
    )

    first = engine.record_hit("p1", "csv")
    second = engine.record_hit("p2", "csv")

    assert first.promoted is True
    assert second.promoted is False
    assert len(engine.get_all_promoted_rules()) == 1


def test_promoted_multiplier_applied(tmp_path: Path):
    db = _connect_db(tmp_path)
    settings = Settings(
        promotion_enabled=True,
        promotion_threshold=1,
        promotion_bonus_multiplier=1.5,
    )
    engine = PromotionEngine(db, settings)
    engine.record_hit("valid_csv_with_header", "csv")
    validator = FileValidator(settings=settings, promotion_engine=engine)

    passed, zod = validator.validate(
        "data_1234_csv_easy.csv",
        "name,age\nAlice,30\nBob,25",
    )

    assert passed is True
    assert zod == pytest.approx(0.45, abs=0.001)


def test_no_multiplier_without_promotion(tmp_path: Path):
    db = _connect_db(tmp_path)
    settings = Settings(promotion_enabled=True, promotion_threshold=3)
    engine = PromotionEngine(db, settings)
    engine.record_hit("valid_csv_with_header", "csv")
    validator = FileValidator(settings=settings, promotion_engine=engine)

    passed, zod = validator.validate(
        "data_1234_csv_easy.csv",
        "name,age\nAlice,30\nBob,25",
    )

    assert passed is True
    assert zod == pytest.approx(0.3, abs=0.001)


def test_promotion_disabled(tmp_path: Path):
    db = _connect_db(tmp_path)
    engine = PromotionEngine(
        db, Settings(promotion_enabled=False, promotion_threshold=1)
    )

    rule = engine.record_hit("any_pattern", "json")

    assert rule.hit_count == 1
    assert rule.promoted is False
    assert engine.get_all_promoted_rules() == []


def test_multiple_families_independent(tmp_path: Path):
    db = _connect_db(tmp_path)
    engine = PromotionEngine(
        db, Settings(promotion_enabled=True, promotion_threshold=3)
    )

    engine.record_hit("shared", "csv")
    engine.record_hit("shared", "csv")
    csv_rule = engine.record_hit("shared", "csv")
    json_rule = engine.record_hit("shared", "json")

    assert csv_rule.promoted is True
    assert json_rule.promoted is False
    assert json_rule.hit_count == 1


def test_promotion_engine_with_orchestrator(tmp_path: Path):
    settings = Settings(
        max_turns=1,
        promotion_enabled=True,
        promotion_threshold=2,
        promotion_bonus_multiplier=1.5,
        decay_rate_per_turn=0.0,
    )
    db = _connect_db(tmp_path)
    sandbox = _make_mock_sandbox()
    orchestrator = AgentOrchestrator(
        settings=settings,
        sandbox_manager=sandbox,
        logging_db=db,
    )
    orchestrator._container_id = "container-test"
    run_id = "promotion-integration-run"
    db.log_run_start(run_id, {"settings": settings.model_dump()})

    pattern = "valid_json_with_records"

    orchestrator._record_overseer_promotion_hits([f"{pattern}::json"])
    orchestrator._record_overseer_promotion_hits([f"{pattern}::json"])

    promoted = db.get_promoted_rules_for_family("json")
    assert len(promoted) == 1
    assert promoted[0]["claimed_pattern"] == pattern
    assert promoted[0]["promoted"] is True
