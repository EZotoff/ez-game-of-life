from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Settings = import_module("petri_dish.config").Settings
EndorphinEngine = import_module("petri_dish.endorphin").EndorphinEngine
TraitVector = import_module("petri_dish.endorphin").TraitVector


def test_trait_vector_defaults() -> None:
    traits = TraitVector()
    assert traits.curiosity == 0.0
    assert traits.thrift == 0.0
    assert traits.sociability == 0.0
    assert traits.persistence == 0.0
    assert traits.shell_affinity == 0.0
    assert traits.file_family_affinity == {"csv": 0.0, "json": 0.0, "log": 0.0}


def test_trait_update_ema() -> None:
    traits = TraitVector()
    traits.update_trait("curiosity", 1.0, ema_alpha=0.3)
    assert traits.curiosity == pytest.approx(0.3)


def test_trait_clamping() -> None:
    traits = TraitVector()
    traits.update_trait("curiosity", 10.0, ema_alpha=1.0)
    assert traits.curiosity == 1.0
    traits.update_trait("curiosity", -10.0, ema_alpha=1.0)
    assert traits.curiosity == -1.0
    traits.update_file_family("csv", 10.0, ema_alpha=1.0)
    assert traits.file_family_affinity["csv"] == 1.0
    traits.update_file_family("csv", -10.0, ema_alpha=1.0)
    assert traits.file_family_affinity["csv"] == -1.0


def test_decay() -> None:
    traits = TraitVector(curiosity=1.0, thrift=-0.5)
    traits.file_family_affinity["json"] = 0.8
    traits.decay(0.5)
    assert traits.curiosity == pytest.approx(0.5)
    assert traits.thrift == pytest.approx(-0.25)
    assert traits.file_family_affinity["json"] == pytest.approx(0.4)


def test_rebirth_carryover() -> None:
    traits = TraitVector(curiosity=0.8, persistence=0.5)
    traits.file_family_affinity["log"] = 0.6
    traits.rebirth_carryover(0.85)
    assert traits.curiosity == pytest.approx(0.68)
    assert traits.persistence == pytest.approx(0.425)
    assert traits.file_family_affinity["log"] == pytest.approx(0.51)


def test_engine_register_agent() -> None:
    engine = EndorphinEngine(Settings())
    engine.register_agent("alice")
    traits = engine.get_traits("alice")
    assert isinstance(traits, TraitVector)


def test_observe_tool_use_shell() -> None:
    engine = EndorphinEngine(Settings())
    before = engine.get_traits("alice").shell_affinity
    engine.observe_tool_use("alice", "shell_exec")
    after = engine.get_traits("alice").shell_affinity
    assert after > before


def test_observe_tool_use_file() -> None:
    engine = EndorphinEngine(Settings())
    before = engine.get_traits("alice").shell_affinity
    engine.observe_tool_use("alice", "file_read")
    after = engine.get_traits("alice").shell_affinity
    assert after < before


def test_observe_tool_use_message() -> None:
    engine = EndorphinEngine(Settings())
    before = engine.get_traits("alice").sociability
    engine.observe_tool_use("alice", "send_message")
    after = engine.get_traits("alice").sociability
    assert after > before


def test_observe_reward() -> None:
    engine = EndorphinEngine(Settings())
    before = engine.get_traits("alice")
    curiosity_before = before.curiosity
    persistence_before = before.persistence
    engine.observe_reward("alice", 15.0)
    after = engine.get_traits("alice")
    assert after.curiosity > curiosity_before
    assert after.persistence > persistence_before


def test_observe_reward_file_family() -> None:
    engine = EndorphinEngine(Settings())
    before = engine.get_traits("alice").file_family_affinity["csv"]
    engine.observe_reward("alice", 15.0, filename="sample_csv_output.csv")
    after = engine.get_traits("alice").file_family_affinity["csv"]
    assert after > before


def test_observe_starvation() -> None:
    engine = EndorphinEngine(Settings())
    traits = engine.get_traits("alice")
    traits.curiosity = 0.5
    traits.sociability = 0.5
    traits.persistence = 0.5
    engine.observe_starvation("alice")
    assert traits.curiosity == pytest.approx(0.4)
    assert traits.sociability == pytest.approx(0.4)
    assert traits.persistence == pytest.approx(0.4)


def test_observe_death_rebirth() -> None:
    engine = EndorphinEngine(Settings(endorphin_rebirth_factor=0.85))
    traits = engine.get_traits("alice")
    traits.curiosity = 0.9
    engine.observe_death("alice")
    assert engine.get_traits("alice").curiosity == pytest.approx(0.765)


def test_observe_empty_turn() -> None:
    engine = EndorphinEngine(Settings())
    traits = engine.get_traits("alice")
    traits.persistence = 0.4
    engine.observe_empty_turn("alice")
    assert traits.persistence < 0.4


def test_end_round_decay() -> None:
    engine = EndorphinEngine(Settings(endorphin_decay_factor=0.95))
    traits = engine.get_traits("alice")
    traits.curiosity = 1.0
    traits.file_family_affinity["json"] = 1.0
    engine.end_round()
    assert traits.curiosity == pytest.approx(0.95)
    assert traits.file_family_affinity["json"] == pytest.approx(0.95)


def test_generate_instincts_empty() -> None:
    engine = EndorphinEngine(Settings(endorphin_instinct_threshold=0.3))
    assert engine.generate_instincts("alice") == ""


def test_generate_instincts_positive() -> None:
    engine = EndorphinEngine(Settings(endorphin_instinct_threshold=0.3))
    traits = engine.get_traits("alice")
    traits.curiosity = 0.31
    traits.thrift = 0.31
    text = engine.generate_instincts("alice")
    assert "You feel drawn to explore new files and areas." in text
    assert "Expensive operations feel risky" in text


def test_generate_instincts_negative() -> None:
    engine = EndorphinEngine(Settings(endorphin_instinct_threshold=0.3))
    traits = engine.get_traits("alice")
    traits.shell_affinity = -0.31
    text = engine.generate_instincts("alice")
    assert "Direct file operations feel more reliable than shell commands." in text


def test_generate_instincts_file_family() -> None:
    engine = EndorphinEngine(Settings(endorphin_instinct_threshold=0.3))
    traits = engine.get_traits("alice")
    traits.file_family_affinity["json"] = 0.6
    text = engine.generate_instincts("alice")
    assert "You feel drawn toward json files." in text


def test_trait_snapshot() -> None:
    engine = EndorphinEngine(Settings())
    traits = engine.get_traits("alice")
    traits.curiosity = 0.12345
    traits.file_family_affinity["csv"] = 0.98765
    snapshot = engine.get_trait_snapshot("alice")
    assert snapshot["agent_id"] == "alice"
    assert snapshot["curiosity"] == 0.123
    assert snapshot["file_family_affinity"]["csv"] == 0.988
