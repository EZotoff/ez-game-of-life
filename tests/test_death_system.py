"""Unit tests for the two-phase death system (economy state machine + tool registry)."""

import sys
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Settings = import_module("petri_dish.config").Settings
CreditEconomy = import_module("petri_dish.economy").CreditEconomy
AgentState = import_module("petri_dish.economy").AgentState


@pytest.fixture
def default_economy():
    return CreditEconomy(Settings())


@pytest.fixture
def tiny_economy():
    return CreditEconomy(
        Settings(initial_balance=0.1, burn_rate_per_turn=0.1, starvation_turns=3)
    )


class TestEconomyStateMachine:
    def test_initial_state_is_active(self, default_economy):
        assert default_economy.state == AgentState.ACTIVE
        assert default_economy.starvation_counter == 0
        assert not default_economy.is_stripped()
        assert not default_economy.is_dead()

    def test_transition_active_to_stripped(self, default_economy):
        assert default_economy.transition_to_stripped() is True
        assert default_economy.state == AgentState.STRIPPED
        assert default_economy.starvation_counter == 0

    def test_stripped_debit_is_noop(self, tiny_economy):
        tiny_economy.debit()
        tiny_economy.transition_to_stripped()
        balance_before = tiny_economy.balance
        tiny_economy.debit()
        assert tiny_economy.balance == balance_before

    def test_starvation_counter_increments(self, tiny_economy):
        tiny_economy.transition_to_stripped()
        for i in range(1, 3):
            tiny_economy.tick_starvation()
            assert tiny_economy.starvation_counter == i
        assert tiny_economy.state == AgentState.STRIPPED

    def test_starvation_death_at_default_threshold(self, default_economy):
        default_economy.transition_to_stripped()
        for _ in range(6):
            assert default_economy.tick_starvation() == AgentState.STRIPPED
        assert default_economy.tick_starvation() == AgentState.DEAD
        assert default_economy.is_dead()

    def test_starvation_death_custom_threshold(self):
        eco = CreditEconomy(Settings(starvation_turns=3))
        eco.transition_to_stripped()
        for _ in range(2):
            eco.tick_starvation()
        assert eco.state == AgentState.STRIPPED
        eco.tick_starvation()
        assert eco.state == AgentState.DEAD

    def test_rescue_resets_to_active(self, tiny_economy):
        tiny_economy.transition_to_stripped()
        tiny_economy.tick_starvation()
        assert tiny_economy.rescue(5.0) is True
        assert tiny_economy.state == AgentState.ACTIVE
        assert tiny_economy.starvation_counter == 0
        assert tiny_economy.balance >= 5.0

    def test_rescue_fails_when_active(self, default_economy):
        assert default_economy.rescue(10.0) is False
        assert default_economy.state == AgentState.ACTIVE

    def test_rescue_fails_when_dead(self):
        eco = CreditEconomy(Settings(starvation_turns=1))
        eco.transition_to_stripped()
        eco.tick_starvation()
        assert eco.rescue(100.0) is False
        assert eco.state == AgentState.DEAD

    def test_dead_is_terminal(self):
        eco = CreditEconomy(Settings(starvation_turns=1))
        eco.transition_to_stripped()
        eco.tick_starvation()
        assert eco.transition_to_stripped() is False
        assert eco.rescue(100.0) is False
        assert eco.state == AgentState.DEAD

    def test_tick_starvation_from_active_can_set_dead(self, default_economy):
        # tick_starvation() does not guard on state — the orchestrator is
        # responsible for only calling it when stripped. Documenting this
        # so a future refactor that adds a state guard doesn't break tests.
        for _ in range(7):
            default_economy.tick_starvation()
        assert default_economy.state == AgentState.DEAD

    def test_get_starvation_remaining(self):
        eco = CreditEconomy(Settings(starvation_turns=3))
        assert eco.get_starvation_remaining() == 3
        eco.transition_to_stripped()
        assert eco.get_starvation_remaining() == 3
        eco.tick_starvation()
        assert eco.get_starvation_remaining() == 2
        eco.tick_starvation()
        assert eco.get_starvation_remaining() == 1
        eco.tick_starvation()
        assert eco.get_starvation_remaining() == 0

    def test_is_stripped_and_is_dead_helpers(self):
        eco = CreditEconomy(Settings(starvation_turns=1))
        assert not eco.is_stripped()
        assert not eco.is_dead()
        eco.transition_to_stripped()
        assert eco.is_stripped()
        assert not eco.is_dead()
        eco.tick_starvation()
        assert not eco.is_stripped()
        assert eco.is_dead()


class TestToolRegistryStripped:
    @pytest.fixture
    def registry(self):
        get_all_tools = import_module("petri_dish.tools").get_all_tools
        return get_all_tools(settings=Settings())

    def test_get_stripped_schemas_returns_only_stripped_tools(self, registry):
        schemas = registry.get_stripped_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert names == {"check_balance", "get_env_info", "file_list"}

    def test_is_tool_allowed_when_stripped(self, registry):
        assert registry.is_tool_allowed_when_stripped("check_balance") is True
        assert registry.is_tool_allowed_when_stripped("get_env_info") is True
        assert registry.is_tool_allowed_when_stripped("file_list") is True

    def test_non_stripped_tool_rejected(self, registry):
        assert registry.is_tool_allowed_when_stripped("file_write") is False
        assert registry.is_tool_allowed_when_stripped("file_read") is False
        assert registry.is_tool_allowed_when_stripped("shell_exec") is False
        assert registry.is_tool_allowed_when_stripped("nonexistent_tool") is False
