"""Tests for inter-agent communication: _MessageStore, send_message, read_messages,
message routing through SQLite, and MultiAgentOrchestrator integration."""

from __future__ import annotations

import asyncio
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Settings = import_module("petri_dish.config").Settings
SharedReserve = import_module("petri_dish.economy").SharedReserve
LoggingDB = import_module("petri_dish.logging_db").LoggingDB
MultiAgentOrchestrator = import_module("petri_dish.orchestrator").MultiAgentOrchestrator
FileValidator = import_module("petri_dish.validators").FileValidator

_MessageStore = import_module("petri_dish.tools.comm_tools")._MessageStore
set_message_store = import_module("petri_dish.tools.comm_tools").set_message_store
get_message_store = import_module("petri_dish.tools.comm_tools").get_message_store
send_message = import_module("petri_dish.tools.comm_tools").send_message
read_messages = import_module("petri_dish.tools.comm_tools").read_messages


@pytest.fixture(autouse=True)
def _reset_message_store():
    yield
    set_message_store(None)


# ── Helpers ──────────────────────────────────────────────────────────────────


class FakeSandboxManager:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.destroyed: list[str] = []

    def create_container(
        self,
        run_id: str,
        memory_host_path: str | None = None,
        shared_volume_host_path: str | None = None,
    ) -> str:
        _ = (memory_host_path, shared_volume_host_path)
        cid = f"fake-{run_id}"
        self.created.append(cid)
        return cid

    def destroy_container(self, cid: str) -> None:
        self.destroyed.append(cid)

    def exec_in_container(self, cid: str, cmd: str) -> str:
        return ""

    def get_container_stats(self, cid: str) -> dict[str, float]:
        return {"cpu_percent": 0.0, "memory_usage_mb": 1.0, "memory_limit_mb": 512.0}

    def read_file(self, cid: str, path: str) -> str:
        return ""

    def list_directory(self, cid: str, path: str) -> str:
        return ""


class FakeOllamaClient:
    def __init__(self, responses: list[tuple[str, list[Any]]] | None = None) -> None:
        self._responses = responses or []
        self._idx = 0

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[Any]] | None:
        _ = (system_prompt, messages, tools)
        if self._idx >= len(self._responses):
            return "", []
        out = self._responses[self._idx]
        self._idx += 1
        return out


def _build_multi_agent(
    settings: Any,
    *,
    llm_responses: dict[str, list[tuple[str, list[Any]]]] | None = None,
    agent_names: list[str] | None = None,
) -> tuple[Any, Any, FakeSandboxManager, Any]:
    names = agent_names or ["agent-a", "agent-b"]
    sandbox = FakeSandboxManager()
    logging_db = LoggingDB(":memory:")
    shared_economy = SharedReserve(settings=settings, agent_ids=names)
    llm_clients = {
        aid: FakeOllamaClient((llm_responses or {}).get(aid, [])) for aid in names
    }
    orchestrator = MultiAgentOrchestrator(
        settings=settings,
        shared_economy=shared_economy,
        sandbox_manager=sandbox,
        logging_db=logging_db,
        file_validator=FileValidator(settings=settings),
        llm_clients=llm_clients,
        agent_names=names,
    )
    return orchestrator, logging_db, sandbox, shared_economy


# ── _MessageStore unit tests ────────────────────────────────────────────────


class TestMessageStore:
    def test_send_and_read(self):
        store = _MessageStore()
        store.configure("run-1", round_num=1, turn=3)
        result = store.send("alice", "bob", "hello bob")
        assert result == "Message sent to bob"
        msgs = store.read("bob")
        assert "alice" in msgs
        assert "hello bob" in msgs
        assert store.read("bob") == "No new messages."

    def test_send_empty_content_rejected(self):
        store = _MessageStore()
        result = store.send("alice", "bob", "  ")
        assert "cannot be empty" in result

    def test_send_empty_recipient_rejected(self):
        store = _MessageStore()
        result = store.send("alice", "  ", "hi")
        assert "cannot be empty" in result

    def test_read_empty_inbox(self):
        store = _MessageStore()
        assert store.read("nobody") == "No new messages."

    def test_multiple_messages_to_same_recipient(self):
        store = _MessageStore()
        store.configure("run-1", round_num=1, turn=1)
        store.send("a1", "b1", "msg1")
        store.send("a2", "b1", "msg2")
        msgs = store.read("b1")
        assert "msg1" in msgs
        assert "msg2" in msgs
        assert store.read("b1") == "No new messages."

    def test_log_fn_called_on_send(self):
        logged: list[tuple[str, str, str, int, int]] = []
        store = _MessageStore()
        store.configure(
            "run-1",
            round_num=2,
            turn=5,
            log_fn=lambda s, r, c, rnd, t: logged.append((s, r, c, rnd, t)),
        )
        store.send("alice", "bob", "test log")
        assert len(logged) == 1
        assert logged[0] == ("alice", "bob", "test log", 2, 5)

    def test_configure_updates_run_id(self):
        store = _MessageStore()
        store.configure("run-1", round_num=1, turn=1)
        assert store.run_id == "run-1"
        store.configure("run-2", round_num=2, turn=3)
        assert store.run_id == "run-2"


# ── send_message / read_messages handlers ───────────────────────────────────


class TestCommHandlers:
    def test_send_message_without_store(self):
        set_message_store(None)
        result = send_message("bob", "hello", _sender_id="alice")
        assert "not available" in result

    def test_read_messages_without_store(self):
        set_message_store(None)
        result = read_messages(_recipient_id="alice")
        assert "not available" in result

    def test_send_message_with_store(self):
        store = _MessageStore()
        set_message_store(store)
        store.configure("run-1", round_num=1, turn=1)
        result = send_message("bob", "hello from alice", _sender_id="alice")
        assert result == "Message sent to bob"
        msgs = store.read("bob")
        assert "hello from alice" in msgs

    def test_read_messages_with_store(self):
        store = _MessageStore()
        set_message_store(store)
        store.configure("run-1", round_num=1, turn=1)
        store.send("alice", "bob", "hello")
        result = read_messages(_recipient_id="bob")
        assert "hello" in result

    def test_send_message_default_sender(self):
        store = _MessageStore()
        set_message_store(store)
        store.configure("run-1", round_num=1, turn=1)
        result = send_message("bob", "test")
        assert "Message sent" in result
        msgs = store.read("bob")
        assert "unknown" in msgs

    def test_set_get_message_store(self):
        store = _MessageStore()
        set_message_store(store)
        assert get_message_store() is store
        set_message_store(None)
        assert get_message_store() is None


# ── SQLite messages table ───────────────────────────────────────────────────


class TestLoggingDBMessages:
    @staticmethod
    def _db_with_run(run_id: str = "run-1") -> LoggingDB:
        db = LoggingDB(":memory:")
        db.connect()
        db.log_run_start(run_id, {"test": True})
        return db

    def test_log_and_get_unread(self):
        db = self._db_with_run()
        db.log_message("run-1", "alice", "bob", "hello", round_num=1, turn=1)
        db.log_message("run-1", "carol", "bob", "hi", round_num=1, turn=2)
        unread = db.get_unread_messages("run-1", "bob")
        assert len(unread) == 2
        assert unread[0]["sender_id"] == "alice"
        assert unread[1]["sender_id"] == "carol"

    def test_mark_messages_read(self):
        db = self._db_with_run()
        db.log_message("run-1", "a", "b", "msg1", round_num=1, turn=1)
        db.log_message("run-1", "a", "b", "msg2", round_num=1, turn=2)
        count = db.mark_messages_read("run-1", "b")
        assert count == 2
        unread = db.get_unread_messages("run-1", "b")
        assert len(unread) == 0

    def test_unread_empty_when_no_messages(self):
        db = self._db_with_run()
        unread = db.get_unread_messages("run-1", "nobody")
        assert unread == []

    def test_messages_isolated_per_run(self):
        db = self._db_with_run()
        db.log_run_start("run-2", {"test": True})
        db.log_message("run-1", "a", "b", "hello", round_num=1, turn=1)
        db.log_message("run-2", "a", "b", "other", round_num=1, turn=1)
        unread_run1 = db.get_unread_messages("run-1", "b")
        assert len(unread_run1) == 1
        assert unread_run1[0]["content"] == "hello"
        unread_run2 = db.get_unread_messages("run-2", "b")
        assert len(unread_run2) == 1
        assert unread_run2[0]["content"] == "other"

    def test_log_message_returns_id(self):
        db = self._db_with_run()
        msg_id = db.log_message("run-1", "a", "b", "test", round_num=1, turn=1)
        assert isinstance(msg_id, int)
        assert msg_id > 0


# ── MultiAgentOrchestrator communication integration ────────────────────────


class TestMultiAgentCommIntegration:
    def test_comm_tools_registered(self):
        from petri_dish.tools import get_all_tools

        settings = Settings(initial_zod=1000, max_turns=1, multi_agent_count=2)
        registry = get_all_tools(settings=settings)
        tool_names = registry.get_tool_names()
        assert "send_message" in tool_names
        assert "read_messages" in tool_names

    def test_message_store_initialized_in_constructor(self):
        settings = Settings(initial_zod=100, max_turns=1, multi_agent_count=2)
        orch, *_ = _build_multi_agent(settings, agent_names=["a1", "a2"])
        assert orch._msg_store is not None
        assert get_message_store() is orch._msg_store

    def test_deliver_pending_messages_injects_into_history(self):
        settings = Settings(initial_zod=100, max_turns=1, multi_agent_count=2)
        orch, logging_db, *_ = _build_multi_agent(
            settings, agent_names=["alice", "bob"]
        )
        logging_db.connect()
        logging_db.log_run_start("run-1", {"test": True})
        logging_db.log_message("run-1", "bob", "alice", "hi alice", round_num=1, turn=1)
        orch._deliver_pending_messages("run-1", "alice")
        msgs = orch._agent_messages["alice"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "hi alice" in msgs[0]["content"]

    def test_deliver_pending_messages_marks_read(self):
        settings = Settings(initial_zod=100, max_turns=1, multi_agent_count=2)
        orch, logging_db, *_ = _build_multi_agent(
            settings, agent_names=["alice", "bob"]
        )
        logging_db.connect()
        logging_db.log_run_start("run-1", {"test": True})
        logging_db.log_message("run-1", "bob", "alice", "hi", round_num=1, turn=1)
        orch._deliver_pending_messages("run-1", "alice")
        unread = logging_db.get_unread_messages("run-1", "alice")
        assert len(unread) == 0

    def test_deliver_pending_messages_noop_when_empty(self):
        settings = Settings(initial_zod=100, max_turns=1, multi_agent_count=2)
        orch, logging_db, *_ = _build_multi_agent(
            settings, agent_names=["alice", "bob"]
        )
        logging_db.connect()
        orch._deliver_pending_messages("run-1", "alice")
        assert len(orch._agent_messages["alice"]) == 0

    def test_full_comm_round_robin(self):
        """Agent-a sends a message to agent-b during its turn.
        Agent-b receives it at the start of its next turn."""
        from petri_dish.tool_parser import ToolCall

        settings = Settings(
            initial_zod=500,
            decay_rate_per_turn=0,
            max_turns=2,
            multi_agent_count=2,
            multi_agent_actions_per_turn=4,
        )

        # Agent-a: send message to bob, then empty turn
        # Agent-b: read messages, then empty turn
        response_a_send = (
            "Sending a message to bob",
            [
                ToolCall(
                    name="send_message",
                    arguments={"recipient": "bob", "content": "hey bob!"},
                )
            ],
        )
        response_a_empty = ("done", [])

        response_b_read = (
            "Reading messages",
            [ToolCall(name="read_messages", arguments={})],
        )
        response_b_empty = ("done", [])

        llm_responses = {
            "alice": [response_a_send, response_a_empty],
            "bob": [response_b_read, response_b_empty],
        }

        orch, logging_db, sandbox, economy = _build_multi_agent(
            settings,
            llm_responses=llm_responses,
            agent_names=["alice", "bob"],
        )

        result = asyncio.run(orch.run("test-comm-run"))

        conn = logging_db._ensure_connection()
        rows = conn.execute(
            "SELECT sender_id, recipient_id, content FROM messages WHERE run_id = ?",
            ("test-comm-run",),
        ).fetchall()
        assert len(rows) >= 1
        assert any(r[2] == "hey bob!" for r in rows)

        bob_msgs = orch._agent_messages["bob"]
        user_msgs = [m for m in bob_msgs if m["role"] == "user"]
        assert len(user_msgs) >= 1
        assert any("hey bob!" in m["content"] for m in user_msgs)

    def test_comm_tools_work_when_stripped(self):
        """Communication tools should be available even in STRIPPED state."""
        from petri_dish.tools import get_all_tools

        settings = Settings()
        registry = get_all_tools(settings=settings)
        assert registry.is_tool_allowed_when_stripped("send_message")
        assert registry.is_tool_allowed_when_stripped("read_messages")
