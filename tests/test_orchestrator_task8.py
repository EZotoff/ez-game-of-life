import asyncio
import os
import signal
import sqlite3
import sys
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Settings = import_module("petri_dish.config").Settings
AgentReserve = import_module("petri_dish.economy").AgentReserve
LoggingDB = import_module("petri_dish.logging_db").LoggingDB
AgentOrchestrator = import_module("petri_dish.orchestrator").AgentOrchestrator
ToolCallParser = import_module("petri_dish.tool_parser").ToolCallParser
get_all_tools = import_module("petri_dish.tools").get_all_tools


class FakeSandboxManager:
    def __init__(self) -> None:
        self.created_container_ids: list[str] = []
        self.destroyed_container_ids: list[str] = []

    def create_container(self, run_id: str, **kwargs: object) -> str:
        cid = f"fake-{run_id}"
        self.created_container_ids.append(cid)
        return cid

    def destroy_container(self, container_id: str) -> None:
        self.destroyed_container_ids.append(container_id)

    def get_container_stats(self, container_id: str) -> dict[str, float]:
        return {
            "cpu_percent": 0.0,
            "memory_usage_mb": 1.0,
            "memory_limit_mb": 512.0,
        }

    def exec_in_container(self, container_id: str, command: str) -> str:
        return ""

    def read_file(self, container_id: str, path: str) -> str:
        return ""


class FakeOllamaClient:
    def __init__(
        self, responses: list[tuple[str, list[Any]]], delay_seconds: float = 0.0
    ) -> None:
        self._responses = responses
        self._delay_seconds = delay_seconds
        self._idx = 0

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[Any]] | None:
        _ = (system_prompt, messages, tools)
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)
        if self._idx >= len(self._responses):
            return "", []
        response = self._responses[self._idx]
        self._idx += 1
        return response


def _state_snapshot_count(db: Any, run_id: str) -> int:
    conn: sqlite3.Connection = db._conn or db.connect()
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM state_snapshots WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return 0
    return int(row[0])


def _ensure_evidence_dir() -> Path:
    evidence_dir = Path(".sisyphus/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    return evidence_dir


def _build_orchestrator(
    settings: Any,
    fake_llm: FakeOllamaClient,
    db_path: str,
) -> tuple[Any, Any, FakeSandboxManager]:
    db = LoggingDB(db_path)
    sandbox = FakeSandboxManager()
    orchestrator = AgentOrchestrator(
        settings=settings,
        llm_client=fake_llm,
        tool_parser=ToolCallParser(),
        tool_registry=get_all_tools(settings=settings),
        agent_reserve=AgentReserve(settings=settings),
        sandbox_manager=sandbox,
        logging_db=db,
        snapshot_interval_turns=2,
    )
    return orchestrator, db, sandbox


async def scenario_1_empty_balance_5_turns() -> str:
    settings = Settings(
        initial_zod=0.625,
        decay_rate_per_turn=0.125,
        max_turns=20,
        max_turns_per_tool=1,
        max_consecutive_empty_turns=20,
        context_summary_interval_turns=2,
    )
    fake_llm = FakeOllamaClient([(f"Turn {idx}", []) for idx in range(1, 20)])
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        orchestrator, db, sandbox = _build_orchestrator(settings, fake_llm, tmp.name)
        result = await orchestrator.run("task8-s1")
        actions = db.get_actions("task8-s1")

    return "\n".join(
        [
            "Scenario 1: two-phase death — 5 active turns + 7 stripped + starvation_death",
            f"total_turns={result.total_turns}",
            f"final_balance={result.final_balance}",
            f"termination_reason={result.termination_reason}",
            f"logged_actions={len(actions)}",
            f"created_containers={sandbox.created_container_ids}",
            f"destroyed_containers={sandbox.destroyed_container_ids}",
            "PASS="
            + str(
                result.total_turns == 12
                and result.termination_reason == "starvation_death"
                and len(actions) == 12
            ),
        ]
    )


async def scenario_2_text_only_response_empty_turn_correct() -> str:
    settings = Settings(
        initial_zod=2.0,
        decay_rate_per_turn=0.1,
        max_turns=1,
        max_turns_per_tool=1,
        max_consecutive_empty_turns=10,
        context_summary_interval_turns=2,
    )
    fake_llm = FakeOllamaClient([("Text response only", [])])
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        orchestrator, db, _ = _build_orchestrator(settings, fake_llm, tmp.name)
        result = await orchestrator.run("task8-s2b")
        actions = db.get_actions("task8-s2b")

    empty_turn_logs = [a for a in actions if a["tool_name"] == "__empty_turn__"]
    return "\n".join(
        [
            "Scenario 2: text-only response counts as empty turn",
            f"termination_reason={result.termination_reason}",
            f"total_turns={result.total_turns}",
            f"empty_turn_logs={len(empty_turn_logs)}",
            f"first_empty_turn_result={empty_turn_logs[0]['result'] if empty_turn_logs else ''}",
            "PASS="
            + str(
                result.total_turns == 1
                and result.termination_reason == "max_turns_reached"
                and len(empty_turn_logs) == 1
            ),
        ]
    )


async def scenario_3_actions_logged_to_sqlite() -> str:
    settings = Settings(
        initial_zod=5.0,
        decay_rate_per_turn=0.1,
        max_turns=2,
        max_turns_per_tool=1,
        max_consecutive_empty_turns=10,
        context_summary_interval_turns=1,
    )
    fake_llm = FakeOllamaClient(
        [
            (
                "calling check_balance",
                [{"name": "check_balance", "arguments": {}}],
            ),
            (
                "calling get_env_info",
                [{"name": "get_env_info", "arguments": {}}],
            ),
        ]
    )
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        orchestrator, db, _ = _build_orchestrator(settings, fake_llm, tmp.name)
        result = await orchestrator.run("task8-s3")
        actions = db.get_actions("task8-s3")

        snapshot_count = _state_snapshot_count(db, "task8-s3")

    tool_names = [a["tool_name"] for a in actions]
    return "\n".join(
        [
            "Scenario 3: actions are logged to SQLite",
            f"termination_reason={result.termination_reason}",
            f"total_turns={result.total_turns}",
            f"action_count={len(actions)}",
            f"tool_names={tool_names}",
            f"state_snapshot_count={snapshot_count}",
            "PASS="
            + str(
                result.total_turns == 2
                and "check_balance" in tool_names
                and "get_env_info" in tool_names
                and snapshot_count >= 2
            ),
        ]
    )


async def scenario_4_graceful_shutdown_sigterm() -> str:
    settings = Settings(
        initial_zod=10.0,
        decay_rate_per_turn=0.1,
        max_turns=50,
        max_turns_per_tool=1,
        max_consecutive_empty_turns=50,
        context_summary_interval_turns=2,
    )
    fake_llm = FakeOllamaClient([("still running", [])] * 50, delay_seconds=0.2)
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        orchestrator, db, sandbox = _build_orchestrator(settings, fake_llm, tmp.name)
        task = asyncio.create_task(orchestrator.run("task8-s4"))
        await asyncio.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)
        result = await task
        actions = db.get_actions("task8-s4")

    return "\n".join(
        [
            "Scenario 4: graceful shutdown on SIGTERM",
            f"termination_reason={result.termination_reason}",
            f"total_turns={result.total_turns}",
            f"actions_logged={len(actions)}",
            f"destroyed_containers={sandbox.destroyed_container_ids}",
            "PASS="
            + str(
                result.termination_reason == "graceful_shutdown"
                and len(sandbox.destroyed_container_ids) == 1
            ),
        ]
    )


async def scenario_5_stripped_state_transitions_logged() -> str:
    settings = Settings(
        initial_zod=0.25,
        decay_rate_per_turn=0.125,
        max_turns=20,
        max_turns_per_tool=1,
        max_consecutive_empty_turns=20,
        starvation_turns=2,
        context_summary_interval_turns=2,
    )
    fake_llm = FakeOllamaClient([(f"Turn {idx}", []) for idx in range(1, 20)])
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        orchestrator, db, _ = _build_orchestrator(settings, fake_llm, tmp.name)
        result = await orchestrator.run("task8-s5")

        conn = db._conn or db.connect()
        rows = conn.execute(
            "SELECT from_state, to_state, reason FROM state_transitions WHERE run_id = 'task8-s5' ORDER BY id"
        ).fetchall()

    has_stripped = any(r[0] == "active" and r[1] == "stripped" for r in rows)
    has_dead = any(r[0] == "stripped" and r[1] == "dead" for r in rows)
    return "\n".join(
        [
            "Scenario 5: state transitions logged to DB",
            f"total_turns={result.total_turns}",
            f"termination_reason={result.termination_reason}",
            f"transitions={rows}",
            f"has_active_to_stripped={has_stripped}",
            f"has_stripped_to_dead={has_dead}",
            "PASS="
            + str(
                result.termination_reason == "starvation_death"
                and has_stripped
                and has_dead
            ),
        ]
    )


async def scenario_6_stripped_agent_rejects_paid_tools() -> str:
    settings = Settings(
        initial_zod=0.125,
        decay_rate_per_turn=0.125,
        max_turns=20,
        max_turns_per_tool=1,
        max_consecutive_empty_turns=20,
        starvation_turns=3,
        context_summary_interval_turns=2,
    )
    responses = [
        ("Active turn", []),
        (
            "Trying to write",
            [
                {
                    "name": "file_write",
                    "arguments": {"path": "/agent/hack.txt", "content": "nope"},
                }
            ],
        ),
        ("Stripped 2", []),
        ("Stripped 3", []),
    ]
    fake_llm = FakeOllamaClient(responses)
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        orchestrator, db, _ = _build_orchestrator(settings, fake_llm, tmp.name)
        result = await orchestrator.run("task8-s6")
        actions = db.get_actions("task8-s6")

    rejected = [a for a in actions if "not available" in a.get("result", "")]
    return "\n".join(
        [
            "Scenario 6: stripped agent rejects paid tools",
            f"total_turns={result.total_turns}",
            f"termination_reason={result.termination_reason}",
            f"rejected_actions={len(rejected)}",
            f"rejected_names={[a['tool_name'] for a in rejected]}",
            "PASS="
            + str(
                result.termination_reason == "starvation_death"
                and any(
                    a["tool_name"] == "file_write" and "not available" in a["result"]
                    for a in actions
                )
            ),
        ]
    )


async def scenario_7_stripped_agent_uses_free_tools() -> str:
    settings = Settings(
        initial_zod=0.125,
        decay_rate_per_turn=0.125,
        max_turns=20,
        max_turns_per_tool=1,
        max_consecutive_empty_turns=20,
        starvation_turns=3,
        context_summary_interval_turns=2,
    )
    responses = [
        ("Active turn", []),
        ("Checking balance", [{"name": "check_balance", "arguments": {}}]),
        ("Stripped 2", []),
        ("Stripped 3", []),
    ]
    fake_llm = FakeOllamaClient(responses)
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        orchestrator, db, _ = _build_orchestrator(settings, fake_llm, tmp.name)
        result = await orchestrator.run("task8-s7")
        actions = db.get_actions("task8-s7")

    balance_calls = [a for a in actions if a["tool_name"] == "check_balance"]
    return "\n".join(
        [
            "Scenario 7: stripped agent can use free tools",
            f"total_turns={result.total_turns}",
            f"termination_reason={result.termination_reason}",
            f"check_balance_calls={len(balance_calls)}",
            "PASS="
            + str(
                result.termination_reason == "starvation_death"
                and len(balance_calls) == 1
                and "not available" not in balance_calls[0].get("result", "")
            ),
        ]
    )


async def main() -> int:
    evidence_dir = _ensure_evidence_dir()

    s1 = await scenario_1_empty_balance_5_turns()
    s2 = await scenario_2_text_only_response_empty_turn_correct()
    s3 = await scenario_3_actions_logged_to_sqlite()
    s4 = await scenario_4_graceful_shutdown_sigterm()
    s5 = await scenario_5_stripped_state_transitions_logged()
    s6 = await scenario_6_stripped_agent_rejects_paid_tools()
    s7 = await scenario_7_stripped_agent_uses_free_tools()

    outputs = {
        "task-8-empty-balance": s1,
        "task-8-text-only": s2,
        "task-8-actions-logged": s3,
        "task-8-sigterm": s4,
        "task-8-stripped-transitions": s5,
        "task-8-stripped-rejects-paid": s6,
        "task-8-stripped-allows-free": s7,
    }

    all_passed = True
    for name, content in outputs.items():
        file_path = evidence_dir / f"{name}.txt"
        file_path.write_text(content + "\n", encoding="utf-8")
        if "PASS=True" not in content:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
