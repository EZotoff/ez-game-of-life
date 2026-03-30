import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from petri_dish.logging_db import LoggingDB


class DashboardServer:
    def __init__(self, db_path: str, host: str = "127.0.0.1", port: int = 8000):
        self.db_path = db_path
        self.host = host
        self.port = port
        self.app = FastAPI(title="Petri Dish Dashboard")
        self.setup_routes()

    def setup_routes(self):
        static_dir = Path(__file__).parent / "dashboard" / "static"
        static_dir.mkdir(parents=True, exist_ok=True)

        @self.app.get("/", response_class=HTMLResponse)
        async def get_index():
            index_path = static_dir / "index.html"
            if index_path.exists():
                return index_path.read_text(encoding="utf-8")
            return "Dashboard index.html not found."

        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @self.app.get("/api/runs")
        async def get_runs():
            with LoggingDB(self.db_path).read_only_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT run_id, start_time, end_time FROM runs ORDER BY start_time DESC"
                )
                return [dict(row) for row in cursor.fetchall()]

        @self.app.get("/api/runs/{run_id}/actions")
        async def get_run_actions(run_id: str):
            with LoggingDB(self.db_path).read_only_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, turn, tool_name, tool_args, result, credits_before, credits_after, duration_ms, agent_id, timestamp 
                    FROM actions 
                    WHERE run_id = ? 
                    ORDER BY turn DESC, timestamp DESC
                    LIMIT 100
                    """,
                    (run_id,),
                )
                actions = []
                for row in cursor.fetchall():
                    action = dict(row)
                    if action["tool_args"]:
                        try:
                            action["tool_args"] = json.loads(action["tool_args"])
                        except Exception:
                            pass
                    actions.append(action)
                return actions

        @self.app.get("/api/runs/{run_id}/balance")
        async def get_run_balance(run_id: str):
            with LoggingDB(self.db_path).read_only_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT timestamp, amount, type, reason, balance_after, agent_id
                    FROM credit_transactions
                    WHERE run_id = ?
                    ORDER BY timestamp ASC, id ASC
                    """,
                    (run_id,),
                )
                return [dict(row) for row in cursor.fetchall()]

        @self.app.get("/api/runs/{run_id}/files")
        async def get_run_files(run_id: str):
            with LoggingDB(self.db_path).read_only_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, filename, file_type, status, dropped_at, processed_at, credits_earned
                    FROM files
                    WHERE run_id = ?
                    ORDER BY dropped_at DESC
                    """,
                    (run_id,),
                )
                return [dict(row) for row in cursor.fetchall()]

        @self.app.get("/api/runs/{run_id}/summary")
        async def get_run_summary(run_id: str):
            with LoggingDB(self.db_path).read_only_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT run_id, start_time, config_snapshot, end_time FROM runs WHERE run_id = ?",
                    (run_id,),
                )
                run_row = cursor.fetchone()
                if not run_row:
                    raise HTTPException(status_code=404, detail="Run not found")

                run_info = dict(run_row)
                if run_info["config_snapshot"]:
                    try:
                        run_info["config_snapshot"] = json.loads(
                            run_info["config_snapshot"]
                        )
                    except Exception:
                        pass

                cursor.execute(
                    "SELECT MAX(turn) as max_turn FROM actions WHERE run_id = ?",
                    (run_id,),
                )
                max_turn_row = cursor.fetchone()
                run_info["turns"] = (
                    max_turn_row["max_turn"]
                    if max_turn_row and max_turn_row["max_turn"] is not None
                    else 0
                )

                cursor.execute(
                    "SELECT balance_after FROM credit_transactions WHERE run_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
                    (run_id,),
                )
                bal_row = cursor.fetchone()
                run_info["current_balance"] = (
                    bal_row["balance_after"] if bal_row else 0.0
                )

                cursor.execute(
                    "SELECT DISTINCT agent_id FROM actions WHERE run_id = ? AND agent_id IS NOT NULL",
                    (run_id,),
                )
                agents_rows = cursor.fetchall()
                agents = []
                for a_row in agents_rows:
                    a_id = a_row["agent_id"]
                    cursor.execute(
                        "SELECT balance_after FROM credit_transactions WHERE run_id = ? AND agent_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
                        (run_id, a_id),
                    )
                    a_bal_row = cursor.fetchone()

                    cursor.execute(
                        "SELECT to_state FROM state_transitions WHERE run_id = ? AND agent_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
                        (run_id, a_id),
                    )
                    a_state_row = cursor.fetchone()

                    agents.append(
                        {
                            "agent_id": a_id,
                            "balance": a_bal_row["balance_after"] if a_bal_row else 0.0,
                            "state": a_state_row["to_state"]
                            if a_state_row
                            else "UNKNOWN",
                        }
                    )
                run_info["agents"] = agents

                cursor.execute(
                    "SELECT agent_id, from_state, to_state, timestamp FROM state_transitions WHERE run_id = ? ORDER BY timestamp DESC",
                    (run_id,),
                )
                run_info["recent_state_changes"] = [dict(r) for r in cursor.fetchall()]

                return run_info

        @self.app.get("/api/runs/{run_id}/messages")
        async def get_run_messages(run_id: str):
            with LoggingDB(self.db_path).read_only_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, sender_id, recipient_id, content, round_num, turn, read, timestamp 
                    FROM messages WHERE run_id = ? ORDER BY timestamp ASC
                    """,
                    (run_id,),
                )
                return [dict(row) for row in cursor.fetchall()]

        @self.app.get("/api/runs/{run_id}/events")
        async def get_run_events(run_id: str):
            with LoggingDB(self.db_path).read_only_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, round_num, agent_id, event_type, details, credit_delta, timestamp 
                    FROM event_ledger WHERE run_id = ? ORDER BY timestamp DESC LIMIT 50
                    """,
                    (run_id,),
                )
                return [dict(row) for row in cursor.fetchall()]

        @self.app.get("/api/runs/{run_id}/state-transitions")
        async def get_run_state_transitions(run_id: str):
            with LoggingDB(self.db_path).read_only_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, turn, from_state, to_state, reason, balance, starvation_counter, agent_id, timestamp 
                    FROM state_transitions WHERE run_id = ? ORDER BY timestamp DESC
                    """,
                    (run_id,),
                )
                return [dict(row) for row in cursor.fetchall()]

        @self.app.get("/api/events")
        async def sse_events(request: Request, run_id: str | None = None):
            async def event_generator() -> AsyncGenerator[str, None]:
                last_action_id = 0
                last_tx_id = 0
                last_message_id = 0
                last_event_id = 0
                current_run_id = run_id

                while True:
                    if await request.is_disconnected():
                        break

                    try:
                        with LoggingDB(self.db_path).read_only_connection() as conn:
                            cursor = conn.cursor()

                            if not current_run_id:
                                cursor.execute(
                                    "SELECT run_id FROM runs ORDER BY start_time DESC LIMIT 1"
                                )
                                row = cursor.fetchone()
                                if row:
                                    current_run_id = row["run_id"]
                                else:
                                    await asyncio.sleep(2)
                                    continue

                            events = []

                            cursor.execute(
                                """
                                SELECT id, turn, tool_name, tool_args, result, credits_before, credits_after, duration_ms, agent_id, timestamp 
                                FROM actions 
                                WHERE run_id = ? AND id > ? 
                                ORDER BY id ASC
                                """,
                                (current_run_id, last_action_id),
                            )
                            actions = []
                            for row in cursor.fetchall():
                                action = dict(row)
                                last_action_id = max(last_action_id, action["id"])
                                if action["tool_args"]:
                                    try:
                                        action["tool_args"] = json.loads(
                                            action["tool_args"]
                                        )
                                    except Exception:
                                        pass
                                actions.append(action)
                            if actions:
                                events.append({"type": "actions", "data": actions})

                            cursor.execute(
                                """
                                SELECT id, timestamp, amount, type, reason, balance_after, agent_id
                                FROM credit_transactions
                                WHERE run_id = ? AND id > ?
                                ORDER BY id ASC
                                """,
                                (current_run_id, last_tx_id),
                            )
                            txs = []
                            for row in cursor.fetchall():
                                tx = dict(row)
                                last_tx_id = max(last_tx_id, tx["id"])
                                txs.append(tx)
                            if txs:
                                events.append({"type": "balance", "data": txs})

                            cursor.execute(
                                """
                                SELECT id, sender_id, recipient_id, content, round_num, turn, read, timestamp
                                FROM messages
                                WHERE run_id = ? AND id > ?
                                ORDER BY id ASC
                                """,
                                (current_run_id, last_message_id),
                            )
                            msgs = []
                            for row in cursor.fetchall():
                                msg = dict(row)
                                last_message_id = max(last_message_id, msg["id"])
                                msgs.append(msg)
                            if msgs:
                                events.append({"type": "messages", "data": msgs})

                            cursor.execute(
                                """
                                SELECT id, round_num, agent_id, event_type, details, credit_delta, timestamp
                                FROM event_ledger
                                WHERE run_id = ? AND id > ?
                                ORDER BY id ASC
                                """,
                                (current_run_id, last_event_id),
                            )
                            evts = []
                            for row in cursor.fetchall():
                                evt = dict(row)
                                last_event_id = max(last_event_id, evt["id"])
                                evts.append(evt)
                            if evts:
                                events.append({"type": "events", "data": evts})

                            cursor.execute(
                                """
                                SELECT status, COUNT(*) as count, SUM(credits_earned) as total_credits
                                FROM files 
                                WHERE run_id = ? 
                                GROUP BY status
                                """,
                                (current_run_id,),
                            )
                            stats = {"by_status": {}}
                            for row in cursor.fetchall():
                                stats["by_status"][row["status"]] = {
                                    "count": row["count"],
                                    "total_credits": row["total_credits"] or 0.0,
                                }

                            cursor.execute(
                                """
                                SELECT 
                                    COUNT(*) as total_files,
                                    SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) as processed_files,
                                    SUM(credits_earned) as total_credits_earned
                                FROM files 
                                WHERE run_id = ?
                                """,
                                (current_run_id,),
                            )
                            row = cursor.fetchone()
                            if row:
                                stats.update(dict(row))

                            if not events:
                                yield f"event: ping\ndata: {{}}\n\n"
                            else:
                                events.append({"type": "file_stats", "data": stats})

                                for event in events:
                                    data_str = json.dumps(event)
                                    yield f"event: {event['type']}\ndata: {data_str}\n\n"

                    except Exception as e:
                        print(f"SSE Error: {e}")

                    await asyncio.sleep(2)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Transfer-Encoding": "chunked",
                },
            )

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Petri Dish Dashboard")
    parser.add_argument(
        "--db", type=str, required=True, help="Path to logging database"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    server = DashboardServer(db_path=args.db, host=args.host, port=args.port)
    server.run()
