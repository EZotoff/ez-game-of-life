"""Inter-agent communication tools.

Tools: send_message, read_messages.

These are host-side tools that require orchestrator context (run_id,
sender/recipient agent IDs, message store). The handlers defined here
are stubs — the MultiAgentOrchestrator replaces them with closures
that capture the current run context.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_message_store: Optional["_MessageStore"] = None


class _MessageStore:
    """In-memory message routing layer used by comm tool handlers.

    The MultiAgentOrchestrator creates one instance per run and sets it
    as the global store. Tool handlers read from it synchronously.
    """

    def __init__(self) -> None:
        self._run_id: str = ""
        self._round: int = 0
        self._turn: int = 0
        self._log_fn = None
        self._pending: Dict[str, List[Dict[str, Any]]] = {}

    def configure(self, run_id: str, round_num: int, turn: int, log_fn=None) -> None:
        self._run_id = run_id
        self._round = round_num
        self._turn = turn
        self._log_fn = log_fn

    @property
    def run_id(self) -> str:
        return self._run_id

    def send(self, sender_id: str, recipient_id: str, content: str) -> str:
        if not content.strip():
            return "Error: message content cannot be empty"
        if not recipient_id.strip():
            return "Error: recipient cannot be empty"

        msg = {
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "content": content,
            "round": self._round,
            "turn": self._turn,
        }
        inbox = self._pending.setdefault(recipient_id, [])
        inbox.append(msg)
        logger.info(
            "Message from %s to %s (round %d)", sender_id, recipient_id, self._round
        )
        if self._log_fn:
            self._log_fn(sender_id, recipient_id, content, self._round, self._turn)
        return f"Message sent to {recipient_id}"

    def read(self, recipient_id: str) -> str:
        inbox = self._pending.get(recipient_id, [])
        if not inbox:
            return "No new messages."

        lines = []
        for msg in inbox:
            lines.append(
                f"[From {msg['sender_id']}, round {msg['round']}, "
                f"turn {msg['turn']}]: {msg['content']}"
            )
        self._pending[recipient_id] = []
        return "\n".join(lines)


def set_message_store(store: Optional[_MessageStore]) -> None:
    global _message_store
    _message_store = store


def get_message_store() -> Optional[_MessageStore]:
    return _message_store


def send_message(recipient: str, content: str, **kwargs: object) -> str:
    store = _message_store
    if store is None:
        return "Error: messaging not available in single-agent mode"
    sender = kwargs.get("_sender_id", "unknown")
    return store.send(str(sender), recipient, content)


def read_messages(**kwargs: object) -> str:
    store = _message_store
    if store is None:
        return "Error: messaging not available in single-agent mode"
    recipient = kwargs.get("_recipient_id", "unknown")
    return store.read(str(recipient))
