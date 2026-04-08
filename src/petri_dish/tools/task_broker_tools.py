"""Stub handler for request_task — orchestrator intercepts before this runs."""


def request_task_stub(task_description: str) -> str:
    return (
        "request_task requires orchestrator context. This stub should never be reached."
    )
