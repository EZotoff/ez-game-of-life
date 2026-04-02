"""Host-side tools that run outside the Docker container.

Tools: check_balance, http_request, overseer_scout.
"""

import logging
import urllib.request
import urllib.error
from types import TracebackType
from typing import Protocol, cast

from petri_dish.tools.overseer_scout import overseer_scout

__all__ = ["check_balance", "http_request", "overseer_scout"]

logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 10 * 1024  # 10KB truncation limit


class _HTTPResponseProtocol(Protocol):
    status: int

    def read(self, amt: int = -1) -> bytes: ...

    def __enter__(self) -> "_HTTPResponseProtocol": ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...


def check_balance(**kwargs: object) -> str:
    """Query current zod balance.

    This is a placeholder — the orchestrator injects the actual balance
    at call time. Returns a stub here; real implementation wired in T8.
    """
    _ = kwargs
    return "Balance check requires orchestrator context. Use via orchestrator."


def http_request(url: str, method: str = "GET") -> str:
    """Make an HTTP request from the host.

    Args:
        url: Target URL.
        method: HTTP method (GET or POST).

    Returns:
        Response body (truncated at 10KB) or error message.
    """
    logger.info("HTTP %s %s", method, url)
    try:
        req = urllib.request.Request(url, method=method.upper())
        response_obj = cast(
            _HTTPResponseProtocol,
            urllib.request.urlopen(req, timeout=15),
        )
        with response_obj as typed_response:
            body = typed_response.read(MAX_RESPONSE_BYTES)
            result = body.decode("utf-8", errors="replace")
            status = typed_response.status
            return f"HTTP {status}\n{result}"
    except urllib.error.HTTPError as e:
        return f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"URL Error: {e.reason}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
