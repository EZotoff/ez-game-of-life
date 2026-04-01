"""Host-side tools that run outside the Docker container.

Tools: check_balance, http_request.
"""

import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 10 * 1024  # 10KB truncation limit


def check_balance(**kwargs: object) -> str:
    """Query current zod balance.

    This is a placeholder — the orchestrator injects the actual balance
    at call time. Returns a stub here; real implementation wired in T8.
    """
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
        with urllib.request.urlopen(req, timeout=15) as response:
            body = response.read(MAX_RESPONSE_BYTES)
            result = body.decode("utf-8", errors="replace")
            status = response.status
            return f"HTTP {status}\n{result}"
    except urllib.error.HTTPError as e:
        return f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"URL Error: {e.reason}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
