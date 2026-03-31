"""Container-side tools that execute inside Docker via docker exec.

Tools: file_read, file_write, file_list, shell_exec, python_exec, pass_turn.
"""

import logging
import shlex
import subprocess

logger = logging.getLogger(__name__)

MAX_OUTPUT_BYTES = 10 * 1024  # 10KB truncation limit for shell_exec
PYTHON_TIMEOUT = 60  # seconds for python_exec


def _docker_exec(container_id: str, command: list[str], timeout: int = 30) -> str:
    """Run a command inside a Docker container.

    Args:
        container_id: Docker container ID or name.
        command: Command and args to execute.
        timeout: Timeout in seconds.

    Returns:
        Combined stdout+stderr output as string.
    """
    cmd = ["docker", "exec", container_id] + command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr
        return output.strip()
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def file_read(container_id: str, path: str) -> str:
    """Read a file from the container filesystem."""
    return _docker_exec(container_id, ["cat", path])


def file_write(container_id: str, path: str, content: str) -> str:
    """Write content to a file in the container filesystem."""
    cmd = ["docker", "exec", container_id, "sh", "-c", f"cat > {shlex.quote(path)}"]
    try:
        result = subprocess.run(
            cmd,
            input=content,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return f"Error writing file: {result.stderr.strip()}"
        return f"Successfully wrote {len(content)} bytes to {path}"
    except subprocess.TimeoutExpired:
        return "Error: write timed out after 30s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def file_list(container_id: str, directory: str = ".") -> str:
    """List directory contents in the container."""
    return _docker_exec(container_id, ["ls", "-la", directory])


def shell_exec(container_id: str, command: str, timeout: int = 30) -> str:
    """Execute a shell command inside the container.

    Output is truncated at 10KB to prevent context overflow.
    """
    output = _docker_exec(container_id, ["sh", "-c", command], timeout=timeout)
    if len(output.encode("utf-8", errors="replace")) > MAX_OUTPUT_BYTES:
        truncated = output.encode("utf-8", errors="replace")[:MAX_OUTPUT_BYTES].decode(
            "utf-8", errors="replace"
        )
        return truncated + "\n... [output truncated at 10KB]"
    return output


def python_exec(container_id: str, code: str, timeout: int = PYTHON_TIMEOUT) -> str:
    """Execute a Python script inside the container.

    Writes the code to a temp file and runs it with python3.
    Output is truncated at 10KB.
    """
    write_cmd = ["docker", "exec", "-i", container_id, "tee", "/tmp/_petri_exec.py"]
    try:
        subprocess.run(
            write_cmd,
            input=code,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as e:
        return f"Error writing script: {type(e).__name__}: {e}"

    output = _docker_exec(
        container_id,
        ["python3", "/tmp/_petri_exec.py"],
        timeout=timeout,
    )
    if len(output.encode("utf-8", errors="replace")) > MAX_OUTPUT_BYTES:
        truncated = output.encode("utf-8", errors="replace")[:MAX_OUTPUT_BYTES].decode(
            "utf-8", errors="replace"
        )
        return truncated + "\n... [output truncated at 10KB]"
    return output


def pass_turn(container_id: str, reason: str = "") -> str:
    """Explicitly end the current turn early."""
    msg = "Turn passed."
    if reason:
        msg += f" Reason: {reason}"
    return msg
