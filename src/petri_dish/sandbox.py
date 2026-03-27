"""Docker sandbox management for Petri Dish agent containers."""

from __future__ import annotations

import logging
import tarfile
import io
from typing import Any, cast

import docker
from docker.errors import DockerException, NotFound, APIError
from docker.models.containers import Container

from petri_dish.config import settings

logger = logging.getLogger(__name__)

MAX_OUTPUT_BYTES = 10 * 1024


class SandboxError(Exception):
    """Raised when a sandbox operation fails."""


class ContainerNotRunningError(SandboxError):
    """Raised when a container is not in running state."""


class SandboxManager:
    """Manages Docker containers as isolated sandboxes for agent runs.

    Each agent run gets its own container with resource limits and
    three mounted volumes: /env/incoming/, /env/outgoing/, /agent/.
    """

    def __init__(self, client: docker.DockerClient | None = None) -> None:
        self._client: docker.DockerClient = client or docker.from_env()

    def _container_name(self, run_id: str) -> str:
        """Generate deterministic container name from run ID."""
        return f"petri-{run_id[:8]}"

    def _get_container(self, container_id: str) -> Container:
        """Fetch container object, raising SandboxError on failure."""
        try:
            return self._client.containers.get(container_id)
        except NotFound:
            raise SandboxError(f"Container {container_id} not found")
        except DockerException as exc:
            raise SandboxError(f"Docker error fetching container {container_id}: {exc}")

    def _health_check(self, container: Container) -> None:
        """Verify container is running; raise if not."""
        container.reload()
        status = container.status
        if status != "running":
            msg = f"Container {container.id} is not running (status={status})"
            logger.error(msg)
            raise ContainerNotRunningError(msg)

    @staticmethod
    def _truncate(output: str) -> str:
        """Truncate output to MAX_OUTPUT_BYTES."""
        encoded = output.encode("utf-8", errors="replace")
        if len(encoded) <= MAX_OUTPUT_BYTES:
            return output
        return (
            encoded[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            + "\n... [truncated at 10KB]"
        )

    # ------------------------------------------------------------------ #
    #  Container lifecycle
    # ------------------------------------------------------------------ #

    def create_container(
        self, run_id: str, *, memory_host_path: str | None = None
    ) -> str:
        """Create an isolated Docker container for a simulation run.

        Args:
            run_id: Unique identifier for the run.
            memory_host_path: Optional host directory to bind-mount as
                /agent/memory/ inside the container. Contents persist across
                runs. Created automatically if it doesn't exist.

        Returns:
            Container ID string.
        """
        name = self._container_name(run_id)

        try:
            old = self._client.containers.get(name)
            logger.warning("Removing leftover container %s", name)
            old.remove(force=True)
        except NotFound:
            pass

        volumes: dict[str, dict[str, str]] = {}
        if memory_host_path:
            from pathlib import Path

            mem_dir = Path(memory_host_path)
            mem_dir.mkdir(parents=True, exist_ok=True)
            volumes[str(mem_dir.resolve())] = {
                "bind": "/agent/memory",
                "mode": "rw",
            }
            logger.info("Persistent memory mounted: %s -> /agent/memory", mem_dir)

        try:
            container: Container = self._client.containers.run(
                image=settings.docker_image,
                name=name,
                detach=True,
                stdin_open=True,
                tty=False,
                network_disabled=True,
                mem_limit=settings.docker_mem_limit,
                cpu_quota=settings.docker_cpu_quota,
                volumes=volumes or None,
                command=["sleep", "infinity"],
                labels={"petri-dish": "sandbox", "run-id": run_id},
            )
        except DockerException as exc:
            raise SandboxError(f"Failed to create container for run {run_id}: {exc}")

        cid: str = container.id or ""
        if not cid:
            raise SandboxError(f"Container for run {run_id} has no ID")
        logger.info("Created container %s (%s) for run %s", name, cid[:12], run_id)

        for path in ("/env/incoming", "/env/outgoing", "/agent"):
            _ = self.exec_in_container(cid, f"mkdir -p {path}")

        return cid

    def destroy_container(self, container_id: str) -> None:
        """Stop and remove a container.

        Args:
            container_id: Docker container ID.
        """
        try:
            container = self._get_container(container_id)
            container.stop(timeout=5)
            container.remove(force=True)
            logger.info("Destroyed container %s", container_id[:12])
        except SandboxError:
            logger.warning("Container %s already gone", container_id[:12])
        except DockerException as exc:
            logger.error("Error destroying container %s: %s", container_id[:12], exc)

    # ------------------------------------------------------------------ #
    #  Execution
    # ------------------------------------------------------------------ #

    def exec_in_container(
        self, container_id: str, command: str, timeout: int = 30
    ) -> str:
        """Execute a command inside the container via docker exec.

        Args:
            container_id: Docker container ID.
            command: Shell command string to execute.
            timeout: Maximum seconds to wait (default 30).

        Returns:
            Command stdout/stderr (truncated to 10KB).
        """
        container = self._get_container(container_id)
        self._health_check(container)

        try:
            exec_result = container.exec_run(
                cmd=["sh", "-c", f"timeout {timeout} sh -c {command!r}"],
                demux=True,
                environment={"PYTHONDONTWRITEBYTECODE": "1"},
            )
            stdout = (
                (exec_result.output[0] or b"").decode("utf-8", errors="replace")
                if exec_result.output
                else ""
            )
            stderr = (
                (exec_result.output[1] or b"").decode("utf-8", errors="replace")
                if exec_result.output
                else ""
            )

            combined = stdout
            if stderr:
                combined += f"\n[stderr]\n{stderr}"

            if exec_result.exit_code != 0:
                combined = f"[exit_code={exec_result.exit_code}]\n{combined}"

            return self._truncate(combined)

        except APIError as exc:
            raise SandboxError(f"Exec failed in {container_id[:12]}: {exc}")

    # ------------------------------------------------------------------ #
    #  File operations
    # ------------------------------------------------------------------ #

    def read_file(self, container_id: str, path: str) -> str:
        """Read a file from the container filesystem.

        Args:
            container_id: Docker container ID.
            path: Absolute path inside the container.

        Returns:
            File content as string (truncated to 10KB).
        """
        container = self._get_container(container_id)
        self._health_check(container)

        try:
            bits, _ = container.get_archive(path)
            buf = io.BytesIO()
            for chunk in bits:
                _ = buf.write(chunk)
            _ = buf.seek(0)

            with tarfile.open(fileobj=buf) as tar:
                member = tar.getmembers()[0]
                f = tar.extractfile(member)
                if f is None:
                    raise SandboxError(f"Cannot read {path}: not a regular file")
                content = f.read().decode("utf-8", errors="replace")
            return self._truncate(content)

        except NotFound:
            raise SandboxError(f"File not found in container: {path}")
        except (APIError, tarfile.TarError) as exc:
            raise SandboxError(f"Failed to read {path} from {container_id[:12]}: {exc}")

    def write_file(self, container_id: str, path: str, content: str) -> str:
        """Write content to a file inside the container.

        Args:
            container_id: Docker container ID.
            path: Absolute path inside the container.
            content: Text content to write.

        Returns:
            Confirmation message.
        """
        container = self._get_container(container_id)
        self._health_check(container)

        try:
            data = content.encode("utf-8")
            tarstream = io.BytesIO()
            with tarfile.open(fileobj=tarstream, mode="w") as tar:
                info = tarfile.TarInfo(name=path.split("/")[-1])
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            _ = tarstream.seek(0)

            dest_dir = "/".join(path.split("/")[:-1]) or "/"
            _ = container.put_archive(dest_dir, tarstream)

            return f"Written {len(data)} bytes to {path}"

        except (APIError, tarfile.TarError) as exc:
            raise SandboxError(f"Failed to write {path} in {container_id[:12]}: {exc}")

    def list_directory(self, container_id: str, path: str) -> str:
        """List files in a directory inside the container.

        Args:
            container_id: Docker container ID.
            path: Absolute directory path inside the container.

        Returns:
            Directory listing string.
        """
        return self.exec_in_container(container_id, f"ls -la {path}")

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    def get_container_stats(self, container_id: str) -> dict[str, float]:
        """Return live CPU and memory statistics for a container.

        Args:
            container_id: Docker container ID.

        Returns:
            Dict with cpu_percent, memory_usage_mb, memory_limit_mb keys.
        """
        container = self._get_container(container_id)
        self._health_check(container)

        try:
            raw = cast(dict[str, Any], container.stats(stream=False))

            mem_stats: dict[str, Any] = raw.get("memory_stats", {})
            mem_usage: int = mem_stats.get("usage", 0)
            mem_limit: int = mem_stats.get("limit", 0)

            cpu_stats: dict[str, Any] = raw.get("cpu_stats", {})
            precpu_stats: dict[str, Any] = raw.get("precpu_stats", {})
            cpu_delta = cpu_stats.get("cpu_usage", {}).get(
                "total_usage", 0
            ) - precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
            system_delta = cpu_stats.get("system_cpu_usage", 0) - precpu_stats.get(
                "system_cpu_usage", 0
            )
            cpu_percent = 0.0
            if system_delta > 0 and cpu_delta > 0:
                online_cpus = cpu_stats.get("online_cpus", 1)
                cpu_percent = (cpu_delta / system_delta) * online_cpus * 100.0

            return {
                "cpu_percent": round(cpu_percent, 2),
                "memory_usage_mb": round(mem_usage / (1024 * 1024), 2),
                "memory_limit_mb": round(mem_limit / (1024 * 1024), 2),
            }

        except (APIError, KeyError) as exc:
            raise SandboxError(f"Failed to get stats for {container_id[:12]}: {exc}")
