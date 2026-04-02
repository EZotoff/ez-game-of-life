"""Test bootstrap helpers."""

from __future__ import annotations

import sys
import types
import typing


def _install_docker_stub() -> None:
    if "docker" in sys.modules:
        return

    docker_mod = types.ModuleType("docker")
    errors_mod = types.ModuleType("docker.errors")
    models_mod = types.ModuleType("docker.models")
    containers_mod = types.ModuleType("docker.models.containers")

    setattr(errors_mod, "DockerException", Exception)
    setattr(errors_mod, "NotFound", Exception)
    setattr(errors_mod, "APIError", Exception)
    setattr(containers_mod, "Container", object)
    setattr(models_mod, "containers", containers_mod)
    setattr(docker_mod, "DockerClient", object)
    setattr(docker_mod, "from_env", lambda: None)

    sys.modules["docker"] = docker_mod
    sys.modules["docker.errors"] = errors_mod
    sys.modules["docker.models"] = models_mod
    sys.modules["docker.models.containers"] = containers_mod


_install_docker_stub()

if not hasattr(typing, "override"):
    setattr(typing, "override", lambda f: f)
