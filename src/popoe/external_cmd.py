"""Shared subprocess boundary for external producer adapters."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ExternalCommand:
    """A subprocess command plus the cwd/env needed by an external repo."""

    argv: tuple[str, ...]
    cwd: str
    env: Mapping[str, str] = field(default_factory=dict)

    def as_subprocess_kwargs(self) -> dict:
        env = os.environ.copy()
        env.update(dict(self.env))
        return {"args": list(self.argv), "cwd": self.cwd, "env": env}

    def shell_line(self) -> str:
        env_prefix = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in sorted(self.env.items())
        )
        if env_prefix:
            env_prefix += " "
        argv = " ".join(shlex.quote(a) for a in self.argv)
        return f"cd {shlex.quote(self.cwd)} && {env_prefix}{argv}"


__all__ = ["ExternalCommand"]
