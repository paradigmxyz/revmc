"""Shared utilities for revmc analysis scripts."""

import os
import re
import subprocess


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cargo_env(rust_log: str | None = None) -> dict[str, str]:
    env = {**os.environ, "NO_COLOR": "1"}
    if rust_log:
        env["RUST_LOG"] = rust_log
    return env


def get_benches(root: str) -> list[str]:
    r = subprocess.run(
        ["cargo", "r", "--", "run", "--list"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def run_cargo(
    args: list[str], root: str, env: dict[str, str] | None = None
) -> str:
    r = subprocess.run(
        ["cargo", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=root,
    )
    return r.stdout
