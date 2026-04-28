"""Shared utilities for revmc analysis scripts."""

import os
import re
import subprocess
import sys

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shared_target_dir() -> str:
    """Return a shared CARGO_TARGET_DIR so worktrees reuse the same build cache."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target")


def cargo_env(rust_log: str | None = None) -> dict[str, str]:
    env = {**os.environ, "NO_COLOR": "1", "CARGO_TARGET_DIR": _shared_target_dir()}
    if rust_log:
        env["RUST_LOG"] = rust_log
    return env


def cargo_build(root: str, incremental: bool = True) -> str:
    """Build the CLI binary and return its path."""
    target_dir = _shared_target_dir()
    env = {**os.environ, "CARGO_TARGET_DIR": target_dir}
    if not incremental:
        env["CARGO_INCREMENTAL"] = "0"
    subprocess.run(["cargo", "build", "--quiet"], check=True, cwd=root, env=env)
    return os.path.join(target_dir, "debug", "revmc")


def get_benches(binary: str) -> list[str]:
    r = subprocess.run(
        [binary, "run", "--list"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def run_cli(
    binary: str,
    args: list[str],
    env: dict[str, str] | None = None,
    capture_stderr: bool = False,
) -> str:
    """Run the CLI binary and return its stdout.

    stderr is printed to the terminal.  When *capture_stderr* is True it is
    also appended to the returned string (needed for trace-log analyses).
    """
    r = subprocess.run(
        [binary, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if r.stderr:
        sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise subprocess.CalledProcessError(
            r.returncode,
            r.args,
            output=r.stdout,
            stderr=r.stderr,
        )
    if capture_stderr:
        return r.stdout + r.stderr
    return r.stdout
