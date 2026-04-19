#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = ["tqdm>=4.67.3"]
# ///
"""Reports dynamic jump resolution stats for all benchmarks."""

import re
import sys

from tqdm import tqdm

from utils import cargo_env, get_benches, repo_root, run_cargo, strip_ansi


def last_match(pattern: str, text: str) -> int:
    """Return the last regex capture, or 0 if none."""
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else 0


def count_matches(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


def analyze(bench: str, root: str, env: dict[str, str]) -> dict[str, int]:
    output = strip_ansi(
        run_cargo(["r", "--", "run", bench, "--parse-only"], root, env)
    )

    local_res = last_match(r"local_jumps.*newly_resolved=(\d+)", output)
    non_adj = count_matches(r"resolved non-adjacent jump", output)
    pcr_res = count_matches(r"resolved via PCR hint", output)
    fixpt_res = last_match(r"ba:.*newly_resolved=(\d+)", output)

    ba_unres_matches = re.findall(
        r"analyze:ba: .*unresolved dynamic jumps remain n=(\d+)", output
    )
    ba_ran = "analyze:ba:" in output

    if ba_unres_matches:
        unresolved = int(ba_unres_matches[-1])
    elif ba_ran:
        unresolved = 0
    else:
        unresolved = last_match(
            r"local_jumps:.*unresolved dynamic jumps remain n=(\d+)", output
        )

    total = local_res + fixpt_res + unresolved

    return {
        "local": local_res,
        "non_adj": non_adj,
        "pcr": pcr_res,
        "fixpt": fixpt_res,
        "unresolved": unresolved,
        "total": total,
    }


def main():
    root = repo_root()
    all_benches = get_benches(root)

    if len(sys.argv) > 1:
        selected = set(sys.argv[1:])
        all_benches = [b for b in all_benches if b in selected]

    env = cargo_env(rust_log="revmc::bytecode=trace")

    header = f"{'BENCHMARK':<25} {'LOCAL':>6} {'NONADJ':>6} {'PCR':>6} {'FIXPT':>6} {'UNRES':>6} {'TOTAL':>10}"
    sep = f"{'---------':<25} {'-----':>6} {'------':>6} {'---':>6} {'-----':>6} {'-----':>6} {'-----':>10}"
    print(header)
    print(sep)

    for bench in tqdm(all_benches, desc="Analyzing", unit="bench", file=sys.stderr):
        s = analyze(bench, root, env)
        print(
            f"{bench:<25} {s['local']:>6d} {s['non_adj']:>6d} {s['pcr']:>6d} "
            f"{s['fixpt']:>6d} {s['unresolved']:>6d} {s['total']:>10d}"
        )


if __name__ == "__main__":
    main()
