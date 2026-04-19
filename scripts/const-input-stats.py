#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = ["tqdm>=4.67.3"]
# ///
"""Aggregate per-opcode constant-input statistics across all benchmarks.

Runs each benchmark with RUST_LOG=revmc::bytecode=trace and --parse-only,
parses the trace output, and prints aggregated stats.
"""

import re
import subprocess
import sys

from tqdm import tqdm

from utils import strip_ansi


def parse_trace(output: str):
    """Parse const input stats from combined output.

    Returns dict: opcode_name -> {opbyte, inputs, total, all_const, const_output, per_input}.
    """
    stats = {}
    current = None
    in_block = False

    for raw_line in output.splitlines():
        line = strip_ansi(raw_line)

        if "const input stats:" in line:
            in_block = True
            continue
        if not in_block:
            continue

        # Opcode header, e.g.:
        # "  ADD              0x01, 86 occ, 2 inputs, all_const=39/86 (45%), const_output=39/86 (45%)"
        m = re.match(
            r"\s{2}(\S+)\s+(0x[0-9a-f]{2}), (\d+) occ, (\d+) inputs, all_const=(\d+)/(\d+)"
            r"(?:.*?const_output=(\d+)/(\d+))?",
            line,
        )
        if m:
            name = m.group(1)
            opbyte = int(m.group(2), 16)
            total = int(m.group(3))
            inputs = int(m.group(4))
            all_const = int(m.group(5))
            const_output = int(m.group(7)) if m.group(7) else 0
            current = name
            stats[name] = {
                "opbyte": opbyte,
                "inputs": inputs,
                "total": total,
                "all_const": all_const,
                "const_output": const_output,
                "per_input": [],
            }
            continue

        # Per-input line: "    [0]: const=70/86 (81%), fits_usize=70/86 (81%)"
        m = re.match(
            r"\s{4}\[(\d+)\]: const=(\d+)/(\d+) \(\d+%\), fits_usize=(\d+)/(\d+)",
            line,
        )
        if m and current:
            c = int(m.group(2))
            t = int(m.group(3))
            u = int(m.group(4))
            stats[current]["per_input"].append([t, c, u])
            continue

        # Any other non-empty, non-indented line ends the block.
        if line.strip() and not line.startswith("  "):
            in_block = False
            current = None

    return stats


def merge(aggregate, stats):
    """Merge per-benchmark stats into aggregate."""
    for name, s in stats.items():
        if name not in aggregate:
            aggregate[name] = {
                "opbyte": s["opbyte"],
                "inputs": s["inputs"],
                "total": 0,
                "all_const": 0,
                "const_output": 0,
                "per_input": [[0, 0, 0] for _ in range(s["inputs"])],
            }
        agg = aggregate[name]
        agg["total"] += s["total"]
        agg["all_const"] += s["all_const"]
        agg["const_output"] += s.get("const_output", 0)
        for i, [t, c, u] in enumerate(s["per_input"]):
            agg["per_input"][i][0] += t
            agg["per_input"][i][1] += c
            agg["per_input"][i][2] += u


def pct(a, b):
    return f"{a / b * 100:.0f}%" if b else "0%"


def print_stats(stats, title):
    print(f"\n=== {title} ===")
    for name in sorted(stats, key=lambda n: stats[n].get("opbyte", 0)):
        s = stats[name]
        t = s["total"]
        if t == 0:
            continue
        ac = s["all_const"]
        co = s.get("const_output", 0)
        line = (
            f"  {name:<16} {t} occ, {s['inputs']} inputs, "
            f"all_const={ac}/{t} ({pct(ac, t)})"
        )
        if co:
            line += f", const_output={co}/{t} ({pct(co, t)})"
        print(line)
        for i, [total, const, usize] in enumerate(s["per_input"]):
            if total > 0:
                print(
                    f"    [{i}]: const={const}/{total} ({pct(const, total)}), "
                    f"fits_usize={usize}/{total} ({pct(usize, total)})"
                )


def main():
    benches = sys.argv[1:] if len(sys.argv) > 1 else None

    root = subprocess.os.path.dirname(subprocess.os.path.dirname(__file__)) or "."

    # Get bench list.
    r = subprocess.run(
        ["cargo", "r", "--", "run", "--list"],
        capture_output=True, text=True, cwd=root,
    )
    all_benches = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    if benches:
        all_benches = [b for b in all_benches if b in benches]

    aggregate = {}
    env = {
        **subprocess.os.environ,
        "RUST_LOG": "revmc::bytecode=trace",
        "NO_COLOR": "1",
    }

    for bench in tqdm(all_benches, desc="Analyzing", unit="bench", file=sys.stderr):
        r = subprocess.run(
            ["cargo", "r", "--", "run", bench, "--parse-only"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env, cwd=root,
        )
        stats = parse_trace(r.stdout)
        if stats:
            merge(aggregate, stats)
            print_stats(stats, bench)

    print_stats(aggregate, f"AGGREGATE ({len(all_benches)} benchmarks)")


if __name__ == "__main__":
    main()
