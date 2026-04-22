#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = ["tabulate>=0.9", "tqdm>=4.67.3"]
# ///
"""Unified benchmark tool for revmc.

Collects codegen stats, compile times, jump resolution, and constant-input
statistics across benchmarks. Supports diffing against a base git revision.

Examples:
    # Codegen + compile time diff against main (default):
    ./scripts/bench.py /tmp/bench --diff main

    # All analyses on current branch only:
    ./scripts/bench.py /tmp/bench --codegen-lines --compile-times --jump-resolution --input-stats

    # Only jump resolution for specific benchmarks:
    ./scripts/bench.py /tmp/bench --jump-resolution seaport usdc_proxy

    # Include mainnet .bin files:
    ./scripts/bench.py /tmp/bench --diff main --extra-dir tmp/mainnet

    # Only compile times:
    ./scripts/bench.py /tmp/bench --diff main --compile-times

    # Only codegen line counts:
    ./scripts/bench.py /tmp/bench --diff main --codegen-lines
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod

from tabulate import tabulate
from tqdm import tqdm
from utils import (
    cargo_build,
    cargo_env,
    eprint,
    get_benches,
    repo_root,
    run_cli,
    strip_ansi,
)

# ---------------------------------------------------------------------------
# Bench discovery
# ---------------------------------------------------------------------------


def find_extra_benches(extra_dirs: list[str], root: str, tmp_dir: str) -> list[str]:
    """Find .bin files in extra directories.

    For directories containing ``bytecode.bin``, creates a uniquely-named
    symlink so the CLI dumps each into its own subdirectory.
    """
    paths = []
    for d in extra_dirs:
        d = os.path.join(root, d) if not os.path.isabs(d) else d
        if not os.path.isdir(d):
            eprint(f"warning: extra dir {d!r} does not exist, skipping")
            continue
        for entry in sorted(os.listdir(d)):
            entry_path = os.path.join(d, entry)
            # Directory containing bytecode.bin — symlink with the dir name.
            bin_file = os.path.join(entry_path, "bytecode.bin")
            if os.path.isdir(entry_path) and os.path.isfile(bin_file):
                link = os.path.join(tmp_dir, entry + ".bin")
                os.symlink(os.path.abspath(bin_file), link)
                paths.append(link)
            # Direct .bin file.
            elif os.path.isfile(entry_path) and entry.endswith(".bin"):
                paths.append(os.path.abspath(entry_path))
    return paths


def bench_name(bench: str) -> str:
    """Short display name for a benchmark."""
    if os.path.isabs(bench) or os.sep in bench:
        parent = os.path.basename(os.path.dirname(bench))
        stem = os.path.splitext(os.path.basename(bench))[0]
        name = parent if parent != "" and stem == "bytecode" else stem
        if len(name) > 16 and name.startswith("0x"):
            return name[:12] + "…"
        return name
    return bench


def dump_subdir(bench: str) -> str:
    """Expected subdirectory name in the dump directory for a benchmark."""
    if os.path.isabs(bench) or os.sep in bench:
        parent = os.path.basename(os.path.dirname(bench))
        stem = os.path.splitext(os.path.basename(bench))[0]
        return parent if parent != "" and stem == "bytecode" else stem
    return bench


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _indicator(delta: float) -> str:
    """🟢 for improvement (negative delta), 🔴 for regression, empty for neutral."""
    if delta < 0:
        return " 🟢"
    if delta > 0:
        return " 🔴"
    return ""


def fmt_diff(base: int, current: int) -> str:
    d = current - base
    return "=" if d == 0 else f"{d:+d}"


def fmt_pct(base: float | int, current: float | int) -> str:
    if base == 0:
        return "-"
    pct = (current - base) / base * 100
    return f"{pct:+.1f}%{_indicator(pct)}"


def fmt_dur(s: float) -> str:
    if s == 0:
        return "-"
    if s < 0.001:
        return f"{s * 1e6:.1f}µs"
    if s < 0.01:
        return f"{s * 1e3:.2f}ms"
    if s < 1.0:
        return f"{s * 1e3:.1f}ms"
    return f"{s:.3f}s"


def line_count(path: str) -> int:
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ns|µs|us|ms|s)")
DURATION_UNITS = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def parse_duration(s: str) -> float:
    """Parse a Rust-style Duration debug string to seconds."""
    m = DURATION_RE.search(s)
    if not m:
        return 0.0
    return float(m.group(1)) * DURATION_UNITS[m.group(2)]


def print_table(headers: list[str], rows: list[list], right_cols: int | None = None):
    """Print a markdown table. All columns except the first are right-aligned by default."""
    n = len(headers)
    if right_cols is None:
        right_cols = n - 1
    align = ("left",) + ("right",) * right_cols + ("left",) * (n - 1 - right_cols)
    print(tabulate(rows, headers=headers, tablefmt="pipe", colalign=align))
    print()


def parse_remarks(dump_dir: str, bench: str) -> dict[str, float]:
    """Parse remarks.txt for a benchmark, returning timing fields in seconds."""
    path = os.path.join(dump_dir, dump_subdir(bench), "remarks.txt")
    result = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("Generated files"):
                    break
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip().lstrip("- ")
                    dur = parse_duration(parts[1])
                    if dur > 0:
                        result[key] = dur
    except FileNotFoundError:
        pass
    return result


# ---------------------------------------------------------------------------
# Analysis classes
# ---------------------------------------------------------------------------


class Analysis(ABC):
    """Base class for benchmark analyses."""

    @abstractmethod
    def needs_codegen(self) -> bool:
        """Whether this analysis needs full compilation with -o dump_dir."""
        ...

    @abstractmethod
    def rust_log(self) -> str | None:
        """RUST_LOG value needed, or None."""
        ...

    @abstractmethod
    def report(self, benches: list[str], dump_dir: str, outputs: dict[str, str]):
        """Print results. dump_dir has codegen artifacts, outputs has stdout per bench."""
        ...

    def report_diff(
        self,
        benches: list[str],
        dump_dir: str,
        outputs: dict[str, str],
        base_dump: str,
        base_outputs: dict[str, str],
        base_label: str,
    ):
        """Print diff results. Default falls back to solo report."""
        self.report(benches, dump_dir, outputs)


class CodegenLines(Analysis):
    def needs_codegen(self) -> bool:
        return True

    def rust_log(self) -> str | None:
        return None

    def _collect(self, benches, dump_dir):
        """Collect line counts per file type, returns (rows, totals)."""
        FILES = ["unopt.ll", "opt.ll", "opt.s"]
        rows = []
        totals = [0] * len(FILES)
        for bench in benches:
            sub = dump_subdir(bench)
            counts = [line_count(os.path.join(dump_dir, sub, f)) for f in FILES]
            if all(c == 0 for c in counts):
                continue
            rows.append((bench_name(bench), counts))
            for i, c in enumerate(counts):
                totals[i] += c
        return rows, totals

    def report(self, benches, dump_dir, outputs):
        rows, totals = self._collect(benches, dump_dir)
        print("### Assembly line counts\n")
        table = [[name, *counts] for name, counts in rows]
        table.append(["**TOTAL**", *[f"**{t}**" for t in totals]])
        print_table(["benchmark", "unopt.ll", "opt.ll", "opt.s"], table)

    def report_diff(
        self, benches, dump_dir, outputs, base_dump, base_outputs, base_label
    ):
        cur_rows, cur_totals = self._collect(benches, dump_dir)
        base_rows, base_totals = self._collect(benches, base_dump)
        base_map = dict(base_rows)

        def diff_cell(b, c):
            return f"{fmt_diff(b, c)} ({fmt_pct(b, c)})"

        # Summary table.
        print("### Assembly line counts\n")
        headers = ["benchmark", "unopt.ll", "opt.ll", "opt.s"]
        table = []
        for name, counts in cur_rows:
            base_counts = base_map.get(name, [0] * 3)
            table.append(
                [name, *[diff_cell(b, c) for b, c in zip(base_counts, counts)]]
            )
        table.append(
            [
                "**TOTAL**",
                *[f"**{diff_cell(b, c)}**" for b, c in zip(base_totals, cur_totals)],
            ]
        )
        print_table(headers, table)

        # Detailed table.
        print("<details><summary>Full line counts</summary>\n")
        detail_headers = ["benchmark"]
        for f in ["unopt.ll", "opt.ll", "opt.s"]:
            detail_headers += [f"{f} ({base_label})", "diff"]
        detail_table = []
        for name, counts in cur_rows:
            base_counts = base_map.get(name, [0] * 3)
            row = [name]
            for b, c in zip(base_counts, counts):
                row += [b, diff_cell(b, c)]
            detail_table.append(row)
        total_row = ["**TOTAL**"]
        for b, c in zip(base_totals, cur_totals):
            total_row += [f"**{b}**", f"**{diff_cell(b, c)}**"]
        detail_table.append(total_row)
        print_table(detail_headers, detail_table)
        print("</details>\n")


PHASES = ["parse", "translate", "finalize"]


class CompileTime(Analysis):
    def needs_codegen(self) -> bool:
        return True

    def rust_log(self) -> str | None:
        return None

    def _collect(self, benches, dump_dir):
        """Collect timing data, returns (rows, totals).

        Each row is (name, {phase: dur}). totals is {phase: dur}.
        """
        keys = ["total"] + PHASES
        rows = []
        totals = {p: 0.0 for p in keys}
        for bench in benches:
            r = parse_remarks(dump_dir, bench)
            if r.get("total", 0.0) == 0:
                continue
            rows.append((bench_name(bench), r))
            for p in keys:
                totals[p] += r.get(p, 0.0)
        return rows, totals

    def report(self, benches, dump_dir, outputs):
        rows, totals = self._collect(benches, dump_dir)
        print("### Compile times\n")
        table = [
            [
                name,
                fmt_dur(r.get("total", 0.0)),
                *[fmt_dur(r.get(p, 0.0)) for p in PHASES],
            ]
            for name, r in rows
        ]
        table.append(["**TOTAL**", f"**{fmt_dur(totals['total'])}**", "", "", ""])
        print_table(["benchmark", "total", "parse", "translate", "finalize"], table)

    def report_diff(
        self, benches, dump_dir, outputs, base_dump, base_outputs, base_label
    ):
        cur_rows, cur_totals = self._collect(benches, dump_dir)
        base_rows, base_totals = self._collect(benches, base_dump)
        base_map = dict(base_rows)

        keys = ["total"] + PHASES

        # Summary table.
        print("### Compile times\n")
        table = []
        for name, cur in cur_rows:
            base = base_map.get(name, {})
            table.append(
                [name, *[fmt_pct(base.get(p, 0.0), cur.get(p, 0.0)) for p in keys]]
            )
        table.append(
            [
                "**TOTAL**",
                *[f"**{fmt_pct(base_totals[p], cur_totals[p])}**" for p in keys],
            ]
        )
        print_table(["benchmark", "total", "parse", "translate", "finalize"], table)

        # Detailed table.
        print("<details><summary>Full compile times</summary>\n")
        detail_headers = ["benchmark", base_label, "branch", "diff"]
        for phase in PHASES:
            detail_headers += [
                f"{phase} ({base_label})",
                f"{phase} (branch)",
                f"{phase} diff",
            ]
        detail_table = []
        for name, cur in cur_rows:
            base = base_map.get(name, {})
            bt, ct = base.get("total", 0.0), cur.get("total", 0.0)
            row = [name, fmt_dur(bt), fmt_dur(ct), fmt_pct(bt, ct)]
            for phase in PHASES:
                bp, cp = base.get(phase, 0.0), cur.get(phase, 0.0)
                row += [fmt_dur(bp), fmt_dur(cp), fmt_pct(bp, cp)]
            detail_table.append(row)
        total_row = [
            "**TOTAL**",
            f"**{fmt_dur(base_totals['total'])}**",
            f"**{fmt_dur(cur_totals['total'])}**",
            f"**{fmt_pct(base_totals['total'], cur_totals['total'])}**",
        ]
        for phase in PHASES:
            total_row += [
                f"**{fmt_dur(base_totals[phase])}**",
                f"**{fmt_dur(cur_totals[phase])}**",
                f"**{fmt_pct(base_totals[phase], cur_totals[phase])}**",
            ]
        detail_table.append(total_row)
        print_table(detail_headers, detail_table)
        print("</details>\n")


# ---------------------------------------------------------------------------
# Jump resolution analysis
# ---------------------------------------------------------------------------


class JumpResolution(Analysis):
    def needs_codegen(self) -> bool:
        return False

    def rust_log(self) -> str | None:
        return "revmc::bytecode=trace"

    @staticmethod
    def _last_match(pattern: str, text: str) -> int:
        matches = re.findall(pattern, text)
        return int(matches[-1]) if matches else 0

    @staticmethod
    def _count_matches(pattern: str, text: str) -> int:
        return len(re.findall(pattern, text))

    @classmethod
    def _analyze(cls, output: str) -> dict[str, int]:
        local_res = cls._last_match(r"local_jumps.*newly_resolved=(\d+)", output)
        non_adj = cls._count_matches(r"resolved non-adjacent jump", output)
        pcr_res = cls._count_matches(r"resolved via PCR hint", output)
        fixpt_res = cls._last_match(r"ba:.*newly_resolved=(\d+)", output)

        ba_unres_matches = re.findall(
            r"analyze:ba: .*unresolved dynamic jumps remain n=(\d+)", output
        )
        ba_ran = "analyze:ba:" in output

        if ba_unres_matches:
            unresolved = int(ba_unres_matches[-1])
        elif ba_ran:
            unresolved = 0
        else:
            unresolved = cls._last_match(
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

    def report(self, benches, dump_dir, outputs):
        print("### Jump resolution\n")
        headers = ["benchmark", "local", "nonadj", "pcr", "fixpt", "unres", "total"]
        table = []
        for bench in benches:
            s = self._analyze(outputs.get(bench, ""))
            table.append(
                [
                    bench_name(bench),
                    s["local"],
                    s["non_adj"],
                    s["pcr"],
                    s["fixpt"],
                    s["unresolved"],
                    s["total"],
                ]
            )
        print_table(headers, table)


# ---------------------------------------------------------------------------
# Constant-input stats analysis
# ---------------------------------------------------------------------------


class InputStats(Analysis):
    def needs_codegen(self) -> bool:
        return False

    def rust_log(self) -> str | None:
        return "revmc::bytecode=trace"

    @staticmethod
    def _pct(a, b):
        return f"{a / b * 100:.0f}%" if b else "0%"

    @staticmethod
    def _parse(output: str):
        """Parse const input stats from trace output.

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

            m = re.match(
                r"\s{2}(\S+)\s+(0x[0-9a-f]{2}), (\d+) occ, (\d+) inputs, all_const=(\d+)/(\d+)"
                r"(?:.*?const_output=(\d+)/(\d+))?",
                line,
            )
            if m:
                name = m.group(1)
                current = name
                stats[name] = {
                    "opbyte": int(m.group(2), 16),
                    "inputs": int(m.group(4)),
                    "total": int(m.group(3)),
                    "all_const": int(m.group(5)),
                    "const_output": int(m.group(7)) if m.group(7) else 0,
                    "per_input": [],
                }
                continue

            m = re.match(
                r"\s{4}\[(\d+)\]: const=(\d+)/(\d+) \(\d+%\), fits_usize=(\d+)/(\d+)",
                line,
            )
            if m and current:
                stats[current]["per_input"].append(
                    [int(m.group(3)), int(m.group(2)), int(m.group(4))]
                )
                continue

            if line.strip() and not line.startswith("  "):
                in_block = False
                current = None

        return stats

    @staticmethod
    def _merge(aggregate, stats):
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

    @classmethod
    def _print(cls, stats, title):
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
                f"all_const={ac}/{t} ({cls._pct(ac, t)})"
            )
            if co:
                line += f", const_output={co}/{t} ({cls._pct(co, t)})"
            print(line)
            for i, [total, const, usize] in enumerate(s["per_input"]):
                if total > 0:
                    print(
                        f"    [{i}]: const={const}/{total} ({cls._pct(const, total)}), "
                        f"fits_usize={usize}/{total} ({cls._pct(usize, total)})"
                    )

    def report(self, benches, dump_dir, outputs):
        print("### Constant-input stats\n")
        aggregate = {}
        for bench in benches:
            stats = self._parse(outputs.get(bench, ""))
            if stats:
                self._merge(aggregate, stats)
                self._print(stats, bench_name(bench))
        self._print(aggregate, f"AGGREGATE ({len(benches)} benchmarks)")
        print()


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect(
    benches: list[str],
    binary: str,
    dump_dir: str | None,
    rust_log: str | None,
) -> dict[str, str]:
    """Run benchmarks once and return captured output per bench.

    If dump_dir is set, runs full compilation with ``-o dump_dir``.
    Otherwise runs ``--parse-only``.
    When rust_log is set, stderr is captured too (trace output goes there).
    """
    env = cargo_env(rust_log) if rust_log else None
    outputs: dict[str, str] = {}
    label = "Compiling" if dump_dir else "Analyzing"
    for bench in tqdm(benches, desc=label, unit="bench"):
        args = ["run", bench]
        if dump_dir:
            sub = os.path.join(dump_dir, dump_subdir(bench))
            if os.path.isdir(sub):
                shutil.rmtree(sub)
            args += ["-o", dump_dir]
        else:
            args.append("--parse-only")
        output = run_cli(binary, args, env, capture_stderr=rust_log is not None)
        outputs[bench] = strip_ansi(output)
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark tool for revmc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dump_dir", help="Directory to dump generated files into")
    parser.add_argument(
        "benches", nargs="*", help="Specific benchmarks to run (default: all)"
    )
    parser.add_argument(
        "--diff",
        metavar="REV",
        dest="base_rev",
        help="Git revision to compare against (e.g. main)",
    )
    parser.add_argument(
        "--extra-dir",
        action="append",
        default=[],
        help="Extra directory with .bin files to benchmark (can be repeated)",
    )

    # Analysis selectors. Default: --codegen-lines --compile-times.
    parser.add_argument(
        "--codegen-lines",
        action="store_true",
        help="Assembly line count tables (default)",
    )
    parser.add_argument(
        "--compile-times",
        action="store_true",
        help="Compile time tables (default)",
    )
    parser.add_argument(
        "--jump-resolution",
        action="store_true",
        help="Report dynamic jump resolution stats",
    )
    parser.add_argument(
        "--input-stats",
        action="store_true",
        help="Report per-opcode constant-input statistics",
    )
    args = parser.parse_args()

    # Default to codegen-lines + compile-times if no analysis flags given.
    if not (
        args.codegen_lines
        or args.compile_times
        or args.jump_resolution
        or args.input_stats
    ):
        args.codegen_lines = True
        args.compile_times = True

    # Build the list of active analyses.
    analyses: list[Analysis] = []
    if args.codegen_lines:
        analyses.append(CodegenLines())
    if args.compile_times:
        analyses.append(CompileTime())
    if args.jump_resolution:
        analyses.append(JumpResolution())
    if args.input_stats:
        analyses.append(InputStats())

    if not analyses:
        eprint("No analyses selected.")
        return

    # Determine collection requirements from the union of all analyses.
    need_codegen = any(a.needs_codegen() for a in analyses)
    rust_log_parts = {r for a in analyses if (r := a.rust_log())}
    rust_log = ",".join(sorted(rust_log_parts)) if rust_log_parts else None

    root = repo_root()
    link_dir = tempfile.mkdtemp(prefix="revmc-bench-links-")

    try:
        binary = cargo_build(root)
        builtin = args.benches or get_benches(binary)
        extra = find_extra_benches(args.extra_dir, root, link_dir)
        benches = builtin + extra

        if not benches:
            eprint("No benchmarks found.")
            return

        eprint(f"Benchmarks: {len(builtin)} built-in + {len(extra)} extra")

        dump_dir = args.dump_dir if need_codegen else None
        outputs = collect(benches, binary, dump_dir, rust_log)

        if args.base_rev:
            base_dump = args.dump_dir + ".base"
            base_worktree = tempfile.mkdtemp()
            try:
                subprocess.run(
                    [
                        "git",
                        "worktree",
                        "add",
                        "--detach",
                        "--quiet",
                        base_worktree,
                        args.base_rev,
                    ],
                    check=True,
                    cwd=root,
                )
                base_binary = cargo_build(base_worktree)
                base_dump_dir = base_dump if need_codegen else None
                base_outputs = collect(benches, base_binary, base_dump_dir, rust_log)
            finally:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", base_worktree],
                    capture_output=True,
                    cwd=root,
                )
                if os.path.isdir(base_worktree):
                    shutil.rmtree(base_worktree)

            for a in analyses:
                a.report_diff(
                    benches,
                    args.dump_dir,
                    outputs,
                    base_dump,
                    base_outputs,
                    args.base_rev,
                )
        else:
            for a in analyses:
                a.report(benches, args.dump_dir, outputs)

    finally:
        shutil.rmtree(link_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
