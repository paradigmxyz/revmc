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
import sys
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


def find_extra_benches(extra_dirs: list[str], root: str) -> list[str]:
    """Find benchmarks in extra directories.

    For directories containing ``bytecode.bin``, returns the directory path —
    the CLI accepts those directly and uses the directory name as the bench
    name. For bare ``.bin`` files, returns the absolute path.
    """
    paths = []
    for d in extra_dirs:
        d = os.path.join(root, d) if not os.path.isabs(d) else d
        if not os.path.isdir(d):
            eprint(f"warning: extra dir {d!r} does not exist, skipping")
            continue
        for entry in sorted(os.listdir(d)):
            entry_path = os.path.join(d, entry)
            # Directory containing bytecode.bin — pass the directory.
            if os.path.isdir(entry_path) and os.path.isfile(
                os.path.join(entry_path, "bytecode.bin")
            ):
                paths.append(os.path.abspath(entry_path))
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
    if round(pct, 1) == 0:
        return "="
    return f"{pct:+.1f}%{_indicator(pct)}"


def fmt_detail(current: str | int, base: str | int, diff: str) -> str:
    return f"{base}<br>{current}<br>{diff}"


def fmt_detail_bold(current: str | int, base: str | int, diff: str) -> str:
    return f"**{fmt_detail(current, base, diff)}**"


def is_noise(base: float | int, current: float | int, noise: float) -> bool:
    """Whether the relative change is within ``noise`` percent (or unmeasurable)."""
    if base == 0:
        return current == 0
    pct = (current - base) / base * 100
    return abs(pct) <= noise


# Noise thresholds for skipping rows in the always-visible summary tables.
# Rows where every change is within the threshold are dropped from the summary
# (the `<details>` table still shows them).
NOISE_CODEGEN = 1.0  # codegen line counts, jit size, spills, reloads
NOISE_TIME = 5.0  # compile times (high run-to-run variance)


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


def fmt_size(b: int) -> str:
    if b == 0:
        return "-"
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KiB"
    return f"{b / 1024 / 1024:.1f} MiB"


def line_count(path: str) -> int:
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def spill_reload_counts(path: str) -> tuple[int, int]:
    """Count spill and reload annotations in an assembly file."""
    spills = 0
    reloads = 0
    try:
        with open(path) as f:
            for line in f:
                if "Spill" in line:
                    spills += 1
                if "Reload" in line:
                    reloads += 1
    except FileNotFoundError:
        pass
    return spills, reloads


def i256_load_store_counts(path: str) -> tuple[int, int]:
    """Count 'load i256' and 'store i256' instructions in an LLVM IR file."""
    loads = 0
    stores = 0
    try:
        with open(path) as f:
            for line in f:
                if "load i256" in line:
                    loads += 1
                if "store i256" in line:
                    stores += 1
    except FileNotFoundError:
        pass
    return loads, stores


def mresize_call_count(path: str) -> int:
    """Count mresize builtin calls in an LLVM IR file."""
    calls = 0
    try:
        with open(path) as f:
            for line in f:
                if "call" in line and "@__revmc_builtin_mresize" in line:
                    calls += 1
    except FileNotFoundError:
        pass
    return calls


DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ns|µs|us|ms|s)")
DURATION_UNITS = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def parse_duration(s: str) -> float:
    """Parse a Rust-style Duration debug string to seconds."""
    m = DURATION_RE.search(s)
    if not m:
        return 0.0
    return float(m.group(1)) * DURATION_UNITS[m.group(2)]


class _Tee:
    """Write to multiple streams simultaneously (e.g. stdout + a file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()

    def isatty(self):
        return False


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


JIT_SIZE_RE = re.compile(r"(?:\(|:\s)(\d+)\s*B\)?")


def parse_jit_size(dump_dir: str, bench: str) -> int:
    """Parse JIT code size in bytes from remarks.txt."""
    path = os.path.join(dump_dir, dump_subdir(bench), "remarks.txt")
    in_section = False
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("JIT code sizes"):
                    in_section = True
                    continue
                if in_section:
                    m = JIT_SIZE_RE.search(line)
                    if m:
                        return int(m.group(1))
    except FileNotFoundError:
        pass
    return 0


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
        """Collect line counts per file type + JIT size + i256 loads/stores + spills/reloads.

        Each row is (name, [line_counts...], jit_size, i256_loads, i256_stores, mresize, spills, reloads).
        """
        FILES = ["unopt.ll", "opt.ll", "opt.s"]
        rows = []
        totals = [0] * len(FILES)
        total_size = 0
        total_i256_loads = 0
        total_i256_stores = 0
        total_mresize = 0
        total_spills = 0
        total_reloads = 0
        for bench in benches:
            sub = dump_subdir(bench)
            counts = [line_count(os.path.join(dump_dir, sub, f)) for f in FILES]
            if all(c == 0 for c in counts):
                continue
            jit_size = parse_jit_size(dump_dir, bench)
            opt_ll = os.path.join(dump_dir, sub, "opt.ll")
            i256_loads, i256_stores = i256_load_store_counts(opt_ll)
            mresize = mresize_call_count(opt_ll)
            spills, reloads = spill_reload_counts(os.path.join(dump_dir, sub, "opt.s"))
            rows.append(
                (
                    bench_name(bench),
                    counts,
                    jit_size,
                    i256_loads,
                    i256_stores,
                    mresize,
                    spills,
                    reloads,
                )
            )
            for i, c in enumerate(counts):
                totals[i] += c
            total_size += jit_size
            total_i256_loads += i256_loads
            total_i256_stores += i256_stores
            total_mresize += mresize
            total_spills += spills
            total_reloads += reloads
        return (
            rows,
            totals,
            total_size,
            total_i256_loads,
            total_i256_stores,
            total_mresize,
            total_spills,
            total_reloads,
        )

    def report(self, benches, dump_dir, outputs):
        (
            rows,
            totals,
            total_size,
            total_i256_ld,
            total_i256_st,
            total_mresize,
            total_spills,
            total_reloads,
        ) = self._collect(benches, dump_dir)
        print("### Codegen statistics\n")
        table = [
            [name, *counts, fmt_size(jit_size), i256_ld, i256_st, mresize, spills, reloads]
            for name, counts, jit_size, i256_ld, i256_st, mresize, spills, reloads in rows
        ]
        table.append(
            [
                "**TOTAL**",
                *[f"**{t}**" for t in totals],
                f"**{fmt_size(total_size)}**",
                f"**{total_i256_ld}**",
                f"**{total_i256_st}**",
                f"**{total_mresize}**",
                f"**{total_spills}**",
                f"**{total_reloads}**",
            ]
        )
        print_table(
            [
                "benchmark",
                "unopt.ll",
                "opt.ll",
                "opt.s",
                "jit size",
                "i256 loads",
                "i256 stores",
                "mresize",
                "spills",
                "reloads",
            ],
            table,
        )

    def report_diff(
        self, benches, dump_dir, outputs, base_dump, base_outputs, base_label
    ):
        cur_rows, cur_totals, cur_total_size, cur_tld, cur_tst, cur_tmresize, cur_tsp, cur_trl = (
            self._collect(benches, dump_dir)
        )
        (
            base_rows,
            base_totals,
            base_total_size,
            base_tld,
            base_tst,
            base_tmresize,
            base_tsp,
            base_trl,
        ) = self._collect(benches, base_dump)
        base_map = {
            name: (counts, jit_size, i256_ld, i256_st, mresize, spills, reloads)
            for name, counts, jit_size, i256_ld, i256_st, mresize, spills, reloads in base_rows
        }

        # Summary table.
        print("### Codegen statistics\n")
        headers = [
            "benchmark",
            "unopt.ll",
            "opt.ll",
            "opt.s",
            "jit size",
            "i256 loads",
            "i256 stores",
            "mresize",
            "spills",
            "reloads",
        ]
        table = []
        n = NOISE_CODEGEN
        for name, counts, jit_size, i256_ld, i256_st, mresize, spills, reloads in cur_rows:
            base_counts, base_jit, base_ld, base_st, base_mresize, base_sp, base_rl = base_map.get(
                name, ([0] * 3, 0, 0, 0, 0, 0, 0)
            )
            pairs = [
                *list(zip(base_counts, counts)),
                (base_jit, jit_size),
                (base_ld, i256_ld),
                (base_st, i256_st),
                (base_mresize, mresize),
                (base_sp, spills),
                (base_rl, reloads),
            ]
            if all(is_noise(b, c, n) for b, c in pairs):
                continue
            table.append([name, *[fmt_pct(b, c) for b, c in pairs]])
        table.append(
            [
                "**TOTAL**",
                *[f"**{fmt_pct(b, c)}**" for b, c in zip(base_totals, cur_totals)],
                f"**{fmt_pct(base_total_size, cur_total_size)}**",
                f"**{fmt_pct(base_tld, cur_tld)}**",
                f"**{fmt_pct(base_tst, cur_tst)}**",
                f"**{fmt_pct(base_tmresize, cur_tmresize)}**",
                f"**{fmt_pct(base_tsp, cur_tsp)}**",
                f"**{fmt_pct(base_trl, cur_trl)}**",
            ]
        )
        print_table(headers, table)

        # Detailed table.
        print("<details><summary>Full details</summary>\n")
        detail_headers = [
            "benchmark",
            "unopt.ll",
            "opt.ll",
            "opt.s",
            "jit size",
            "i256 loads",
            "i256 stores",
            "mresize",
            "spills",
            "reloads",
        ]
        detail_table = []
        for name, counts, jit_size, i256_ld, i256_st, mresize, spills, reloads in cur_rows:
            base_counts, base_jit, base_ld, base_st, base_mresize, base_sp, base_rl = base_map.get(
                name, ([0] * 3, 0, 0, 0, 0, 0, 0)
            )
            row = [name]
            for b, c in zip(base_counts, counts):
                row.append(fmt_detail(c, b, fmt_pct(b, c)))
            row.append(
                fmt_detail(fmt_size(jit_size), fmt_size(base_jit), fmt_pct(base_jit, jit_size))
            )
            row.append(fmt_detail(i256_ld, base_ld, fmt_pct(base_ld, i256_ld)))
            row.append(fmt_detail(i256_st, base_st, fmt_pct(base_st, i256_st)))
            row.append(fmt_detail(mresize, base_mresize, fmt_pct(base_mresize, mresize)))
            row.append(fmt_detail(spills, base_sp, fmt_pct(base_sp, spills)))
            row.append(fmt_detail(reloads, base_rl, fmt_pct(base_rl, reloads)))
            detail_table.append(row)
        total_row = ["**TOTAL**"]
        for b, c in zip(base_totals, cur_totals):
            total_row.append(fmt_detail_bold(c, b, fmt_pct(b, c)))
        total_row.append(
            fmt_detail_bold(
                fmt_size(cur_total_size),
                fmt_size(base_total_size),
                fmt_pct(base_total_size, cur_total_size),
            )
        )
        total_row.append(fmt_detail_bold(cur_tld, base_tld, fmt_pct(base_tld, cur_tld)))
        total_row.append(fmt_detail_bold(cur_tst, base_tst, fmt_pct(base_tst, cur_tst)))
        total_row.append(fmt_detail_bold(cur_tmresize, base_tmresize, fmt_pct(base_tmresize, cur_tmresize)))
        total_row.append(fmt_detail_bold(cur_tsp, base_tsp, fmt_pct(base_tsp, cur_tsp)))
        total_row.append(fmt_detail_bold(cur_trl, base_trl, fmt_pct(base_trl, cur_trl)))
        detail_table.append(total_row)
        print_table(detail_headers, detail_table)
        print("</details>\n")


PHASES = ["parse", "translate", "finalize", "codegen"]


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
        table.append(["**TOTAL**", f"**{fmt_dur(totals['total'])}**", "", "", "", ""])
        print_table(
            ["benchmark", "total", "parse", "translate", "finalize", "codegen"], table
        )

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
        n = NOISE_TIME
        for name, cur in cur_rows:
            base = base_map.get(name, {})
            pairs = [(base.get(p, 0.0), cur.get(p, 0.0)) for p in keys]
            if all(is_noise(b, c, n) for b, c in pairs):
                continue
            table.append([name, *[fmt_pct(b, c) for b, c in pairs]])
        table.append(
            [
                "**TOTAL**",
                *[f"**{fmt_pct(base_totals[p], cur_totals[p])}**" for p in keys],
            ]
        )
        print_table(
            ["benchmark", "total", "parse", "translate", "finalize", "codegen"], table
        )

        # Detailed table.
        print("<details><summary>Full compile times</summary>\n")
        detail_headers = ["benchmark", "total", *PHASES]
        detail_table = []
        for name, cur in cur_rows:
            base = base_map.get(name, {})
            bt, ct = base.get("total", 0.0), cur.get("total", 0.0)
            row = [name, fmt_detail(fmt_dur(ct), fmt_dur(bt), fmt_pct(bt, ct))]
            for phase in PHASES:
                bp, cp = base.get(phase, 0.0), cur.get(phase, 0.0)
                row.append(fmt_detail(fmt_dur(cp), fmt_dur(bp), fmt_pct(bp, cp)))
            detail_table.append(row)
        total_row = [
            "**TOTAL**",
            fmt_detail_bold(
                fmt_dur(cur_totals["total"]),
                fmt_dur(base_totals["total"]),
                fmt_pct(base_totals["total"], cur_totals["total"]),
            ),
        ]
        for phase in PHASES:
            total_row.append(
                fmt_detail_bold(
                    fmt_dur(cur_totals[phase]),
                    fmt_dur(base_totals[phase]),
                    fmt_pct(base_totals[phase], cur_totals[phase]),
                )
            )
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
        return "revmc_codegen::bytecode=trace"

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
# IR stats analysis
# ---------------------------------------------------------------------------

IR_STAT_KEYS = [
    "total_insts",
    "live",
    "dead",
    "noops",
    "suspends",
    "blocks",
    "block_min",
    "block_max",
    "block_avg",
    "block_median",
]


class BlockStats(Analysis):
    def needs_codegen(self) -> bool:
        return False

    def rust_log(self) -> str | None:
        return "revmc_codegen::bytecode=trace"

    @staticmethod
    def _parse(output: str) -> dict[str, float] | None:
        m = re.search(
            r"ir stats"
            r" total_insts=(\d+)"
            r" live=(\d+)"
            r" dead=(\d+)"
            r" noops=(\d+)"
            r" suspends=(\d+)"
            r" blocks=(\d+)"
            r" block_min=(\d+)"
            r" block_max=(\d+)"
            r" block_avg=([\d.]+)"
            r" block_median=(\d+)",
            output,
        )
        if not m:
            return None
        return {
            "total_insts": int(m.group(1)),
            "live": int(m.group(2)),
            "dead": int(m.group(3)),
            "noops": int(m.group(4)),
            "suspends": int(m.group(5)),
            "blocks": int(m.group(6)),
            "block_min": int(m.group(7)),
            "block_max": int(m.group(8)),
            "block_avg": float(m.group(9)),
            "block_median": int(m.group(10)),
        }

    def report(self, benches, dump_dir, outputs):
        print("### IR stats\n")
        headers = ["benchmark"] + IR_STAT_KEYS
        table = []
        for bench in benches:
            s = self._parse(outputs.get(bench, ""))
            if not s:
                continue
            table.append([bench_name(bench)] + [s[k] for k in IR_STAT_KEYS])
        print_table(headers, table)

    def report_diff(
        self, benches, dump_dir, outputs, base_dump, base_outputs, base_label
    ):
        print("### IR stats\n")
        headers = ["benchmark"] + IR_STAT_KEYS
        table = []
        for bench in benches:
            cur = self._parse(outputs.get(bench, ""))
            base = self._parse(base_outputs.get(bench, ""))
            if not cur:
                continue
            if not base:
                table.append([bench_name(bench)] + [cur[k] for k in IR_STAT_KEYS])
            else:
                table.append(
                    [bench_name(bench)]
                    + [fmt_pct(base[k], cur[k]) for k in IR_STAT_KEYS]
                )
        print_table(headers, table)


# ---------------------------------------------------------------------------
# Constant-input stats analysis
# ---------------------------------------------------------------------------


class InputStats(Analysis):
    def needs_codegen(self) -> bool:
        return False

    def rust_log(self) -> str | None:
        return "revmc_codegen::bytecode=trace"

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
        default=None,
        help="Extra directory with .bin files to benchmark (can be repeated). "
        "Defaults to ['tmp/mainnet'] when running all benchmarks; "
        "pass any --extra-dir to override or to include extras with explicit "
        "benchmarks.",
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
    parser.add_argument(
        "--block-stats",
        action="store_true",
        help="Report IR stats (inst counts, block size distribution, suspends)",
    )
    args = parser.parse_args()

    # Default to codegen-lines + compile-times if no analysis flags given.
    if not (
        args.codegen_lines
        or args.compile_times
        or args.jump_resolution
        or args.input_stats
        or args.block_stats
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
    if args.block_stats:
        analyses.append(BlockStats())

    if not analyses:
        eprint("No analyses selected.")
        return

    # Determine collection requirements from the union of all analyses.
    need_codegen = any(a.needs_codegen() for a in analyses)
    rust_log_parts = {r for a in analyses if (r := a.rust_log())}
    rust_log = ",".join(sorted(rust_log_parts)) if rust_log_parts else None

    root = repo_root()

    binary = cargo_build(root)
    builtin = args.benches or get_benches(binary)
    extra_dirs = (
        args.extra_dir
        if args.extra_dir is not None
        else ([] if args.benches else ["tmp/mainnet"])
    )
    extra = find_extra_benches(extra_dirs, root)
    benches = builtin + extra

    if not benches:
        eprint("No benchmarks found.")
        return

    eprint(f"Benchmarks: {len(builtin)} built-in + {len(extra)} extra")

    dump_dir = args.dump_dir if need_codegen else None
    outputs = collect(benches, binary, dump_dir, rust_log)

    if args.base_rev:
        # Detect self-diff: abort if base_rev resolves to the same commit as HEAD.
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        ).stdout.strip()
        base_sha = subprocess.run(
            ["git", "rev-parse", args.base_rev],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        ).stdout.strip()
        if head_sha == base_sha:
            eprint(
                f"error: --diff {args.base_rev!r} resolves to HEAD ({head_sha[:12]}); "
                f"diffing a branch against itself is pointless"
            )
            raise SystemExit(1)

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
            base_binary = cargo_build(base_worktree, incremental=False)
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

        def run_reports():
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

        def run_reports():
            for a in analyses:
                a.report(benches, args.dump_dir, outputs)

    os.makedirs(args.dump_dir, exist_ok=True)
    results_path = os.path.join(args.dump_dir, "results.md")
    with open(results_path, "w") as md:
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, md)
        try:
            run_reports()
        finally:
            sys.stdout = old_stdout
    eprint(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
