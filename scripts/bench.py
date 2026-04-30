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


def _indicator(delta: float, lower_is_better: bool = True) -> str:
    """🟢 for improvement, 🔴 for regression, empty for neutral."""
    score = delta if lower_is_better else -delta
    if score < 0:
        return " 🟢"
    if score > 0:
        return " 🔴"
    return ""


def fmt_diff(base: int, current: int) -> str:
    d = current - base
    return "=" if d == 0 else f"{d:+d}"


def fmt_pct(
    base: float | int, current: float | int, lower_is_better: bool = True
) -> str:
    if base == 0:
        return "-"
    pct = (current - base) / base * 100
    if round(pct, 1) == 0:
        return "="
    return f"{pct:+.1f}%{_indicator(pct, lower_is_better)}"


def fmt_change(
    base: float | int, current: float | int, lower_is_better: bool = True
) -> str:
    if base == 0:
        delta = current - base
        if delta == 0:
            return "="
        return f"{delta:+g}{_indicator(delta, lower_is_better)}"
    return fmt_pct(base, current, lower_is_better)


def fmt_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "-"
    return f"{numerator}/{denominator} ({numerator / denominator * 100:.0f}%)"


def fmt_ratio_change(
    base_numerator: int,
    base_denominator: int,
    current_numerator: int,
    current_denominator: int,
    lower_is_better: bool = False,
) -> str:
    base = base_numerator / base_denominator if base_denominator else 0.0
    current = current_numerator / current_denominator if current_denominator else 0.0
    return fmt_change(base, current, lower_is_better)


def ratio_is_noise(
    base_numerator: int,
    base_denominator: int,
    current_numerator: int,
    current_denominator: int,
    noise: float,
) -> bool:
    base = base_numerator / base_denominator if base_denominator else 0.0
    current = current_numerator / current_denominator if current_denominator else 0.0
    return is_noise(base, current, noise)


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
NOISE_ANALYSIS = 0.0  # analysis counters are deterministic


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
    KEYS = ["local", "non_adj", "pcr", "fixpt", "unresolved", "total"]
    HEADERS = ["benchmark", "local", "nonadj", "pcr", "fixpt", "unres", "total"]
    LOWER_IS_BETTER = {
        "local": False,
        "non_adj": False,
        "pcr": False,
        "fixpt": False,
        "unresolved": True,
        "total": False,
    }

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

    @classmethod
    def _collect(cls, benches, outputs):
        rows = [(bench_name(bench), cls._analyze(outputs.get(bench, ""))) for bench in benches]
        totals = {k: 0 for k in cls.KEYS}
        for _, stats in rows:
            for key in cls.KEYS:
                totals[key] += stats[key]
        return rows, totals

    def report(self, benches, dump_dir, outputs):
        print("### Jump resolution\n")
        rows, totals = self._collect(benches, outputs)
        table = [[name, *[s[k] for k in self.KEYS]] for name, s in rows]
        table.append(["**TOTAL**", *[f"**{totals[k]}**" for k in self.KEYS]])
        print_table(self.HEADERS, table)

    def report_diff(
        self, benches, dump_dir, outputs, base_dump, base_outputs, base_label
    ):
        cur_rows, cur_totals = self._collect(benches, outputs)
        base_rows, base_totals = self._collect(benches, base_outputs)
        base_map = dict(base_rows)

        print("### Jump resolution\n")
        table = []
        for name, cur in cur_rows:
            base = base_map.get(name)
            if not base:
                table.append([name, *[cur[k] for k in self.KEYS]])
                continue
            if all(base[k] == cur[k] for k in self.KEYS):
                continue
            table.append(
                [
                    name,
                    *[
                        fmt_change(
                            base[k],
                            cur[k],
                            lower_is_better=self.LOWER_IS_BETTER[k],
                        )
                        for k in self.KEYS
                    ],
                ]
            )
        table.append(
            [
                "**TOTAL**",
                *[
                    f"**{fmt_change(base_totals[k], cur_totals[k], lower_is_better=self.LOWER_IS_BETTER[k])}**"
                    for k in self.KEYS
                ],
            ]
        )
        print_table(self.HEADERS, table)

        print("<details><summary>Full jump resolution</summary>\n")
        detail_table = []
        for name, cur in cur_rows:
            base = base_map.get(name, {k: 0 for k in self.KEYS})
            detail_table.append(
                [
                    name,
                    *[
                        fmt_detail(
                            cur[k],
                            base[k],
                            fmt_change(
                                base[k],
                                cur[k],
                                lower_is_better=self.LOWER_IS_BETTER[k],
                            ),
                        )
                        for k in self.KEYS
                    ],
                ]
            )
        detail_table.append(
            [
                "**TOTAL**",
                *[
                    fmt_detail_bold(
                        cur_totals[k],
                        base_totals[k],
                        fmt_change(
                            base_totals[k],
                            cur_totals[k],
                            lower_is_better=self.LOWER_IS_BETTER[k],
                        ),
                    )
                    for k in self.KEYS
                ],
            ]
        )
        print_table(self.HEADERS, detail_table)
        print("</details>\n")


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

    @staticmethod
    def _collect(benches, outputs):
        rows = []
        for bench in benches:
            stats = BlockStats._parse(outputs.get(bench, ""))
            if stats:
                rows.append((bench_name(bench), stats))
        return rows

    def report(self, benches, dump_dir, outputs):
        print("### IR stats\n")
        headers = ["benchmark"] + IR_STAT_KEYS
        rows = self._collect(benches, outputs)
        table = [[name] + [s[k] for k in IR_STAT_KEYS] for name, s in rows]
        print_table(headers, table)

    def report_diff(
        self, benches, dump_dir, outputs, base_dump, base_outputs, base_label
    ):
        cur_rows = self._collect(benches, outputs)
        base_rows = self._collect(benches, base_outputs)
        base_map = dict(base_rows)

        print("### IR stats\n")
        headers = ["benchmark"] + IR_STAT_KEYS
        table = []
        for name, cur in cur_rows:
            base = base_map.get(name)
            if not base:
                table.append([name] + [cur[k] for k in IR_STAT_KEYS])
                continue
            pairs = [(base[k], cur[k]) for k in IR_STAT_KEYS]
            if all(is_noise(b, c, NOISE_ANALYSIS) for b, c in pairs):
                continue
            table.append([name] + [fmt_change(b, c) for b, c in pairs])
        print_table(headers, table)

        print("<details><summary>Full IR stats</summary>\n")
        detail_table = []
        for name, cur in cur_rows:
            base = base_map.get(name, {k: 0 for k in IR_STAT_KEYS})
            detail_table.append(
                [
                    name,
                    *[
                        fmt_detail(cur[k], base[k], fmt_change(base[k], cur[k]))
                        for k in IR_STAT_KEYS
                    ],
                ]
            )
        print_table(headers, detail_table)
        print("</details>\n")


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
    def _collect(cls, benches, outputs):
        rows = []
        aggregate = {}
        for bench in benches:
            stats = cls._parse(outputs.get(bench, ""))
            if stats:
                rows.append((bench_name(bench), stats))
                cls._merge(aggregate, stats)
        return rows, aggregate

    @staticmethod
    def _per_input_totals(stats):
        total = sum(i[0] for i in stats["per_input"])
        const = sum(i[1] for i in stats["per_input"])
        usize = sum(i[2] for i in stats["per_input"])
        return total, const, usize

    @classmethod
    def _stat_table(cls, stats):
        rows = []
        for name in sorted(stats, key=lambda n: stats[n].get("opbyte", 0)):
            s = stats[name]
            total = s["total"]
            if total == 0:
                continue
            input_total, input_const, input_usize = cls._per_input_totals(s)
            rows.append(
                [
                    name,
                    total,
                    s["inputs"],
                    fmt_ratio(s["all_const"], total),
                    fmt_ratio(s.get("const_output", 0), total),
                    fmt_ratio(input_const, input_total),
                    fmt_ratio(input_usize, input_total),
                ]
            )
        return rows

    @classmethod
    def _stat_detail(cls, stats):
        parts = []
        for i, (total, const, usize) in enumerate(stats["per_input"]):
            if total > 0:
                parts.append(
                    f"[{i}] const={fmt_ratio(const, total)}, usize={fmt_ratio(usize, total)}"
                )
        return "; ".join(parts) if parts else "-"

    def report(self, benches, dump_dir, outputs):
        print("### Constant-input stats\n")
        rows, aggregate = self._collect(benches, outputs)
        headers = [
            "opcode",
            "occ",
            "inputs",
            "all const",
            "const output",
            "input const",
            "fits usize",
        ]
        print_table(headers, self._stat_table(aggregate))

        print("<details><summary>Per-benchmark constant-input stats</summary>\n")
        for name, stats in rows:
            print(f"#### {name}\n")
            print_table(headers, self._stat_table(stats))
        print("</details>\n")

    def report_diff(
        self, benches, dump_dir, outputs, base_dump, base_outputs, base_label
    ):
        _, cur = self._collect(benches, outputs)
        _, base = self._collect(benches, base_outputs)
        names = sorted(
            set(cur) | set(base),
            key=lambda n: cur.get(n, base.get(n, {})).get("opbyte", 0),
        )

        print("### Constant-input stats\n")
        headers = [
            "opcode",
            "occ",
            "all const",
            "const output",
            "input const",
            "fits usize",
        ]
        table = []
        for name in names:
            c = cur.get(name)
            b = base.get(name)
            if not c:
                continue
            if not b:
                table.append([name, c["total"], "-", "-", "-", "-"])
                continue

            c_input_total, c_input_const, c_input_usize = self._per_input_totals(c)
            b_input_total, b_input_const, b_input_usize = self._per_input_totals(b)
            changes = [
                not is_noise(b["total"], c["total"], NOISE_ANALYSIS),
                not ratio_is_noise(
                    b["all_const"], b["total"], c["all_const"], c["total"], NOISE_ANALYSIS
                ),
                not ratio_is_noise(
                    b.get("const_output", 0),
                    b["total"],
                    c.get("const_output", 0),
                    c["total"],
                    NOISE_ANALYSIS,
                ),
                not ratio_is_noise(
                    b_input_const,
                    b_input_total,
                    c_input_const,
                    c_input_total,
                    NOISE_ANALYSIS,
                ),
                not ratio_is_noise(
                    b_input_usize,
                    b_input_total,
                    c_input_usize,
                    c_input_total,
                    NOISE_ANALYSIS,
                ),
            ]
            if not any(changes):
                continue
            table.append(
                [
                    name,
                    fmt_change(b["total"], c["total"]),
                    fmt_ratio_change(b["all_const"], b["total"], c["all_const"], c["total"]),
                    fmt_ratio_change(
                        b.get("const_output", 0),
                        b["total"],
                        c.get("const_output", 0),
                        c["total"],
                    ),
                    fmt_ratio_change(
                        b_input_const,
                        b_input_total,
                        c_input_const,
                        c_input_total,
                    ),
                    fmt_ratio_change(
                        b_input_usize,
                        b_input_total,
                        c_input_usize,
                        c_input_total,
                    ),
                ]
            )
        print_table(headers, table)

        print("<details><summary>Full constant-input stats</summary>\n")
        detail_headers = [
            "opcode",
            "occ",
            "inputs",
            "all const",
            "const output",
            "input const",
            "fits usize",
            "per input",
        ]
        detail_table = []
        for name in names:
            c = cur.get(name)
            b = base.get(name)
            if not c:
                continue
            if not b:
                detail_table.append(
                    [
                        name,
                        c["total"],
                        c["inputs"],
                        fmt_ratio(c["all_const"], c["total"]),
                        fmt_ratio(c.get("const_output", 0), c["total"]),
                        "-",
                        "-",
                        self._stat_detail(c),
                    ]
                )
                continue

            c_input_total, c_input_const, c_input_usize = self._per_input_totals(c)
            b_input_total, b_input_const, b_input_usize = self._per_input_totals(b)
            detail_table.append(
                [
                    name,
                    fmt_detail(
                        c["total"],
                        b["total"],
                        fmt_change(b["total"], c["total"]),
                    ),
                    fmt_detail(c["inputs"], b["inputs"], fmt_diff(b["inputs"], c["inputs"])),
                    fmt_detail(
                        fmt_ratio(c["all_const"], c["total"]),
                        fmt_ratio(b["all_const"], b["total"]),
                        fmt_ratio_change(
                            b["all_const"], b["total"], c["all_const"], c["total"]
                        ),
                    ),
                    fmt_detail(
                        fmt_ratio(c.get("const_output", 0), c["total"]),
                        fmt_ratio(b.get("const_output", 0), b["total"]),
                        fmt_ratio_change(
                            b.get("const_output", 0),
                            b["total"],
                            c.get("const_output", 0),
                            c["total"],
                        ),
                    ),
                    fmt_detail(
                        fmt_ratio(c_input_const, c_input_total),
                        fmt_ratio(b_input_const, b_input_total),
                        fmt_ratio_change(
                            b_input_const,
                            b_input_total,
                            c_input_const,
                            c_input_total,
                        ),
                    ),
                    fmt_detail(
                        fmt_ratio(c_input_usize, c_input_total),
                        fmt_ratio(b_input_usize, b_input_total),
                        fmt_ratio_change(
                            b_input_usize,
                            b_input_total,
                            c_input_usize,
                            c_input_total,
                        ),
                    ),
                    fmt_detail(
                        self._stat_detail(c),
                        self._stat_detail(b),
                        "=" if self._stat_detail(c) == self._stat_detail(b) else "changed",
                    ),
                ]
            )
        print_table(detail_headers, detail_table)
        print("</details>\n")


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
