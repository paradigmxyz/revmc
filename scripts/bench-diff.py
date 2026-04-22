#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = ["tqdm>=4.67.3"]
# ///
"""Benchmark current branch vs a base revision, collecting codegen and compile-time stats.

Outputs markdown tables suitable for PR descriptions.

Examples:
    # All built-in benchmarks vs main:
    ./scripts/bench-diff.py /tmp/bench main

    # Specific benchmarks:
    ./scripts/bench-diff.py /tmp/bench main seaport usdc_proxy

    # Include mainnet .bin files from a directory:
    ./scripts/bench-diff.py /tmp/bench main --extra-dir tmp/mainnet

    # Current branch only (no diff):
    ./scripts/bench-diff.py /tmp/bench

    # Skip codegen line counts (compile time only):
    ./scripts/bench-diff.py /tmp/bench main --no-codegen

    # Skip compile times:
    ./scripts/bench-diff.py /tmp/bench main --no-compile-time
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

from tqdm import tqdm

from utils import get_benches, repo_root, run_cargo


def find_extra_benches(extra_dirs: list[str], root: str, tmp_dir: str) -> list[str]:
    """Find .bin files in extra directories.

    For directories containing ``bytecode.bin``, creates a uniquely-named
    symlink so the CLI dumps each into its own subdirectory.
    """
    paths = []
    for d in extra_dirs:
        d = os.path.join(root, d) if not os.path.isabs(d) else d
        if not os.path.isdir(d):
            print(f"warning: extra dir {d!r} does not exist, skipping", file=sys.stderr)
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
        # /path/to/0xabcd1234.../bytecode.bin -> 0xabcd1234...
        parent = os.path.basename(os.path.dirname(bench))
        stem = os.path.splitext(os.path.basename(bench))[0]
        name = parent if parent != "" and stem == "bytecode" else stem
        # Truncate long hex names.
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


def collect(benches: list[str], dump_dir: str, root: str):
    """Run all benchmarks and dump IR/asm into dump_dir."""
    subprocess.run(["cargo", "build", "--quiet"], capture_output=True, cwd=root)
    for bench in tqdm(benches, desc="Collecting", unit="bench", file=sys.stderr):
        sub = os.path.join(dump_dir, dump_subdir(bench))
        if os.path.isdir(sub):
            shutil.rmtree(sub)
        run_cargo(["r", "--quiet", "--", "run", bench, "-o", dump_dir], root)


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


def parse_remarks(dump_dir: str, bench: str) -> dict[str, float]:
    """Parse remarks.txt for a benchmark, returning timing fields in seconds."""
    path = os.path.join(dump_dir, dump_subdir(bench), "remarks.txt")
    result = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                # Stop at the "Generated files" section.
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


def print_codegen_table(benches, dump_dir, base_dump, base_label):
    """Print markdown codegen line-count diff table."""
    # Collect data.
    rows = []
    t_bu = t_bo = t_ba = t_mu = t_mo = t_ma = 0
    for bench in benches:
        sub = dump_subdir(bench)
        name = bench_name(bench)
        bu = line_count(os.path.join(dump_dir, sub, "unopt.ll"))
        bo = line_count(os.path.join(dump_dir, sub, "opt.ll"))
        ba = line_count(os.path.join(dump_dir, sub, "opt.s"))
        mu = line_count(os.path.join(base_dump, sub, "unopt.ll"))
        mo = line_count(os.path.join(base_dump, sub, "opt.ll"))
        ma = line_count(os.path.join(base_dump, sub, "opt.s"))
        if bu == 0 and mu == 0:
            continue
        rows.append((name, mu, bu, mo, bo, ma, ba))
        t_bu += bu; t_bo += bo; t_ba += ba
        t_mu += mu; t_mo += mo; t_ma += ma

    # Summary table (diffs only).
    print("### Assembly line counts\n")
    print("| benchmark | unopt.ll | opt.ll | opt.s |")
    print("|:--|--:|--:|--:|")
    for name, mu, bu, mo, bo, ma, ba in rows:
        print(
            f"| {name} "
            f"| {fmt_diff(mu, bu)} ({fmt_pct(mu, bu)}) "
            f"| {fmt_diff(mo, bo)} ({fmt_pct(mo, bo)}) "
            f"| {fmt_diff(ma, ba)} ({fmt_pct(ma, ba)}) |"
        )
    print(
        f"| **TOTAL** "
        f"| **{fmt_diff(t_mu, t_bu)} ({fmt_pct(t_mu, t_bu)})** "
        f"| **{fmt_diff(t_mo, t_bo)} ({fmt_pct(t_mo, t_bo)})** "
        f"| **{fmt_diff(t_ma, t_ba)} ({fmt_pct(t_ma, t_ba)})** |"
    )
    print()

    # Full table under <details>.
    print(f"<details><summary>Full line counts</summary>\n")
    print(f"| benchmark | unopt.ll ({base_label}) | diff | opt.ll ({base_label}) | diff | opt.s ({base_label}) | diff |")
    print("|:--|--:|--:|--:|--:|--:|--:|")
    for name, mu, bu, mo, bo, ma, ba in rows:
        print(
            f"| {name} "
            f"| {mu} | {fmt_diff(mu, bu)} ({fmt_pct(mu, bu)}) "
            f"| {mo} | {fmt_diff(mo, bo)} ({fmt_pct(mo, bo)}) "
            f"| {ma} | {fmt_diff(ma, ba)} ({fmt_pct(ma, ba)}) |"
        )
    print(
        f"| **TOTAL** "
        f"| **{t_mu}** | **{fmt_diff(t_mu, t_bu)} ({fmt_pct(t_mu, t_bu)})** "
        f"| **{t_mo}** | **{fmt_diff(t_mo, t_bo)} ({fmt_pct(t_mo, t_bo)})** "
        f"| **{t_ma}** | **{fmt_diff(t_ma, t_ba)} ({fmt_pct(t_ma, t_ba)})** |"
    )
    print("\n</details>\n")


def print_codegen_table_solo(benches, dump_dir):
    """Print codegen line counts without diff."""
    print("### Assembly line counts\n")
    print("| benchmark | unopt.ll | opt.ll | opt.s |")
    print("|:--|--:|--:|--:|")

    t_u = t_o = t_a = 0
    for bench in benches:
        sub = dump_subdir(bench)
        name = bench_name(bench)
        u = line_count(os.path.join(dump_dir, sub, "unopt.ll"))
        o = line_count(os.path.join(dump_dir, sub, "opt.ll"))
        a = line_count(os.path.join(dump_dir, sub, "opt.s"))
        if u == 0:
            continue
        print(f"| {name} | {u} | {o} | {a} |")
        t_u += u; t_o += o; t_a += a

    print(f"| **TOTAL** | **{t_u}** | **{t_o}** | **{t_a}** |")
    print()


PHASES = ["parse", "translate", "finalize"]


def print_compile_time_table(benches, dump_dir, base_dump, base_label):
    """Print markdown compile-time diff table with per-phase breakdown."""
    # Collect data.
    rows = []
    totals_base = {p: 0.0 for p in ["total"] + PHASES}
    totals_cur = {p: 0.0 for p in ["total"] + PHASES}
    for bench in benches:
        name = bench_name(bench)
        cur = parse_remarks(dump_dir, bench)
        base = parse_remarks(base_dump, bench)
        ct = cur.get("total", 0.0)
        bt = base.get("total", 0.0)
        if ct == 0 and bt == 0:
            continue
        totals_cur["total"] += ct
        totals_base["total"] += bt
        phase_data = {}
        for phase in PHASES:
            bp = base.get(phase, 0.0)
            cp = cur.get(phase, 0.0)
            totals_base[phase] += bp
            totals_cur[phase] += cp
            phase_data[phase] = (bp, cp)
        rows.append((name, bt, ct, phase_data))

    # Summary table (diffs only).
    print("### Compile times\n")
    print("| benchmark | total | parse | translate | finalize |")
    print("|:--|--:|--:|--:|--:|")
    for name, bt, ct, phase_data in rows:
        row = f"| {name} | {fmt_pct(bt, ct)} |"
        for phase in PHASES:
            bp, cp = phase_data[phase]
            row += f" {fmt_pct(bp, cp)} |"
        print(row)
    row = f"| **TOTAL** | **{fmt_pct(totals_base['total'], totals_cur['total'])}** |"
    for phase in PHASES:
        row += f" **{fmt_pct(totals_base[phase], totals_cur[phase])}** |"
    print(row)
    print()

    # Full table under <details>.
    print("<details><summary>Full compile times</summary>\n")
    header = f"| benchmark | {base_label} | branch | diff |"
    for phase in PHASES:
        header += f" {phase} ({base_label}) | {phase} (branch) | {phase} diff |"
    print(header)
    print("|:--|--:|--:|--:|" + "--:|--:|--:|" * len(PHASES))
    for name, bt, ct, phase_data in rows:
        row = f"| {name} | {fmt_dur(bt)} | {fmt_dur(ct)} | {fmt_pct(bt, ct)} |"
        for phase in PHASES:
            bp, cp = phase_data[phase]
            row += f" {fmt_dur(bp)} | {fmt_dur(cp)} | {fmt_pct(bp, cp)} |"
        print(row)
    row = (
        f"| **TOTAL** | **{fmt_dur(totals_base['total'])}** | **{fmt_dur(totals_cur['total'])}** "
        f"| **{fmt_pct(totals_base['total'], totals_cur['total'])}** |"
    )
    for phase in PHASES:
        row += (
            f" **{fmt_dur(totals_base[phase])}** | **{fmt_dur(totals_cur[phase])}** "
            f"| **{fmt_pct(totals_base[phase], totals_cur[phase])}** |"
        )
    print(row)
    print("\n</details>\n")


def print_compile_time_table_solo(benches, dump_dir):
    """Print compile times without diff."""
    print("### Compile times\n")
    print("| benchmark | total | parse | translate | finalize |")
    print("|:--|--:|--:|--:|--:|")

    total = 0.0
    for bench in benches:
        name = bench_name(bench)
        r = parse_remarks(dump_dir, bench)
        t = r.get("total", 0.0)
        if t == 0:
            continue
        total += t
        print(
            f"| {name} | {fmt_dur(t)} | {fmt_dur(r.get('parse', 0.0))} "
            f"| {fmt_dur(r.get('translate', 0.0))} | {fmt_dur(r.get('finalize', 0.0))} |"
        )

    print(f"| **TOTAL** | **{fmt_dur(total)}** | | | |")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark current branch vs a base revision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dump_dir", help="Directory to dump generated files into")
    parser.add_argument("base_rev", nargs="?", help="Git revision to compare against (e.g. main)")
    parser.add_argument("benches", nargs="*", help="Specific benchmarks to run (default: all)")
    parser.add_argument(
        "--extra-dir", action="append", default=[],
        help="Extra directory with .bin files to benchmark (can be repeated)",
    )
    parser.add_argument("--no-codegen", action="store_true", help="Skip codegen line counts")
    parser.add_argument("--no-compile-time", action="store_true", help="Skip compile time table")
    args = parser.parse_args()

    root = repo_root()
    link_dir = tempfile.mkdtemp(prefix="revmc-bench-links-")

    try:
        # Build bench list.
        builtin = args.benches or get_benches(root)
        extra = find_extra_benches(args.extra_dir, root, link_dir)
        benches = builtin + extra

        if not benches:
            print("No benchmarks found.", file=sys.stderr)
            return

        print(f"Benchmarks: {len(builtin)} built-in + {len(extra)} extra", file=sys.stderr)

        # Collect current branch.
        collect(benches, args.dump_dir, root)

        if args.base_rev:
            base_dump = args.dump_dir + ".base"
            base_worktree = tempfile.mkdtemp()
            try:
                subprocess.run(
                    ["git", "worktree", "add", "--detach", "--quiet", base_worktree, args.base_rev],
                    capture_output=True, cwd=root,
                )
                collect(benches, base_dump, base_worktree)
            finally:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", base_worktree],
                    capture_output=True, cwd=root,
                )
                if os.path.isdir(base_worktree):
                    shutil.rmtree(base_worktree)

            if not args.no_codegen:
                print_codegen_table(benches, args.dump_dir, base_dump, args.base_rev)
            if not args.no_compile_time:
                print_compile_time_table(benches, args.dump_dir, base_dump, args.base_rev)
        else:
            if not args.no_codegen:
                print_codegen_table_solo(benches, args.dump_dir)
            if not args.no_compile_time:
                print_compile_time_table_solo(benches, args.dump_dir)
    finally:
        shutil.rmtree(link_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
