#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = ["tqdm>=4.67.3"]
# ///
"""Reports line counts of generated LLVM IR and assembly for all benchmarks."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from tqdm import tqdm

from utils import get_benches, repo_root, run_cargo


def collect(benches: list[str], dump_dir: str, root: str):
    subprocess.run(
        ["cargo", "build", "--quiet"],
        capture_output=True,
        cwd=root,
    )
    for bench in tqdm(benches, desc="Collecting", unit="bench", file=sys.stderr):
        bench_dir = os.path.join(dump_dir, bench)
        if os.path.isdir(bench_dir):
            shutil.rmtree(bench_dir)
        run_cargo(["r", "--", "run", bench, "-o", dump_dir], root)


def line_count(path: str) -> int:
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def fmt_diff(base: int, current: int) -> str:
    d = current - base
    return "    =" if d == 0 else f"{d:+5d}"


def fmt_pct(base: int, current: int) -> str:
    if base == 0:
        return "    -"
    return f"{(current - base) / base * 100:+.1f}%"


def main():
    parser = argparse.ArgumentParser(
        description="Report line counts of generated LLVM IR and assembly."
    )
    parser.add_argument("dump_dir", help="Directory to dump generated files into")
    parser.add_argument("--diff", metavar="REV", help="Compare against a git revision")
    parser.add_argument("benches", nargs="*", help="Specific benchmarks to run")
    args = parser.parse_args()

    root = repo_root()
    benches = args.benches or get_benches(root)

    # Collect current results.
    collect(benches, args.dump_dir, root)

    if args.diff:
        base_dump = args.dump_dir + ".base"
        base_worktree = tempfile.mkdtemp()
        try:
            subprocess.run(
                ["git", "worktree", "add", "--detach", "--quiet", base_worktree, args.diff],
                capture_output=True,
                cwd=root,
            )
            collect(benches, base_dump, base_worktree)
            print(f"Base dump saved to: {base_dump}", file=sys.stderr)

            print(
                f"{'':22s} {'unopt.ll':>18s} {'opt.ll':>18s} {'opt.s':>18s}"
            )
            print(
                f"{'benchmark':22s} {args.diff:>8s} {'diff':>9s} "
                f"{args.diff:>8s} {'diff':>9s} {args.diff:>8s} {'diff':>9s}"
            )
            print("-" * 76)

            t_bu = t_bo = t_ba = t_mu = t_mo = t_ma = 0
            for bench in benches:
                bu = line_count(os.path.join(args.dump_dir, bench, "unopt.ll"))
                bo = line_count(os.path.join(args.dump_dir, bench, "opt.ll"))
                ba = line_count(os.path.join(args.dump_dir, bench, "opt.s"))
                mu = line_count(os.path.join(base_dump, bench, "unopt.ll"))
                mo = line_count(os.path.join(base_dump, bench, "opt.ll"))
                ma = line_count(os.path.join(base_dump, bench, "opt.s"))

                print(
                    f"{bench:22s} {mu:6d} {fmt_diff(mu, bu):>5s} {fmt_pct(mu, bu):>5s}  "
                    f"{mo:6d} {fmt_diff(mo, bo):>5s} {fmt_pct(mo, bo):>5s}  "
                    f"{ma:6d} {fmt_diff(ma, ba):>5s} {fmt_pct(ma, ba):>5s}"
                )

                t_bu += bu
                t_bo += bo
                t_ba += ba
                t_mu += mu
                t_mo += mo
                t_ma += ma

            print("-" * 76)
            print(
                f"{'TOTAL':22s} {t_mu:6d} {fmt_diff(t_mu, t_bu):>5s} {fmt_pct(t_mu, t_bu):>5s}  "
                f"{t_mo:6d} {fmt_diff(t_mo, t_bo):>5s} {fmt_pct(t_mo, t_bo):>5s}  "
                f"{t_ma:6d} {fmt_diff(t_ma, t_ba):>5s} {fmt_pct(t_ma, t_ba):>5s}"
            )
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", base_worktree],
                capture_output=True,
                cwd=root,
            )
            if os.path.isdir(base_worktree):
                shutil.rmtree(base_worktree)
    else:
        print(f"{'benchmark':22s} {'unopt':>7s} {'opt.ll':>7s} {'opt.s':>7s}")
        print("-" * 46)

        t_u = t_o = t_a = 0
        for bench in benches:
            u = line_count(os.path.join(args.dump_dir, bench, "unopt.ll"))
            o = line_count(os.path.join(args.dump_dir, bench, "opt.ll"))
            a = line_count(os.path.join(args.dump_dir, bench, "opt.s"))
            print(f"{bench:22s} {u:7d} {o:7d} {a:7d}")
            t_u += u
            t_o += o
            t_a += a

        print("-" * 46)
        print(f"{'TOTAL':22s} {t_u:7d} {t_o:7d} {t_a:7d}")


if __name__ == "__main__":
    main()
