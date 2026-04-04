#!/usr/bin/env bash
# Reports line counts of generated LLVM IR and assembly for all benchmarks.
#
# Usage:
#   ./scripts/codegen-lines.sh /tmp/dump                          # all benchmarks
#   ./scripts/codegen-lines.sh /tmp/dump usdc_proxy weth           # specific benchmarks
#   ./scripts/codegen-lines.sh /tmp/dump --diff main               # compare against rev
#   ./scripts/codegen-lines.sh /tmp/dump --diff main usdc_proxy    # compare specific benchmarks

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dump-dir> [--diff <rev>] [bench...]" >&2
    exit 1
fi

DUMP_DIR="$1"; shift
COMPARE_REV=""
if [[ "${1:-}" == "--diff" ]]; then
    shift
    COMPARE_REV="${1:?missing rev for --diff}"; shift
fi
BENCHES=("$@")

collect() {
    local dir="$1"
    cargo build --quiet 2>/dev/null

    if [[ ${#BENCHES[@]} -eq 0 ]]; then
        mapfile -t BENCHES < <(cargo r -q -- run --list)
    fi

    for bench in "${BENCHES[@]}"; do
        rm -rf "$dir/$bench"
        cargo r -q -- run "$bench" -o "$dir" >/dev/null 2>&1 || true
    done
}

line_count() {
    if [[ -f "$1" ]]; then wc -l < "$1"; else echo 0; fi
}

fmt_diff() {
    local d=$(($2 - $1))
    if [[ $d -eq 0 ]]; then printf "    ="; else printf "%+5d" "$d"; fi
}

fmt_pct() {
    if [[ $1 -eq 0 ]]; then printf "    -"; return; fi
    awk "BEGIN { printf \"%+.1f%%\", ($2 - $1) / $1 * 100 }"
}

# Collect current results.
collect "$DUMP_DIR"

if [[ -n "$COMPARE_REV" ]]; then
    BASE_DUMP="$DUMP_DIR.base"
    BASE_WORKTREE=$(mktemp -d)
    trap 'git worktree remove --force "$BASE_WORKTREE" 2>/dev/null; rm -rf "$BASE_WORKTREE"' EXIT
    git worktree add --detach --quiet "$BASE_WORKTREE" "$COMPARE_REV" 2>/dev/null
    (cd "$BASE_WORKTREE" && collect "$BASE_DUMP")
    echo "Base dump saved to: $BASE_DUMP" >&2

    printf "%-22s %18s %18s %18s\n" "" "unopt.ll" "opt.ll" "opt.s"
    printf "%-22s %8s %9s %8s %9s %8s %9s\n" \
        "benchmark" "$COMPARE_REV" "diff" "$COMPARE_REV" "diff" "$COMPARE_REV" "diff"
    printf '%s\n' "$(printf '%.0s-' {1..76})"

    t_bu=0; t_bo=0; t_ba=0; t_mu=0; t_mo=0; t_ma=0
    for bench in "${BENCHES[@]}"; do
        bu=$(line_count "$DUMP_DIR/$bench/unopt.ll")
        bo=$(line_count "$DUMP_DIR/$bench/opt.ll")
        ba=$(line_count "$DUMP_DIR/$bench/opt.s")
        mu=$(line_count "$BASE_DUMP/$bench/unopt.ll")
        mo=$(line_count "$BASE_DUMP/$bench/opt.ll")
        ma=$(line_count "$BASE_DUMP/$bench/opt.s")

        printf "%-22s %6d %5s %5s  %6d %5s %5s  %6d %5s %5s\n" \
            "$bench" "$mu" "$(fmt_diff "$mu" "$bu")" "$(fmt_pct "$mu" "$bu")" \
            "$mo" "$(fmt_diff "$mo" "$bo")" "$(fmt_pct "$mo" "$bo")" \
            "$ma" "$(fmt_diff "$ma" "$ba")" "$(fmt_pct "$ma" "$ba")"

        t_bu=$((t_bu + bu)); t_bo=$((t_bo + bo)); t_ba=$((t_ba + ba))
        t_mu=$((t_mu + mu)); t_mo=$((t_mo + mo)); t_ma=$((t_ma + ma))
    done

    printf '%s\n' "$(printf '%.0s-' {1..76})"
    printf "%-22s %6d %5s %5s  %6d %5s %5s  %6d %5s %5s\n" \
        "TOTAL" "$t_mu" "$(fmt_diff "$t_mu" "$t_bu")" "$(fmt_pct "$t_mu" "$t_bu")" \
        "$t_mo" "$(fmt_diff "$t_mo" "$t_bo")" "$(fmt_pct "$t_mo" "$t_bo")" \
        "$t_ma" "$(fmt_diff "$t_ma" "$t_ba")" "$(fmt_pct "$t_ma" "$t_ba")"
else
    printf "%-22s %7s %7s %7s\n" "benchmark" "unopt" "opt.ll" "opt.s"
    printf '%s\n' "$(printf '%.0s-' {1..46})"

    t_u=0; t_o=0; t_a=0
    for bench in "${BENCHES[@]}"; do
        u=$(line_count "$DUMP_DIR/$bench/unopt.ll")
        o=$(line_count "$DUMP_DIR/$bench/opt.ll")
        a=$(line_count "$DUMP_DIR/$bench/opt.s")
        printf "%-22s %7d %7d %7d\n" "$bench" "$u" "$o" "$a"
        t_u=$((t_u + u)); t_o=$((t_o + o)); t_a=$((t_a + a))
    done

    printf '%s\n' "$(printf '%.0s-' {1..46})"
    printf "%-22s %7d %7d %7d\n" "TOTAL" "$t_u" "$t_o" "$t_a"
fi
