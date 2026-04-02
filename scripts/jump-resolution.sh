#!/usr/bin/env bash
# Reports dynamic jump resolution stats for all benchmarks.
#
# Usage:
#   ./scripts/jump-resolution.sh              # all benchmarks
#   ./scripts/jump-resolution.sh usdc_proxy weth  # specific benchmarks

set -euo pipefail

cargo build --quiet 2>/dev/null

if [ $# -gt 0 ]; then
    BENCHES=("$@")
else
    mapfile -t BENCHES < <(cargo r -q -- run --list)
fi

last_match() {
    # Print the last matching capture, or 0 if none.
    rg -o "$1" -r '$1' | tail -1 || echo 0
}

printf "%-25s %6s %6s %6s %6s %6s %10s\n" "BENCHMARK" "LOCAL" "NONADJ" "PCR" "FIXPT" "UNRES" "TOTAL"
printf "%-25s %6s %6s %6s %6s %6s %10s\n" "---------" "-----" "------" "---" "-----" "-----" "-----"

for bench in "${BENCHES[@]}"; do
    output=$(NO_COLOR=1 RUST_LOG=revmc::bytecode=trace cargo r -q -- run "$bench" --parse-only 2>&1)

    local_res=$(echo "$output" | { rg -o 'local_jumps.*newly_resolved=(\d+)' -r '$1' | tail -1 || echo 0; })
    non_adj=$(echo "$output" | { rg -c 'resolved non-adjacent jump' || echo 0; })
    pcr_res=$(echo "$output" | { rg -c 'resolved via PCR hint' || echo 0; })
    fixpt_res=$(echo "$output" | { rg -o 'ba:.*newly_resolved=(\d+)' -r '$1' | tail -1 || echo 0; })
    # Use the last "unresolved" line (after all passes).
    unresolved=$(echo "$output" | { rg -o 'unresolved dynamic jumps remain n=(\d+)' -r '$1' | tail -1 || echo 0; })
    total=$((local_res + fixpt_res + unresolved))

    printf "%-25s %6d %6d %6d %6d %6d %10d\n" "$bench" "$local_res" "$non_adj" "$pcr_res" "$fixpt_res" "$unresolved" "$total"
done
