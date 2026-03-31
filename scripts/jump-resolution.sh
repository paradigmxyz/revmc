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

sum_matches() {
    local total=0
    while IFS= read -r n; do
        total=$((total + n))
    done
    echo "$total"
}

printf "%-25s %10s %10s %10s\n" "BENCHMARK" "RESOLVED" "UNRESOLVED" "TOTAL"
printf "%-25s %10s %10s %10s\n" "---------" "--------" "----------" "-----"

for bench in "${BENCHES[@]}"; do
    output=$(NO_COLOR=1 RUST_LOG=debug cargo r -q -- run "$bench" --parse-only --display 2>&1)

    resolved=$(echo "$output" | { rg -o 'newly_resolved=(\d+)' -r '$1' || true; } | sum_matches)
    unresolved=$(echo "$output" | { rg -o 'unresolved dynamic jumps remain n=(\d+)' -r '$1' || true; } | sum_matches)
    total=$((resolved + unresolved))

    printf "%-25s %10d %10d %10d\n" "$bench" "$resolved" "$unresolved" "$total"
done
