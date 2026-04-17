#!/usr/bin/env bash
# Download and extract test fixtures to match revm's test coverage.
#
# Fixtures are cached in `test-fixtures/` at the repo root. Pass `clean` as
# the first argument to wipe and re-download everything.
#
# Directory layout after running:
#   test-fixtures/
#   ├── main/
#   │   ├── stable/state_tests/...    # execution-spec-tests stable
#   │   └── develop/state_tests/...   # execution-spec-tests develop
#   └── legacytests/
#       ├── Cancun/GeneralStateTests/...
#       └── Constantinople/GeneralStateTests/...

set -euo pipefail

MAIN_VERSION="v5.3.0"
BASE_URL="https://github.com/ethereum/execution-spec-tests/releases/download"

# Resolve repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FIXTURES_DIR="$REPO_ROOT/test-fixtures"

if [[ "${1:-}" == "clean" ]]; then
    echo "Cleaning test fixtures..."
    rm -rf "$FIXTURES_DIR"
fi

MAIN_STABLE_DIR="$FIXTURES_DIR/main/stable"
MAIN_DEVELOP_DIR="$FIXTURES_DIR/main/develop"
LEGACY_DIR="$FIXTURES_DIR/legacytests"

download_and_extract() {
    local url="$1"
    local dest="$2"

    if [[ -d "$dest" ]]; then
        echo "  Already exists: $dest"
        return
    fi

    echo "  Downloading: $url"
    mkdir -p "$dest"
    curl -sSfL "$url" | tar xz --strip-components=1 -C "$dest"

    # Remove dirs we don't need (blockchain engine tests).
    rm -rf "$dest/blockchain_tests_engine" \
           "$dest/blockchain_tests_engine_x" \
           "$dest/blockchain_tests_sync"
}

echo "=== Downloading execution-spec-tests ($MAIN_VERSION) ==="
download_and_extract \
    "$BASE_URL/$MAIN_VERSION/fixtures_stable.tar.gz" \
    "$MAIN_STABLE_DIR"
download_and_extract \
    "$BASE_URL/$MAIN_VERSION/fixtures_develop.tar.gz" \
    "$MAIN_DEVELOP_DIR"

echo "=== Cloning ethereum/legacytests ==="
if [[ -d "$LEGACY_DIR" ]]; then
    echo "  Already exists: $LEGACY_DIR"
else
    git clone --depth 1 https://github.com/ethereum/legacytests.git "$LEGACY_DIR"
fi

echo "=== Done ==="
echo "Fixture directories:"
for d in "$MAIN_STABLE_DIR" "$MAIN_DEVELOP_DIR" "$LEGACY_DIR"; do
    if [[ -d "$d" ]]; then
        n=$(find "$d" -name '*.json' | wc -l)
        echo "  $d ($n JSON files)"
    fi
done
