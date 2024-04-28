#!/usr/bin/env bash
set -eo pipefail

v=${1:-$LLVM_VERSION}
bins=(clang llvm-config lld ld.lld FileCheck)
llvm_sh=$(mktemp)

wget https://apt.llvm.org/llvm.sh -O "$llvm_sh"
chmod +x "$llvm_sh"
"$llvm_sh" "$v" all
for bin in "${bins[@]}"; do
    ln -fs "$(which "$bin-$v")" "/usr/bin/$bin"
done
