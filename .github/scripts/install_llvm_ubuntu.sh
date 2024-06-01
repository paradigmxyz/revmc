#!/usr/bin/env bash
set -eo pipefail

v=${1:-$LLVM_VERSION}
bins=(clang llvm-config lld ld.lld FileCheck)
llvm_sh=$(mktemp)

echo "::group::Download llvm.sh"
wget https://apt.llvm.org/llvm.sh -O "$llvm_sh"
echo "::endgroup::"

chmod +x "$llvm_sh"

echo "::group::Install LLVM $v"
"$llvm_sh" "$v" all
for bin in "${bins[@]}"; do
    sudo ln -fs "$(which "$bin-$v")" "/usr/bin/$bin"
done
echo "::endgroup::"

echo "GITHUB_ENV=$GITHUB_ENV"
prefix="$(llvm-config --prefix)"
{
    echo "LLVM_SYS_${v}0_PREFIX=$prefix"
    echo "MLIR_SYS_${v}0_PREFIX=$prefix"
    echo "TABLEGEN_${v}0_PREFIX=$prefix"
} >> "$GITHUB_ENV"
