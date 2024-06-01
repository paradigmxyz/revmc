#!/usr/bin/env bash
set -eo pipefail

v=${1:-$LLVM_VERSION}
bins=(clang llvm-config lld ld.lld FileCheck)

echo "::group::Download llvm.sh"
wget https://apt.llvm.org/llvm.sh
echo "::endgroup::"

chmod +x ./llvm.sh

echo "::group::Install LLVM $v"
sudo ./llvm.sh "$v" all
echo "::endgroup::"

echo "::group::Install MLIR $v"
sudo apt-get install -y libmlir-18-dev mlir-18-tools
echo "::endgroup::"

for bin in "${bins[@]}"; do
    sudo ln -fs "$(which "$bin-$v")" "/usr/bin/$bin"
done

echo "GITHUB_ENV=$GITHUB_ENV"
prefix="$(llvm-config --prefix)"
{
    echo "LLVM_SYS_${v}0_PREFIX=$prefix"
    echo "MLIR_SYS_${v}0_PREFIX=$prefix"
    echo "TABLEGEN_${v}0_PREFIX=$prefix"
} >> "$GITHUB_ENV"
