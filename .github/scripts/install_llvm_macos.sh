#!/usr/bin/env bash
set -eo pipefail

v=${1:-$LLVM_VERSION}

brew install "llvm@${v}"
prefix="$(brew --prefix llvm@${v})"
echo "LLVM_SYS_${v}0_PREFIX=$prefix" >> $GITHUB_ENV
echo "MLIR_SYS_${v}0_PREFIX=$prefix" >> $GITHUB_ENV
echo "TABLEGEN_${v}0_PREFIX=$prefix" >> $GITHUB_ENV
echo "$prefix/bin" >> $GITHUB_PATH
