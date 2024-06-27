# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0](https://github.com/paradigmxyz/revmc/releases/tag/v0.1.0) - 2024-06-27

### Bug Fixes

- Use `effective_gas_price` in GASPRICE ([#32](https://github.com/paradigmxyz/revmc/issues/32))
- Dereferenceable attribute on builtins ([#25](https://github.com/paradigmxyz/revmc/issues/25))
- Memory leak in LLVM diagnostic handler
- Manually delete LLVM IR to reduce memory consumption
- End a section at SSTORE too (EIP-1706) ([#21](https://github.com/paradigmxyz/revmc/issues/21))
- Build script for macos ([#20](https://github.com/paradigmxyz/revmc/issues/20))
- Linker args
- Stack overflows
- Make comments work on instrs ([#7](https://github.com/paradigmxyz/revmc/issues/7))
- Correct linker flag
- Pass -zundefs to linker
- Only allow setting compilation mode at instantiation
- Features 2
- Features
- Stack io for suspending instructions
- Fix some TODOs
- Ci
- Add rust-src component to rust-toolchain.toml

### Dependencies

- Bump revm
- [deps] Bump all
- Bump revm to 9
- [deps] Cargo update
- Bump to revm main
- Bump revm, fix gas overflows
- Bump
- Bump
- Bump revm patch

### Documentation

- Add changelogs
- Add requirements and examples ([#29](https://github.com/paradigmxyz/revmc/issues/29))
- Wording

### Features

- Add target configuration ([#39](https://github.com/paradigmxyz/revmc/issues/39))
- Improve DX around statically linked bytecodes, add example ([#37](https://github.com/paradigmxyz/revmc/issues/37))
- Implement IR builtins ([#36](https://github.com/paradigmxyz/revmc/issues/36))
- Make builtins no_std ([#27](https://github.com/paradigmxyz/revmc/issues/27))
- Add llvm-prefer-* features ([#24](https://github.com/paradigmxyz/revmc/issues/24))
- More LLVM overrides ([#23](https://github.com/paradigmxyz/revmc/issues/23))
- Add another benchmark ([#22](https://github.com/paradigmxyz/revmc/issues/22))
- Allow running after aot
- [llvm] Use intel syntax for x86 assembly, set diagnostic handler
- Build script support, fix revm update fallout
- Allow disabling stack length overflow checks
- Add linker helper
- Also build builtins as a dynamic library
- Implement AOT compilation
- New backend API ([#3](https://github.com/paradigmxyz/revmc/issues/3))
- Pass most statetest tests
- Implement CALL instructions
- Pre-built callbacks bitcode module
- Consolidate stack IO calculations, fix some instructions
- Add LLVM ORC wrappers
- Unify returns, implement suspend/resume for CALL/CREATE
- Implement dynamic jumps
- Finish core instruction implementations and tests (2/2)
- Finish core instruction implementations and tests (1)
- Implement most instructions (untested)
- Implement some env-fetching instructions
- Don't allocate tmp in SWAP
- Move attributes to the builder trait
- Implement GAS, add some more tests
- Ignore unreachable `JUMPDEST`s in DCE
- Basic DCE, restructure bytecode module
- Implement SIGNEXTEND, refactor test suite
- Implement opcode info ourselves
- Implement callbacks
- Criterion bench
- Debug_assertions/panics, function call hooks, gas in tests, benchmarks
- Implement basic gas accounting
- Test fibonacci
- Codegen jumps
- Internal bytecode representation, add more docs
- Implement cold blocks in llvm
- More implementations, add tests
- Core backend abstraction, add LLVM
- Basic implementation

### Miscellaneous Tasks

- Exclude examples from auto changelogs
- Replace timing macros with tracy ([#40](https://github.com/paradigmxyz/revmc/issues/40))
- Add release configuration
- Update some comments
- Rebrand to `revmc` ([#33](https://github.com/paradigmxyz/revmc/issues/33))
- Patch revm to reth pin ([#31](https://github.com/paradigmxyz/revmc/issues/31))
- Add more benchmarks
- Remove config file
- Add a TODO
- Make DIFFICULTY a builtin ([#26](https://github.com/paradigmxyz/revmc/issues/26))
- Add some todos
- Comment
- Adjustments
- Update benchmarks
- Update revm gas ([#17](https://github.com/paradigmxyz/revmc/issues/17))
- Cleanups
- Add iai-callgrind benchmarks ([#14](https://github.com/paradigmxyz/revmc/issues/14))
- Fix some TODOs, cleanup ([#11](https://github.com/paradigmxyz/revmc/issues/11))
- [jit] Use return_imm builder method ([#9](https://github.com/paradigmxyz/revmc/issues/9))
- Remove one-array
- Rewrite native fibonacci
- Dedup code
- Renames, remove unnecessary dll linking, clippy
- Adjustments
- Update to latest revm
- Minor updates
- Split compiler.rs
- Update to latest revm main
- Make utils private again
- Rename callback to builtin
- Move all callbacks to the crate
- Remove precompiled bitcode
- Improve call_with_interpreter
- Update
- Pass spec_id after stack pointer
- Default stack config to true
- Separate opcode (bytes) from instructions (higher level)
- Core -> backend
- Clippy

### Other

- Update README.md
- Add config file w/ top 250 contracts
- Fix iai on push ([#18](https://github.com/paradigmxyz/revmc/issues/18))
- Run macos in CI ([#6](https://github.com/paradigmxyz/revmc/issues/6))
- Fix checks
- Merge pull request [#2](https://github.com/paradigmxyz/revmc/issues/2) from DaniPopes/remove-precompiled-bitcode
- Callbacks bitcode
- Add callbacks crate
- Merge pull request [#1](https://github.com/paradigmxyz/revmc/issues/1) from DaniPopes/fighting-ci
- Print llvm-config
- Download llvm
- Restructure
- Stuff
- Initial commit

### Performance

- Pay base gas of dynamic opcodes in sections ([#19](https://github.com/paradigmxyz/revmc/issues/19))
- Re-order gas check ([#15](https://github.com/paradigmxyz/revmc/issues/15))
- Fix resume switch branch weights
- Set weight metadata on switches too ([#13](https://github.com/paradigmxyz/revmc/issues/13))
- Set branch weight metadata instead of cold blocks ([#12](https://github.com/paradigmxyz/revmc/issues/12))
- Add attributes to builtin functions' params ([#8](https://github.com/paradigmxyz/revmc/issues/8))
- Unify failure blocks ([#5](https://github.com/paradigmxyz/revmc/issues/5))
- Batch gas and stack length checks ([#4](https://github.com/paradigmxyz/revmc/issues/4))

### Styling

- Clippy, add more lints ([#34](https://github.com/paradigmxyz/revmc/issues/34))
- Fmt

### Testing

- Ensure SELFDESTRUCT stops execution
- Add codegen tests ([#10](https://github.com/paradigmxyz/revmc/issues/10))
- Test suspend-resume
- Cross-reference with revm interpreter

<!-- generated by git-cliff -->
