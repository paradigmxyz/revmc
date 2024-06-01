# revm-jit

Experimental [JIT] and [AOT] compiler for the [Ethereum Virtual Machine][EVM].

The compiler implementation is abstracted over an intermediate representation backend.

This repository hosts two backend implementations:
- [LLVM] ([`revm-jit-llvm`]): main backend with full test coverage;
- [Cranelift] ([`revm-jit-cranelift`]); currently not functional due to missing `i256` support in Cranelift. This will likely require a custom fork of Cranelift.

[JIT]: https://en.wikipedia.org/wiki/Just-in-time_compilation
[AOT]: https://en.wikipedia.org/wiki/Ahead-of-time_compilation
[EVM]: https://ethereum.org/en/developers/docs/evm/
[LLVM]: https://llvm.org/
[`revm-jit-llvm`]: /crates/revm-jit-llvm
[Cranelift]: https://cranelift.dev/
[`revm-jit-cranelift`]: /crates/revm-jit-cranelift

## Requirements

- Linux or macOS, Windows is not supported
- Latest stable Rust version
- LLVM 17
  - On Debian-based Linux distros: see [apt.llvm.org](https://apt.llvm.org/)
  - On Arch-based Linux distros: `pacman -S llvm`
  - On macOS: `brew install llvm@17`
  - The following environment variables may be required:
    ```bash
    prefix=$(llvm-config --prefix)
    # or
    #prefix=$(llvm-config-17 --prefix)
    # on macOS:
    #prefix=$(brew --prefix llvm@17)
    export LLVM_SYS_170_PREFIX=$prefix
    ```

## Usage

The compiler is implemented as a library and can be used as such through the `revm-jit` crate.

A minimal runtime is required to run AOT-compiled bytecodes. A default runtime implementation is
provided through symbols exported in the `revm-jit-builtins` crate and must be exported in the final
binary. This can be achieved with the following build script:
```rust,ignore
fn main() {
    revm_jit_build::emit();
}
```

You can check out the [examples](/examples) directory for example usage.

## Credits

The initial compiler implementation was inspired by [`paradigmxyz/jitevm`](https://github.com/paradigmxyz/jitevm).

#### License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in these crates by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
</sub>
