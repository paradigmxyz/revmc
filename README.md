# revm-jit

Experimental [JIT] and [AOT] compiler for the [Ethereum Virtual Machine][EVM].

The compiler implementation is abstracted over an intermediate representation backend.

This repository hosts two backend implementations:

- [LLVM] ([`revm-jit-llvm`]): main backend with full test coverage;
- [Cranelift] ([`revm-jit-cranelift`]); currently not functional due to missing `i256` support in Cranelift. This will likely require a custom fork of Cranelift.

Compiling the crate requires LLVM 17.

[JIT]: https://en.wikipedia.org/wiki/Just-in-time_compilation
[AOT]: https://en.wikipedia.org/wiki/Ahead-of-time_compilation
[EVM]: https://ethereum.org/en/developers/docs/evm/
[LLVM]: https://llvm.org/
[`revm-jit-llvm`]: /crates/revm-jit-llvm
[Cranelift]: https://cranelift.dev/
[`revm-jit-cranelift`]: /crates/revm-jit-cranelift

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
