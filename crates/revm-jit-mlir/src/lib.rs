/// The MLIR-based EVM bytecode compiler backend.
#[derive(Debug)]
#[must_use]
pub struct EvmMlirBackend<'ctx> {
    ctx: &'ctx Melior,
}
