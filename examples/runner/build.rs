use revmc::{new_llvm_backend, primitives::SpecId, EvmCompiler, OptimizationLevel, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    // Emit the configuration to run compiled bytecodes.
    // This not used if we are only using statically linked bytecodes.
    revmc_build::emit();

    // Compile and statically link a bytecode.
    let name = "fibonacci";
    let bytecode = revmc::primitives::hex!(
        "5f355f60015b8215601a578181019150909160019003916005565b9150505f5260205ff3"
    );
    println!("cargo:rustc-env=FIB_HASH={}", revmc::primitives::keccak256(&bytecode));

    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let context = revmc::llvm::inkwell::context::Context::create();
    let backend = new_llvm_backend(&context, true, OptimizationLevel::Aggressive)?;
    let mut compiler = EvmCompiler::new(backend);
    compiler.translate(Some(name), &bytecode, SpecId::CANCUN)?;
    let object = out_dir.join(name).with_extension("o");
    compiler.write_object_to_file(&object)?;

    cc::Build::new().object(&object).static_flag(true).compile(name);

    Ok(())
}
