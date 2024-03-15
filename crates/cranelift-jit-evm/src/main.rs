#![allow(missing_docs)]

use std::path::PathBuf;

use cranelift_jit_evm::Ret;
use revm_interpreter::opcode as op;

fn main() -> color_eyre::Result<()> {
    let _ = color_eyre::install();

    let bytecode = [
        //
        op::PUSH0,
        op::PUSH1,
        0x69,
        op::STOP,
    ];
    let mut jit = cranelift_jit_evm::JitEvm::new();
    jit.dump_ir_to(Some(PathBuf::from("./target/")));
    let id = jit.compile(&bytecode)?;
    let f = jit.get_fn(id);
    let ret = f();
    assert_eq!(ret, Ret::Stop);

    Ok(())
}
