#![allow(missing_docs)]

use revm_interpreter::opcode as op;
use revm_jit_cranelift::Ret;
use std::path::PathBuf;

fn main() -> color_eyre::Result<()> {
    let _ = color_eyre::install();

    let bytecode = [
        //
        op::PUSH0,
        op::PUSH0,
        op::ADD,
        op::STOP,
    ];
    let mut jit = revm_jit_cranelift::JitEvm::new();
    jit.dump_ir_to(Some(PathBuf::from("./target/")));
    let id = jit.compile(&bytecode)?;
    let f = jit.get_fn(id);
    let ret = f();
    assert_eq!(ret, Ret::Stop);

    Ok(())
}
