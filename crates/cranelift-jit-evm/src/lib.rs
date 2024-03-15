#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(unused_extern_crates))]

use cranelift::{
    codegen::ir::{Inst, StackSlot},
    prelude::*,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use revm_interpreter::opcode as op;
use revm_primitives::U256;
use std::{fmt::Write, path::PathBuf, slice};

mod error;
pub use error::{Error, Result};

mod pretty_clif;
use pretty_clif::CommentWriter;

/// The signature of a JIT'd EVM bytecode.
pub type Sig = fn() -> Ret;

/// EVM bytecode JIT compiler.
#[allow(missing_debug_implementations)]
#[must_use]
pub struct JitEvm {
    /// The function builder context, which is reused across multiple FunctionBuilder instances.
    builder_context: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    ctx: codegen::Context,

    /// The module, with the jit backend, which manages the JIT'd functions.
    module: JITModule,

    ir_out_dir: Option<PathBuf>,

    function_counter: usize,
}

impl JitEvm {
    /// Returns `Ok(())` if the current architecture is supported, or `Err(())` if the host machine
    /// is not supported in the current configuration.
    pub fn is_supported() -> Result<(), &'static str> {
        cranelift_native::builder().map(drop)
    }

    /// Creates a new instance of the JIT compiler.
    ///
    /// # Panics
    ///
    /// Panics if the current architecture is not supported. See
    /// [`is_supported`](Self::is_supported).
    #[track_caller]
    pub fn new() -> Self {
        let builder = JITBuilder::with_flags(
            &[("opt_level", "speed")],
            cranelift_module::default_libcall_names(),
        )
        .unwrap();
        let module = JITModule::new(builder);
        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module,
            ir_out_dir: None,
            function_counter: 0,
        }
    }

    /// Dumps the IR to the given directory after compilation.
    ///
    /// Creates a `clif` directory in the given directory and writes `function.unopt.clif` and
    /// `function.opt.clif` files for each function that's compiled.
    pub fn dump_ir_to(&mut self, output_dir: Option<PathBuf>) {
        self.ctx.want_disasm = output_dir.is_some();
        self.ir_out_dir = output_dir;
    }

    fn new_name(&mut self) -> String {
        let name = format!("__evm_bytecode_{}", self.function_counter);
        self.function_counter += 1;
        name
    }

    /// Compiles the given EVM bytecode into machine code.
    pub fn compile(&mut self, bytecode: &[u8]) -> Result<FuncId> {
        let clif_comments = self.translate(bytecode)?;

        // Next, declare the function to jit. Functions must be declared
        // before they can be called, or defined.
        let name = self.new_name();
        let id = self.module.declare_function(&name, Linkage::Export, &self.ctx.func.signature)?;

        // Print the unoptimized IR.
        if let Some(output_dir) = &self.ir_out_dir {
            crate::pretty_clif::write_clif_file(
                output_dir,
                &name,
                "unopt",
                self.module.isa(),
                &self.ctx.func,
                &clif_comments,
            );
        }

        // Define the function to jit. This finishes compilation, although
        // there may be outstanding relocations to perform. Currently, jit
        // cannot finish relocations until all functions to be called are
        // defined. For this toy demo for now, we'll just finalize the
        // function below.
        self.module.define_function(id, &mut self.ctx)?;

        // Print the optimized IR.
        if let Some(output_dir) = &self.ir_out_dir {
            crate::pretty_clif::write_clif_file(
                output_dir,
                &name,
                "opt",
                self.module.isa(),
                &self.ctx.func,
                &clif_comments,
            );
        }

        // Now that compilation is finished, we can clear out the context state.
        self.module.clear_context(&mut self.ctx);

        // Finalize the functions which we just defined, which resolves any outstanding relocations
        // (patching in addresses, now that they're available).
        self.module.finalize_definitions()?;

        Ok(id)
    }

    /// Returns the function pointer of a finalized function.
    pub fn get_fn(&self, id: FuncId) -> Sig {
        unsafe { std::mem::transmute(self.get_fn_address(id)) }
    }

    /// Returns the address of a finalized function.
    pub fn get_fn_address(&self, id: FuncId) -> *const u8 {
        self.module.get_finalized_function(id)
    }

    /// Translates bytecode opcodes to Cranelift IR.
    fn translate(&mut self, bytecode: &[u8]) -> Result<CommentWriter> {
        let pointer_type = self.module.target_config().pointer_type();
        let return_type = types::I32;

        self.ctx.func.signature.returns.push(AbiParam::new(return_type));
        let mut bcx = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        // Set up the first block.
        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        bcx.seal_block(entry);

        // Function prologue: allocate the stack.
        let stack = EvmStack::new(&mut bcx, pointer_type);

        let mut fx = FunctionCx {
            bcx,
            // module: &mut self.module,
            clif_comments: CommentWriter::new(),
            pointer_type,
            return_type,
            stack,
        };
        fx.translate_bytecode(bytecode)?;
        Ok(fx.clif_comments)
    }
}

struct FunctionCx<'a> {
    bcx: FunctionBuilder<'a>,
    // module: &'a mut JITModule,
    pointer_type: Type,
    return_type: Type,
    clif_comments: CommentWriter,

    stack: EvmStack,
}

#[allow(dead_code)]
impl<'a> FunctionCx<'a> {
    fn translate_bytecode(&mut self, bytecode: &[u8]) -> Result<()> {
        let mut iter = bytecode.iter();
        while let Some(&opcode) = iter.next() {
            self.translate_opcode(opcode, &mut iter)?;
        }
        Ok(())
    }

    fn translate_opcode(&mut self, opcode: u8, iter: &mut slice::Iter<'_, u8>) -> Result<()> {
        self.bcx.ins().nop();
        if self.clif_comments.enabled() {
            let inst = self.last_inst();
            let mut comment = String::with_capacity(16);
            if let Some(as_str) = op::OPCODE_JUMPMAP[opcode as usize] {
                comment.push_str(as_str);
            } else {
                write!(comment, "UNKNOWN(0x{opcode:02x})").unwrap();
            }
            if let op::PUSH1..=op::PUSH32 = opcode {
                let n = (opcode - op::PUSH0) as usize;
                let slice = iter.as_slice();
                let slice = slice.get(..n.min(slice.len())).unwrap();
                write!(comment, " 0x").unwrap();
                for &b in slice {
                    write!(comment, "{b:02x}").unwrap();
                }
                if slice.len() != n {
                    write!(comment, " (truncated to {})", slice.len()).unwrap();
                }
            }
            self.add_comment(inst, comment);
        }

        let mut get_imm = |len: usize| {
            let r = iter.as_slice().get(..len);
            // This also takes care of the `None` case.
            // TODO: Use `advance_by` when stable.
            iter.take(len).for_each(drop);
            r
        };
        match opcode {
            op::STOP => self.build_return(Ret::Stop),

            op::ADD => {}
            op::MUL => {}
            op::SUB => {}
            op::DIV => {}
            op::SDIV => {}
            op::MOD => {}
            op::SMOD => {}
            op::ADDMOD => {}
            op::MULMOD => {}
            op::EXP => {}
            op::SIGNEXTEND => {}

            op::LT => {}
            op::GT => {}
            op::SLT => {}
            op::SGT => {}
            op::EQ => {}
            op::ISZERO => {}
            op::AND => {}
            op::OR => {}
            op::XOR => {}
            op::NOT => {}
            op::BYTE => {}
            op::SHL => {}
            op::SHR => {}
            op::SAR => {}

            op::KECCAK256 => {}

            op::ADDRESS => {}
            op::BALANCE => {}
            op::ORIGIN => {}
            op::CALLER => {}
            op::CALLVALUE => {}
            op::CALLDATALOAD => {}
            op::CALLDATASIZE => {}
            op::CALLDATACOPY => {}
            op::CODESIZE => {}
            op::CODECOPY => {}

            op::GASPRICE => {}
            op::EXTCODESIZE => {}
            op::EXTCODECOPY => {}
            op::RETURNDATASIZE => {}
            op::RETURNDATACOPY => {}
            op::EXTCODEHASH => {}
            op::BLOCKHASH => {}
            op::COINBASE => {}
            op::TIMESTAMP => {}
            op::NUMBER => {}
            op::DIFFICULTY => {}
            op::GASLIMIT => {}
            op::CHAINID => {}
            op::SELFBALANCE => {}
            op::BASEFEE => {}
            op::BLOBHASH => {}
            op::BLOBBASEFEE => {}

            op::POP => {}
            op::MLOAD => {}
            op::MSTORE => {}
            op::MSTORE8 => {}
            op::SLOAD => {}
            op::SSTORE => {}
            op::JUMP => {}
            op::JUMPI => {}
            op::PC => {}
            op::MSIZE => {}
            op::GAS => {}
            op::JUMPDEST => {}
            op::TLOAD => {}
            op::TSTORE => {}

            op::PUSH0 => {
                let value = self.iconst_256(U256::ZERO);
                self.push(value);
            }
            op::PUSH1..=op::PUSH32 => {
                let n = opcode - op::PUSH0;
                let value = get_imm(n as usize).map(U256::from_be_slice).unwrap_or_default();
                let value = self.iconst_256(value);
                self.push(value);
            }

            op::DUP1..=op::DUP16 => self.dup(opcode - op::DUP1 + 1),

            op::SWAP1..=op::SWAP16 => self.swap(opcode - op::SWAP1 + 1),

            op::LOG0..=op::LOG4 => {
                let _n = opcode - op::LOG0;
            }

            op::CREATE => {}
            op::CALL => {}
            op::CALLCODE => {}
            op::RETURN => {}
            op::DELEGATECALL => {}
            op::CREATE2 => {}
            op::STATICCALL => {}
            op::REVERT => {}
            op::INVALID => {}
            op::SELFDESTRUCT => {}

            _ => todo!("unknown opcode: {opcode}"),
        }

        Ok(())
    }

    fn iconst_256(&mut self, value: U256) -> EvmWord {
        if self.clif_comments.enabled() {
            self.bcx.ins().nop();
            let inst = self.last_inst();
            self.add_comment(inst, format!("{value} (0x{value:x})"));
        }
        EvmWord {
            values: value.into_limbs().map(|limb| self.bcx.ins().iconst(types::I64, limb as i64)),
        }
    }

    fn push(&mut self, value: EvmWord) {
        self.pushn(&[value]);
    }

    fn pushn(&mut self, values: &[EvmWord]) {
        debug_assert!(values.len() <= EvmStack::CAPACITY);

        let len = self.load_len();
        let failure_cond = self.bcx.ins().icmp_imm(
            IntCC::UnsignedGreaterThan,
            len,
            (EvmStack::CAPACITY - values.len()) as i64,
        );
        self.build_failure(failure_cond, Ret::StackOverflow);

        self.pushn_unchecked(values);
    }

    fn push_unchecked(&mut self, value: EvmWord) {
        self.pushn_unchecked(&[value]);
    }

    fn pushn_unchecked(&mut self, values: &[EvmWord]) {
        let len = self.load_len();
        for value in values {
            let len = self.bcx.ins().iadd_imm(len, 1);
            let sp = self.sp_at(len);
            for (i, value) in value.iter().enumerate() {
                self.bcx.ins().store(MemFlags::trusted(), value, sp, i as i32 * 8);
            }
        }
        self.store_len(len);
    }

    /// Removes the topmost element from the stack and returns it.
    fn pop(&mut self) -> EvmWord {
        self.popn::<1>()[0]
    }

    /// Removes the topmost `N` elements from the stack and returns them.
    fn popn<const N: usize>(&mut self) -> [EvmWord; N] {
        debug_assert_ne!(N, 0);

        let mut len = self.load_len();
        let failure_cond = self.bcx.ins().icmp_imm(IntCC::UnsignedLessThan, len, N as i64);
        self.build_failure(failure_cond, Ret::StackUnderflow);

        let ret = std::array::from_fn(|_i| {
            len = self.bcx.ins().iadd_imm(len, -1);
            let sp = self.sp_at(len);
            self.load_word(sp)
        });
        self.store_len(len);
        ret
    }

    /// Duplicates the `n`th value from the top of the stack.
    /// `n` cannot be `0`.
    fn dup(&mut self, n: u8) {
        debug_assert_ne!(n, 0);

        let len = self.load_len();
        let failure_cond = self.bcx.ins().icmp_imm(IntCC::UnsignedLessThan, len, n as i64);
        self.build_failure(failure_cond, Ret::StackUnderflow);

        let sp = self.sp_from_top(len, n as usize);
        let value = self.load_word(sp);
        self.push(value);
    }

    /// Swaps the topmost value with the `n`th value from the top.
    fn swap(&mut self, n: u8) {
        debug_assert_ne!(n, 0);

        let len = self.load_len();
        let failure_cond = self.bcx.ins().icmp_imm(IntCC::UnsignedLessThan, len, n as i64);
        self.build_failure(failure_cond, Ret::StackUnderflow);

        // let tmp;
        let tmp = self.bcx.create_sized_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: EvmWord::SIZE as _,
        });
        // tmp = a;
        let a_sp = self.sp_from_top(len, n as usize);
        let a = self.load_word(a_sp);
        for (i, value) in a.iter().enumerate() {
            let offset = (i * EvmWord::LIMB) as i32;
            self.bcx.ins().stack_store(value, tmp, offset);
        }
        // a = b;
        let b_sp = self.sp_from_top(len, 1);
        let b = self.load_word(b_sp);
        for (i, value) in b.iter().enumerate() {
            let offset = (i * EvmWord::LIMB) as i32;
            self.bcx.ins().store(MemFlags::trusted(), value, a_sp, offset);
        }
        // b = tmp;
        for i in 0..EvmWord::N_LIMBS {
            let offset = (i * EvmWord::LIMB) as i32;
            let v = self.bcx.ins().stack_load(types::I64, tmp, offset);
            self.bcx.ins().store(MemFlags::trusted(), v, b_sp, offset);
        }
    }

    /// Loads an EVM word at `ptr`.
    fn load_word(&mut self, ptr: Value) -> EvmWord {
        EvmWord {
            values: std::array::from_fn(|i| {
                self.bcx.ins().load(
                    types::I64,
                    MemFlags::trusted(),
                    ptr,
                    (i * EvmWord::LIMB) as i32,
                )
            }),
        }
    }

    /// Loads the stack length.
    fn load_len(&mut self) -> Value {
        self.bcx.ins().stack_load(self.pointer_type, self.stack.len, 0)
    }

    /// Stores the stack length.
    fn store_len(&mut self, value: Value) {
        self.bcx.ins().stack_store(value, self.stack.len, 0);
    }

    /// Returns the stack pointer at `len` (`sp + len * WORD`).
    fn sp_at(&mut self, len: Value) -> Value {
        let len = self.bcx.ins().imul_imm(len, EvmWord::SIZE as i64);
        self.bcx.ins().iadd(self.stack.sp, len)
    }

    /// Returns the stack pointer at `len` from the top (`sp + (len - n) * WORD`).
    fn sp_from_top(&mut self, len: Value, n: usize) -> Value {
        let len = self.bcx.ins().iadd_imm(len, -(n as i64));
        self.sp_at(len)
    }

    /// `if failure_cond { return ret } else { ... }`
    fn build_failure(&mut self, failure_cond: Value, ret: Ret) {
        let failure = self.bcx.create_block();
        let target = self.bcx.create_block();
        self.bcx.ins().brif(failure_cond, failure, &[], target, &[]);

        self.bcx.set_cold_block(failure);
        self.bcx.seal_block(failure);
        self.bcx.switch_to_block(failure);
        self.build_return(ret);

        self.bcx.seal_block(target);
        self.bcx.switch_to_block(target);
    }

    /// `return ret`
    fn build_return(&mut self, ret: Ret) {
        let old_block = self.bcx.current_block().unwrap();
        if self.clif_comments.enabled() {
            self.bcx.ins().nop();
            let inst = self.bcx.func.layout.last_inst(old_block).unwrap();
            self.add_comment(inst, format!("return {ret:?}"));
        }
        let ret = self.bcx.ins().iconst(self.return_type, ret as i64);
        self.bcx.ins().return_(&[ret]);
        // self.bcx.seal_block(old_block);

        let new_block = self.bcx.create_block();
        self.bcx.switch_to_block(new_block);
        // self.bcx.ensure_inserted_block();
    }

    fn last_inst(&self) -> Inst {
        let block = self.bcx.current_block().unwrap();
        self.bcx.func.layout.last_inst(block).unwrap()
    }
}

/// The return value of a JIT'd EVM bytecode function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(isize)]
pub enum Ret {
    /// `STOP` instruction.
    Stop,
    /// Stack underflow.
    StackUnderflow,
    /// Stack overflow.
    StackOverflow,
}

struct EvmStack {
    sp: Value,
    len: StackSlot,
}

impl EvmStack {
    const CAPACITY: usize = 1024;

    fn new(bcx: &mut FunctionBuilder<'_>, pointer_type: Type) -> Self {
        let slot = bcx.create_sized_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: (Self::CAPACITY * EvmWord::SIZE) as u32,
        });
        let sp = bcx.ins().stack_addr(pointer_type, slot, 0);
        let len = bcx.create_sized_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: pointer_type.bytes(),
        });
        let zero = bcx.ins().iconst(pointer_type, 0);
        bcx.ins().stack_store(zero, len, 0);
        Self { sp, len }
    }
}

#[derive(Clone, Copy)]
struct EvmWord {
    /// `[u64; 4]`
    values: [Value; 4],
}

impl EvmWord {
    const LIMB: usize = 8;
    const N_LIMBS: usize = 4;
    const SIZE: usize = 32;

    fn iter(&self) -> impl ExactSizeIterator<Item = Value> + '_ {
        self.values.iter().copied()
    }
}
