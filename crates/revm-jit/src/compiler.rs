use crate::{Backend, Builder, IntCC, JitEvmFn, Result, Ret};
use revm_interpreter::opcode as op;
use revm_primitives::U256;
use std::{path::PathBuf, slice};

const STACK_CAP: usize = 1024;
// const WORD_SIZE: usize = 32;

// TODO: cannot find function if `compile` is called a second time

/// JIT compiler for EVM bytecode.
#[derive(Debug)]
pub struct JitEvm<B> {
    backend: B,

    out_dir: Option<PathBuf>,
    function_counter: usize,
}

impl<B: Backend + Default> Default for JitEvm<B> {
    fn default() -> Self {
        Self::new(B::default())
    }
}

impl<B: Backend> JitEvm<B> {
    /// Creates a new instance of the JIT compiler with the given backend.
    pub fn new(backend: B) -> Self {
        Self { backend, out_dir: None, function_counter: 0 }
    }

    /// Dumps the IR and potential to the given directory after compilation.
    ///
    /// Disables dumping if `output_dir` is `None`.
    ///
    /// Creates a subdirectory with the name of the backend in the given directory.
    pub fn dump_to(&mut self, output_dir: Option<PathBuf>) {
        self.backend.set_is_dumping(output_dir.is_some());
        self.out_dir = output_dir;
    }

    /// Don't optimize the generated code.
    pub fn no_optimize(&mut self) {
        self.backend.no_optimize();
    }

    fn new_name(&mut self) -> String {
        let name = format!("__evm_bytecode_{}", self.function_counter);
        self.function_counter += 1;
        name
    }

    /// Compiles the given EVM bytecode into a JIT function.
    pub fn compile(&mut self, bytecode: &[u8]) -> Result<JitEvmFn> {
        let name = &self.new_name()[..];
        let bcx = self.backend.build_function(name)?;
        translate(bcx, bytecode)?;
        if let Some(dir) = &self.out_dir {
            let filename = format!("{name}.unopt.{}", self.backend.ir_extension());
            self.backend.dump_ir(&dir.join(filename))?;

            let filename = format!("{name}.unopt.s");
            self.backend.dump_disasm(&dir.join(filename))?;
        }

        self.backend.optimize_function(name)?;
        if let Some(dir) = &self.out_dir {
            let filename = format!("{name}.opt.{}", self.backend.ir_extension());
            self.backend.dump_ir(&dir.join(filename))?;

            let filename = format!("{name}.opt.s");
            self.backend.dump_disasm(&dir.join(filename))?;
        }

        self.backend.get_function(name)
    }
}

fn translate<B: Builder>(mut bcx: B, bytecode: &[u8]) -> Result<()> {
    let isize_type = bcx.type_ptr_sized_int();
    let return_type = bcx.type_int(32);
    let word_type = bcx.type_int(256);
    // let stack_type = bcx.type_array(word_type, STACK_CAP as _);

    let stack_len = bcx.new_stack_slot(isize_type);
    let zero = bcx.iconst(isize_type, 0);
    bcx.stack_store(zero, stack_len, 0);

    let sp = bcx.fn_param(0);

    FunctionCx { isize_type, word_type, return_type, stack_len, sp, bcx }
        .translate_bytecode(bytecode)
}

struct FunctionCx<B: Builder> {
    bcx: B,

    isize_type: B::Type,
    word_type: B::Type,
    return_type: B::Type,

    stack_len: B::StackSlot,
    sp: B::Value,
}

impl<B: Builder> FunctionCx<B> {
    fn translate_bytecode(&mut self, bytecode: &[u8]) -> Result<()> {
        let mut iter = bytecode.iter();
        while let Some(&opcode) = iter.next() {
            self.translate_opcode(opcode, &mut iter)?;
        }
        Ok(())
    }

    fn translate_opcode(&mut self, opcode: u8, iter: &mut slice::Iter<'_, u8>) -> Result<()> {
        self.bcx.nop();
        /*
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
        */

        let mut get_imm = |len: usize| {
            let r = iter.as_slice().get(..len);
            // This also takes care of the `None` case.
            // TODO: Use `advance_by` when stable.
            iter.take(len).for_each(drop);
            r
        };
        match opcode {
            op::STOP => self.build_return(Ret::Stop),

            op::ADD => {
                let [a, b] = self.popn();
                let r = self.bcx.iadd(a, b);
                self.push_unchecked(r);
            }
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

            op::POP => {
                self.pop();
            }
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
                let value = self.bcx.iconst_256(U256::ZERO);
                self.push(value);
            }
            op::PUSH1..=op::PUSH32 => {
                let n = opcode - op::PUSH0;
                let value = get_imm(n as usize).map(U256::from_be_slice).unwrap_or_default();
                let value = self.bcx.iconst_256(value);
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

    fn push(&mut self, value: B::Value) {
        self.pushn(&[value]);
    }

    fn pushn(&mut self, values: &[B::Value]) {
        debug_assert!(values.len() <= STACK_CAP);

        let len = self.load_len();
        let failure_cond =
            self.bcx.icmp_imm(IntCC::UnsignedGreaterThan, len, (STACK_CAP - values.len()) as i64);
        self.build_failure(failure_cond, Ret::StackOverflow);

        self.pushn_unchecked(values);
    }

    fn push_unchecked(&mut self, value: B::Value) {
        self.pushn_unchecked(&[value]);
    }

    fn pushn_unchecked(&mut self, values: &[B::Value]) {
        let mut len = self.load_len();
        for &value in values {
            let sp = self.sp_at(len);
            self.bcx.store(value, sp, 0);
            len = self.bcx.iadd_imm(len, 1);
        }
        self.store_len(len);
    }

    /// Removes the topmost element from the stack and returns it.
    fn pop(&mut self) -> B::Value {
        self.popn::<1>()[0]
    }

    /// Removes the topmost `N` elements from the stack and returns them.
    fn popn<const N: usize>(&mut self) -> [B::Value; N] {
        debug_assert_ne!(N, 0);

        let mut len = self.load_len();
        let failure_cond = self.bcx.icmp_imm(IntCC::UnsignedLessThan, len, N as i64);
        self.build_failure(failure_cond, Ret::StackUnderflow);

        let ret = std::array::from_fn(|_i| {
            len = self.bcx.iadd_imm(len, -1);
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
        let failure_cond = self.bcx.icmp_imm(IntCC::UnsignedLessThan, len, n as i64);
        self.build_failure(failure_cond, Ret::StackUnderflow);

        let sp = self.sp_from_top(len, n as usize);
        let value = self.bcx.load(self.word_type, sp, 0);
        self.push(value);
    }

    /// Swaps the topmost value with the `n`th value from the top.
    fn swap(&mut self, n: u8) {
        debug_assert_ne!(n, 0);

        let len = self.load_len();
        let failure_cond = self.bcx.icmp_imm(IntCC::UnsignedLessThan, len, n as i64);
        self.build_failure(failure_cond, Ret::StackUnderflow);

        // let tmp;
        /* let tmp = self.bcx.create_sized_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: WORD_SIZE as _,
        });
        // tmp = a;
        let a_sp = self.sp_from_top(len, n as usize);
        let a = self.load_word(a_sp);
        for (i, value) in a.iter().enumerate() {
            let offset = (i * EvmWord::LIMB) as i32;
            self.bcx.stack_store(value, tmp, offset);
        }
        // a = b;
        let b_sp = self.sp_from_top(len, 1);
        let b = self.load_word(b_sp);
        for (i, value) in b.iter().enumerate() {
            let offset = (i * EvmWord::LIMB) as i32;
            self.bcx.store(MemFlags::trusted(), value, a_sp, offset);
        }
        // b = tmp;
        for i in 0..EvmWord::N_LIMBS {
            let offset = (i * EvmWord::LIMB) as i32;
            let v = self.bcx.stack_load(types::I64, tmp, offset);
            self.bcx.store(MemFlags::trusted(), v, b_sp, offset);
        } */
    }

    /// Loads the word at the given pointer.
    fn load_word(&mut self, ptr: B::Value) -> B::Value {
        self.bcx.load(self.word_type, ptr, 0)
    }

    /// Loads the stack length.
    fn load_len(&mut self) -> B::Value {
        self.bcx.stack_load(self.isize_type, self.stack_len, 0)
    }

    /// Stores the stack length.
    fn store_len(&mut self, value: B::Value) {
        self.bcx.stack_store(value, self.stack_len, 0);
    }

    /// Returns the stack pointer at `len` (`&stack[len]`).
    fn sp_at(&mut self, len: B::Value) -> B::Value {
        self.bcx.gep_add(self.word_type, self.sp, len)
    }

    /// Returns the stack pointer at `len` from the top (`&stack[CAPACITY - len]`).
    fn sp_from_top(&mut self, len: B::Value, n: usize) -> B::Value {
        let len = self.bcx.iadd_imm(len, -(n as i64));
        self.sp_at(len)
    }

    /// `if failure_cond { return ret } else { ... }`
    fn build_failure(&mut self, failure_cond: B::Value, ret: Ret) {
        let failure = self.bcx.create_block();
        let target = self.bcx.create_block();
        self.bcx.brif(failure_cond, failure, target);

        self.bcx.set_cold_block(failure);
        self.bcx.switch_to_block(failure);
        self.build_return(ret);

        self.bcx.switch_to_block(target);
    }

    /// `return ret`
    fn build_return(&mut self, ret: Ret) {
        let old_block = self.bcx.current_block();
        /* if self.clif_comments.enabled() {
            self.bcx.nop();
            let inst = self.bcx.func.layout.last_inst(old_block).unwrap();
            self.add_comment(inst, format!("return {ret:?}"));
        } */
        let ret = self.bcx.iconst(self.return_type, ret as i64);
        self.bcx.ret(&[ret]);
        if let Some(old_block) = old_block {
            self.bcx.seal_block(old_block);
        }

        let new_block = self.bcx.create_block();
        self.bcx.switch_to_block(new_block);
        // self.bcx.ensure_inserted_block();
    }

    // fn last_inst(&self) -> Inst {
    //     let block = self.bcx.current_block().unwrap();
    //     self.bcx.func.layout.last_inst(block).unwrap()
    // }
}
