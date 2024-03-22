//! JIT compiler implementation.

use crate::{Backend, Builder, Bytecode, IntCC, JitEvmFn, OpcodeData, Result};
use revm_interpreter::{opcode as op, InstructionResult};
use revm_primitives::{SpecId, U256};
use std::path::PathBuf;

const STACK_CAP: usize = 1024;
// const WORD_SIZE: usize = 32;

// TODO: indexvec or something
type Opcode = usize;

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
    #[instrument(level = "debug", skip_all, ret)]
    pub fn compile(&mut self, bytecode: &[u8], spec: SpecId) -> Result<JitEvmFn> {
        let bytecode = debug_time!("parse", || self.parse_bytecode(bytecode, spec))?;
        debug_time!("compile", || self.compile_bytecode(&bytecode))
    }

    fn parse_bytecode<'a>(&mut self, bytecode: &'a [u8], spec: SpecId) -> Result<Bytecode<'a>> {
        trace!(bytecode = revm_primitives::hex::encode(bytecode));
        let mut bytecode = trace_time!("new bytecode", || Bytecode::new(bytecode, spec));
        trace_time!("analyze", || bytecode.analyze())?;
        Ok(bytecode)
    }

    fn compile_bytecode(&mut self, bytecode: &Bytecode<'_>) -> Result<JitEvmFn> {
        let name = &self.new_name()[..];
        let bcx = self.backend.build_function(name)?;

        trace_time!("translate", || FunctionCx::translate(bcx, &bytecode))?;

        let verify = |b: &mut B| trace_time!("verify", || b.verify_function(name));
        if let Some(dir) = &self.out_dir {
            trace_time!("dump unopt IR", || {
                let filename = format!("{name}.unopt.{}", self.backend.ir_extension());
                self.backend.dump_ir(&dir.join(filename))
            })?;

            // Dump IR before verifying for better debugging.
            verify(&mut self.backend)?;

            trace_time!("dump unopt disasm", || {
                let filename = format!("{name}.unopt.s");
                self.backend.dump_disasm(&dir.join(filename))
            })?;
        } else {
            verify(&mut self.backend)?;
        }

        trace_time!("optimize", || self.backend.optimize_function(name)?);

        if let Some(dir) = &self.out_dir {
            trace_time!("dump opt IR", || {
                let filename = format!("{name}.opt.{}", self.backend.ir_extension());
                self.backend.dump_ir(&dir.join(filename))
            })?;

            trace_time!("dump opt disasm", || {
                let filename = format!("{name}.opt.s");
                self.backend.dump_disasm(&dir.join(filename))
            })?;
        }

        self.backend.get_function(name)
    }
}

struct FunctionCx<'a, B: Builder> {
    /// The backend's function builder.
    bcx: B,

    // Common types.
    isize_type: B::Type,
    word_type: B::Type,
    return_type: B::Type,

    /// The stack length slot.
    stack_len: B::StackSlot,
    /// The stack pointer. Constant throughout the function, passed in the arguments.
    sp: B::Value,

    /// The bytecode being translated.
    bytecode: &'a Bytecode<'a>,
    /// All entry blocks for each opcode.
    op_blocks: Vec<B::BasicBlock>,
    /// The current opcode being translated.
    ///
    /// Note that `self.op_blocks[current_opcode]` does not necessarily equal the builder's current
    /// block.
    current_opcode: Opcode,
}

impl<'a, B: Builder> FunctionCx<'a, B> {
    fn translate(mut bcx: B, bytecode: &'a Bytecode<'a>) -> Result<()> {
        // Get common types.
        let isize_type = bcx.type_ptr_sized_int();
        let return_type = bcx.type_int(32);
        let word_type = bcx.type_int(256);
        // let stack_type = bcx.type_array(word_type, STACK_CAP as _);

        // Set up entry block.
        let stack_len = bcx.new_stack_slot(isize_type, "len_slot");
        let zero = bcx.iconst(isize_type, 0);
        bcx.stack_store(zero, stack_len);

        let sp = bcx.fn_param(0);

        // Create all opcode entry blocks.
        let op_blocks: Vec<_> = bytecode
            .opcodes
            .iter()
            .enumerate()
            .map(|(i, opcode)| bcx.create_block(&op_block_name_with(i, *opcode, "")))
            .collect();

        // Branch to the first opcode. The bytecode is guaranteed to have at least one opcode.
        assert!(!op_blocks.is_empty(), "got empty bytecode");
        bcx.br(op_blocks[0]);

        let mut fx = FunctionCx {
            isize_type,
            word_type,
            return_type,
            stack_len,
            sp,
            bcx,
            bytecode,
            op_blocks,
            current_opcode: 0,
        };
        for i in 0..bytecode.opcodes.len() {
            fx.current_opcode = i;
            fx.translate_opcode()?;
        }

        Ok(())
    }

    fn translate_opcode(&mut self) -> Result<()> {
        let opcode = self.current_opcode;
        let data = self.bytecode.opcodes[opcode];
        let block = self.op_blocks[opcode];
        self.bcx.switch_to_block(block);

        let op_byte = data.opcode;

        macro_rules! unop {
            ($op:ident) => {{
                let mut a = self.pop();
                a = self.bcx.$op(a);
                self.push_unchecked(a);
            }};
        }

        macro_rules! binop {
            ($op:ident) => {{
                let [a, b] = self.popn();
                let r = self.bcx.$op(a, b);
                self.push_unchecked(r);
            }};
            (@if_not_zero $op:ident $(, $extra_cond:expr)?) => {{
                let [a, b] = self.popn();
                let cond = self.bcx.icmp_imm(IntCC::Equal, b, 0);
                let r = self.bcx.lazy_select(
                    cond,
                    self.word_type,
                    |bcx, block| {
                        bcx.set_cold_block(block);
                        bcx.iconst_256(U256::ZERO)
                    },
                    |bcx, _op_block| {
                        // TODO: segfault ??
                        // $(
                        //     let cond = $extra_cond(bcx, a, b);
                        //     return bcx.lazy_select(
                        //         cond,
                        //         self.word_type,
                        //         |bcx, block| {
                        //             bcx.set_cold_block(block);
                        //             bcx.iconst_256(U256::ZERO)
                        //         },
                        //         |bcx, _block| bcx.$op(a, b),
                        //     );
                        //     #[allow(unreachable_code)]
                        // )?
                        bcx.$op(a, b)
                    },
                );
                self.push_unchecked(r);
            }};
        }

        macro_rules! cmp_op {
            ($op:ident) => {{
                let [a, b] = self.popn();
                let r = self.bcx.icmp(IntCC::$op, a, b);
                let r = self.bcx.zext(self.word_type, r);
                self.push_unchecked(r);
            }};
        }

        match data.opcode {
            op::STOP => {
                self.build_return(InstructionResult::Stop);
                return Ok(());
            }

            op::ADD => binop!(iadd),
            op::MUL => binop!(imul),
            op::SUB => binop!(isub),
            op::DIV => binop!(@if_not_zero udiv),
            op::SDIV => binop!(@if_not_zero sdiv, |bcx: &mut B, a, b| {
                let min = bcx.iconst_256(I256_MIN);
                let a_is_min = bcx.icmp(IntCC::Equal, a, min);
                let b_is_neg1 = bcx.icmp_imm(IntCC::Equal, b, -1);
                bcx.bitand(a_is_min, b_is_neg1)
            }),
            op::MOD => binop!(@if_not_zero urem),
            op::SMOD => binop!(@if_not_zero srem),
            op::ADDMOD => {
                // TODO
                // let [a, b, c] = self.popn();
            }
            op::MULMOD => {
                // TODO
                // let [a, b, c] = self.popn();
            }
            op::EXP => {
                // TODO
                // let [base, exponent] = self.popn();
                // let r = self.bcx.ipow(base, exponent);
                // self.push_unchecked(r);
            }
            op::SIGNEXTEND => {
                // TODO
                // let [a, b] = self.popn();
                // let r = self.bcx.sign_extend(a, b);
                // self.push_unchecked(r);
            }

            op::LT => cmp_op!(UnsignedLessThan),
            op::GT => cmp_op!(UnsignedGreaterThan),
            op::SLT => cmp_op!(SignedLessThan),
            op::SGT => cmp_op!(SignedGreaterThan),
            op::EQ => cmp_op!(Equal),
            op::ISZERO => {
                let a = self.pop();
                let r = self.bcx.icmp_imm(IntCC::Equal, a, 0);
                let r = self.bcx.zext(self.word_type, r);
                self.push_unchecked(r);
            }
            op::AND => binop!(bitand),
            op::OR => binop!(bitor),
            op::XOR => binop!(bitxor),
            op::NOT => unop!(bitnot),
            op::BYTE => {
                // TODO
                // let [index, value] = self.popn();
                // let cond = self.bcx.icmp_imm(IntCC::UnsignedLessThan, value, 32);
                // let zero = self.bcx.iconst_256(U256::ZERO);
                // let r = self.bcx.select(cond, cond, zero);
                // self.push_unchecked(r);
            }
            op::SHL => binop!(ishl),
            op::SHR => binop!(ushr),
            op::SAR => binop!(sshr),

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
            op::JUMPDEST => {
                self.bcx.nop();
            }
            op::TLOAD => {}
            op::TSTORE => {}

            op::PUSH0 => {
                let value = self.bcx.iconst_256(U256::ZERO);
                self.push(value);
            }
            op::PUSH1..=op::PUSH32 => {
                let imm = self.bytecode.get_imm_of(data);
                let value = imm.map(U256::from_be_slice).unwrap_or_default();
                let value = self.bcx.iconst_256(value);
                self.push(value);
            }

            op::DUP1..=op::DUP16 => self.dup(op_byte - op::DUP1 + 1),

            op::SWAP1..=op::SWAP16 => self.swap(op_byte - op::SWAP1 + 1),

            op::LOG0..=op::LOG4 => {
                let _n = op_byte - op::LOG0;
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

            _ => {}
        }

        if let Some(next) = self.op_blocks.get(opcode + 1) {
            self.bcx.br(*next);
        }
        self.bcx.seal_block(block);

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
        self.build_failure(failure_cond, InstructionResult::StackOverflow);

        self.pushn_unchecked(values);
    }

    fn push_unchecked(&mut self, value: B::Value) {
        self.pushn_unchecked(&[value]);
    }

    fn pushn_unchecked(&mut self, values: &[B::Value]) {
        let mut len = self.load_len();
        for &value in values {
            let sp = self.sp_at(len);
            self.bcx.store(value, sp);
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
        debug_assert!(N < 26, "too many pops");

        let mut len = self.load_len();
        let failure_cond = self.bcx.icmp_imm(IntCC::UnsignedLessThan, len, N as i64);
        self.build_failure(failure_cond, InstructionResult::StackUnderflow);

        let ret = std::array::from_fn(|i| {
            len = self.bcx.isub_imm(len, 1);
            let sp = self.sp_at(len);
            let name = b'a' + i as u8;
            self.load_word(sp, core::str::from_utf8(&[name]).unwrap())
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
        self.build_failure(failure_cond, InstructionResult::StackUnderflow);

        let sp = self.sp_from_top(len, n as usize);
        let value = self.load_word(sp, &format!("dup{n}"));
        self.push(value);
    }

    /// Swaps the topmost value with the `n`th value from the top.
    fn swap(&mut self, n: u8) {
        debug_assert_ne!(n, 0);

        let len = self.load_len();
        let failure_cond = self.bcx.icmp_imm(IntCC::UnsignedLessThan, len, n as i64);
        self.build_failure(failure_cond, InstructionResult::StackUnderflow);

        // let tmp;
        let tmp = self.bcx.new_stack_slot(self.word_type, "tmp");
        // tmp = a;
        let a_sp = self.sp_from_top(len, n as usize);
        let a = self.load_word(a_sp, "a");
        self.bcx.stack_store(a, tmp);
        // a = b;
        let b_sp = self.sp_from_top(len, 1);
        let b = self.load_word(b_sp, "b");
        self.bcx.store(b, a_sp);
        // b = tmp;
        let tmp = self.bcx.stack_load(self.word_type, tmp, "tmp");
        self.bcx.store(tmp, b_sp);
    }

    /// Loads the word at the given pointer.
    fn load_word(&mut self, ptr: B::Value, name: &str) -> B::Value {
        self.bcx.load(self.word_type, ptr, name)
    }

    /// Loads the stack length.
    fn load_len(&mut self) -> B::Value {
        self.bcx.stack_load(self.isize_type, self.stack_len, "len")
    }

    /// Stores the stack length.
    fn store_len(&mut self, value: B::Value) {
        self.bcx.stack_store(value, self.stack_len);
    }

    /// Returns the stack pointer at `len` (`&stack[len]`).
    fn sp_at(&mut self, len: B::Value) -> B::Value {
        self.bcx.gep(self.word_type, self.sp, len)
    }

    /// Returns the stack pointer at `len` from the top (`&stack[CAPACITY - len]`).
    fn sp_from_top(&mut self, len: B::Value, n: usize) -> B::Value {
        let len = self.bcx.iadd_imm(len, -(n as i64));
        self.sp_at(len)
    }

    /// `if failure_cond { return ret } else { ... }`
    fn build_failure(&mut self, failure_cond: B::Value, ret: InstructionResult) {
        let failure = self.create_block_after_op("fail");
        let target = self.create_block_after(failure, "contd");
        self.bcx.brif(failure_cond, failure, target);

        self.bcx.set_cold_block(failure);
        self.bcx.switch_to_block(failure);
        self.build_return(ret);

        self.bcx.switch_to_block(target);
    }

    /// Builds `return ret`.
    fn build_return(&mut self, ret: InstructionResult) {
        let old_block = self.bcx.current_block();
        if self.comments_enabled() {
            self.bcx.nop();
            self.add_comment(&format!("return {ret:?}"));
        }
        let ret = self.bcx.iconst(self.return_type, ret as i64);
        self.bcx.ret(&[ret]);
        if let Some(old_block) = old_block {
            self.bcx.seal_block(old_block);
        }
    }

    /// Returns whether comments are enabled.
    fn comments_enabled(&self) -> bool {
        true
    }

    /// Adds a comment to the current instruction.
    fn add_comment(&mut self, comment: &str) {
        self.bcx.nop();
        self.bcx.add_comment_to_current_inst(comment);
    }

    /// Creates a named block.
    #[allow(dead_code)]
    fn create_block(&mut self, name: &str) -> B::BasicBlock {
        let name = self.op_block_name(name);
        self.bcx.create_block(&name)
    }

    /// Creates a named block after the given block.
    fn create_block_after(&mut self, after: B::BasicBlock, name: &str) -> B::BasicBlock {
        let name = self.op_block_name(name);
        self.bcx.create_block_after(after, &name)
    }

    /// Creates a named block after the current opcode.
    fn create_block_after_op(&mut self, name: &str) -> B::BasicBlock {
        let after = self.op_blocks[self.current_opcode];
        self.create_block_after(after, name)
    }

    /// Returns the block name for the current opcode with the given suffix.
    fn op_block_name(&self, name: &str) -> String {
        op_block_name_with(self.current_opcode, self.bytecode.opcodes[self.current_opcode], name)
    }
}

fn op_block_name_with(op: Opcode, data: OpcodeData, with: &str) -> String {
    let data = data.to_raw();
    if with.is_empty() {
        format!("_{op}_{data}")
    } else {
        format!("_{op}_{data}_{with}")
    }
}

#[cfg(test)]
#[allow(dead_code, unused_imports)]
mod tests {
    use super::*;
    use crate::*;
    use revm_interpreter::opcode as op;
    use revm_primitives::ruint::uint;

    #[cfg(feature = "llvm")]
    #[test]
    fn test_llvm() {
        let context = revm_jit_llvm::inkwell::context::Context::create();
        run_tests_with_backend(|| crate::JitEvmLlvmBackend::new(&context).unwrap());
    }

    struct TestCase<'a> {
        name: &'a str,
        bytecode: &'a [u8],

        expected_return: InstructionResult,
        expected_stack: &'a [U256],
    }

    const fn bytecode_unop(op: u8, a: U256) -> [u8; 36] {
        let mut code = [0; 36];

        let mut i = 0;

        code[i] = op::PUSH32;
        i += 1;
        {
            let mut j = 0;
            let bytes = a.to_be_bytes::<32>();
            while j < 32 {
                code[i] = bytes[j];
                i += 1;
                j += 1;
            }
        }

        code[i] = op;
        i += 1;
        code[i] = op::STOP;
        // i += 1;

        code
    }

    const fn bytecode_binop(op: u8, a: U256, b: U256) -> [u8; 68] {
        // NOTE: push `b` first.

        let mut code = [0; 68];

        let mut i = 0;

        code[i] = op::PUSH32;
        i += 1;
        {
            let mut j = 0;
            let bytes = b.to_be_bytes::<32>();
            while j < 32 {
                code[i] = bytes[j];
                i += 1;
                j += 1;
            }
        }

        code[i] = op::PUSH32;
        i += 1;
        {
            let mut j = 0;
            let bytes = a.to_be_bytes::<32>();
            while j < 32 {
                code[i] = bytes[j];
                i += 1;
                j += 1;
            }
        }

        code[i] = op;
        i += 1;
        code[i] = op::STOP;
        // i += 1;

        code
    }

    macro_rules! testcases {
        (@op $op:expr, $a:expr) => { bytecode_unop($op, $a) };
        (@op $op:expr, $a:expr, $b:expr) => { bytecode_binop($op, $a, $b) };

        ($($name:ident($op:expr, $a:expr $(, $b:expr)? => $ret:expr)),* $(,)?) => {
            uint!([$(TestCase {
                name: stringify!($name),
                bytecode: &testcases!(@op $op, $a $(, $b)?),
                expected_return: InstructionResult::Stop,
                expected_stack: &[$ret],
            }),*])
        };
    }

    static BASIC_CASES: &[TestCase<'static>] = &[
        TestCase {
            name: "empty",
            bytecode: &[],
            expected_return: InstructionResult::Stop,
            expected_stack: &[],
        },
        TestCase {
            name: "stop",
            bytecode: &[op::STOP],
            expected_return: InstructionResult::Stop,
            expected_stack: &[],
        },
        TestCase {
            name: "underflow1",
            bytecode: &[op::ADD, op::STOP],
            expected_return: InstructionResult::StackUnderflow,
            expected_stack: &[],
        },
        TestCase {
            name: "underflow2",
            bytecode: &[op::PUSH0, op::ADD, op::STOP],
            expected_return: InstructionResult::StackUnderflow,
            expected_stack: &[U256::ZERO],
        },
    ];

    static ARITH_CASES: &[TestCase<'static>] = &testcases![
        add1(op::ADD, 0_U256, 0_U256 => 0_U256),
        add2(op::ADD, 1_U256, 2_U256 => 3_U256),
        add3(op::ADD, 255_U256, 255_U256 => 510_U256),
        add4(op::ADD, U256::MAX, 1_U256 => 0_U256),
        add5(op::ADD, U256::MAX, 2_U256 => 1_U256),

        sub1(op::SUB, 3_U256, 2_U256 => 1_U256),
        sub2(op::SUB, 1_U256, 2_U256 => MINUS_1),
        sub2(op::SUB, 1_U256, 3_U256 => MINUS_1.wrapping_sub(1_U256)),
        sub3(op::SUB, 255_U256, 255_U256 => 0_U256),

        mul1(op::MUL, 1_U256, 2_U256 => 2_U256),
        mul2(op::MUL, 32_U256, 32_U256 => 1024_U256),
        mul3(op::MUL, U256::MAX, 2_U256 => U256::MAX.wrapping_sub(1_U256)),

        div1(op::DIV, 32_U256, 32_U256 => 1_U256),
        div2(op::DIV, 1_U256, 2_U256 => 0_U256),
        div3(op::DIV, 2_U256, 2_U256 => 1_U256),
        div4(op::DIV, 3_U256, 2_U256 => 1_U256),
        div5(op::DIV, 4_U256, 2_U256 => 2_U256),
        div_by_zero(op::DIV, 32_U256, 0_U256 => 0_U256),

        rem1(op::MOD, 32_U256, 32_U256 => 0_U256),
        rem2(op::MOD, 1_U256, 2_U256 => 1_U256),
        rem3(op::MOD, 2_U256, 2_U256 => 0_U256),
        rem4(op::MOD, 3_U256, 2_U256 => 1_U256),
        rem5(op::MOD, 4_U256, 2_U256 => 0_U256),
        rem_by_zero(op::MOD, 32_U256, 0_U256 => 0_U256),

        sdiv1(op::SDIV, 32_U256, 32_U256 => 1_U256),
        sdiv2(op::SDIV, 1_U256, 2_U256 => 0_U256),
        sdiv3(op::SDIV, 2_U256, 2_U256 => 1_U256),
        sdiv4(op::SDIV, 3_U256, 2_U256 => 1_U256),
        sdiv5(op::SDIV, 4_U256, 2_U256 => 2_U256),
        sdiv_by_zero(op::SDIV, 32_U256, 0_U256 => 0_U256),
        sdiv_min_by_1(op::SDIV, I256_MIN, 1_U256 => I256_MIN.wrapping_neg()),
        sdiv_min_by_minus_1(op::SDIV, I256_MIN, MINUS_1 => I256_MIN),
        sdiv_max1(op::SDIV, I256_MAX, 1_U256 => I256_MAX),
        sdiv_max2(op::SDIV, I256_MAX, MINUS_1 => I256_MAX.wrapping_neg()),

        srem1(op::SMOD, 32_U256, 32_U256 => 0_U256),
        srem2(op::SMOD, 1_U256, 2_U256 => 1_U256),
        srem3(op::SMOD, 2_U256, 2_U256 => 0_U256),
        srem4(op::SMOD, 3_U256, 2_U256 => 1_U256),
        srem5(op::SMOD, 4_U256, 2_U256 => 0_U256),
        srem_by_zero(op::SMOD, 32_U256, 0_U256 => 0_U256),

        // TODO:
        // ADDMOD
        // MULMOD
        // EXP
        // SIGNEXTEND
    ];

    static CMP_CASES: &[TestCase<'static>] = &testcases![
        lt1(op::LT, 1_U256, 2_U256 => 1_U256),
        lt2(op::LT, 2_U256, 1_U256 => 0_U256),
        lt3(op::LT, 1_U256, 1_U256 => 0_U256),
        lt4(op::LT, MINUS_1, 1_U256 => 0_U256),

        gt1(op::GT, 1_U256, 2_U256 => 0_U256),
        gt2(op::GT, 2_U256, 1_U256 => 1_U256),
        gt3(op::GT, 1_U256, 1_U256 => 0_U256),
        gt4(op::GT, MINUS_1, 1_U256 => 1_U256),

        slt1(op::SLT, 1_U256, 2_U256 => 1_U256),
        slt2(op::SLT, 2_U256, 1_U256 => 0_U256),
        slt3(op::SLT, 1_U256, 1_U256 => 0_U256),
        slt4(op::SLT, MINUS_1, 1_U256 => 1_U256),

        sgt1(op::SGT, 1_U256, 2_U256 => 0_U256),
        sgt2(op::SGT, 2_U256, 1_U256 => 1_U256),
        sgt3(op::SGT, 1_U256, 1_U256 => 0_U256),
        sgt4(op::SGT, MINUS_1, 1_U256 => 0_U256),

        eq1(op::EQ, 1_U256, 2_U256 => 0_U256),
        eq2(op::EQ, 2_U256, 1_U256 => 0_U256),
        eq3(op::EQ, 1_U256, 1_U256 => 1_U256),

        iszero1(op::ISZERO, 0_U256 => 1_U256),
        iszero2(op::ISZERO, 1_U256 => 0_U256),
        iszero3(op::ISZERO, 2_U256 => 0_U256),
    ];

    static BITWISE_CASES: &[TestCase<'static>] = &testcases![
        and1(op::AND, 0_U256, 0_U256 => 0_U256),
        and2(op::AND, 1_U256, 1_U256 => 1_U256),
        and3(op::AND, 1_U256, 2_U256 => 0_U256),
        and4(op::AND, 255_U256, 255_U256 => 255_U256),

        or1(op::OR, 0_U256, 0_U256 => 0_U256),
        or2(op::OR, 1_U256, 2_U256 => 3_U256),
        or3(op::OR, 1_U256, 3_U256 => 3_U256),
        or4(op::OR, 2_U256, 2_U256 => 2_U256),

        xor1(op::XOR, 0_U256, 0_U256 => 0_U256),
        xor2(op::XOR, 1_U256, 2_U256 => 3_U256),
        xor3(op::XOR, 1_U256, 3_U256 => 2_U256),
        xor4(op::XOR, 2_U256, 2_U256 => 0_U256),

        not1(op::NOT, 0_U256 => U256::MAX),
        not2(op::NOT, U256::MAX => 0_U256),
        not3(op::NOT, 1_U256 => U256::MAX.wrapping_sub(1_U256)),

        // TODO
        // byte1(op::BYTE, 0_U256, 0_U256 => 0_U256),
        // byte2(op::BYTE, 0_U256, 1_U256 => 0_U256),
        // byte3(op::BYTE, 0_U256, 2_U256 => 0_U256),

        shl1(op::SHL, 1_U256, 0_U256 => 1_U256),
        shl2(op::SHL, 1_U256, 1_U256 => 2_U256),
        shl3(op::SHL, 1_U256, 2_U256 => 4_U256),

        shr1(op::SHR, 1_U256, 0_U256 => 1_U256),
        shr2(op::SHR, 2_U256, 1_U256 => 1_U256),
        shr3(op::SHR, 4_U256, 2_U256 => 1_U256),

        sar1(op::SAR, 1_U256, 0_U256 => 1_U256),
        sar2(op::SAR, 2_U256, 1_U256 => 1_U256),
        sar3(op::SAR, 4_U256, 2_U256 => 1_U256),
        sar4(op::SAR, MINUS_1, 1_U256 => MINUS_1),
        sar5(op::SAR, MINUS_1, 2_U256 => MINUS_1),
    ];

    static ALL_TEST_CASES: &[(&str, &[TestCase<'static>])] = &[
        ("basic", BASIC_CASES),
        ("arithmetic", ARITH_CASES),
        ("comparison", CMP_CASES),
        ("bitwise", BITWISE_CASES),
    ];

    // TODO: Have to create a new backend per call for now
    fn run_tests_with_backend<B: Backend>(make_backend: impl Fn() -> B) {
        let backend_name = std::any::type_name::<B>().split("::").last().unwrap();
        for &(group_name, cases) in ALL_TEST_CASES {
            println!("Running test group `{group_name}` for backend `{backend_name}`");
            run_test_group(&make_backend, cases);
            println!();
        }
    }

    fn run_test_group<B: Backend>(make_backend: impl Fn() -> B, cases: &[TestCase<'_>]) {
        for (i, &TestCase { name, bytecode, expected_return, expected_stack }) in
            cases.iter().enumerate()
        {
            let mut jit = JitEvm::new(make_backend());
            // TODO: segfaults if we don't disable IR optimizations
            jit.no_optimize();

            println!("Running test case {i:2}: {name}");
            println!("  bytecode: {}", format_bytecode(bytecode));
            let f = jit.compile(bytecode, SpecId::LATEST).unwrap();

            let mut stack = ContextStack::new();
            let actual_return = unsafe { f(&mut stack) };
            assert_eq!(actual_return, expected_return);

            for (j, (chunk, expected)) in
                stack.as_slice().chunks_exact(32).zip(expected_stack).enumerate()
            {
                let bytes: [u8; 32] = chunk.try_into().unwrap();
                let actual = if cfg!(target_endian = "big") {
                    U256::from_be_bytes(bytes)
                } else {
                    U256::from_le_bytes(bytes)
                };
                assert_eq!(actual, *expected, "stack item {j} does not match");
            }
        }
    }
}
