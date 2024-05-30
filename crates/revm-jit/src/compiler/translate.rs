//! EVM to IR translation.

use crate::{
    Backend, Builder, Bytecode, EvmContext, Inst, InstData, InstFlags, IntCC, Result, I256_MIN,
};
use revm_interpreter::{opcode as op, Contract, InstructionResult};
use revm_jit_backend::{BackendTypes, Pointer, TypeMethods};
use revm_jit_builtins::{Builtin, Builtins, CallKind, CreateKind};
use revm_primitives::{BlockEnv, CfgEnv, Env, TxEnv, U256};
use std::{fmt::Write, mem, sync::atomic::AtomicPtr};

const STACK_CAP: usize = 1024;
// const WORD_SIZE: usize = 32;

#[derive(Clone, Copy, Debug)]
pub(super) struct FcxConfig {
    pub(super) comments: bool,
    pub(super) debug_assertions: bool,
    pub(super) frame_pointers: bool,

    pub(super) local_stack: bool,
    pub(super) inspect_stack_length: bool,
    pub(super) stack_bound_checks: bool,
    pub(super) gas_metering: bool,
}

impl Default for FcxConfig {
    fn default() -> Self {
        Self {
            debug_assertions: cfg!(debug_assertions),
            comments: false,
            frame_pointers: cfg!(debug_assertions),
            local_stack: false,
            inspect_stack_length: false,
            stack_bound_checks: true,
            gas_metering: true,
        }
    }
}

/// A list of incoming values for a block. Represents a `phi` node.
type Incoming<B> = Vec<(<B as BackendTypes>::Value, <B as BackendTypes>::BasicBlock)>;

/// A list of `switch` targets.
type SwitchTargets<B> = Vec<(u64, <B as BackendTypes>::BasicBlock)>;

pub(super) struct FunctionCx<'a, B: Backend> {
    // Configuration.
    config: FcxConfig,

    /// The backend's function builder.
    bcx: B::Builder<'a>,

    // Common types.
    isize_type: B::Type,
    word_type: B::Type,
    address_type: B::Type,
    i8_type: B::Type,

    // Locals.
    /// The stack length. Either passed in the arguments as a pointer or allocated locally.
    stack_len: Pointer<B::Builder<'a>>,
    /// The stack value. Constant throughout the function, either passed in the arguments as a
    /// pointer or allocated locally.
    stack: Pointer<B::Builder<'a>>,
    /// The amount of gas remaining. `i64`. See `Gas`.
    gas_remaining: Pointer<B::Builder<'a>>,
    /// The environment. Constant throughout the function.
    env: B::Value,
    /// The contract. Constant throughout the function.
    contract: B::Value,
    /// The EVM context. Opaque pointer, only passed to builtins.
    ecx: B::Value,
    /// Stack length before the current instruction.
    len_before: B::Value,
    /// Stack length offset for the current instruction, used for push/pop.
    len_offset: i8,

    /// The bytecode being translated.
    bytecode: &'a Bytecode<'a>,
    /// All entry blocks for each instruction.
    inst_entries: Vec<B::BasicBlock>,
    /// The current instruction being translated.
    current_inst: Inst,

    /// `dynamic_jump_table` incoming values.
    incoming_dynamic_jumps: Incoming<B>,
    /// The dynamic jump table block where all dynamic jumps branch to.
    dynamic_jump_table: B::BasicBlock,

    /// `failure_block` incoming values.
    incoming_failures: Incoming<B>,
    /// The block that all failures branch to.
    failure_block: B::BasicBlock,
    /// `return_block` incoming values.
    incoming_returns: Incoming<B>,
    /// The return block that all return instructions branch to.
    return_block: B::BasicBlock,

    /// `resume_block` switch values.
    resume_blocks: SwitchTargets<B>,
    /// `suspend_block` incoming values.
    suspend_blocks: Incoming<B>,
    /// The suspend block that all suspend instructions branch to.
    suspend_block: B::BasicBlock,

    /// Builtins.
    builtins: &'a mut Builtins<B>,
}

impl<'a, B: Backend> FunctionCx<'a, B> {
    /// Translates an EVM bytecode into a native function.
    ///
    /// Example pseudo-code:
    ///
    /// ```ignore (pseudo-code)
    /// // `cfg(will_suspend) = bytecode.will_suspend()`: `true` if it contains a
    /// // `*CALL*` or `CREATE*` instruction.
    /// fn evm_bytecode(args: ...) {
    ///     setup_locals();
    ///
    ///     #[cfg(debug_assertions)]
    ///     if args.<ptr>.is_null() { panic!("...") };
    ///
    ///     load_arguments();
    ///
    ///     #[cfg(will_suspend)]
    ///     resume: {
    ///         goto match ecx.resume_at {
    ///             0 => inst0,
    ///             1 => first_call_or_create_inst + 1, // + 1 as in the block after.
    ///             2 => second_call_or_create_inst + 1,
    ///             ... => ...,
    ///             _ => unreachable, // Assumed to be valid.
    ///         };
    ///     };
    ///     
    ///     op.inst0: { /* ... */ };
    ///     op.inst1: { /* ... */ };
    ///     // ...
    ///     #[cfg(will_suspend)]
    ///     first_call_or_create_inst: {
    ///          // ...
    ///          goto suspend(1);
    ///     };
    ///     // ...
    ///     // There will always be at least one diverging instruction.
    ///     op.stop: {
    ///         goto return(InstructionResult::Stop);
    ///     };
    ///
    ///     #[cfg(will_suspend)]
    ///     suspend(resume_at: u32): {
    ///         ecx.resume_at = resume_at;
    ///         goto return(InstructionResult::CallOrCreate);
    ///     };
    ///
    ///     // All paths lead to here.
    ///     return(ir: InstructionResult): {
    ///         #[cfg(inspect_stack_length)]
    ///         *args.stack_len = stack_len;
    ///         return ir;
    ///     }
    /// }
    /// ```
    #[allow(rustdoc::invalid_rust_codeblocks)] // Syntax highlighting.
    pub(super) fn translate(
        mut bcx: B::Builder<'a>,
        config: FcxConfig,
        builtins: &'a mut Builtins<B>,
        bytecode: &'a Bytecode<'a>,
    ) -> Result<()> {
        // Get common types.
        let isize_type = bcx.type_ptr_sized_int();
        let i8_type = bcx.type_int(8);
        let i64_type = bcx.type_int(64);
        let address_type = bcx.type_int(160);
        let word_type = bcx.type_int(256);

        // Set up entry block.
        let gas_ptr = bcx.fn_param(0);
        let gas_remaining = {
            let offset = bcx.iconst(i64_type, mem::offset_of!(pf::Gas, remaining) as i64);
            let name = "gas.remaining.addr";
            Pointer::new_address(i64_type, bcx.gep(i8_type, gas_ptr, &[offset], name))
        };

        let sp_arg = bcx.fn_param(1);
        let stack = if config.local_stack {
            bcx.new_stack_slot(word_type, "stack.addr")
        } else {
            Pointer::new_address(word_type, sp_arg)
        };

        let stack_len_arg = bcx.fn_param(2);
        // This is initialized later in `post_entry_block`.
        let stack_len = bcx.new_stack_slot(isize_type, "len.addr");

        let env = bcx.fn_param(3);
        let contract = bcx.fn_param(4);
        let ecx = bcx.fn_param(5);

        // Create all instruction entry blocks.
        let unreachable_block = bcx.create_block("unreachable");
        let inst_entries: Vec<_> = bytecode
            .iter_all_insts()
            .map(|(i, data)| {
                if data.is_dead_code() {
                    unreachable_block
                } else {
                    bcx.create_block(&op_block_name_with(i, data, ""))
                }
            })
            .collect();
        assert!(!inst_entries.is_empty(), "translating empty bytecode");

        let dynamic_jump_table = bcx.create_block("dynamic_jump_table");
        let suspend_block = bcx.create_block("suspend");
        let failure_block = bcx.create_block("failure");
        let return_block = bcx.create_block("return");

        let mut fx = FunctionCx {
            config,

            isize_type,
            address_type,
            word_type,
            i8_type,
            stack_len,
            stack,
            gas_remaining,
            env,
            contract,
            ecx,
            len_before: bcx.iconst(isize_type, 0),
            len_offset: 0,
            bcx,

            bytecode,
            inst_entries,
            current_inst: usize::MAX,

            incoming_dynamic_jumps: Vec::new(),
            dynamic_jump_table,

            incoming_failures: Vec::new(),
            failure_block,
            incoming_returns: Vec::new(),
            return_block,

            resume_blocks: Vec::new(),
            suspend_blocks: Vec::new(),
            suspend_block,

            builtins,
        };

        // We store the stack length if requested or necessary due to the bytecode.
        let store_stack_length = config.inspect_stack_length || bytecode.will_suspend();

        // Add debug assertions for the parameters.
        if config.debug_assertions {
            fx.pointer_panic_with_bool(
                config.gas_metering,
                gas_ptr,
                "gas pointer",
                "gas metering is enabled",
            );
            fx.pointer_panic_with_bool(
                !config.local_stack,
                sp_arg,
                "stack pointer",
                "local stack is disabled",
            );
            fx.pointer_panic_with_bool(
                store_stack_length,
                stack_len_arg,
                "stack length pointer",
                if config.inspect_stack_length {
                    "stack length inspection is enabled"
                } else {
                    "bytecode suspends execution"
                },
            );
            fx.pointer_panic_with_bool(true, env, "env pointer", "");
            fx.pointer_panic_with_bool(true, contract, "contract pointer", "");
            fx.pointer_panic_with_bool(true, ecx, "EVM context pointer", "");
        }

        // The bytecode is guaranteed to have at least one instruction.
        let first_inst_block = fx.inst_entries[0];
        let current_block = fx.current_block();
        let post_entry_block = fx.bcx.create_block_after(current_block, "entry.post");
        let resume_block = fx.bcx.create_block_after(post_entry_block, "resume");
        fx.bcx.br(post_entry_block);
        // Important: set the first resume target to be the start of the instructions.
        if fx.bytecode.will_suspend() {
            fx.add_resume_at(first_inst_block);
        }

        // Translate individual instructions into their respective blocks.
        for (inst, _) in bytecode.iter_insts() {
            fx.translate_inst(inst)?;
        }

        // Finalize the dynamic jump table.
        fx.bcx.switch_to_block(unreachable_block);
        fx.bcx.unreachable();
        let i32_type = fx.bcx.type_int(32);
        if bytecode.has_dynamic_jumps() {
            fx.bcx.switch_to_block(fx.dynamic_jump_table);
            // TODO: Manually reduce to i32?
            let jumpdests =
                fx.bytecode.iter_insts().filter(|(_, data)| data.opcode == op::JUMPDEST);
            // let max_pc =
            //     jumpdests.clone().map(|(_, data)| data.pc).next_back().expect("no jumpdests");
            let targets = jumpdests
                .map(|(inst, data)| (data.pc as u64, fx.inst_entries[inst]))
                .collect::<Vec<_>>();
            let index = fx.bcx.phi(fx.word_type, &fx.incoming_dynamic_jumps);
            // let target =
            //     fx.bcx.create_block_after(fx.dynamic_jump_table, "dynamic_jump_table.contd");
            // let overflow = fx.bcx.icmp_imm(IntCC::UnsignedGreaterThan, index, max_pc as i64);
            // fx.bcx.brif(overflow, default, target);

            // fx.bcx.switch_to_block(target);
            // let index = fx.bcx.ireduce(i32_type, index);
            fx.add_invalid_jump();
            fx.bcx.switch(index, return_block, &targets, true);
        } else {
            // No dynamic jumps.
            debug_assert!(fx.incoming_dynamic_jumps.is_empty());
            fx.bcx.switch_to_block(fx.dynamic_jump_table);
            fx.bcx.unreachable();
        }

        // Finalize the suspend and resume blocks. Must come before the return block.
        // Also here is where the stack length is initialized.
        let load_len_at_start = |fx: &mut Self| {
            // Loaded from args only for the config.
            if config.inspect_stack_length {
                let stack_len = fx.bcx.load(fx.isize_type, stack_len_arg, "stack_len");
                fx.stack_len.store(&mut fx.bcx, stack_len);
            } else {
                fx.stack_len.store_imm(&mut fx.bcx, 0);
            }
        };
        if bytecode.will_suspend() {
            let get_ecx_resume_at = |fx: &mut Self| {
                let offset =
                    fx.bcx.iconst(fx.isize_type, mem::offset_of!(EvmContext<'_>, resume_at) as i64);
                let name = "ecx.resume_at.addr";
                fx.bcx.gep(fx.i8_type, fx.ecx, &[offset], name)
            };

            // Resume block: load the `resume_at` value and switch to the corresponding block.
            // Invalid values are treated as unreachable.
            {
                let default = fx.bcx.create_block_after(resume_block, "resume_invalid");
                fx.bcx.switch_to_block(default);
                fx.call_panic("invalid `resume_at` value");

                // Special-case the zero block to load 0 into the length if possible.
                let resume_is_zero_block =
                    fx.bcx.create_block_after(resume_block, "resume_is_zero");

                fx.bcx.switch_to_block(post_entry_block);
                let resume_at = get_ecx_resume_at(&mut fx);
                let resume_at = fx.bcx.load(i32_type, resume_at, "resume_at");
                let is_resume_zero = fx.bcx.icmp_imm(IntCC::Equal, resume_at, 0);
                fx.bcx.brif(is_resume_zero, resume_is_zero_block, resume_block);

                fx.bcx.switch_to_block(resume_is_zero_block);
                load_len_at_start(&mut fx);
                fx.bcx.br(first_inst_block);

                // Dispatch to the resume block.
                fx.bcx.switch_to_block(resume_block);
                let stack_len = fx.bcx.load(fx.isize_type, stack_len_arg, "stack_len");
                fx.stack_len.store(&mut fx.bcx, stack_len);
                let targets = &fx.resume_blocks[1..]; // Zero case is handled above.
                fx.bcx.switch(resume_at, default, targets, true);
            }

            // Suspend block: store the `resume_at` value and return `CallOrCreate`.
            {
                fx.bcx.switch_to_block(fx.suspend_block);
                let resume_value = fx.bcx.phi(i32_type, &fx.suspend_blocks);
                let resume_at = get_ecx_resume_at(&mut fx);
                fx.bcx.store(resume_value, resume_at);

                fx.build_return_imm(InstructionResult::CallOrCreate);
            }
        } else {
            debug_assert!(fx.resume_blocks.is_empty());
            debug_assert!(fx.suspend_blocks.is_empty());

            fx.bcx.switch_to_block(post_entry_block);
            load_len_at_start(&mut fx);
            fx.bcx.br(first_inst_block);

            fx.bcx.switch_to_block(resume_block);
            fx.bcx.unreachable();
            fx.bcx.switch_to_block(fx.suspend_block);
            fx.bcx.unreachable();
        }

        // Finalize the failure block.
        fx.bcx.switch_to_block(fx.failure_block);
        let failure_value = fx.bcx.phi(fx.i8_type, &fx.incoming_failures);
        fx.bcx.set_current_block_cold();
        fx.build_return(failure_value);

        // Finalize the return block.
        fx.bcx.switch_to_block(fx.return_block);
        let return_value = fx.bcx.phi(fx.i8_type, &fx.incoming_returns);
        if store_stack_length {
            let len = fx.stack_len.load(&mut fx.bcx, "stack_len");
            fx.bcx.store(len, stack_len_arg);
        }
        fx.bcx.ret(&[return_value]);

        fx.bcx.seal_all_blocks();

        Ok(())
    }

    fn translate_inst(&mut self, inst: Inst) -> Result<()> {
        self.current_inst = inst;
        let data = self.bytecode.inst(inst);
        let entry_block = self.inst_entries[inst];
        self.bcx.switch_to_block(entry_block);

        let opcode = data.opcode;

        let branch_to_next_opcode = |this: &mut Self| {
            debug_assert!(
                !this.bytecode.is_instr_diverging(inst),
                "attempted to branch to next instruction in a diverging instruction: {data:?}",
            );
            if let Some(next) = this.inst_entries.get(inst + 1) {
                this.bcx.br(*next);
            }
        };
        // Currently a noop.
        // let epilogue = |this: &mut Self| {
        //     this.bcx.seal_block(entry_block);
        // };

        /// Makes sure to run cleanup code and return.
        /// Use `no_branch` to skip the branch to the next opcode.
        /// Use `build` to build the return instruction and skip the branch.
        macro_rules! goto_return {
            (no_branch $($comment:expr)?) => {
                $(
                    if self.config.comments {
                        self.add_comment($comment);
                    }
                )?
                // epilogue(self);
                return Ok(());
            };
            (build $ret:expr) => {{
                self.build_return_imm($ret);
                goto_return!(no_branch);
            }};
            (fail $ret:expr) => {{
                self.build_fail_imm($ret);
                goto_return!(no_branch);
            }};
            ($($comment:expr)?) => {
                branch_to_next_opcode(self);
                goto_return!(no_branch $($comment)?);
            };
        }

        // Assert that we already skipped the block.
        debug_assert!(!data.flags.contains(InstFlags::DEAD_CODE));

        #[cfg(test)]
        if opcode == crate::TEST_SUSPEND {
            self.suspend();
            goto_return!(no_branch);
        }

        // Disabled instructions don't pay gas.
        if data.flags.contains(InstFlags::DISABLED) {
            goto_return!(fail InstructionResult::NotActivated);
        }
        if data.flags.contains(InstFlags::UNKNOWN) {
            goto_return!(fail InstructionResult::OpcodeNotFound);
        }

        // Pay static gas for the current section.
        self.gas_cost_imm(data.section.gas_cost as u64);

        if data.flags.contains(InstFlags::SKIP_LOGIC) {
            goto_return!("skipped");
        }

        // Reset the stack length offset for this instruction.
        self.len_offset = 0;
        self.len_before = self.stack_len.load(&mut self.bcx, "stack_len");

        // Check stack length for the current section.
        if self.config.stack_bound_checks {
            let inp = data.section.inputs;
            let diff = data.section.max_growth as i64;

            if diff > revm_jit_context::EvmStack::CAPACITY as i64 {
                goto_return!(fail InstructionResult::StackOverflow);
            }

            let underflow = |this: &mut Self| {
                debug_assert!(inp > 0);
                this.bcx.icmp_imm(IntCC::UnsignedLessThan, this.len_before, inp as i64)
            };
            let overflow = |this: &mut Self| {
                debug_assert!(diff > 0 && diff <= STACK_CAP as i64);
                this.bcx.icmp_imm(
                    IntCC::UnsignedGreaterThan,
                    this.len_before,
                    STACK_CAP as i64 - diff,
                )
            };

            let may_underflow = inp > 0;
            let may_overflow = diff > 0;
            if may_underflow && may_overflow {
                let underflow = underflow(self);
                let overflow = overflow(self);
                let cond = self.bcx.bitor(underflow, overflow);
                let ret = {
                    let under =
                        self.bcx.iconst(self.i8_type, InstructionResult::StackUnderflow as i64);
                    let over =
                        self.bcx.iconst(self.i8_type, InstructionResult::StackOverflow as i64);
                    self.bcx.select(underflow, under, over)
                };
                let target = self.build_check_inner(true, cond, ret);
                self.bcx.switch_to_block(target);
            } else if may_underflow {
                let cond = underflow(self);
                self.build_check(cond, InstructionResult::StackUnderflow);
            } else if may_overflow {
                let cond = overflow(self);
                self.build_check(cond, InstructionResult::StackOverflow);
            }
        }

        // Update the stack length for this instruction.
        {
            let (inp, out) = data.stack_io();
            let diff = out as i64 - inp as i64;
            if diff != 0 {
                let mut diff = diff;
                // HACK: For now all opcodes that suspend (minus the test one, which does not reach
                // here) return exactly one value. This value is pushed onto the stack by the
                // caller, so we don't account for it here.
                if data.will_suspend() {
                    diff -= 1;
                }
                let len_changed = self.bcx.iadd_imm(self.len_before, diff);
                self.stack_len.store(&mut self.bcx, len_changed);
            }
        }

        // Macro utils.
        macro_rules! unop {
            ($op:ident) => {{
                let mut a = self.pop();
                a = self.bcx.$op(a);
                self.push(a);
            }};
        }

        macro_rules! binop {
            ($op:ident) => {{
                let [a, b] = self.popn();
                let r = self.bcx.$op(a, b);
                self.push(r);
            }};
            (@shift $op:ident, | $value:ident, $shift:ident | $default:expr) => {{
                let [$shift, $value] = self.popn();
                let r = self.bcx.$op($value, $shift);
                let overflow = self.bcx.icmp_imm(IntCC::UnsignedGreaterThan, $shift, 255);
                let default = $default;
                let r = self.bcx.select(overflow, default, r);
                self.push(r);
            }};
            (@if_not_zero $op:ident) => {{
                // TODO: `select` might not have the same semantics in all backends.
                let [a, b] = self.popn();
                let b_is_zero = self.bcx.icmp_imm(IntCC::Equal, b, 0);
                let zero = self.bcx.iconst_256(U256::ZERO);
                let op_result = self.bcx.$op(a, b);
                let r = self.bcx.select(b_is_zero, zero, op_result);
                self.push(r);
            }};
        }

        macro_rules! field {
            // Gets the pointer to a field.
            ($field:ident; @get $($paths:path),*; $($spec:tt).*) => {
                self.get_field(self.$field, 0 $(+ mem::offset_of!($paths, $spec))*, stringify!($field.$($spec).*.addr))
            };
            // Gets and loads the pointer to a field.
            // The value is loaded as a native-endian 256-bit integer.
            // `@[endian]` is the endianness of the value. If native, omit it.
            ($field:ident; @load $(@[endian = $endian:tt])? $ty:expr, $($paths:path),*; $($spec:tt).*) => {{
                let ptr = field!($field; @get $($paths),*; $($spec).*);
                #[allow(unused_mut)]
                let mut value = self.bcx.load($ty, ptr, stringify!($field.$($spec).*));
                $(
                    if !cfg!(target_endian = $endian) {
                        value = self.bcx.bswap(value);
                    }
                )?
                value
            }};
            // Gets, loads, extends (if necessary), and pushes the value of a field to the stack.
            // `@[endian]` is the endianness of the value. If native, omit it.
            ($field:ident; @push $(@[endian = $endian:tt])? $ty:expr, $($rest:tt)*) => {{
                let mut value = field!($field; @load $(@[endian = $endian])? $ty, $($rest)*);
                if self.bcx.type_bit_width($ty) < 256 {
                    value = self.bcx.zext(self.word_type, value);
                }
                self.push(value);
            }};
        }
        macro_rules! env_field {
            ($($tt:tt)*) => { field!(env; $($tt)*) };
        }
        macro_rules! contract_field {
            ($($tt:tt)*) => { field!(contract; $($tt)*) };
        }

        match data.opcode {
            op::STOP => goto_return!(build InstructionResult::Stop),

            op::ADD => binop!(iadd),
            op::MUL => binop!(imul),
            op::SUB => binop!(isub),
            op::DIV => binop!(@if_not_zero udiv),
            op::SDIV => {
                let [a, b] = self.popn();
                let b_is_zero = self.bcx.icmp_imm(IntCC::Equal, b, 0);
                let r = self.bcx.lazy_select(
                    b_is_zero,
                    self.word_type,
                    |bcx| bcx.iconst_256(U256::ZERO),
                    |bcx| {
                        let min = bcx.iconst_256(I256_MIN);
                        let is_weird_sdiv_edge_case = {
                            let a_is_min = bcx.icmp(IntCC::Equal, a, min);
                            let b_is_neg1 = bcx.icmp_imm(IntCC::Equal, b, -1);
                            bcx.bitand(a_is_min, b_is_neg1)
                        };
                        let sdiv_result = bcx.sdiv(a, b);
                        bcx.select(is_weird_sdiv_edge_case, min, sdiv_result)
                    },
                );
                self.push(r);
            }
            op::MOD => binop!(@if_not_zero urem),
            op::SMOD => binop!(@if_not_zero srem),
            op::ADDMOD => {
                let sp = self.sp_after_inputs();
                let _ = self.builtin(Builtin::AddMod, &[sp]);
            }
            op::MULMOD => {
                let sp = self.sp_after_inputs();
                let _ = self.builtin(Builtin::MulMod, &[sp]);
            }
            op::EXP => {
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.builtin_ir(Builtin::Exp, &[self.ecx, sp, spec_id]);
            }
            op::SIGNEXTEND => {
                // From the yellow paper:
                /*
                let [ext, x] = stack.pop();
                let t = 256 - 8 * (ext + 1);
                let mut result = x;
                result[..t] = [x[t]; t]; // Index by bits.
                */

                let [ext, x] = self.popn();
                // For 31 we also don't need to do anything.
                let might_do_something = self.bcx.icmp_imm(IntCC::UnsignedLessThan, ext, 31);
                let r = self.bcx.lazy_select(
                    might_do_something,
                    self.word_type,
                    |bcx| {
                        // Adapted from revm: https://github.com/bluealloy/revm/blob/fda371f73aba2c30a83c639608be78145fd1123b/crates/interpreter/src/instructions/arithmetic.rs#L89
                        // let bit_index = 8 * ext + 7;
                        // let bit = (x >> bit_index) & 1 != 0;
                        // let mask = (1 << bit_index) - 1;
                        // let r = if bit { x | !mask } else { *x & mask };

                        // let bit_index = 8 * ext + 7;
                        let bit_index = bcx.imul_imm(ext, 8);
                        let bit_index = bcx.iadd_imm(bit_index, 7);

                        // let bit = (x >> bit_index) & 1 != 0;
                        let one = bcx.iconst_256(U256::from(1));
                        let bit = bcx.ushr(x, bit_index);
                        let bit = bcx.bitand(bit, one);
                        let bit = bcx.icmp_imm(IntCC::NotEqual, bit, 0);

                        // let mask = (1 << bit_index) - 1;
                        let mask = bcx.ishl(one, bit_index);
                        let mask = bcx.isub_imm(mask, 1);

                        // let r = if bit { x | !mask } else { *x & mask };
                        let not_mask = bcx.bitnot(mask);
                        let sext = bcx.bitor(x, not_mask);
                        let zext = bcx.bitand(x, mask);
                        bcx.select(bit, sext, zext)
                    },
                    |_bcx| x,
                );
                self.push(r);
            }

            op::LT | op::GT | op::SLT | op::SGT | op::EQ => {
                let cond = match opcode {
                    op::LT => IntCC::UnsignedLessThan,
                    op::GT => IntCC::UnsignedGreaterThan,
                    op::SLT => IntCC::SignedLessThan,
                    op::SGT => IntCC::SignedGreaterThan,
                    op::EQ => IntCC::Equal,
                    _ => unreachable!(),
                };

                let [a, b] = self.popn();
                let r = self.bcx.icmp(cond, a, b);
                let r = self.bcx.zext(self.word_type, r);
                self.push(r);
            }
            op::ISZERO => {
                let a = self.pop();
                let r = self.bcx.icmp_imm(IntCC::Equal, a, 0);
                let r = self.bcx.zext(self.word_type, r);
                self.push(r);
            }
            op::AND => binop!(bitand),
            op::OR => binop!(bitor),
            op::XOR => binop!(bitxor),
            op::NOT => unop!(bitnot),
            op::BYTE => {
                let [index, value] = self.popn();
                let cond = self.bcx.icmp_imm(IntCC::UnsignedLessThan, index, 32);
                let byte = {
                    // (value >> (31 - index) * 8) & 0xFF
                    let thirty_one = self.bcx.iconst_256(U256::from(31));
                    let shift = self.bcx.isub(thirty_one, index);
                    let shift = self.bcx.imul_imm(shift, 8);
                    let shifted = self.bcx.ushr(value, shift);
                    let mask = self.bcx.iconst_256(U256::from(0xFF));
                    self.bcx.bitand(shifted, mask)
                };
                let zero = self.bcx.iconst_256(U256::ZERO);
                let r = self.bcx.select(cond, byte, zero);
                self.push(r);
            }
            op::SHL => binop!(@shift ishl, |value, shift| self.bcx.iconst_256(U256::ZERO)),
            op::SHR => binop!(@shift ushr, |value, shift| self.bcx.iconst_256(U256::ZERO)),
            op::SAR => binop!(@shift sshr, |value, shift| {
                let is_negative = self.bcx.icmp_imm(IntCC::SignedLessThan, value, 0);
                let max = self.bcx.iconst_256(U256::MAX);
                let zero = self.bcx.iconst_256(U256::ZERO);
                self.bcx.select(is_negative, max, zero)
            }),

            op::KECCAK256 => {
                let sp = self.sp_after_inputs();
                self.builtin_ir(Builtin::Keccak256, &[self.ecx, sp]);
            }

            op::ADDRESS => {
                contract_field!(@push @[endian = "big"] self.address_type, Contract; target_address)
            }
            op::BALANCE => {
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.builtin_ir(Builtin::Balance, &[self.ecx, sp, spec_id]);
            }
            op::ORIGIN => {
                env_field!(@push @[endian = "big"] self.address_type, Env, TxEnv; tx.caller)
            }
            op::CALLER => {
                contract_field!(@push @[endian = "big"] self.address_type, Contract; caller)
            }
            op::CALLVALUE => {
                contract_field!(@push @[endian = "little"] self.word_type, Contract; call_value)
            }
            op::CALLDATALOAD => {
                let index = self.pop();
                let len_ptr = contract_field!(@get Contract, pf::Bytes; input.len);
                let len = self.bcx.load(self.isize_type, len_ptr, "input.len");
                let len = self.bcx.zext(self.word_type, len);
                let in_bounds = self.bcx.icmp(IntCC::UnsignedLessThan, index, len);

                let ptr = contract_field!(@get Contract, pf::Bytes; input.ptr);
                let zero = self.bcx.iconst_256(U256::ZERO);
                let isize_type = self.isize_type;
                let i8_type = self.i8_type;
                let word_type = self.word_type;
                let value = self.bcx.lazy_select(
                    in_bounds,
                    self.word_type,
                    |bcx| {
                        let ptr = bcx.load(bcx.type_ptr(), ptr, "contract.input.ptr");
                        let index = bcx.ireduce(isize_type, index);
                        let calldata = bcx.gep(i8_type, ptr, &[index], "calldata.addr");

                        // `32.min(contract.input.len() - index)`
                        let slice_len = {
                            let len = bcx.ireduce(isize_type, len);
                            let diff = bcx.isub(len, index);
                            let max = bcx.iconst(isize_type, 32);
                            bcx.umin(diff, max)
                        };

                        let tmp = bcx.new_stack_slot(word_type, "calldata.addr");
                        tmp.store(bcx, zero);
                        let tmp_addr = tmp.addr(bcx);
                        bcx.memcpy(tmp_addr, calldata, slice_len);
                        let mut value = tmp.load(bcx, "calldata.i256");
                        if cfg!(target_endian = "little") {
                            value = bcx.bswap(value);
                        }
                        value
                    },
                    |_| zero,
                );
                self.push(value);
            }
            op::CALLDATASIZE => {
                contract_field!(@push self.isize_type, Contract, pf::Bytes; input.len)
            }
            op::CALLDATACOPY => {
                let sp = self.sp_after_inputs();
                self.builtin_ir(Builtin::CallDataCopy, &[self.ecx, sp]);
            }
            op::CODESIZE => {
                let size = self.builtin(Builtin::CodeSize, &[self.ecx]).unwrap();
                let size = self.bcx.zext(self.word_type, size);
                self.push(size);
            }
            op::CODECOPY => {
                let sp = self.sp_after_inputs();
                self.builtin_ir(Builtin::CodeCopy, &[self.ecx, sp]);
            }

            op::GASPRICE => {
                env_field!(@push @[endian = "little"] self.word_type, Env, TxEnv; tx.gas_price)
            }
            op::EXTCODESIZE => {
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.builtin_ir(Builtin::ExtCodeSize, &[self.ecx, sp, spec_id]);
            }
            op::EXTCODECOPY => {
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.builtin_ir(Builtin::ExtCodeCopy, &[self.ecx, sp, spec_id]);
            }
            op::RETURNDATASIZE => {
                field!(ecx; @push self.isize_type, EvmContext<'_>, pf::Slice; return_data.len)
            }
            op::RETURNDATACOPY => {
                let sp = self.sp_after_inputs();
                self.builtin_ir(Builtin::ReturnDataCopy, &[self.ecx, sp]);
            }
            op::EXTCODEHASH => {
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.builtin_ir(Builtin::ExtCodeHash, &[self.ecx, sp, spec_id]);
            }
            op::BLOCKHASH => {
                let sp = self.sp_after_inputs();
                self.builtin_ir(Builtin::BlockHash, &[self.ecx, sp]);
            }
            op::COINBASE => {
                env_field!(@push @[endian = "big"] self.address_type, Env, BlockEnv; block.coinbase)
            }
            op::TIMESTAMP => {
                env_field!(@push @[endian = "little"] self.word_type, Env, BlockEnv; block.timestamp)
            }
            op::NUMBER => {
                env_field!(@push @[endian = "little"] self.word_type, Env, BlockEnv; block.number)
            }
            op::DIFFICULTY => {
                let slot = self.sp_at_top();
                let spec_id = self.const_spec_id();
                let _ = self.builtin(Builtin::Difficulty, &[self.ecx, slot, spec_id]);
            }
            op::GASLIMIT => {
                env_field!(@push @[endian = "little"] self.word_type, Env, BlockEnv; block.gas_limit)
            }
            op::CHAINID => env_field!(@push self.bcx.type_int(64), Env, CfgEnv; cfg.chain_id),
            op::SELFBALANCE => {
                let slot = self.sp_at_top();
                self.builtin_ir(Builtin::SelfBalance, &[self.ecx, slot]);
            }
            op::BASEFEE => {
                env_field!(@push @[endian = "little"] self.word_type, Env, BlockEnv; block.basefee)
            }
            op::BLOBHASH => {
                let sp = self.sp_after_inputs();
                let _ = self.builtin(Builtin::BlobHash, &[self.ecx, sp]);
            }
            op::BLOBBASEFEE => {
                let len = self.len_before();
                let slot = self.sp_at(len);
                let _ = self.builtin(Builtin::BlobBaseFee, &[self.ecx, slot]);
            }

            op::POP => { /* Already handled in stack_io */ }
            op::MLOAD => {
                let sp = self.sp_after_inputs();
                self.builtin_ir(Builtin::Mload, &[self.ecx, sp]);
            }
            op::MSTORE => {
                let sp = self.sp_after_inputs();
                self.builtin_ir(Builtin::Mstore, &[self.ecx, sp]);
            }
            op::MSTORE8 => {
                let sp = self.sp_after_inputs();
                self.builtin_ir(Builtin::Mstore8, &[self.ecx, sp]);
            }
            op::SLOAD => {
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.builtin_ir(Builtin::Sload, &[self.ecx, sp, spec_id]);
            }
            op::SSTORE => {
                self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.builtin_ir(Builtin::Sstore, &[self.ecx, sp, spec_id]);
            }
            op::JUMP | op::JUMPI => {
                let is_invalid = data.flags.contains(InstFlags::INVALID_JUMP);
                if is_invalid && opcode == op::JUMP {
                    // NOTE: We can't early return for `JUMPI` since the jump target is evaluated
                    // lazily.
                    self.build_fail_imm(InstructionResult::InvalidJump);
                } else {
                    let target = if is_invalid {
                        debug_assert_eq!(*data, op::JUMPI);
                        // The jump target is invalid, but we still need to account for the stack.
                        self.len_offset -= 1;
                        self.return_block
                    } else if data.flags.contains(InstFlags::STATIC_JUMP) {
                        let target_inst = data.data as usize;
                        debug_assert_eq!(
                            *self.bytecode.inst(target_inst),
                            op::JUMPDEST,
                            "jumping to non-JUMPDEST; target_inst={target_inst}",
                        );
                        self.inst_entries[target_inst]
                    } else {
                        // Dynamic jump.
                        debug_assert!(self.bytecode.has_dynamic_jumps());
                        let target = self.pop();
                        self.incoming_dynamic_jumps
                            .push((target, self.bcx.current_block().unwrap()));
                        self.dynamic_jump_table
                    };

                    if opcode == op::JUMPI {
                        let cond_word = self.pop();
                        let cond = self.bcx.icmp_imm(IntCC::NotEqual, cond_word, 0);
                        let next = self.inst_entries[inst + 1];
                        if target == self.return_block {
                            self.add_invalid_jump();
                        }
                        self.bcx.brif(cond, target, next);
                    } else {
                        self.bcx.br(target);
                    }
                    self.inst_entries[inst] = self.bcx.current_block().unwrap();
                }

                goto_return!(no_branch);
            }
            op::PC => {
                let pc = self.bcx.iconst_256(U256::from(data.pc));
                self.push(pc);
            }
            op::MSIZE => {
                let msize = self.builtin(Builtin::Msize, &[self.ecx]).unwrap();
                let msize = self.bcx.zext(self.word_type, msize);
                self.push(msize);
            }
            op::GAS => {
                let remaining = self.load_gas_remaining();
                let remaining = self.bcx.zext(self.word_type, remaining);
                self.push(remaining);
            }
            op::JUMPDEST => {
                self.bcx.nop();
            }
            op::TLOAD => {
                let sp = self.sp_after_inputs();
                let _ = self.builtin(Builtin::Tload, &[self.ecx, sp]);
            }
            op::TSTORE => {
                self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
                let sp = self.sp_after_inputs();
                let _ = self.builtin(Builtin::Tstore, &[self.ecx, sp]);
            }
            op::MCOPY => {
                let sp = self.sp_after_inputs();
                self.builtin_ir(Builtin::Mcopy, &[self.ecx, sp]);
            }

            op::PUSH0 => {
                let value = self.bcx.iconst_256(U256::ZERO);
                self.push(value);
            }
            op::PUSH1..=op::PUSH32 => {
                // NOTE: This can be None if the bytecode is invalid.
                let imm = self.bytecode.get_imm_of(data);
                let value = imm.map(U256::from_be_slice).unwrap_or_default();
                let value = self.bcx.iconst_256(value);
                self.push(value);
            }

            op::DUP1..=op::DUP16 => self.dup(opcode - op::DUP1 + 1),

            op::SWAP1..=op::SWAP16 => self.swap(opcode - op::SWAP1 + 1),

            op::LOG0..=op::LOG4 => {
                self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
                let n = opcode - op::LOG0;
                let sp = self.sp_after_inputs();
                let n = self.bcx.iconst(self.i8_type, n as i64);
                self.builtin_ir(Builtin::Log, &[self.ecx, sp, n]);
            }

            op::CREATE => {
                self.create_common(CreateKind::Create);
                goto_return!(no_branch);
            }
            op::CALL => {
                self.call_common(CallKind::Call);
                goto_return!(no_branch);
            }
            op::CALLCODE => {
                self.call_common(CallKind::CallCode);
                goto_return!(no_branch);
            }
            op::RETURN => {
                self.return_common(InstructionResult::Return);
                goto_return!(no_branch);
            }
            op::DELEGATECALL => {
                self.call_common(CallKind::DelegateCall);
                goto_return!(no_branch);
            }
            op::CREATE2 => {
                self.create_common(CreateKind::Create2);
                goto_return!(no_branch);
            }

            op::STATICCALL => {
                self.call_common(CallKind::StaticCall);
                goto_return!(no_branch);
            }

            op::REVERT => {
                self.return_common(InstructionResult::Revert);
                goto_return!(no_branch);
            }
            op::INVALID => goto_return!(fail InstructionResult::InvalidFEOpcode),
            op::SELFDESTRUCT => {
                self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.builtin_ir(Builtin::SelfDestruct, &[self.ecx, sp, spec_id]);
                goto_return!(build InstructionResult::SelfDestruct);
            }

            _ => unreachable!("unimplemented instruction: {data:?}"),
        }

        goto_return!("normal exit");
    }

    /// Pushes a 256-bit value onto the stack.
    fn push(&mut self, value: B::Value) {
        self.pushn(&[value]);
    }

    /// Pushes 256-bit values onto the stack.
    fn pushn(&mut self, values: &[B::Value]) {
        let len_start = self.len_before();
        for &value in values {
            let len = if self.len_offset != 0 {
                self.bcx.iadd_imm(len_start, self.len_offset as i64)
            } else {
                len_start
            };
            self.len_offset += 1;
            let sp = self.sp_at(len);
            self.bcx.store(value, sp);
        }
    }

    /// Removes the topmost element from the stack and returns it.
    fn pop(&mut self) -> B::Value {
        self.popn::<1>()[0]
    }

    /// Removes the topmost `N` elements from the stack and returns them.
    ///
    /// If `load` is `false`, returns just the pointers.
    fn popn<const N: usize>(&mut self) -> [B::Value; N] {
        debug_assert_ne!(N, 0);

        let len_start = self.len_before();
        std::array::from_fn(|i| {
            self.len_offset -= 1;
            let len = if self.len_offset != 0 {
                self.bcx.iadd_imm(len_start, self.len_offset as i64)
            } else {
                len_start
            };
            let sp = self.sp_at(len);
            let name = b'a' + i as u8;
            self.load_word(sp, std::str::from_utf8(&[name]).unwrap())
        })
    }

    /// Returns an error if the current context is a static call.
    fn fail_if_staticcall(&mut self, ret: InstructionResult) {
        let ptr =
            self.get_field(self.ecx, mem::offset_of!(EvmContext<'_>, is_static), "ecx.is_static");
        let bool = self.bcx.type_int(1);
        let is_static = self.bcx.load(bool, ptr, "is_static");
        self.build_check(is_static, ret)
    }

    /// Duplicates the `n`th value from the top of the stack.
    /// `n` cannot be `0`.
    fn dup(&mut self, n: u8) {
        debug_assert_ne!(n, 0);
        let len = self.len_before();
        let sp = self.sp_from_top(len, n as usize);
        let value = self.load_word(sp, &format!("dup{n}"));
        self.push(value);
    }

    /// Swaps the topmost value with the `n`th value from the top.
    /// `n` cannot be `0`.
    fn swap(&mut self, n: u8) {
        debug_assert_ne!(n, 0);
        let len = self.len_before();
        // Load a.
        let a_sp = self.sp_from_top(len, n as usize + 1);
        let a = self.load_word(a_sp, "swap.a");
        // Load b.
        let b_sp = self.sp_from_top(len, 1);
        let b = self.load_word(b_sp, "swap.top");
        // Store.
        self.bcx.store(a, b_sp);
        self.bcx.store(b, a_sp);
    }

    /// `RETURN` or `REVERT` instruction.
    fn return_common(&mut self, ir: InstructionResult) {
        let sp = self.sp_after_inputs();
        let ir_const = self.bcx.iconst(self.i8_type, ir as i64);
        self.builtin_ir(Builtin::DoReturn, &[self.ecx, sp, ir_const]);
        self.build_return_imm(ir);
    }

    /// `CREATE` or `CREATE2` instruction.
    fn create_common(&mut self, create_kind: CreateKind) {
        self.fail_if_staticcall(InstructionResult::StateChangeDuringStaticCall);
        let sp = self.sp_after_inputs();
        let spec_id = self.const_spec_id();
        let create_kind = self.bcx.iconst(self.i8_type, create_kind as i64);
        self.builtin_ir(Builtin::Create, &[self.ecx, sp, spec_id, create_kind]);
        self.suspend();
    }

    /// `*CALL*` instruction.
    fn call_common(&mut self, call_kind: CallKind) {
        let sp = self.sp_after_inputs();
        let spec_id = self.const_spec_id();
        let call_kind = self.bcx.iconst(self.i8_type, call_kind as i64);
        self.builtin_ir(Builtin::Call, &[self.ecx, sp, spec_id, call_kind]);
        self.suspend();
    }

    /// Suspend execution, storing the resume point in the context.
    fn suspend(&mut self) {
        // Register the next instruction as the resume block.
        let value = self.add_resume_at(self.inst_entries[self.current_inst + 1]);

        // Register the current block as the suspend block.
        let value = self.bcx.iconst(self.bcx.type_int(32), value as i64);
        self.suspend_blocks.push((value, self.bcx.current_block().unwrap()));

        // Branch to the suspend block.
        self.bcx.br(self.suspend_block);
    }

    /// Adds a resume point and returns its index.
    fn add_resume_at(&mut self, block: B::BasicBlock) -> usize {
        let value = self.resume_blocks.len();
        self.resume_blocks.push((value as u64, block));
        value
    }

    /// Loads the word at the given pointer.
    fn load_word(&mut self, ptr: B::Value, name: &str) -> B::Value {
        self.bcx.load(self.word_type, ptr, name)
    }

    /// Gets the stack length before the current instruction.
    fn len_before(&mut self) -> B::Value {
        self.len_before
    }

    /// Returns the spec ID as a value.
    fn const_spec_id(&mut self) -> B::Value {
        self.bcx.iconst(self.i8_type, self.bytecode.spec_id as i64)
    }

    /// Gets the environment field at the given offset.
    fn get_field(&mut self, ptr: B::Value, offset: usize, name: &str) -> B::Value {
        let offset = self.bcx.iconst(self.isize_type, offset as i64);
        self.bcx.gep(self.i8_type, ptr, &[offset], name)
    }

    /// Loads the gas used.
    fn load_gas_remaining(&mut self) -> B::Value {
        self.gas_remaining.load(&mut self.bcx, "gas_remaining")
    }

    /// Stores the gas used.
    fn store_gas_remaining(&mut self, value: B::Value) {
        self.gas_remaining.store(&mut self.bcx, value);
    }

    /// Returns the stack pointer at the top (`&stack[stack.len]`).
    fn sp_at_top(&mut self) -> B::Value {
        let len = self.len_before();
        self.sp_at(len)
    }

    /// Returns the stack pointer after the input has been popped
    /// (`&stack[stack.len - op.input()]`).
    fn sp_after_inputs(&mut self) -> B::Value {
        let mut len = self.len_before();
        let (inputs, _) = self.current_inst().stack_io();
        if inputs > 0 {
            len = self.bcx.isub_imm(len, inputs as i64);
        }
        self.sp_at(len)
    }

    /// Returns the stack pointer at `len` (`&stack[len]`).
    fn sp_at(&mut self, len: B::Value) -> B::Value {
        let ptr = self.stack.addr(&mut self.bcx);
        self.bcx.gep(self.word_type, ptr, &[len], "sp")
    }

    /// Returns the stack pointer at `len` from the top (`&stack[CAPACITY - len]`).
    fn sp_from_top(&mut self, len: B::Value, n: usize) -> B::Value {
        debug_assert_ne!(n, 0);
        let len = self.bcx.isub_imm(len, n as i64);
        self.sp_at(len)
    }

    /// Builds a gas cost deduction for an immediate value.
    fn gas_cost_imm(&mut self, cost: u64) {
        if !self.config.gas_metering || cost == 0 {
            return;
        }
        let value = self.bcx.iconst(self.isize_type, cost as i64);
        self.gas_cost(value);
    }

    /// Builds a gas cost deduction for a value.
    fn gas_cost(&mut self, cost: B::Value) {
        if !self.config.gas_metering {
            return;
        }

        // Modified from `Gas::record_cost`.
        // This can overflow the gas counters, which has to be adjusted for after the call.
        let gas_remaining = self.load_gas_remaining();
        let (res, overflow) = self.bcx.usub_overflow(gas_remaining, cost);
        if self.bytecode.is_small() {
            // Storing the result before the check significantly increases time spent in
            // `llvm::MemoryDependenceResults::getNonLocalPointerDependency`, but it might produce
            // slightly better code.
            self.store_gas_remaining(res);
            self.build_check(overflow, InstructionResult::OutOfGas);
        } else {
            self.build_check(overflow, InstructionResult::OutOfGas);
            self.store_gas_remaining(res);
        }
    }

    /*
    /// Builds a check, failing if the condition is false.
    ///
    /// `if success_cond { ... } else { return ret }`
    fn build_failure_inv(&mut self, success_cond: B::Value, ret: InstructionResult) {
        self.build_failure_imm_inner(false, success_cond, ret);
    }
    */

    /// Builds a check, failing if the condition is true.
    ///
    /// `if failure_cond { return ret } else { ... }`
    fn build_check(&mut self, failure_cond: B::Value, ret: InstructionResult) {
        self.build_check_imm_inner(true, failure_cond, ret);
    }

    fn build_check_imm_inner(&mut self, is_failure: bool, cond: B::Value, ret: InstructionResult) {
        let ret_value = self.bcx.iconst(self.i8_type, ret as i64);
        let target = self.build_check_inner(is_failure, cond, ret_value);
        if self.config.comments {
            self.add_comment(&format!("check {ret:?}"));
        }
        self.bcx.switch_to_block(target);
    }

    #[must_use]
    fn build_check_inner(
        &mut self,
        is_failure: bool,
        cond: B::Value,
        ret: B::Value,
    ) -> B::BasicBlock {
        let current_block = self.current_block();
        let target = self.create_block_after(current_block, "contd");

        self.incoming_returns.push((ret, current_block));
        let return_block = self.return_block;
        let then_block = if is_failure { return_block } else { target };
        let else_block = if is_failure { target } else { return_block };
        self.bcx.brif_cold(cond, then_block, else_block, is_failure);

        target
    }

    /// Builds a branch to the failure block.
    fn build_fail_imm(&mut self, ret: InstructionResult) {
        let ret_value = self.bcx.iconst(self.i8_type, ret as i64);
        self.build_fail(ret_value);
        if self.config.comments {
            self.add_comment(&format!("fail {ret:?}"));
        }
    }

    /// Builds a branch to the failure block.
    fn build_fail(&mut self, ret: B::Value) {
        self.incoming_failures.push((ret, self.bcx.current_block().unwrap()));
        self.bcx.br(self.failure_block);
    }

    /// Builds a branch to the return block.
    fn build_return_imm(&mut self, ret: InstructionResult) {
        let ret_value = self.bcx.iconst(self.i8_type, ret as i64);
        self.build_return(ret_value);
        if self.config.comments {
            self.add_comment(&format!("return {ret:?}"));
        }
    }

    /// Builds a branch to the return block.
    fn build_return(&mut self, ret: B::Value) {
        self.incoming_returns.push((ret, self.bcx.current_block().unwrap()));
        self.bcx.br(self.return_block);
    }

    fn add_invalid_jump(&mut self) {
        self.incoming_returns.push((
            self.bcx.iconst(self.i8_type, InstructionResult::InvalidJump as i64),
            self.bcx.current_block().unwrap(),
        ));
    }

    // Pointer must not be null if `must_be_set` is true.
    fn pointer_panic_with_bool(
        &mut self,
        must_be_set: bool,
        ptr: B::Value,
        name: &str,
        extra: &str,
    ) {
        if !must_be_set {
            return;
        }
        let panic_cond = self.bcx.is_null(ptr);
        let mut msg = format!("revm_jit panic: {name} must not be null");
        if !extra.is_empty() {
            write!(msg, " ({extra})").unwrap();
        }
        self.build_panic_cond(panic_cond, &msg);
    }

    fn build_panic_cond(&mut self, cond: B::Value, msg: &str) {
        let failure = self.create_block_after_current("panic");
        let target = self.create_block_after(failure, "contd");
        self.bcx.brif(cond, failure, target);

        // `panic` is already marked as a cold function call.
        // self.bcx.set_cold_block(failure);
        self.bcx.switch_to_block(failure);
        self.call_panic(msg);

        self.bcx.switch_to_block(target);
    }

    fn call_panic(&mut self, msg: &str) {
        let function = self.builtin_function(Builtin::Panic);
        let ptr = self.bcx.str_const(msg);
        let len = self.bcx.iconst(self.isize_type, msg.len() as i64);
        let _ = self.bcx.call(function, &[ptr, len]);
        self.bcx.unreachable();
    }

    /// Build a call to a builtin that returns an [`InstructionResult`].
    fn builtin_ir(&mut self, builtin: Builtin, args: &[B::Value]) {
        let ret = self.builtin(builtin, args).expect("builtin does not return a value");
        let failure = self.bcx.icmp_imm(IntCC::NotEqual, ret, InstructionResult::Continue as i64);
        let target = self.build_check_inner(true, failure, ret);
        self.bcx.switch_to_block(target);
    }

    /// Build a call to a builtin.
    #[must_use]
    fn builtin(&mut self, builtin: Builtin, args: &[B::Value]) -> Option<B::Value> {
        let function = self.builtin_function(builtin);
        self.bcx.call(function, args)
    }

    fn builtin_function(&mut self, builtin: Builtin) -> B::Function {
        self.builtins.get(builtin, &mut self.bcx)
    }

    /// Adds a comment to the current instruction.
    fn add_comment(&mut self, comment: &str) {
        if comment.is_empty() {
            return;
        }
        self.bcx.add_comment_to_current_inst(comment);
    }

    /// Returns the current instruction.
    fn current_inst(&self) -> &InstData {
        self.bytecode.inst(self.current_inst)
    }

    /// Returns the current block.
    fn current_block(&mut self) -> B::BasicBlock {
        // There always is a block present.
        self.bcx.current_block().expect("no blocks")
    }

    /*
    /// Creates a named block.
    fn create_block(&mut self, name: &str) -> B::BasicBlock {
        let name = self.op_block_name(name);
        self.bcx.create_block(&name)
    }
    */

    /// Creates a named block after the given block.
    fn create_block_after(&mut self, after: B::BasicBlock, name: &str) -> B::BasicBlock {
        let name = self.op_block_name(name);
        self.bcx.create_block_after(after, &name)
    }

    /// Creates a named block after the current block.
    fn create_block_after_current(&mut self, name: &str) -> B::BasicBlock {
        let after = self.current_block();
        self.create_block_after(after, name)
    }

    /// Returns the block name for the current opcode with the given suffix.
    fn op_block_name(&self, name: &str) -> String {
        if self.current_inst == usize::MAX {
            return format!("entry.{name}");
        }
        op_block_name_with(self.current_inst, self.bytecode.inst(self.current_inst), name)
    }
}

// HACK: Need these structs' fields to be public for `offset_of!`.
// `pf == private_fields`.
#[allow(dead_code)]
mod pf {
    use super::*;

    pub(super) struct Bytes {
        pub(super) ptr: *const u8,
        pub(super) len: usize,
        // inlined "trait object"
        data: AtomicPtr<()>,
        vtable: &'static Vtable,
    }
    const _: [(); mem::size_of::<revm_primitives::Bytes>()] = [(); mem::size_of::<Bytes>()];
    struct Vtable {
        /// fn(data, ptr, len)
        clone: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> Bytes,
        /// fn(data, ptr, len)
        ///
        /// takes `Bytes` to value
        to_vec: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> Vec<u8>,
        /// fn(data, ptr, len)
        drop: unsafe fn(&mut AtomicPtr<()>, *const u8, usize),
    }

    #[repr(C)] // See core::ptr::metadata::PtrComponents
    pub(super) struct Slice {
        pub(super) ptr: *const u8,
        pub(super) len: usize,
    }
    const _: [(); mem::size_of::<&'static [u8]>()] = [(); mem::size_of::<Slice>()];

    pub(super) struct Gas {
        /// The initial gas limit. This is constant throughout execution.
        pub(super) limit: u64,
        /// The remaining gas.
        pub(super) remaining: u64,
        /// Refunded gas. This is used only at the end of execution.
        refunded: i64,
    }
    const _: [(); mem::size_of::<revm_interpreter::Gas>()] = [(); mem::size_of::<Gas>()];
}

fn op_block_name_with(op: Inst, data: &InstData, with: &str) -> String {
    let data = data.to_op();
    if with.is_empty() {
        format!("op.{op}.{data}")
    } else {
        format!("op.{op}.{data}.{with}")
    }
}
