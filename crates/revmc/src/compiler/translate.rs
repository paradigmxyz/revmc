//! EVM to IR translation.

use super::default_attrs;
use crate::{Backend, Builder, Bytecode, EvmContext, Inst, InstData, InstFlags, IntCC, Result};
use revm_bytecode::opcode as op;
use revm_interpreter::{InputsImpl, InstructionResult};
use revm_primitives::U256;
use revmc_backend::{Attribute, BackendTypes, FunctionAttributeLocation, Pointer, TypeMethods};
use revmc_builtins::{Builtin, Builtins, CallKind, CreateKind};
use std::{fmt::Write, mem, sync::atomic::AtomicPtr};

const STACK_CAP: usize = 1024;
// const WORD_SIZE: usize = 32;

#[derive(Clone, Copy, Debug)]
pub(super) struct FcxConfig {
    pub(super) comments: bool,
    pub(super) debug_assertions: bool,
    pub(super) frame_pointers: bool,

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
            inspect_stack_length: false,
            stack_bound_checks: true,
            gas_metering: true,
        }
    }
}

/// A list of incoming values for a block. Represents a `phi` node.
type Incoming<B> = Vec<(<B as BackendTypes>::Value, <B as BackendTypes>::BasicBlock)>;

/// A list of `switch` targets.
#[allow(dead_code)]
type SwitchTargets<B> = Vec<(u64, <B as BackendTypes>::BasicBlock)>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResumeKind {
    /// Use `indirectbr`.
    Blocks,
    /// Use a switch over `0..N`.
    Indexes,
}

pub(super) struct FunctionCx<'a, B: Backend> {
    // Configuration.
    config: FcxConfig,

    /// The backend's function builder.
    bcx: B::Builder<'a>,

    // Common types.
    ptr_type: B::Type,
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
    /// The stack argument pointer. Only used when `local_stack` is enabled and the stack needs
    /// to be copied in/out at entry/exit boundaries.
    sp_arg: Option<B::Value>,
    /// The amount of gas remaining. `i64`. See `Gas`.
    gas_remaining: Pointer<B::Builder<'a>>,
    /// The input. Constant throughout the function.
    input: B::Value,
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

    // Basic blocks are `None` when outside of a main function.
    /// `dynamic_jump_table` incoming values.
    incoming_dynamic_jumps: Incoming<B>,
    /// The dynamic jump table block where all dynamic jumps branch to.
    dynamic_jump_table: B::BasicBlock,

    /// `failure_block` incoming values.
    incoming_failures: Incoming<B>,
    /// The block that all failures branch to.
    failure_block: Option<B::BasicBlock>,
    /// `return_block` incoming values.
    incoming_returns: Incoming<B>,
    /// The return block that all return instructions branch to.
    return_block: Option<B::BasicBlock>,

    /// The kind of resume mechanism to use.
    resume_kind: ResumeKind,
    /// `resume_block` switch values.
    resume_blocks: Vec<B::BasicBlock>,
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
    /// // `cfg(may_suspend) = bytecode.may_suspend()`: `true` if it contains a
    /// // `*CALL*` or `CREATE*` instruction.
    /// fn evm_bytecode(args: ...) {
    ///     setup_locals();
    ///
    ///     #[cfg(debug_assertions)]
    ///     if args.<ptr>.is_null() { panic!("...") };
    ///
    ///     load_arguments();
    ///
    ///     #[cfg(may_suspend)]
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
    ///     #[cfg(may_suspend)]
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
    ///     #[cfg(may_suspend)]
    ///     suspend(resume_at: u32): {
    ///         ecx.resume_at = resume_at;
    ///         goto return(InstructionResult::Stop);  // Caller checks next_action
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
        let entry_block = bcx.current_block().unwrap();

        // Get common types.
        let ptr_type = bcx.type_ptr();
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
        // Use a local alloca for the stack to allow the backend to eliminate dead stores to
        // stack slots above `stack_len` at function exit (e.g. `PUSH0 POP`).
        // Disabled when `inspect_stack_length` is set because the caller observes every store.
        let local_stack = !config.inspect_stack_length;
        let stack = if local_stack {
            let stack_type = bcx.type_array(word_type, STACK_CAP as u32);
            bcx.new_stack_slot(stack_type, "stack.addr")
        } else {
            Pointer::new_address(word_type, sp_arg)
        };

        let stack_len_arg = bcx.fn_param(2);
        // This is initialized later in `post_entry_block`.
        let stack_len = bcx.new_stack_slot(isize_type, "len.addr");

        let input = bcx.fn_param(3);
        let ecx = bcx.fn_param(4);

        // Create all instruction entry blocks.
        let unreachable_block = bcx.create_block("unreachable");
        let inst_entries: Vec<_> = bytecode
            .iter_all_insts()
            .map(|(i, data)| {
                if data.is_dead_code() {
                    unreachable_block
                } else {
                    bcx.create_block(&bytecode.op_block_name(i, ""))
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

            ptr_type,
            isize_type,
            address_type,
            word_type,
            i8_type,
            stack_len,
            stack,
            sp_arg: local_stack.then_some(sp_arg),
            gas_remaining,
            input,
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
            failure_block: Some(failure_block),
            incoming_returns: Vec::new(),
            return_block: Some(return_block),

            resume_kind: ResumeKind::Indexes,
            resume_blocks: Vec::new(),
            suspend_blocks: Vec::new(),
            suspend_block,

            builtins,
        };

        // We store the stack length if requested or necessary due to the bytecode.
        let stack_length_observable = config.inspect_stack_length || bytecode.may_suspend();

        // Add debug assertions for the parameters.
        if config.debug_assertions {
            fx.pointer_panic_with_bool(
                config.gas_metering,
                gas_ptr,
                "gas pointer",
                "gas metering is enabled",
            );
            fx.pointer_panic_with_bool(
                !local_stack,
                sp_arg,
                "stack pointer",
                "local stack is disabled",
            );
            fx.pointer_panic_with_bool(
                stack_length_observable,
                stack_len_arg,
                "stack length pointer",
                if config.inspect_stack_length {
                    "stack length inspection is enabled"
                } else {
                    "bytecode suspends execution"
                },
            );
            fx.pointer_panic_with_bool(true, input, "input pointer", "");
            fx.pointer_panic_with_bool(true, ecx, "EVM context pointer", "");
        }

        // The bytecode is guaranteed to have at least one instruction.
        let first_inst_block = fx.inst_entries[0];
        let post_entry_block = fx.bcx.create_block_after(entry_block, "entry.post");
        let resume_block = fx.bcx.create_block_after(post_entry_block, "resume");
        fx.bcx.br(post_entry_block);

        // Translate individual instructions into their respective blocks.
        for (inst, _) in bytecode.iter_insts() {
            fx.translate_inst(inst)?;
        }

        // Finalize the dynamic jump table.
        fx.bcx.switch_to_block(unreachable_block);
        fx.bcx.unreachable();
        if bytecode.has_dynamic_jumps() {
            fx.bcx.switch_to_block(fx.dynamic_jump_table);
            // TODO: Manually reduce to i32?
            let jumpdests = bytecode.iter_insts().filter(|(_, data)| data.opcode == op::JUMPDEST);
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
                fx.copy_stack_from_arg(stack_len);
            } else {
                fx.stack_len.store_imm(&mut fx.bcx, 0);
            }
        };
        let generate_resume = bytecode.may_suspend();
        if generate_resume {
            let get_ecx_resume_at_ptr = |fx: &mut Self| {
                fx.get_field(
                    fx.ecx,
                    mem::offset_of!(EvmContext<'_>, resume_at),
                    "ecx.resume_at.addr",
                )
            };

            let kind = fx.resume_kind;
            let resume_ty = match kind {
                ResumeKind::Blocks => fx.ptr_type,
                ResumeKind::Indexes => fx.isize_type,
            };

            // Resume block: load the `resume_at` value and switch to the corresponding block.
            // Invalid values are treated as unreachable.
            {
                // Special-case the no resume case to load 0 into the length if possible.
                let no_resume_block = fx.bcx.create_block_after(resume_block, "no_resume");

                fx.bcx.switch_to_block(post_entry_block);
                let resume_at = get_ecx_resume_at_ptr(&mut fx);
                let resume_at = fx.bcx.load_aligned(resume_ty, resume_at, 1, "ecx.resume_at");
                let no_resume = match kind {
                    ResumeKind::Blocks => fx.bcx.is_null(resume_at),
                    ResumeKind::Indexes => fx.bcx.icmp_imm(IntCC::Equal, resume_at, 0),
                };
                fx.bcx.brif(no_resume, no_resume_block, resume_block);

                fx.bcx.switch_to_block(no_resume_block);
                load_len_at_start(&mut fx);
                fx.bcx.br(first_inst_block);

                // Dispatch to the resume block.
                fx.bcx.switch_to_block(resume_block);
                let stack_len = fx.bcx.load(fx.isize_type, stack_len_arg, "stack_len");
                fx.stack_len.store(&mut fx.bcx, stack_len);
                fx.copy_stack_from_arg(stack_len);
                match kind {
                    ResumeKind::Blocks => {
                        fx.bcx.br_indirect(resume_at, &fx.resume_blocks);
                    }
                    ResumeKind::Indexes => {
                        let default = fx.bcx.create_block_after(resume_block, "resume_invalid");
                        fx.bcx.switch_to_block(default);
                        fx.call_panic("invalid `resume_at` value");

                        fx.bcx.switch_to_block(resume_block);
                        let targets = fx
                            .resume_blocks
                            .iter()
                            .enumerate()
                            .map(|(i, b)| (i as u64 + 1, *b))
                            .collect::<Vec<_>>();
                        fx.bcx.switch(resume_at, default, &targets, true);
                    }
                }
            }

            // Suspend block: store the `resume_at` value and return `Stop`.
            {
                fx.bcx.switch_to_block(fx.suspend_block);
                let resume_value = fx.bcx.phi(resume_ty, &fx.suspend_blocks);
                let resume_at = get_ecx_resume_at_ptr(&mut fx);
                fx.bcx.store_aligned(resume_value, resume_at, 1);

                // Signal that execution suspended - caller checks next_action for Call/Create
                fx.build_return_imm(InstructionResult::Stop);
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
        fx.bcx.switch_to_block(fx.failure_block.unwrap());
        if !fx.incoming_failures.is_empty() {
            let failure_value = fx.bcx.phi(fx.i8_type, &fx.incoming_failures);
            fx.bcx.set_current_block_cold();
            fx.build_return(failure_value);
        } else {
            fx.bcx.unreachable();
        }

        // Finalize the return block.
        fx.bcx.switch_to_block(fx.return_block.unwrap());
        if !fx.incoming_returns.is_empty() {
            let return_value = fx.bcx.phi(fx.i8_type, &fx.incoming_returns);
            if stack_length_observable {
                fx.copy_stack_to_arg();
                fx.save_stack_len();
            }
            fx.bcx.ret(&[return_value]);
        } else {
            fx.bcx.unreachable();
        }

        fx.bcx.seal_all_blocks();

        Ok(())
    }

    #[instrument(level = "debug", skip_all, fields(inst = %self.bytecode.inst(inst).to_op()))]
    fn translate_inst(&mut self, inst: Inst) -> Result<()> {
        self.current_inst = inst;
        let data = self.bytecode.inst(inst);
        let opcode = data.opcode;
        let entry_block = self.inst_entries[inst];
        self.bcx.switch_to_block(entry_block);

        // self.call_printf(format_printf!("{}\n", self.op_block_name("")), &[]);

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

            if diff > revmc_context::EvmStack::CAPACITY as i64 {
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
                if data.may_suspend() {
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
                // Use align=1 because the pointer comes from a byte-offset GEP and may not
                // satisfy the type's natural alignment.
                #[allow(unused_mut)]
                let mut value = self.bcx.load_aligned($ty, ptr, 1, stringify!($field.$($spec).*));
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
        macro_rules! input_field {
            ($($tt:tt)*) => { field!(input; $($tt)*) };
        }

        match data.opcode {
            op::STOP => goto_return!(build InstructionResult::Stop),

            op::ADD => binop!(iadd),
            op::MUL => binop!(imul),
            op::SUB => binop!(isub),
            op::DIV => {
                let sp = self.sp_after_inputs();
                let _ = self.call_builtin(Builtin::UDiv, &[sp]);
            }
            op::SDIV => {
                let sp = self.sp_after_inputs();
                let _ = self.call_builtin(Builtin::SDiv, &[sp]);
            }
            op::MOD => {
                let sp = self.sp_after_inputs();
                let _ = self.call_builtin(Builtin::URem, &[sp]);
            }
            op::SMOD => {
                let sp = self.sp_after_inputs();
                let _ = self.call_builtin(Builtin::SRem, &[sp]);
            }
            op::ADDMOD => {
                let sp = self.sp_after_inputs();
                let _ = self.call_builtin(Builtin::AddMod, &[sp]);
            }
            op::MULMOD => {
                let sp = self.sp_after_inputs();
                let _ = self.call_builtin(Builtin::MulMod, &[sp]);
            }
            op::EXP => {
                let sp = self.sp_after_inputs();
                self.call_fallible_builtin(Builtin::Exp, &[self.ecx, sp]);
            }
            op::SIGNEXTEND => {
                let [ext, x] = self.popn();
                let r = self.call_signextend(ext, x);
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
                let r = self.call_byte(index, value);
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
            op::CLZ => unop!(clz),

            op::KECCAK256 => {
                let sp = self.sp_after_inputs();
                self.call_fallible_builtin(Builtin::Keccak256, &[self.ecx, sp]);
            }

            op::ADDRESS => {
                input_field!(@push @[endian = "big"] self.address_type, InputsImpl; target_address)
            }
            op::BALANCE => {
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.call_fallible_builtin(Builtin::Balance, &[self.ecx, sp, spec_id]);
            }
            op::ORIGIN => {
                let slot = self.sp_at_top();
                let _ = self.call_builtin(Builtin::Origin, &[self.ecx, slot]);
            }
            op::CALLER => {
                input_field!(@push @[endian = "big"] self.address_type, InputsImpl; caller_address)
            }
            op::CALLVALUE => {
                input_field!(@push @[endian = "little"] self.word_type, InputsImpl; call_value)
            }
            op::CALLDATALOAD => {
                let sp = self.sp_after_inputs();
                let _ = self.call_builtin(Builtin::CallDataLoad, &[self.ecx, sp]);
            }
            op::CALLDATASIZE => {
                let size = self.call_builtin(Builtin::CallDataSize, &[self.ecx]).unwrap();
                let size = self.bcx.zext(self.word_type, size);
                self.push(size);
            }
            op::CALLDATACOPY => {
                let sp = self.sp_after_inputs();
                self.call_fallible_builtin(Builtin::CallDataCopy, &[self.ecx, sp]);
            }
            op::CODESIZE => {
                let len = self.bcx.iconst(self.word_type, self.bytecode.code.len() as i64);
                self.push(len);
            }
            op::CODECOPY => {
                let sp = self.sp_after_inputs();
                self.call_fallible_builtin(Builtin::CodeCopy, &[self.ecx, sp]);
            }

            op::GASPRICE => {
                let sp = self.sp_after_inputs();
                let _ = self.call_builtin(Builtin::GasPrice, &[self.ecx, sp]);
            }
            op::EXTCODESIZE => {
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.call_fallible_builtin(Builtin::ExtCodeSize, &[self.ecx, sp, spec_id]);
            }
            op::EXTCODECOPY => {
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.call_fallible_builtin(Builtin::ExtCodeCopy, &[self.ecx, sp, spec_id]);
            }
            op::RETURNDATASIZE => {
                field!(ecx; @push self.isize_type, EvmContext<'_>, pf::Slice; return_data.len);
            }
            op::RETURNDATACOPY => {
                let sp = self.sp_after_inputs();
                self.call_fallible_builtin(Builtin::ReturnDataCopy, &[self.ecx, sp]);
            }
            op::EXTCODEHASH => {
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.call_fallible_builtin(Builtin::ExtCodeHash, &[self.ecx, sp, spec_id]);
            }
            op::BLOCKHASH => {
                let sp = self.sp_after_inputs();
                self.call_fallible_builtin(Builtin::BlockHash, &[self.ecx, sp]);
            }
            op::COINBASE => {
                let slot = self.sp_at_top();
                let _ = self.call_builtin(Builtin::Coinbase, &[self.ecx, slot]);
            }
            op::TIMESTAMP => {
                let slot = self.sp_at_top();
                let _ = self.call_builtin(Builtin::Timestamp, &[self.ecx, slot]);
            }
            op::NUMBER => {
                let slot = self.sp_at_top();
                let _ = self.call_builtin(Builtin::Number, &[self.ecx, slot]);
            }
            op::DIFFICULTY => {
                let slot = self.sp_at_top();
                let spec_id = self.const_spec_id();
                let _ = self.call_builtin(Builtin::Difficulty, &[self.ecx, slot, spec_id]);
            }
            op::GASLIMIT => {
                let slot = self.sp_at_top();
                let _ = self.call_builtin(Builtin::GasLimit, &[self.ecx, slot]);
            }
            op::CHAINID => {
                let slot = self.sp_at_top();
                let _ = self.call_builtin(Builtin::ChainId, &[self.ecx, slot]);
            }
            op::SELFBALANCE => {
                let slot = self.sp_at_top();
                self.call_fallible_builtin(Builtin::SelfBalance, &[self.ecx, slot]);
            }
            op::BASEFEE => {
                let slot = self.sp_at_top();
                let _ = self.call_builtin(Builtin::Basefee, &[self.ecx, slot]);
            }
            op::BLOBHASH => {
                let sp = self.sp_after_inputs();
                let _ = self.call_builtin(Builtin::BlobHash, &[self.ecx, sp]);
            }
            op::BLOBBASEFEE => {
                let len = self.len_before();
                let slot = self.sp_at(len);
                let _ = self.call_builtin(Builtin::BlobBaseFee, &[self.ecx, slot]);
            }

            op::POP => { /* Already handled in stack_io */ }
            op::MLOAD => {
                let sp = self.sp_after_inputs();
                self.call_fallible_builtin(Builtin::Mload, &[self.ecx, sp]);
            }
            op::MSTORE => {
                let sp = self.sp_after_inputs();
                self.call_fallible_builtin(Builtin::Mstore, &[self.ecx, sp]);
            }
            op::MSTORE8 => {
                let sp = self.sp_after_inputs();
                self.call_fallible_builtin(Builtin::Mstore8, &[self.ecx, sp]);
            }
            op::SLOAD => {
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.call_fallible_builtin(Builtin::Sload, &[self.ecx, sp, spec_id]);
            }
            op::SSTORE => {
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.call_fallible_builtin(Builtin::Sstore, &[self.ecx, sp, spec_id]);
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
                        self.return_block.unwrap()
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
                        if target == self.return_block.unwrap() {
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
                let msize = self.call_builtin(Builtin::Msize, &[self.ecx]).unwrap();
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
                let _ = self.call_builtin(Builtin::Tload, &[self.ecx, sp]);
            }
            op::TSTORE => {
                let sp = self.sp_after_inputs();
                self.call_fallible_builtin(Builtin::Tstore, &[self.ecx, sp]);
            }
            op::MCOPY => {
                let sp = self.sp_after_inputs();
                self.call_fallible_builtin(Builtin::Mcopy, &[self.ecx, sp]);
            }

            op::PUSH0 => {
                let value = self.bcx.iconst_256(U256::ZERO);
                self.push(value);
            }
            op::PUSH1..=op::PUSH32 => {
                // NOTE: This can be None if the bytecode is invalid.
                let imm = self.bytecode.get_imm(data);
                let value = imm.map(U256::from_be_slice).unwrap_or_default();
                let value = self.bcx.iconst_256(value);
                self.push(value);
            }

            op::DUP1..=op::DUP16 => self.dup((opcode - op::DUP1 + 1) as usize),

            op::SWAP1..=op::SWAP16 => self.swap((opcode - op::SWAP1 + 1) as usize),

            op::LOG0..=op::LOG4 => {
                let n = opcode - op::LOG0;
                let sp = self.sp_after_inputs();
                let n = self.bcx.iconst(self.i8_type, n as i64);
                self.call_fallible_builtin(Builtin::Log, &[self.ecx, sp, n]);
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
                let sp = self.sp_after_inputs();
                let spec_id = self.const_spec_id();
                self.call_fallible_builtin(Builtin::SelfDestruct, &[self.ecx, sp, spec_id]);
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

    /// Duplicates the `n`th value from the top of the stack.
    /// `n` cannot be `0`.
    fn dup(&mut self, n: usize) {
        debug_assert_ne!(n, 0);
        let len = self.len_before();
        let sp = self.sp_from_top(len, n);
        let value = self.load_word(sp, &format!("dup{n}"));
        self.push(value);
    }

    /// Swaps the topmost value with the `n`th value from the top.
    /// `n` cannot be `0`.
    fn swap(&mut self, n: usize) {
        self.exchange(0, n);
    }

    /// Exchange two values on the stack.
    /// `n` is the first index, and the second index is calculated as `n + m`.
    /// `m` cannot be `0`.
    fn exchange(&mut self, n: usize, m: usize) {
        debug_assert_ne!(m, 0);
        let len = self.len_before();
        // Load a.
        let a_sp = self.sp_from_top(len, n + 1);
        let a = self.load_word(a_sp, "swap.a");
        // Load b.
        let b_sp = self.sp_from_top(len, n + m + 1);
        let b = self.load_word(b_sp, "swap.b");
        // Store.
        self.bcx.store(a, b_sp);
        self.bcx.store(b, a_sp);
    }

    /// `RETURN` or `REVERT` instruction.
    fn return_common(&mut self, ir: InstructionResult) {
        let sp = self.sp_after_inputs();
        let ir_const = self.bcx.iconst(self.i8_type, ir as i64);
        self.call_fallible_builtin(Builtin::DoReturn, &[self.ecx, sp, ir_const]);
        self.build_return_imm(ir);
    }

    /// Builds a `CREATE` or `CREATE2` instruction.
    fn create_common(&mut self, create_kind: CreateKind) {
        let sp = self.sp_after_inputs();
        let spec_id = self.const_spec_id();
        let create_kind = self.bcx.iconst(self.i8_type, create_kind as i64);
        self.call_fallible_builtin(Builtin::Create, &[self.ecx, sp, spec_id, create_kind]);
        self.suspend();
    }

    /// Builds `*CALL*` instructions.
    fn call_common(&mut self, call_kind: CallKind) {
        let sp = self.sp_after_inputs();
        let spec_id = self.const_spec_id();
        let call_kind = self.bcx.iconst(self.i8_type, call_kind as i64);
        self.call_fallible_builtin(Builtin::Call, &[self.ecx, sp, spec_id, call_kind]);
        self.suspend();
    }

    /// Suspend execution, storing the resume point in the context.
    fn suspend(&mut self) {
        // Register the next instruction as the resume block.
        let idx = self.resume_blocks.len();
        let value = self.add_resume_at(self.inst_entries[self.current_inst + 1]);

        // Register the current block as the suspend block.
        let value = match value {
            Some(value) => value,
            None => self.bcx.iconst(self.isize_type, idx as i64 + 1),
        };
        self.suspend_blocks.push((value, self.bcx.current_block().unwrap()));

        // Branch to the suspend block.
        self.bcx.br(self.suspend_block);
    }

    /// Adds a resume point and returns its index.
    fn add_resume_at(&mut self, block: B::BasicBlock) -> Option<B::Value> {
        let value = self.bcx.block_addr(block);
        if self.resume_blocks.is_empty() {
            self.resume_kind =
                if value.is_some() { ResumeKind::Blocks } else { ResumeKind::Indexes };
        }
        self.resume_blocks.push(block);
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

    /// Gets a field at the given offset.
    fn get_field(&mut self, ptr: B::Value, offset: usize, name: &str) -> B::Value {
        get_field(&mut self.bcx, ptr, offset, name)
    }

    /// Loads the gas used.
    fn load_gas_remaining(&mut self) -> B::Value {
        self.gas_remaining.load(&mut self.bcx, "gas.remaining")
    }

    /// Stores the gas used.
    fn store_gas_remaining(&mut self, value: B::Value) {
        self.gas_remaining.store(&mut self.bcx, value);
    }

    /// Saves the local `stack_len` to `stack_len_arg`.
    fn save_stack_len(&mut self) {
        let len = self.stack_len.load(&mut self.bcx, "stack_len");
        let ptr = self.stack_len_arg();
        self.bcx.store(len, ptr);
    }

    /// Copies the live prefix of the stack from the argument to the local alloca.
    /// `len` is the number of live stack elements.
    fn copy_stack_from_arg(&mut self, len: B::Value) {
        if let Some(src) = self.sp_arg {
            let dst = self.stack.addr(&mut self.bcx);
            let word_size = 32i64;
            let byte_len = self.bcx.imul_imm(len, word_size);
            self.bcx.memcpy(dst, src, byte_len);
        }
    }

    /// Copies the live prefix of the stack from the local alloca to the argument.
    fn copy_stack_to_arg(&mut self) {
        if let Some(dst) = self.sp_arg {
            let len = self.stack_len.load(&mut self.bcx, "stack_len");
            let src = self.stack.addr(&mut self.bcx);
            let word_size = 32i64;
            let byte_len = self.bcx.imul_imm(len, word_size);
            self.bcx.memcpy(dst, src, byte_len);
        }
    }

    /// Returns the stack length argument.
    fn stack_len_arg(&mut self) -> B::Value {
        self.bcx.fn_param(2)
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

    /// Builds a check, failing if `ret` is not `InstructionResult::Continue`.
    fn build_check_instruction_result(&mut self, ret: B::Value) {
        // Continue was 0 in old revm, use Stop (1) as the "continue" marker
        let failure = self.bcx.icmp_imm(IntCC::NotEqual, ret, InstructionResult::Stop as i64);
        let target = self.build_check_inner(true, failure, ret);
        self.bcx.switch_to_block(target);
    }

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

        let return_block = if let Some(return_block) = self.return_block {
            self.incoming_returns.push((ret, current_block));
            return_block
        } else {
            self.create_block_after(target, "return")
        };
        let then_block = if is_failure { return_block } else { target };
        let else_block = if is_failure { target } else { return_block };
        self.bcx.brif_cold(cond, then_block, else_block, is_failure);

        if self.return_block.is_none() {
            self.bcx.switch_to_block(return_block);
            self.bcx.ret(&[ret]);
        }

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
        if let Some(block) = self.failure_block {
            self.incoming_failures.push((ret, self.bcx.current_block().unwrap()));
            self.bcx.br(block);
        } else {
            self.bcx.ret(&[ret]);
        }
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
        if let Some(block) = self.return_block {
            self.incoming_returns.push((ret, self.bcx.current_block().unwrap()));
            self.bcx.br(block);
        } else {
            self.bcx.ret(&[ret]);
        }
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
        let mut msg = format!("revmc panic: {name} must not be null");
        if !extra.is_empty() {
            write!(msg, " ({extra})").unwrap();
        }
        self.build_assertion(panic_cond, &msg);
    }

    fn build_assertion(&mut self, cond: B::Value, msg: &str) {
        let failure = self.create_block_after_current("panic");
        let target = self.create_block_after(failure, "contd");
        self.bcx.brif(cond, failure, target);

        // `panic` is already marked as a cold function call.
        // self.bcx.set_cold_block(failure);
        self.bcx.switch_to_block(failure);
        self.call_panic(msg);

        self.bcx.switch_to_block(target);
    }

    /// Build a call to the panic builtin.
    fn call_panic(&mut self, msg: &str) {
        let function = self.builtin_function(Builtin::Panic);
        let ptr = self.bcx.str_const(msg);
        let len = self.bcx.iconst(self.isize_type, msg.len() as i64);
        let _ = self.bcx.call(function, &[ptr, len]);
        self.bcx.unreachable();
    }

    #[allow(dead_code)]
    fn call_printf(&mut self, template: &std::ffi::CStr, values: &[B::Value]) {
        let mut args = Vec::with_capacity(values.len() + 1);
        args.push(self.bcx.cstr_const(template));
        args.extend_from_slice(values);
        let printf = self.bcx.get_printf_function();
        let _ = self.bcx.call(printf, &args);
    }

    /// Build a call to a builtin that returns an [`InstructionResult`].
    fn call_fallible_builtin(&mut self, builtin: Builtin, args: &[B::Value]) {
        let ret = self.call_builtin(builtin, args).expect("builtin does not return a value");
        self.build_check_instruction_result(ret);
    }

    /// Build a call to a builtin.
    #[must_use]
    fn call_builtin(&mut self, builtin: Builtin, args: &[B::Value]) -> Option<B::Value> {
        let function = self.builtin_function(builtin);
        // self.call_printf(
        //     format_printf!("{} - calling {}\n", self.op_block_name(""), builtin.name()),
        //     &[],
        // );
        self.bcx.call(function, args)
    }

    /// Gets the function for the given builtin.
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
        self.bytecode.op_block_name(self.current_inst, name)
    }
}

/// IR builtins.
impl<B: Backend> FunctionCx<'_, B> {
    fn call_byte(&mut self, index: B::Value, value: B::Value) -> B::Value {
        self.call_ir_binop_builtin("byte", index, value, Self::build_byte)
    }

    /// Builds: `fn byte(index: u256, value: u256) -> u256`
    fn build_byte(&mut self) {
        let index = self.bcx.fn_param(0);
        let value = self.bcx.fn_param(1);

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

        self.bcx.ret(&[r]);
    }

    fn call_signextend(&mut self, ext: B::Value, x: B::Value) -> B::Value {
        self.call_ir_binop_builtin("signextend", ext, x, Self::build_signextend)
    }

    /// Builds: `fn signextend(ext: u256, x: u256) -> u256`
    fn build_signextend(&mut self) {
        // From the yellow paper:
        /*
        let [ext, x] = stack.pop();
        let t = 256 - 8 * (ext + 1);
        let mut result = x;
        result[..t] = [x[t]; t]; // Index by bits.
        */

        let ext = self.bcx.fn_param(0);
        let x = self.bcx.fn_param(1);

        // For 31 we also don't need to do anything.
        let might_do_something = self.bcx.icmp_imm(IntCC::UnsignedLessThan, ext, 31);
        let r = self.bcx.lazy_select(
            might_do_something,
            self.bcx.type_int(256),
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
        self.bcx.ret(&[r]);
    }

    fn call_ir_binop_builtin(
        &mut self,
        name: &str,
        x1: B::Value,
        x2: B::Value,
        build: fn(&mut Self),
    ) -> B::Value {
        let word = self.word_type;
        self.call_ir_builtin(name, &[x1, x2], &[word, word], Some(word), build).unwrap()
    }

    #[must_use]
    fn call_ir_builtin(
        &mut self,
        name: &str,
        args: &[B::Value],
        arg_types: &[B::Type],
        ret: Option<B::Type>,
        build: impl FnOnce(&mut Self),
    ) -> Option<B::Value> {
        let prefix = "__revmc_ir_builtin_";
        let name = &format!("{prefix}{name}")[..];

        // self.call_printf(format_printf!("{} - calling {name}\n", self.op_block_name("")), &[]);

        debug_assert_eq!(args.len(), arg_types.len());
        let linkage = revmc_backend::Linkage::Private;
        // SAFETY: `this` aliases `self` to work around the borrow checker: we need `self.bcx`
        // borrowed by `get_or_build_function` while also swapping `self.bcx` inside the closure.
        // The closure's `bcx` is a fresh builder (not `self.bcx`), so the swap is safe.
        let this = unsafe { &mut *(self as *mut Self) };
        let f = self.bcx.get_or_build_function(name, arg_types, ret, linkage, |bcx| {
            let prev_return_block = this.return_block.take();
            let prev_failure_block = this.failure_block.take();
            // SAFETY: `this.bcx` and `bcx` are non-overlapping (bcx is a fresh builder).
            unsafe { std::ptr::swap(&mut this.bcx, bcx) };

            for attr in default_attrs::for_fn().chain(std::iter::once(Attribute::NoUnwind)) {
                this.bcx.add_function_attribute(None, attr, FunctionAttributeLocation::Function)
            }
            for i in 0..this.bcx.num_fn_params() as u32 {
                for attr in default_attrs::for_param() {
                    this.bcx.add_function_attribute(None, attr, FunctionAttributeLocation::Param(i))
                }
            }
            build(this);

            // SAFETY: same as above.
            unsafe { std::ptr::swap(&mut this.bcx, bcx) };
            this.failure_block = prev_failure_block;
            this.return_block = prev_return_block;
        });
        self.bcx.call(f, args)
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
        /// Memory gas tracking (words_num: usize, expansion_cost: u64)
        memory: MemoryGas,
    }

    #[repr(C)]
    struct MemoryGas {
        words_num: usize,
        expansion_cost: u64,
    }
    const _: [(); mem::size_of::<revm_interpreter::Gas>()] = [(); mem::size_of::<Gas>()];
}

fn get_field<B: Builder>(bcx: &mut B, ptr: B::Value, offset: usize, name: &str) -> B::Value {
    let offset = bcx.iconst(bcx.type_ptr_sized_int(), offset as i64);
    bcx.gep(bcx.type_int(8), ptr, &[offset], name)
}

#[allow(unused)]
macro_rules! format_printf {
    ($($t:tt)*) => {
        &std::ffi::CString::new(format!($($t)*)).unwrap()
    };
}
#[allow(unused)]
use format_printf;
