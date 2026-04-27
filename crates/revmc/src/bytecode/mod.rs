//! Internal EVM bytecode and opcode representation.

use crate::FxHashMap;
use bitvec::vec::BitVec;
use oxc_index::IndexVec;
use revm_bytecode::opcode as op;
use revm_primitives::{U256, hardfork::SpecId};
use revmc_backend::Result;
use smallvec::SmallVec;
use std::{borrow::Cow, cell::RefCell};

pub(crate) use revm_context_interface::cfg::GasParams;

mod passes;
pub(crate) use passes::{Block, Cfg, GasSection, SectionsAnalysis, Snapshots, StackSection};

mod asm;
pub use asm::parse_asm;

mod fmt;

mod interner;
pub(crate) use interner::Interner;

mod info;
pub use info::*;

mod opcode;
pub use opcode::*;

/// Noop opcode used to test suspend-resume.
#[cfg(any(feature = "__fuzzing", test))]
pub(crate) const TEST_SUSPEND: u8 = 0x25;

/// Implements `Display` for a nonmax index type using a format string.
macro_rules! impl_index_display {
    ($ty:ty, $fmt:literal) => {
        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, $fmt, self.index())
            }
        }
    };
}
pub(crate) use impl_index_display;

oxc_index::define_nonmax_u32_index_type! {
    /// An EVM instruction index into [`Bytecode`] instructions.
    ///
    /// Also known as `ic`, or instruction counter; not to be confused with SSA `inst`s.
    pub(crate) struct Inst;
}
impl_index_display!(Inst, "ic{}");

oxc_index::define_nonmax_u32_index_type! {
    /// Index into the deduplicated U256 constant pool.
    pub(crate) struct U256Idx;
}
impl_index_display!(U256Idx, "{}");

/// Type alias for the bytecode-level U256 constant interner.
pub(crate) type U256Interner = Interner<U256Idx, U256, alloy_primitives::map::FbBuildHasher<32>>;

/// Compact handle to a `U256` constant.
///
/// Encodes either:
/// - An inline `u8` value (when the constant fits in `0..=255`), tagged with the high bit: raw =
///   `INLINE_TAG | value`.
/// - A reference into a [`U256Interner`], with the high bit clear: raw = `idx`. Up to 2^31 distinct
///   non-inline constants per [`Bytecode`] are supported.
///
/// Distinguishing the two representations is a single bit test against `INLINE_TAG`.
/// Constructing through [`U256Imm::new`] is canonical: equal values always produce equal
/// `U256Imm`s, so `Eq`/`Hash` work on the raw bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct U256Imm(u32);

impl U256Imm {
    /// High bit set means the handle is an inline `u8` value.
    const INLINE_TAG: u32 = 1 << 31;

    /// Constructs a handle for `value`, inlining `u8`-sized values and otherwise interning.
    #[inline]
    pub(crate) fn new(value: U256, interner: &mut U256Interner) -> Self {
        if value.bit_len() <= 8 {
            Self(Self::INLINE_TAG | value.byte(0) as u32)
        } else {
            let idx = interner.intern(value).index() as u32;
            debug_assert!(
                idx & Self::INLINE_TAG == 0,
                "U256 interner exceeded 2^31 entries; bump `U256Imm` encoding",
            );
            Self(idx)
        }
    }

    /// Returns the underlying `U256` value.
    #[inline]
    pub(crate) fn get(self, interner: &U256Interner) -> U256 {
        if self.0 & Self::INLINE_TAG != 0 {
            U256::from(self.0 as u8)
        } else {
            *interner.get(U256Idx::from_usize(self.0 as usize))
        }
    }

    /// Returns the inline `u8` value, if any.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn as_u8(self) -> Option<u8> {
        (self.0 & Self::INLINE_TAG != 0).then_some(self.0 as u8)
    }

    /// Encodes this handle as a `u32` for storage in [`InstData::data`].
    #[inline]
    pub(crate) fn to_raw(self) -> u32 {
        self.0
    }

    /// Decodes a handle previously produced by [`Self::to_raw`].
    #[inline]
    pub(crate) fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
}

impl std::fmt::Debug for U256Imm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_u8() {
            Some(b) => write!(f, "U256Imm::Inline(0x{b:02x})"),
            None => write!(f, "U256Imm::Idx({})", self.0),
        }
    }
}

bitflags::bitflags! {
    /// Controls which analysis passes run during [`Bytecode::analyze`].
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct AnalysisConfig: u8 {
        /// Run block deduplication.
        const DEDUP = 1 << 0;
        /// The stack is observable outside the function (`inspect_stack` mode).
        /// When set, DSE must not assume diverging terminators kill the stack.
        const INSPECT_STACK = 1 << 1;
        /// Run dead store elimination.
        const DSE = 1 << 2;

        /// All passes enabled.
        const ALL = Self::DEDUP.bits() | Self::DSE.bits();
    }
}

impl Default for AnalysisConfig {
    #[inline]
    fn default() -> Self {
        Self::ALL
    }
}

/// Default compiler gas limit for compile-time evaluation (100k gas).
pub(crate) const DEFAULT_COMPILER_GAS_LIMIT: u64 = 100_000;

/// EVM bytecode.
#[doc(hidden)] // Not public API.
pub struct Bytecode<'a> {
    /// The original bytecode.
    pub(crate) code: Cow<'a, [u8]>,
    /// The instructions.
    insts: IndexVec<Inst, InstData>,
    /// `JUMPDEST` opcode map. `jumpdests[pc]` is `true` if `code[pc] == op::JUMPDEST`.
    jumpdests: BitVec,
    /// The [`SpecId`].
    pub(crate) spec_id: SpecId,
    /// Gas parameters for dynamic gas folding. Defaults to `GasParams::new_spec(spec_id)`.
    pub(crate) gas_params: GasParams,
    /// Whether the bytecode contains dynamic jumps.
    has_dynamic_jumps: bool,
    /// Whether the bytecode may suspend execution.
    may_suspend: bool,
    /// Mapping from program counter to instruction.
    pc_to_inst: FxHashMap<u32, Inst>,
    /// Mapping from instruction index to its program counter (`code[pc]` is the opcode byte).
    ///
    /// `JUMPDEST` and `PC` mirror this in `data` for hot codegen access; everything else
    /// reads it through [`Bytecode::pc`].
    inst_to_pc: IndexVec<Inst, u32>,

    /// Deduplicated constant pool for U256 values.
    u256_interner: RefCell<U256Interner>,

    /// Per-instruction operand snapshots computed by block analysis.
    snapshots: Snapshots,
    /// Multi-target jump table: maps a JUMP/JUMPI instruction to its set of known targets.
    /// Only populated for jumps resolved to multiple targets by block analysis.
    multi_jump_targets: FxHashMap<Inst, SmallVec<[Inst; 4]>>,

    /// Instruction index to 1-based line number in the formatted dump, built during formatting.
    inst_lines: RefCell<IndexVec<Inst, u32>>,

    /// Dead-block redirects: maps the first instruction of a dead/merged block to the target
    /// instruction. Used mainly for fallthrough jumps.
    pub(crate) redirects: FxHashMap<Inst, Inst>,
    /// Basic-block CFG, rebuilt by [`Bytecode::rebuild_cfg`].
    cfg: Cfg,
    /// Controls which analysis passes are enabled.
    pub(crate) config: AnalysisConfig,
    /// Gas budget for compile-time evaluation of user-supplied bytecode.
    ///
    /// The compiler evaluates EVM operations at compile time during analysis passes. Without a
    /// budget, adversarial bytecode (e.g. thousands of `EXP(U256::MAX, U256::MAX)`) can make
    /// compilation arbitrarily slow. This limit uses the EVM gas schedule to bound work.
    ///
    /// When exhausted, further compile-time evaluation is skipped (values remain dynamic).
    /// Defaults to 100k gas.
    pub(crate) compiler_gas_limit: u64,
    /// Cumulative compiler gas consumed so far.
    pub(crate) compiler_gas_used: u64,
}

impl<'a> Bytecode<'a> {
    pub(crate) fn new(
        code: impl Into<Cow<'a, [u8]>>,
        spec_id: SpecId,
        gas_params: Option<GasParams>,
    ) -> Self {
        Self::new_mono(code.into(), spec_id, gas_params)
    }

    #[cfg(test)]
    pub(crate) fn test(code: impl Into<Cow<'a, [u8]>>) -> Self {
        Self::new(code, crate::tests::DEF_SPEC, None)
    }

    #[instrument(name = "Bytecode::new", level = "debug", skip_all)]
    fn new_mono(code: Cow<'a, [u8]>, spec_id: SpecId, gas_params: Option<GasParams>) -> Self {
        let gas_params = gas_params.unwrap_or_else(|| GasParams::new_spec(spec_id));
        let mut insts = IndexVec::with_capacity(code.len() + 8);
        let mut inst_to_pc: IndexVec<Inst, u32> = IndexVec::with_capacity(code.len() + 8);
        let mut jumpdests = BitVec::repeat(false, code.len());
        let mut pc_to_inst = FxHashMap::with_capacity_and_hasher(code.len(), Default::default());
        let mut u256_interner: U256Interner = Interner::new();
        let op_infos = op_info_map(spec_id);
        let mut iter = OpcodesIter::new(&code, spec_id).with_pc();
        while let Some((pc, Opcode { opcode, immediate })) = iter.next() {
            let inst: Inst = insts.next_idx();
            pc_to_inst.insert(pc as u32, inst);
            inst_to_pc.push(pc as u32);

            if opcode == op::JUMPDEST {
                jumpdests.set(pc, true)
            }

            let mut flags = InstFlags::empty();
            let info = op_infos[opcode as usize];
            if info.is_unknown() {
                flags |= InstFlags::UNKNOWN;
            }
            if info.is_disabled() {
                flags |= InstFlags::DISABLED;
            }
            let (base_gas, stack_io) = if info.is_unknown() || info.is_disabled() {
                (0, (0, 0))
            } else {
                (info.base_gas(), compute_stack_io(opcode, immediate))
            };

            // Pre-compute the immediate so the rest of the compiler never has to access
            // the raw bytecode. PUSH values are wrapped in `U256Imm`, which inlines
            // u8-sized values directly and otherwise interns them. DUPN/SWAPN/EXCHANGE
            // store their immediate byte as an always-inline `U256Imm`. JUMPDEST and PC
            // store their pc inline so jump dispatch and the PC opcode read it directly
            // from `data`.
            let data = match opcode {
                op::PUSH0 => U256Imm::new(U256::ZERO, &mut u256_interner).to_raw(),
                op::PUSH1..=op::PUSH32 => {
                    let imm_len = min_imm_len(opcode) as usize;
                    // `OpcodesIter` returns `None` for truncated EOF immediates; recover
                    // the available bytes directly from `code` and right-pad with zeros
                    // per EVM spec.
                    let start = pc + 1;
                    let avail = code.get(start..).map(|s| &s[..s.len().min(imm_len)]);
                    let value = match avail {
                        Some(slice) if slice.len() == imm_len => U256::from_be_slice(slice),
                        Some(slice) if !slice.is_empty() => {
                            let mut padded = [0u8; 32];
                            padded[..slice.len()].copy_from_slice(slice);
                            U256::from_be_slice(&padded[..imm_len])
                        }
                        _ => U256::ZERO,
                    };
                    U256Imm::new(value, &mut u256_interner).to_raw()
                }
                op::DUPN | op::SWAPN | op::EXCHANGE => {
                    let b = code.get(pc + 1).copied().unwrap_or(0);
                    U256Imm::new(U256::from(b), &mut u256_interner).to_raw()
                }
                op::JUMPDEST | op::PC => pc as u32,
                _ => 0,
            };

            let gas_section = GasSection::default();
            let stack_section = StackSection::default();

            insts.push(InstData {
                opcode,
                flags,
                base_gas,
                stack_io,
                data,
                gas_section,
                stack_section,
            });

            // EIP-8024: JUMPDEST analysis is unchanged by DUPN/SWAPN/EXCHANGE. Their
            // immediate byte is NOT masked, so 0x5b in that position is a valid jump
            // target. When the immediate is 0x5b, rewind the iterator so it is re-yielded
            // and processed as a normal JUMPDEST by the loop.
            if matches!(opcode, op::DUPN | op::SWAPN | op::EXCHANGE)
                && !info.is_unknown()
                && !info.is_disabled()
                && immediate == Some(&[op::JUMPDEST])
            {
                // SAFETY: we just consumed the 1-byte immediate.
                unsafe { iter.rewind(1) };
            }
        }

        // Pad code to ensure there is at least one diverging instruction.
        if insts.last().is_none_or(|last| last.can_fall_through()) {
            trace!("adding STOP padding");
            insts.push(InstData::new(op::STOP));
            // Synthetic padding lives one past the real bytecode.
            inst_to_pc.push(code.len() as u32);
        }

        Self {
            code,
            insts,
            jumpdests,
            spec_id,
            gas_params,
            has_dynamic_jumps: false,
            may_suspend: false,
            snapshots: Snapshots::default(),
            u256_interner: RefCell::new(u256_interner),
            multi_jump_targets: FxHashMap::default(),
            pc_to_inst,
            inst_to_pc,
            inst_lines: RefCell::new(IndexVec::new()),
            redirects: FxHashMap::default(),
            cfg: Cfg::default(),
            config: AnalysisConfig::default(),
            compiler_gas_limit: 100_000,
            compiler_gas_used: 0,
        }
    }

    /// Takes the instruction-to-line map built during formatting.
    ///
    /// If the bytecode has not been formatted yet, formats it first to build the map.
    pub(crate) fn take_inst_lines(&self) -> IndexVec<Inst, u32> {
        if self.inst_lines.borrow().is_empty() {
            self.collect_lines();
        }
        self.inst_lines.take()
    }

    /// Returns an iterator over the opcodes.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn opcodes(&self) -> OpcodesIter<'_> {
        OpcodesIter::new(&self.code, self.spec_id)
    }

    /// Returns the instruction at the given instruction counter.
    #[inline]
    #[track_caller]
    pub(crate) fn inst(&self, inst: Inst) -> &InstData {
        &self.insts[inst]
    }

    /// Returns a mutable reference the instruction at the given instruction counter.
    #[inline]
    #[track_caller]
    #[allow(dead_code)]
    pub(crate) fn inst_mut(&mut self, inst: Inst) -> &mut InstData {
        &mut self.insts[inst]
    }

    /// Returns the opcode + immediate slice for the given instruction.
    #[inline]
    pub(crate) fn opcode(&self, inst: Inst) -> Opcode<'_> {
        Opcode { opcode: self.inst(inst).opcode, immediate: self.get_imm(inst) }
    }

    /// Returns an iterator over the instructions.
    #[inline]
    pub(crate) fn iter_insts(&self) -> impl DoubleEndedIterator<Item = (Inst, &InstData)> + Clone {
        self.iter_all_insts().filter(|(_, data)| !data.is_dead_code())
    }

    /// Returns an iterator over the instructions.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn iter_mut_insts(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (Inst, &mut InstData)> {
        self.iter_mut_all_insts().filter(|(_, data)| !data.is_dead_code())
    }

    /// Returns an iterator over all the instructions, including dead code.
    #[inline]
    pub(crate) fn iter_all_insts(
        &self,
    ) -> impl DoubleEndedIterator<Item = (Inst, &InstData)> + ExactSizeIterator + Clone {
        self.insts.iter_enumerated()
    }

    /// Returns an iterator over all the instructions, including dead code.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn iter_mut_all_insts(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (Inst, &mut InstData)> + ExactSizeIterator {
        self.insts.iter_mut_enumerated()
    }

    /// Runs a list of analysis passes on the instructions.
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn analyze(&mut self) -> Result<()> {
        self.recompute_has_dynamic_jumps();
        self.mark_dead_code();
        self.rebuild_cfg();

        self.block_analysis_local();
        self.mark_dead_code();
        self.rebuild_cfg();

        let local_snapshots = self.snapshots.clone();

        self.block_analysis(&local_snapshots);
        self.mark_dead_code();
        self.rebuild_cfg();

        if self.config.contains(AnalysisConfig::DEDUP) {
            self.dedup_blocks(&local_snapshots);
            self.mark_dead_code();
            self.rebuild_cfg();
        }
        drop(local_snapshots);

        if self.config.contains(AnalysisConfig::DSE) {
            self.dead_store_elim();
        }

        self.calc_may_suspend();

        self.construct_sections();

        debug!(
            compiler_gas_used = self.compiler_gas_used,
            compiler_gas_limit = self.compiler_gas_limit,
            "constant folding gas budget",
        );

        if tracing::enabled!(tracing::Level::TRACE) {
            self.log_ir_stats();
            self.log_const_input_stats();
        }

        Ok(())
    }

    /// Mark unreachable instructions as `DEAD_CODE` to not generate any code for them.
    ///
    /// This pass is technically unnecessary as the backend will very likely optimize any
    /// unreachable code that we generate, but this is trivial for us to do and significantly speeds
    /// up code generation.
    ///
    /// We can simply mark all instructions that are between diverging instructions and
    /// `JUMPDEST`s.
    #[instrument(name = "dce", level = "debug", skip_all)]
    fn mark_dead_code(&mut self) {
        let mut iter = self.insts.iter_mut_enumerated();
        while let Some((i, data)) = iter.next() {
            if !data.can_fall_through() {
                let mut end = i;
                let mut any_new = false;
                for (j, data) in &mut iter {
                    end = j;
                    if data.is_reachable_jumpdest(self.has_dynamic_jumps) {
                        break;
                    }
                    if !data.flags.contains(InstFlags::DEAD_CODE) {
                        any_new = true;
                    }
                    data.flags |= InstFlags::DEAD_CODE;
                }
                let start = i + 1;
                if any_new && end > start {
                    debug!("found dead code: {start}..{end}");
                }
            }
        }
    }

    /// Calculates whether the bytecode suspend suspend execution.
    ///
    /// This can only happen if the bytecode contains `*CALL*` or `*CREATE*` instructions.
    #[instrument(name = "suspend", level = "debug", skip_all)]
    fn calc_may_suspend(&mut self) {
        let may_suspend = self.iter_insts().any(|(_, data)| data.may_suspend());
        self.may_suspend = may_suspend;
    }

    /// Constructs the sections in the bytecode.
    #[instrument(name = "sections", level = "debug", skip_all)]
    fn construct_sections(&mut self) {
        let mut analysis = SectionsAnalysis::default();
        for inst in self.insts.indices() {
            if !self.inst(inst).is_dead_code() {
                analysis.process(self, inst);
            }
        }
        analysis.finish(self);
    }

    /// Returns the immediate value of the given instruction, if any.
    ///
    /// For truncated immediates at EOF, returns the available bytes (which may be shorter than
    /// `imm_len`). Returns `None` only when `imm_len` is 0 or `start` is completely out of bounds.
    pub(crate) fn get_imm(&self, inst: Inst) -> Option<&[u8]> {
        let data = self.inst(inst);
        let imm_len = data.imm_len() as usize;
        if imm_len == 0 {
            return None;
        }
        let start = self.pc(inst) as usize + 1;
        let end = (start + imm_len).min(self.code.len());
        if start >= end {
            return None;
        }
        Some(&self.code[start..end])
    }

    /// Returns the program counter (offset into the original bytecode) of the given instruction.
    #[inline]
    pub(crate) fn pc(&self, inst: Inst) -> u32 {
        self.inst_to_pc[inst]
    }

    /// Returns the value of a PUSH instruction, right-padding truncated EOF immediates with zeros
    /// per EVM spec.
    #[cfg(test)]
    pub(crate) fn get_push_value(&self, data: &InstData) -> U256 {
        debug_assert!(matches!(data.opcode, op::PUSH0..=op::PUSH32));
        data.imm().get(&self.u256_interner.borrow())
    }

    /// Returns the length of the original bytecode in bytes (for `CODESIZE`).
    #[inline]
    pub(crate) fn codesize(&self) -> usize {
        self.code.len()
    }

    /// Returns `true` if the given program counter is a valid jump destination.
    fn is_valid_jump(&self, pc: usize) -> bool {
        self.jumpdests.get(pc).as_deref().copied() == Some(true)
    }

    /// Returns `true` if the bytecode has dynamic jumps.
    pub(crate) fn has_dynamic_jumps(&self) -> bool {
        self.has_dynamic_jumps
    }

    /// Returns `true` if the bytecode may suspend execution, to be resumed later.
    pub(crate) fn may_suspend(&self) -> bool {
        self.may_suspend
    }

    /// Returns `true` if any dead-block redirects exist.
    pub(crate) fn has_redirects(&self) -> bool {
        !self.redirects.is_empty()
    }

    /// Returns `true` if the instruction is diverging.
    pub(crate) fn is_instr_diverging(&self, inst: Inst) -> bool {
        self.insts[inst].is_diverging()
    }

    /// Converts a program counter (`self.code[pc]`) to an instruction (`self.inst(inst)`).
    #[inline]
    pub(crate) fn pc_to_inst(&self, pc: usize) -> Inst {
        match self.pc_to_inst.get(&(pc as u32)) {
            Some(&inst) => inst,
            None => panic!("pc out of bounds: {pc}"),
        }
    }

    /// Returns the multi-target jump table entry for the given instruction, if any.
    pub(crate) fn multi_jump_targets(&self, inst: Inst) -> Option<&[Inst]> {
        self.multi_jump_targets.get(&inst).map(|v| v.as_slice())
    }

    /// Returns the known constant value of a stack operand at the given instruction.
    ///
    /// `depth` 0 is TOS (first popped by this instruction), 1 is second, etc.
    /// Returns `None` if the value is unknown or the analysis didn't cover this instruction.
    #[allow(dead_code)]
    pub(crate) fn const_operand(&self, inst: Inst, depth: usize) -> Option<U256> {
        let snap = &self.snapshots.inputs[inst];
        let imm = snap.get(snap.len().checked_sub(1 + depth)?)?.as_const()?;
        Some(imm.get(&self.u256_interner.borrow()))
    }

    /// Returns the known constant output value of the given instruction.
    ///
    /// Returns `None` if the output is unknown, the instruction has no output, or the
    /// analysis didn't cover this instruction.
    #[allow(dead_code)]
    pub(crate) fn const_output(&self, inst: Inst) -> Option<U256> {
        let imm = self.snapshots.outputs[inst]?.as_const()?;
        Some(imm.get(&self.u256_interner.borrow()))
    }

    /// Logs per-opcode constant-input statistics at trace level.
    #[inline(never)]
    fn log_const_input_stats(&self) {
        use op::*;
        use std::fmt::Write;

        // per_input[depth] = [total, const_count, fits_usize_count].
        struct Entry {
            inputs: usize,
            outputs: usize,
            total: u32,
            all_const: u32,
            const_output: u32,
            per_input: Vec<[u32; 3]>,
        }
        let mut stats = [const { None }; 256];
        for (inst, data) in self.iter_insts() {
            let (inputs, outputs) = data.stack_io();
            if matches!(data.opcode, PUSH0..=PUSH32 | DUP1..=DUP16 | SWAP1..=SWAP16 | DUPN | SWAPN)
            {
                continue;
            }
            let entry = stats[data.opcode as usize].get_or_insert_with(|| Entry {
                inputs: inputs as usize,
                outputs: outputs as usize,
                total: 0,
                all_const: 0,
                const_output: 0,
                per_input: vec![[0u32; 3]; inputs as usize],
            });
            entry.total += 1;
            let mut all = true;
            for depth in 0..inputs as usize {
                entry.per_input[depth][0] += 1;
                if let Some(val) = self.const_operand(inst, depth) {
                    entry.per_input[depth][1] += 1;
                    if val <= usize::MAX {
                        entry.per_input[depth][2] += 1;
                    }
                } else {
                    all = false;
                }
            }
            if all {
                entry.all_const += 1;
            }
            if self.const_output(inst).is_some() {
                entry.const_output += 1;
            }
        }

        let mut buf = String::from("const input stats:");
        for (opcode, entry) in stats.iter().enumerate() {
            let Some(entry) = entry else { continue };
            if entry.total == 0 {
                continue;
            }
            let name = OpCode::new(opcode as u8)
                .map_or_else(|| format!("0x{opcode:02x}"), |o| o.as_str().to_string());
            let all_pct = entry.all_const as f64 / entry.total as f64 * 100.0;
            let _ = write!(
                buf,
                "\n  {name:<16} 0x{opcode:02x}, {total} occ, {inputs} inputs, all_const={ac}/{total} ({all_pct:.0}%)",
                opcode = opcode,
                total = entry.total,
                inputs = entry.inputs,
                ac = entry.all_const,
            );
            if entry.outputs > 0 {
                let co = entry.const_output;
                let co_pct = co as f64 / entry.total as f64 * 100.0;
                let _ = write!(buf, ", const_output={co}/{t} ({co_pct:.0}%)", t = entry.total);
            }
            for (i, [t, c, usize_c]) in entry.per_input.iter().enumerate() {
                if *t > 0 {
                    let pct = *c as f64 / *t as f64 * 100.0;
                    let usize_pct = *usize_c as f64 / *t as f64 * 100.0;
                    let _ = write!(
                        buf,
                        "\n    [{i}]: const={c}/{t} ({pct:.0}%), fits_usize={usize_c}/{t} ({usize_pct:.0}%)",
                    );
                }
            }
        }
        trace!("{buf}");
    }

    /// Collects IR statistics: instruction counts and block size distribution.
    fn ir_stats(&self) -> IrStats {
        let total = self.insts.len();
        let mut live = 0usize;
        let mut noops = 0usize;
        let mut dead = 0usize;
        let mut suspends = 0usize;
        for (_inst, data) in self.iter_all_insts() {
            if data.is_dead_code() {
                dead += 1;
            } else {
                live += 1;
                if data.flags.contains(InstFlags::NOOP) {
                    noops += 1;
                }
                if data.may_suspend() {
                    suspends += 1;
                }
            }
        }

        let mut lens: Vec<usize> = self.cfg.blocks.iter().map(|data| data.insts().len()).collect();
        let n = lens.len();
        lens.sort_unstable();
        let block_min = lens.first().copied().unwrap_or(0);
        let block_max = lens.last().copied().unwrap_or(0);
        let sum: usize = lens.iter().sum();
        let block_avg = if n > 0 { sum as f64 / n as f64 } else { 0.0 };
        let block_median = if n > 0 { lens[n / 2] } else { 0 };

        IrStats {
            total,
            live,
            dead,
            noops,
            suspends,
            blocks: n,
            block_min,
            block_max,
            block_avg,
            block_median,
        }
    }

    /// Logs IR statistics and top 5 longest blocks at trace level.
    #[inline(never)]
    fn log_ir_stats(&self) {
        use std::fmt::Write;

        let s = self.ir_stats();
        trace!(
            total_insts = s.total,
            live = s.live,
            dead = s.dead,
            noops = s.noops,
            suspends = s.suspends,
            blocks = s.blocks,
            block_min = s.block_min,
            block_max = s.block_max,
            block_avg = format_args!("{:.1}", s.block_avg),
            block_median = s.block_median,
            "ir stats",
        );

        let mut lens: Vec<(Block, usize)> = self
            .cfg
            .blocks
            .iter_enumerated()
            .map(|(block, data)| (block, data.insts().len()))
            .collect();
        lens.sort_unstable_by_key(|b| std::cmp::Reverse(b.1));
        lens.truncate(5);
        let mut buf = String::from("top 5 longest blocks:");
        for (block, len) in &lens {
            let data = &self.cfg.blocks[*block];
            let _ = write!(buf, "\n  {block} (pc={}, len={len})", self.pc(data.insts.start));
        }
        trace!("{buf}");
    }

    /// Returns the name for a basic block.
    pub(crate) fn op_block_name(&self, inst: Option<Inst>, name: &str) -> String {
        use std::fmt::Write;

        let Some(inst) = inst else {
            return format!("entry.{name}");
        };
        let data = self.inst(inst);

        let mut s = String::with_capacity(64);
        if let Some(block) = self.cfg.inst_to_block[inst] {
            let _ = write!(s, "{block}.");
        }
        let _ = write!(s, "{inst}.{}", data.to_op());
        if !name.is_empty() {
            let _ = write!(s, ".{name}");
        }
        s
    }
}

/// Summary IR statistics for a compiled bytecode.
pub(crate) struct IrStats {
    pub(crate) total: usize,
    pub(crate) live: usize,
    pub(crate) dead: usize,
    pub(crate) noops: usize,
    pub(crate) suspends: usize,
    pub(crate) blocks: usize,
    pub(crate) block_min: usize,
    pub(crate) block_max: usize,
    pub(crate) block_avg: f64,
    pub(crate) block_median: usize,
}

/// A single instruction in the bytecode.
#[derive(Clone, Default)]
pub(crate) struct InstData {
    /// The opcode byte.
    pub(crate) opcode: u8,
    /// Flags.
    pub(crate) flags: InstFlags,
    /// The base gas cost of the opcode.
    ///
    /// This may not be the final/full gas cost of the opcode as it may also have a dynamic cost.
    base_gas: u16,
    /// Stack inputs and outputs.
    stack_io: (u8, u8),
    /// Instruction-specific data, populated at parse time and refined by analysis.
    /// Interpretation depends on `opcode`; access via the typed methods on [`InstData`]:
    /// - `PUSH0..=PUSH32`, `DUPN | SWAPN | EXCHANGE`: encoded [`U256Imm`].
    /// - `PC`: program counter of this instruction.
    /// - `JUMPDEST`: program counter in low 31 bits, reachable flag in high bit
    ///   ([`InstData::JUMPDEST_REACHABLE`]).
    /// - `JUMP | JUMPI` with [`InstFlags::STATIC_JUMP`]: target instruction index.
    /// - otherwise: unused.
    pub(crate) data: u32,
    /// The gas section this instruction belongs to.
    pub(crate) gas_section: GasSection,
    /// The stack section this instruction belongs to.
    pub(crate) stack_section: StackSection,
}

impl PartialEq<u8> for InstData {
    #[inline]
    fn eq(&self, other: &u8) -> bool {
        self.opcode == *other
    }
}

impl PartialEq<InstData> for u8 {
    #[inline]
    fn eq(&self, other: &InstData) -> bool {
        *self == other.opcode
    }
}

impl InstData {
    /// Creates a new instruction data with the given opcode byte.
    /// Note that this may not be a valid instruction.
    #[inline]
    fn new(opcode: u8) -> Self {
        Self { opcode, stack_io: stack_io(opcode), ..Default::default() }
    }

    /// Returns the length of the immediate data of this instruction.
    #[inline]
    pub(crate) const fn imm_len(&self) -> u8 {
        min_imm_len(self.opcode)
    }

    /// Returns the number of input and output stack elements of this instruction.
    #[inline]
    pub(crate) fn stack_io(&self) -> (u8, u8) {
        self.stack_io
    }

    /// Converts this instruction to a raw opcode. Note that the immediate data is not resolved.
    #[inline]
    pub(crate) const fn to_op(&self) -> Opcode<'static> {
        Opcode { opcode: self.opcode, immediate: None }
    }

    /// Returns `true` if this instruction is a jump instruction (`JUMP`/`JUMPI`).
    #[inline]
    pub(crate) fn is_jump(&self) -> bool {
        matches!(self.opcode, op::JUMP | op::JUMPI)
    }

    /// Returns `true` if this instruction is a jump instruction (`JUMP`/`JUMPI`), and the
    /// target known statically.
    #[inline]
    pub(crate) fn is_static_jump(&self) -> bool {
        self.is_jump() && self.flags.contains(InstFlags::STATIC_JUMP)
    }

    /// Returns `true` if this instruction is a `JUMPDEST`.
    #[inline]
    pub(crate) const fn is_jumpdest(&self) -> bool {
        self.opcode == op::JUMPDEST
    }

    /// Returns `true` if this instruction is a reachable `JUMPDEST`.
    #[inline]
    pub(crate) fn is_reachable_jumpdest(&self, has_dynamic_jumps: bool) -> bool {
        self.is_jumpdest() && (has_dynamic_jumps || self.is_jumpdest_reachable())
    }

    /// Returns the [`U256Imm`] of a `PUSH*`/`DUPN`/`SWAPN`/`EXCHANGE` instruction.
    #[inline]
    pub(crate) fn imm(&self) -> U256Imm {
        debug_assert!(matches!(
            self.opcode,
            op::PUSH0..=op::PUSH32 | op::DUPN | op::SWAPN | op::EXCHANGE
        ));
        U256Imm::from_raw(self.data)
    }

    /// Returns the inline `u8` immediate of a `DUPN`/`SWAPN`/`EXCHANGE` instruction.
    #[inline]
    pub(crate) fn imm_byte(&self) -> u8 {
        debug_assert!(matches!(self.opcode, op::DUPN | op::SWAPN | op::EXCHANGE));
        // Parse-time encoding for these opcodes always produces an inline `u8`.
        self.imm().as_u8().unwrap()
    }

    /// Returns the program counter stored inline for a `PC` instruction.
    #[inline]
    pub(crate) fn pc_imm(&self) -> u32 {
        debug_assert_eq!(self.opcode, op::PC);
        self.data
    }

    /// High bit of `data` set when a `JUMPDEST` is a reachable static jump target.
    pub(crate) const JUMPDEST_REACHABLE: u32 = 1 << 31;

    /// Returns the program counter of a `JUMPDEST`.
    #[inline]
    pub(crate) fn jumpdest_pc(&self) -> u32 {
        debug_assert_eq!(self.opcode, op::JUMPDEST);
        self.data & !Self::JUMPDEST_REACHABLE
    }

    /// Returns `true` if this `JUMPDEST` is the target of a resolved static jump.
    #[inline]
    pub(crate) fn is_jumpdest_reachable(&self) -> bool {
        debug_assert_eq!(self.opcode, op::JUMPDEST);
        self.data & Self::JUMPDEST_REACHABLE != 0
    }

    /// Marks this `JUMPDEST` as a reachable static jump target.
    #[inline]
    pub(crate) fn set_jumpdest_reachable(&mut self) {
        debug_assert_eq!(self.opcode, op::JUMPDEST);
        self.data |= Self::JUMPDEST_REACHABLE;
    }

    /// Returns the static target of a `JUMP`/`JUMPI` with [`InstFlags::STATIC_JUMP`].
    #[inline]
    pub(crate) fn static_jump_target(&self) -> Inst {
        debug_assert!(self.is_static_jump());
        Inst::from_usize(self.data as usize)
    }

    /// Sets the static jump target. Caller must also set [`InstFlags::STATIC_JUMP`].
    #[inline]
    pub(crate) fn set_static_jump_target(&mut self, target: Inst) {
        debug_assert!(self.is_jump());
        self.data = target.index() as u32;
    }

    /// Returns `true` if this instruction starts a new stack section.
    #[inline]
    pub(crate) fn is_stack_section_head(&self) -> bool {
        self.flags.contains(InstFlags::STACK_SECTION_HEAD)
    }

    /// Returns `true` if this instruction is dead code.
    pub(crate) fn is_dead_code(&self) -> bool {
        self.flags.contains(InstFlags::DEAD_CODE)
    }

    /// Returns `true` if this instruction requires to know `gasleft()`.
    /// Note that this does not include CALL and CREATE.
    #[inline]
    pub(crate) fn requires_gasleft(&self, spec_id: SpecId) -> bool {
        // For SSTORE, see `revm_interpreter::gas::sstore_cost`.
        self.opcode == op::GAS
            || (self.opcode == op::SSTORE && spec_id.is_enabled_in(SpecId::ISTANBUL))
    }

    /// Returns `true` if execution can fall through to the next sequential instruction.
    #[inline]
    pub(crate) fn can_fall_through(&self) -> bool {
        !self.is_diverging() && self.opcode != op::JUMP
    }

    /// Returns `true` if we know that this instruction will branch or stop execution.
    #[inline]
    pub(crate) fn is_branching(&self) -> bool {
        self.is_jump() || self.is_diverging()
    }

    /// Returns `true` if we know that this instruction will stop execution.
    #[inline]
    pub(crate) fn is_diverging(&self) -> bool {
        #[cfg(test)]
        if self.opcode == TEST_SUSPEND {
            return false;
        }

        (self.opcode == op::JUMP && self.flags.contains(InstFlags::INVALID_JUMP))
            || self.flags.contains(InstFlags::DISABLED)
            || self.flags.contains(InstFlags::UNKNOWN)
            || matches!(
                self.opcode,
                op::STOP | op::RETURN | op::REVERT | op::INVALID | op::SELFDESTRUCT
            )
    }

    /// Returns `true` if this instruction may suspend execution.
    #[inline]
    pub(crate) const fn may_suspend(&self) -> bool {
        #[cfg(test)]
        if self.opcode == TEST_SUSPEND {
            return true;
        }

        matches!(
            self.opcode,
            op::CALL | op::CALLCODE | op::DELEGATECALL | op::STATICCALL | op::CREATE | op::CREATE2
        )
    }
}

bitflags::bitflags! {
    /// [`InstrData`] flags.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) struct InstFlags: u8 {
        /// The `JUMP`/`JUMPI` target is known at compile time.
        const STATIC_JUMP = 1 << 0;
        /// The jump target is known to be invalid.
        /// Always returns [`InstructionResult::InvalidJump`] at runtime.
        const INVALID_JUMP = 1 << 1;
        /// The jump has multiple known targets (see `Bytecode::multi_jump_targets`).
        /// The target value is still on the stack and must be popped and switched on at runtime.
        const MULTI_JUMP = 1 << 2;

        /// The instruction is disabled in this EVM version.
        /// Always returns [`InstructionResult::NotActivated`] at runtime.
        const DISABLED = 1 << 3;
        /// The instruction is unknown.
        /// Always returns [`InstructionResult::NotFound`] at runtime.
        const UNKNOWN = 1 << 4;

        /// Instruction is a no-op: skip generating logic, but keep the gas calculation.
        const NOOP = 1 << 5;
        /// This instruction starts a new stack section.
        const STACK_SECTION_HEAD = 1 << 6;
        /// Don't generate any code.
        const DEAD_CODE = 1 << 7;
    }
}

fn bitvec_as_bytes<T: bitvec::store::BitStore, O: bitvec::order::BitOrder>(
    bitvec: &BitVec<T, O>,
) -> &[u8] {
    slice_as_bytes(bitvec.as_raw_slice())
}

fn slice_as_bytes<T>(a: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), std::mem::size_of_val(a)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm_bytecode::opcode::OPCODE_INFO;

    #[test]
    fn test_suspend_is_free() {
        assert_eq!(OPCODE_INFO[TEST_SUSPEND as usize], None);
    }

    #[test]
    fn truncated_push_imm_right_pads_with_zeros() {
        // PUSH2 followed by a single byte 0x42 — truncated at EOF.
        // EVM spec: missing bytes are zero, so the value is 0x4200.
        let code = [op::PUSH2, 0x42];
        let bc = Bytecode::test(&code);
        let inst = Inst::from_usize(0);
        let data = bc.inst(inst);
        assert_eq!(bc.get_imm(inst), Some([0x42].as_slice()));
        assert_eq!(bc.get_push_value(data), U256::from(0x4200));

        // PUSH3 followed by two bytes — truncated at EOF.
        let code = [op::PUSH3, 0xAB, 0xCD];
        let bc = Bytecode::test(&code);
        let inst = Inst::from_usize(0);
        let data = bc.inst(inst);
        assert_eq!(bc.get_imm(inst), Some([0xAB, 0xCD].as_slice()));
        assert_eq!(bc.get_push_value(data), U256::from(0xABCD00));

        // PUSH1 with no immediate bytes at all.
        let code = [op::PUSH1];
        let bc = Bytecode::test(&code);
        let inst = Inst::from_usize(0);
        let data = bc.inst(inst);
        assert_eq!(bc.get_imm(inst), None);
        assert_eq!(bc.get_push_value(data), U256::ZERO);

        // Non-truncated PUSH2 — full immediate.
        let code = [op::PUSH2, 0x42, 0xFF];
        let bc = Bytecode::test(&code);
        let data = bc.inst(Inst::from_usize(0));
        assert_eq!(bc.get_push_value(data), U256::from(0x42FF));
    }
}
