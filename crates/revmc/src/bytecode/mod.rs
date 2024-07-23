//! Internal EVM bytecode and opcode representation.

use bitvec::vec::BitVec;
use either::Either;
use revm_interpreter::opcode as op;
use revm_primitives::{hex, Eof, SpecId};
use revmc_backend::{eyre::ensure, Result};
use rustc_hash::FxHashMap;
use std::{borrow::Cow, fmt};

mod sections;
use sections::{Section, SectionAnalysis};

mod info;
pub use info::*;

mod opcode;
pub use opcode::*;

/// Noop opcode used to test suspend-resume.
#[cfg(any(feature = "__fuzzing", test))]
pub(crate) const TEST_SUSPEND: u8 = 0x25;

// TODO: Use `indexvec`.
/// An EVM instruction is a high level internal representation of an EVM opcode.
///
/// This is an index into [`Bytecode`] instructions.
///
/// Also known as `ic`, or instruction counter; not to be confused with SSA `inst`s.
pub(crate) type Inst = usize;

/// EVM bytecode.
#[doc(hidden)] // Not public API.
pub struct Bytecode<'a> {
    /// The original bytecode slice.
    pub(crate) code: &'a [u8],
    /// The parsed EOF container, if any.
    eof: Option<Cow<'a, Eof>>,
    /// The instructions.
    insts: Vec<InstData>,
    /// `JUMPDEST` opcode map. `jumpdests[pc]` is `true` if `code[pc] == op::JUMPDEST`.
    jumpdests: BitVec,
    /// The [`SpecId`].
    pub(crate) spec_id: SpecId,
    /// Whether the bytecode contains dynamic jumps. Always false in EOF.
    has_dynamic_jumps: bool,
    /// Whether the bytecode may suspend execution.
    may_suspend: bool,
    /// Mapping from program counter to instruction.
    pc_to_inst: FxHashMap<u32, u32>,
    /// Mapping from EOF code section index to the list of instructions that call it.
    eof_called_by: Vec<Vec<Inst>>,
}

impl<'a> Bytecode<'a> {
    #[instrument(name = "new_bytecode", level = "debug", skip_all)]
    pub(crate) fn new(mut code: &'a [u8], eof: Option<Cow<'a, Eof>>, spec_id: SpecId) -> Self {
        if let Some(eof) = &eof {
            code = unsafe {
                std::slice::from_raw_parts(
                    eof.body.code_section.first().unwrap().as_ptr(),
                    eof.header.sum_code_sizes,
                )
            };
        }

        let is_eof = eof.is_some();

        let mut insts = Vec::with_capacity(code.len() + 8);
        // JUMPDEST analysis is not done in EOF.
        let mut jumpdests = if is_eof { BitVec::new() } else { BitVec::repeat(false, code.len()) };
        let mut pc_to_inst = FxHashMap::with_capacity_and_hasher(code.len(), Default::default());
        let op_infos = op_info_map(spec_id);
        for (inst, (pc, Opcode { opcode, immediate: _ })) in
            OpcodesIter::new(code, spec_id).with_pc().enumerate()
        {
            pc_to_inst.insert(pc as u32, inst as u32);

            if !is_eof && opcode == op::JUMPDEST {
                jumpdests.set(pc, true)
            }

            let data = 0;

            let mut flags = InstFlags::empty();
            let info = op_infos[opcode as usize];
            if info.is_unknown() {
                flags |= InstFlags::UNKNOWN;
            }
            if info.is_disabled() {
                flags |= InstFlags::DISABLED;
            }
            if info.is_eof_only() {
                flags |= InstFlags::EOF_ONLY;
            }
            let base_gas = info.base_gas();

            let section = Section::default();

            insts.push(InstData { opcode, flags, base_gas, data, pc: pc as u32, section });
        }

        let mut bytecode = Self {
            code,
            eof,
            insts,
            jumpdests,
            spec_id,
            has_dynamic_jumps: false,
            may_suspend: false,
            pc_to_inst,
            eof_called_by: vec![],
        };

        // Pad code to ensure there is at least one diverging instruction.
        // EOF enforces this, so there is no need to pad it ourselves.
        if !is_eof && bytecode.insts.last().map_or(true, |last| !last.is_diverging(false)) {
            bytecode.insts.push(InstData::new(op::STOP));
        }

        bytecode
    }

    /// Returns an iterator over the opcodes.
    #[inline]
    pub(crate) fn opcodes(&self) -> OpcodesIter<'a> {
        OpcodesIter::new(self.code, self.spec_id)
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

    /// Returns the opcode at the given instruction counter.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn opcode(&self, inst: Inst) -> Opcode<'_> {
        self.inst(inst).to_op_in(self)
    }

    /// Returns an iterator over the instructions.
    #[inline]
    pub(crate) fn iter_insts(
        &self,
    ) -> impl DoubleEndedIterator<Item = (usize, &InstData)> + Clone + '_ {
        self.iter_all_insts().filter(|(_, data)| !data.is_dead_code())
    }

    /// Returns an iterator over the instructions.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn iter_mut_insts(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (usize, &mut InstData)> + '_ {
        self.iter_mut_all_insts().filter(|(_, data)| !data.is_dead_code())
    }

    /// Returns an iterator over all the instructions, including dead code.
    #[inline]
    pub(crate) fn iter_all_insts(
        &self,
    ) -> impl DoubleEndedIterator<Item = (usize, &InstData)> + ExactSizeIterator + Clone + '_ {
        self.insts.iter().enumerate()
    }

    /// Returns an iterator over all the instructions, including dead code.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn iter_mut_all_insts(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (usize, &mut InstData)> + ExactSizeIterator + '_ {
        self.insts.iter_mut().enumerate()
    }

    /// Runs a list of analysis passes on the instructions.
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn analyze(&mut self) -> Result<()> {
        if !self.is_eof() {
            self.static_jump_analysis();
            // NOTE: `mark_dead_code` must run after `static_jump_analysis` as it can mark
            // unreachable `JUMPDEST`s as dead code.
            self.mark_dead_code();
        }

        self.calc_may_suspend();

        if self.is_eof() {
            self.calc_eof_called_by()?;
            self.eof_mark_jumpdests();
        }

        self.construct_sections();

        Ok(())
    }

    /// Mark `PUSH<N>` followed by `JUMP[I]` as `STATIC_JUMP` and resolve the target.
    #[instrument(name = "sj", level = "debug", skip_all)]
    fn static_jump_analysis(&mut self) {
        debug_assert!(!self.is_eof());

        for jump_inst in 0..self.insts.len() {
            let jump = &self.insts[jump_inst];
            let Some(push_inst) = jump_inst.checked_sub(1) else {
                if jump.is_legacy_jump() {
                    trace!(jump_inst, target=?None::<()>, "found jump");
                    self.has_dynamic_jumps = true;
                }
                continue;
            };

            let push = &self.insts[push_inst];
            if !(push.is_push() && jump.is_legacy_jump()) {
                if jump.is_legacy_jump() {
                    trace!(jump_inst, target=?None::<()>, "found jump");
                    self.has_dynamic_jumps = true;
                }
                continue;
            }

            let imm_opt = self.get_imm(push);
            if push.opcode != op::PUSH0 && imm_opt.is_none() {
                continue;
            }
            let imm = imm_opt.unwrap_or(&[]);
            self.insts[jump_inst].flags |= InstFlags::STATIC_JUMP;

            const USIZE_SIZE: usize = std::mem::size_of::<usize>();
            if imm.len() > USIZE_SIZE {
                trace!(jump_inst, "jump target too large");
                self.insts[jump_inst].flags |= InstFlags::INVALID_JUMP;
                continue;
            }

            let mut padded = [0; USIZE_SIZE];
            padded[USIZE_SIZE - imm.len()..].copy_from_slice(imm);
            let target_pc = usize::from_be_bytes(padded);
            if !self.is_valid_jump(target_pc) {
                trace!(jump_inst, target_pc, "invalid jump target");
                self.insts[jump_inst].flags |= InstFlags::INVALID_JUMP;
                continue;
            }

            self.insts[push_inst].flags |= InstFlags::SKIP_LOGIC;
            let target = self.pc_to_inst(target_pc);

            // Mark the `JUMPDEST` as reachable.
            debug_assert_eq!(
                self.insts[target],
                op::JUMPDEST,
                "is_valid_jump returned true for non-JUMPDEST: \
                 jump_inst={jump_inst} target_pc={target_pc} target={target}",
            );
            self.insts[target].data = 1;

            // Set the target on the `JUMP` instruction.
            trace!(jump_inst, target, "found jump");
            self.insts[jump_inst].data = target as u32;
        }
    }

    /// Mark `RJUMP*` targets with `EOF_JUMPDEST` flag.
    #[instrument(name = "eof_sj", level = "debug", skip_all)]
    fn eof_mark_jumpdests(&mut self) {
        debug_assert!(self.is_eof());

        for inst in 0..self.insts.len() {
            let data = self.inst(inst);
            if data.is_eof_jump() {
                for (_, pc) in self.iter_rjump_targets(data) {
                    let target_inst = self.pc_to_inst(pc);
                    self.inst_mut(target_inst).flags |= InstFlags::EOF_JUMPDEST;
                }
            }
        }
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
        debug_assert!(!self.is_eof());

        let mut iter = self.insts.iter_mut().enumerate();
        while let Some((i, data)) = iter.next() {
            if data.is_diverging(false) {
                let mut end = i;
                for (j, data) in &mut iter {
                    end = j;
                    if data.is_reachable_jumpdest(false, self.has_dynamic_jumps) {
                        break;
                    }
                    data.flags |= InstFlags::DEAD_CODE;
                }
                let start = i + 1;
                if end > start {
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
        let is_eof = self.is_eof();
        let may_suspend = self.iter_insts().any(|(_, data)| data.may_suspend(is_eof));
        self.may_suspend = may_suspend;
    }

    /// Constructs the sections in the bytecode.
    #[instrument(name = "sections", level = "debug", skip_all)]
    fn construct_sections(&mut self) {
        let mut analysis = SectionAnalysis::default();
        for inst in 0..self.insts.len() {
            if !self.inst(inst).is_dead_code() {
                analysis.process(self, inst);
            }
        }
        analysis.finish(self);
    }

    /// Constructs the sections in the bytecode.
    #[instrument(name = "sections", level = "debug", skip_all)]
    #[cfg(any())]
    fn construct_sections_default(&mut self) {
        for inst in &mut self.insts {
            let (inp, out) = inst.stack_io();
            let stack_diff = out as i16 - inp as i16;
            inst.section =
                Section { gas_cost: inst.base_gas as _, inputs: inp as _, max_growth: stack_diff }
        }
    }

    /// Calculates the list of instructions that call each EOF section.
    ///
    /// This is done to compute the `indirectbr` destinations of `RETF` instructions.
    #[instrument(name = "eof_called_by", level = "debug", skip_all)]
    fn calc_eof_called_by(&mut self) -> Result<()> {
        let code_sections_len = self.expect_eof().body.code_section.len();
        if code_sections_len <= 1 {
            return Ok(());
        }

        // First, collect all `CALLF` targets.
        let mut eof_called_by = vec![Vec::new(); code_sections_len];
        for (inst, data) in self.iter_all_insts() {
            if data.opcode == op::CALLF {
                let imm = self.get_imm(data).unwrap();
                let target_section = u16::from_be_bytes(imm.try_into().unwrap()) as usize;
                eof_called_by[target_section].push(inst);
            }
        }

        // Then, propagate `JUMPF` calls.
        const MAX_ITERATIONS: usize = 32;
        let mut any_progress = true;
        let mut i = 0usize;
        let first_section_inst = self.eof_section_inst(1);
        while any_progress && i < MAX_ITERATIONS {
            any_progress = false;

            for (_inst, data) in self.iter_all_insts().skip(first_section_inst) {
                if data.opcode == op::JUMPF {
                    let source_section = self.pc_to_eof_section(data.pc as usize);
                    debug_assert!(source_section != 0);

                    let imm = self.get_imm(data).unwrap();
                    let target_section = u16::from_be_bytes(imm.try_into().unwrap()) as usize;

                    let (source_section, target_section) =
                        get_two_mut(&mut eof_called_by, source_section, target_section);

                    for &source_call in &*source_section {
                        if !target_section.contains(&source_call) {
                            any_progress = true;
                            target_section.push(source_call);
                        }
                    }
                }
            }

            i += 1;
        }
        // TODO: Is this actually reachable?
        // If so, we should remove this error and handle this case properly by making all `CALLF`
        // reachable.
        ensure!(i < MAX_ITERATIONS, "`calc_eof_called_by` did not converge");
        self.eof_called_by = eof_called_by;
        Ok(())
    }

    /// Returns the list of instructions that call the given EOF section.
    pub(crate) fn eof_section_called_by(&self, section: usize) -> &[Inst] {
        &self.eof_called_by[section]
    }

    /// Returns the immediate value of the given instruction data, if any.
    /// Returns `None` if out of bounds too.
    pub(crate) fn get_imm(&self, data: &InstData) -> Option<&'a [u8]> {
        let mut imm_len = data.imm_len() as usize;
        if imm_len == 0 {
            return None;
        }
        let start = data.pc as usize + 1;
        if data.opcode == op::RJUMPV {
            imm_len += (*self.code.get(start)? as usize + 1) * 2;
        }
        self.code.get(start..start + imm_len)
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

    /// Returns `true` if the bytecode is EOF.
    pub(crate) fn is_eof(&self) -> bool {
        self.eof.is_some()
    }

    /// Returns `true` if the bytecode is small.
    ///
    /// This is arbitrarily chosen to speed up compilation for larger contracts.
    pub(crate) fn is_small(&self) -> bool {
        self.insts.len() < 2000
    }

    /// Returns `true` if the instruction is diverging.
    pub(crate) fn is_instr_diverging(&self, inst: Inst) -> bool {
        self.insts[inst].is_diverging(self.is_eof())
    }

    /// Converts a program counter (`self.code[pc]`) to an instruction (`self.inst(inst)`).
    #[inline]
    pub(crate) fn pc_to_inst(&self, pc: usize) -> usize {
        match self.pc_to_inst.get(&(pc as u32)) {
            Some(&inst) => inst as usize,
            None => panic!("pc out of bounds: {pc}"),
        }
    }

    /*
    /// Converts an instruction to a program counter.
    fn inst_to_pc(&self, inst: Inst) -> usize {
        self.insts[inst].pc as usize
    }
    */

    /// Returns the program counter of the given EOF section index.
    pub(crate) fn eof_section_pc(&self, section: usize) -> Inst {
        let code = &self.expect_eof().body.code_section;
        let first = code.first().unwrap().as_ptr();
        let section_ptr = code[section].as_ptr();
        section_ptr as usize - first as usize
    }

    /// Returns the first instruction of the given EOF section index.
    pub(crate) fn eof_section_inst(&self, section: usize) -> Inst {
        self.pc_to_inst(self.eof_section_pc(section))
    }

    pub(crate) fn pc_to_eof_section(&self, pc: usize) -> usize {
        (0..self.expect_eof().body.code_section.len())
            .rev()
            .find(|&section| pc >= self.eof_section_pc(section))
            .unwrap()
    }

    /// Iterates over the index and `RJUMP` target instructions of the given instruction.
    pub(crate) fn iter_rjump_target_insts(
        &self,
        data: &InstData,
    ) -> impl Iterator<Item = (usize, Inst)> + '_ {
        let from = data.pc;
        self.iter_rjump_targets(data).map(move |(i, pc)| {
            self.eof_assert_jump_in_bounds(from as usize, pc);
            (i, self.pc_to_inst(pc))
        })
    }

    /// Iterates over the index and `RJUMP` target PCs of the given instruction.
    pub(crate) fn iter_rjump_targets(
        &self,
        data: &InstData,
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        let opcode = data.opcode;
        let pc = data.pc;
        let imm = self.get_imm(data).unwrap();

        debug_assert!(InstData::new(opcode).is_eof_jump());
        if matches!(opcode, op::RJUMP | op::RJUMPI) {
            let offset = i16::from_be_bytes(imm.try_into().unwrap());
            let base_pc = pc + 3;
            let target_pc = (base_pc as usize).wrapping_add(offset as usize);
            return Either::Left(std::iter::once((0, target_pc)));
        }

        let max_index = imm[0] as usize;
        let base_pc = pc + 2 + (max_index as u32 + 1) * 2;
        Either::Right(imm[1..].chunks(2).enumerate().map(move |(i, chunk)| {
            debug_assert!(i <= max_index);
            debug_assert_eq!(chunk.len(), 2);
            let offset = i16::from_be_bytes(chunk.try_into().unwrap());
            let target_pc = (base_pc as usize).wrapping_add(offset as usize);
            (i, target_pc)
        }))
    }

    /// Asserts that the given jump target is in bounds.
    pub(crate) fn eof_assert_jump_in_bounds(&self, from: usize, to: usize) {
        debug_assert_eq!(
            self.pc_to_eof_section(from),
            self.pc_to_eof_section(to),
            "RJUMP* target out of bounds: {from} -> {to}"
        );
    }

    /// Returns the `Eof` container, panicking if it is not set.
    #[track_caller]
    #[inline]
    pub(crate) fn expect_eof(&self) -> &Eof {
        self.eof.as_deref().expect("EOF container not set")
    }

    /// Returns the name for a basic block.
    pub(crate) fn op_block_name(&self, mut inst: usize, name: &str) -> String {
        use std::fmt::Write;

        if inst == usize::MAX {
            return format!("entry.{name}");
        }
        let mut section = None;
        let data = self.inst(inst);
        if self.is_eof() {
            let section_index = self.pc_to_eof_section(data.pc as usize);
            section = Some(section_index);
            inst -= self.eof_section_inst(section_index);
        }

        let mut s = String::new();
        if let Some(section) = section {
            let _ = write!(s, "S{section}.");
        }
        let _ = write!(s, "OP{inst}.{}", data.to_op());
        if !name.is_empty() {
            let _ = write!(s, ".{name}");
        }
        s
    }
}

impl fmt::Display for Bytecode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header = format!("{:^6} | {:^6} | {:^80} | {}", "ic", "pc", "opcode", "instruction");
        writeln!(f, "{header}")?;
        writeln!(f, "{}", "-".repeat(header.len()))?;
        for (inst, (pc, opcode)) in self.opcodes().with_pc().enumerate() {
            let data = self.inst(inst);
            let opcode = opcode.to_string();
            writeln!(f, "{inst:>6} | {pc:>6} | {opcode:<80} | {data:?}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Bytecode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bytecode")
            .field("code", &hex::encode(self.code))
            .field("eof", &self.eof)
            .field("insts", &self.insts)
            .field("jumpdests", &hex::encode(bitvec_as_bytes(&self.jumpdests)))
            .field("spec_id", &self.spec_id)
            .field("has_dynamic_jumps", &self.has_dynamic_jumps)
            .field("may_suspend", &self.may_suspend)
            .finish()
    }
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
    /// Instruction-specific data:
    /// - if the instruction has immediate data, this is a packed offset+length into the bytecode;
    /// - `JUMP{,I} && STATIC_JUMP in kind`: the jump target, `Instr`;
    /// - `JUMPDEST`: `1` if the jump destination is reachable, `0` otherwise;
    /// - otherwise: no meaning.
    pub(crate) data: u32,
    /// The program counter, meaning `code[pc]` is this instruction's opcode.
    pub(crate) pc: u32,
    /// The section this instruction belongs to.
    pub(crate) section: Section,
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

impl fmt::Debug for InstData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InstData")
            .field("opcode", &self.to_op())
            .field("flags", &format_args!("{:?}", self.flags))
            .field("data", &self.data)
            .field("pc", &self.pc)
            .field("section", &self.section)
            .finish()
    }
}

impl InstData {
    /// Creates a new instruction data with the given opcode byte.
    /// Note that this may not be a valid instruction.
    #[inline]
    fn new(opcode: u8) -> Self {
        Self { opcode, ..Default::default() }
    }

    /// Returns the length of the immediate data of this instruction.
    #[inline]
    pub(crate) const fn imm_len(&self) -> u8 {
        min_imm_len(self.opcode)
    }

    /// Returns the number of input and output stack elements of this instruction.
    #[inline]
    pub(crate) fn stack_io(&self) -> (u8, u8) {
        let (mut inp, out) = stack_io(self.opcode);
        if self.is_legacy_static_jump()
            && !(self.opcode == op::JUMPI && self.flags.contains(InstFlags::INVALID_JUMP))
        {
            inp -= 1;
        }
        (inp, out)
    }

    /// Converts this instruction to a raw opcode. Note that the immediate data is not resolved.
    #[inline]
    pub(crate) const fn to_op(&self) -> Opcode<'static> {
        Opcode { opcode: self.opcode, immediate: None }
    }

    /// Converts this instruction to a raw opcode in the given bytecode.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn to_op_in<'a>(&self, bytecode: &Bytecode<'a>) -> Opcode<'a> {
        Opcode { opcode: self.opcode, immediate: bytecode.get_imm(self) }
    }

    /// Returns `true` if this instruction is a push instruction.
    #[inline]
    pub(crate) fn is_push(&self) -> bool {
        matches!(self.opcode, op::PUSH0..=op::PUSH32)
    }

    /// Returns `true` if this instruction is a jump instruction.
    #[inline]
    fn is_jump(&self, is_eof: bool) -> bool {
        if is_eof {
            self.is_eof_jump()
        } else {
            self.is_legacy_jump()
        }
    }

    /// Returns `true` if this instruction is an EOF jump instruction (`RJUMP`/`RJUMPI`/`RJUMPV`).
    #[inline]
    fn is_eof_jump(&self) -> bool {
        matches!(self.opcode, op::RJUMP | op::RJUMPI | op::RJUMPV)
    }

    /// Returns `true` if this instruction is a legacy jump instruction (`JUMP`/`JUMPI`).
    #[inline]
    pub(crate) fn is_legacy_jump(&self) -> bool {
        matches!(self.opcode, op::JUMP | op::JUMPI)
    }

    /// Returns `true` if this instruction is a legacy jump instruction (`JUMP`/`JUMPI`), and the
    /// target known statically.
    #[inline]
    pub(crate) fn is_legacy_static_jump(&self) -> bool {
        self.is_legacy_jump() && self.flags.contains(InstFlags::STATIC_JUMP)
    }

    /// Returns `true` if this instruction is a `JUMPDEST`.
    #[inline]
    pub(crate) const fn is_jumpdest(&self) -> bool {
        self.opcode == op::JUMPDEST
    }

    /// Returns `true` if this instruction is a reachable `JUMPDEST`.
    #[inline]
    pub(crate) const fn is_reachable_jumpdest(
        &self,
        is_eof: bool,
        has_dynamic_jumps: bool,
    ) -> bool {
        if is_eof {
            self.flags.contains(InstFlags::EOF_JUMPDEST)
        } else {
            self.is_jumpdest() && (has_dynamic_jumps || self.data == 1)
        }
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

    /// Returns `true` if we know that this instruction will branch or stop execution.
    #[inline]
    pub(crate) fn is_branching(&self, is_eof: bool) -> bool {
        self.is_jump(is_eof) || self.is_diverging(is_eof)
    }

    /// Returns `true` if we know that this instruction will stop execution.
    #[inline]
    pub(crate) fn is_diverging(&self, is_eof: bool) -> bool {
        #[cfg(test)]
        if self.opcode == TEST_SUSPEND {
            return false;
        }

        (self.opcode == op::JUMP && self.flags.contains(InstFlags::INVALID_JUMP))
            || self.flags.contains(InstFlags::DISABLED)
            || self.flags.contains(InstFlags::UNKNOWN)
            || matches!(self.opcode, op::STOP | op::RETURN | op::REVERT | op::INVALID)
            || (!is_eof && matches!(self.opcode, op::SELFDESTRUCT))
            || (is_eof && matches!(self.opcode, op::JUMPF | op::RETF | op::RETURNCONTRACT))
    }

    /// Returns `true` if this instruction may suspend execution.
    #[inline]
    pub(crate) const fn may_suspend(&self, is_eof: bool) -> bool {
        #[cfg(test)]
        if self.opcode == TEST_SUSPEND {
            return true;
        }

        if is_eof {
            matches!(
                self.opcode,
                op::EXTCALL | op::EXTDELEGATECALL | op::EXTSTATICCALL | op::EOFCREATE
            )
        } else {
            matches!(
                self.opcode,
                op::CALL
                    | op::CALLCODE
                    | op::DELEGATECALL
                    | op::STATICCALL
                    | op::CREATE
                    | op::CREATE2
            )
        }
    }
}

bitflags::bitflags! {
    /// [`InstrData`] flags.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) struct InstFlags: u8 {
        /// The `JUMP`/`JUMPI` target is known at compile time.
        /// This is implied for other jump instructions which are always static.
        const STATIC_JUMP = 1 << 0;
        /// The jump target is known to be invalid.
        /// Always returns [`InstructionResult::InvalidJump`] at runtime.
        const INVALID_JUMP = 1 << 1;
        /// The instruction is a target of at least one `RJUMP*` instruction.
        const EOF_JUMPDEST = 1 << 2;

        /// The instruction is disabled in this EVM version.
        /// Always returns [`InstructionResult::NotActivated`] at runtime.
        const DISABLED = 1 << 3;
        /// The instruction is unknown.
        /// Always returns [`InstructionResult::NotFound`] at runtime.
        const UNKNOWN = 1 << 4;
        /// The instruction is only enabled in EOF bytecodes.
        /// Always returns [`InstructionResult::EOFOpcodeDisabledInLegacy`] at runtime.
        const EOF_ONLY = 1 << 5;

        /// Skip generating instruction logic, but keep the gas calculation.
        const SKIP_LOGIC = 1 << 6;
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

fn get_two_mut<T>(sl: &mut [T], idx_1: usize, idx_2: usize) -> (&mut T, &mut T) {
    assert!(idx_1 != idx_2 && idx_1 < sl.len() && idx_2 < sl.len());
    let ptr = sl.as_mut_ptr();
    unsafe { (&mut *ptr.add(idx_1), &mut *ptr.add(idx_2)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suspend_is_free() {
        assert_eq!(op::OPCODE_INFO_JUMPTABLE[TEST_SUSPEND as usize], None);
    }
}
