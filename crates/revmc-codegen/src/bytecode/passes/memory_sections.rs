use crate::bytecode::{Bytecode, Inst};
use core::fmt;
use oxc_index::{IndexVec, index_vec};

/// A memory section tracks the maximum constant memory size required by a sequence of instructions.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MemorySection {
    /// The maximum constant memory size required by the section.
    pub(crate) memory_size: u64,
}

impl fmt::Debug for MemorySection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str("MemorySection::EMPTY")
        } else {
            f.debug_struct("MemorySection").field("memory_size", &self.memory_size).finish()
        }
    }
}

impl MemorySection {
    /// Returns `true` if the section is empty.
    #[inline]
    pub(crate) fn is_empty(self) -> bool {
        self == Self::default()
    }
}

/// Memory section analysis state.
pub(crate) struct MemorySectionAnalysis {
    memory_size: u64,
    start_inst: Inst,
    sections: IndexVec<Inst, MemorySection>,
}

impl MemorySectionAnalysis {
    pub(crate) fn new(bytecode: &Bytecode<'_>) -> Self {
        Self {
            memory_size: 0,
            start_inst: Inst::from_usize(0),
            sections: index_vec![MemorySection::default(); bytecode.insts.len()],
        }
    }

    /// Runs memory section analysis.
    pub(crate) fn run(mut self, bytecode: &Bytecode<'_>) -> IndexVec<Inst, MemorySection> {
        for inst in bytecode.insts.indices() {
            if !bytecode.inst(inst).is_dead_code() {
                self.process_inst(bytecode, inst);
            }
        }
        self.finish(bytecode)
    }

    /// Processes a single instruction.
    fn process_inst(&mut self, bytecode: &Bytecode<'_>, inst: Inst) {
        if bytecode.inst(inst).is_reachable_jumpdest(bytecode.has_dynamic_jumps()) {
            self.save_reset(bytecode, inst);
        }

        let data = bytecode.inst(inst);
        if let Some((offset, len)) = bytecode.const_memory_access(inst) {
            self.process_access(offset, len);
        }

        if data.may_suspend() || data.is_branching() {
            self.save_reset(bytecode, inst + 1);
        }
    }

    /// Accumulates a constant memory access.
    #[inline]
    fn process_access(&mut self, offset: Option<u64>, len: Option<u64>) {
        let memory_size = offset.unwrap_or(0).saturating_add(len.unwrap_or(0));
        let memory_size = memory_size.saturating_add(31) / 32 * 32;
        self.memory_size = self.memory_size.max(memory_size);
    }

    fn save_reset(&mut self, bytecode: &Bytecode<'_>, next_section_inst: Inst) {
        self.save(bytecode, next_section_inst);
        self.memory_size = 0;
        self.start_inst = next_section_inst;
    }

    /// Saves the current memory section.
    fn save(&mut self, bytecode: &Bytecode<'_>, next_section_inst: Inst) {
        if self.start_inst >= bytecode.insts.len_idx() || self.memory_size == 0 {
            return;
        }
        let _ = next_section_inst;
        let mut insts = bytecode.insts[self.start_inst..].iter_enumerated();
        if let Some((inst, _)) = insts.find(|(_, inst)| !inst.is_dead_code()) {
            self.sections[inst] = MemorySection { memory_size: self.memory_size };
        }
    }

    /// Finishes the analysis.
    fn finish(mut self, bytecode: &Bytecode<'_>) -> IndexVec<Inst, MemorySection> {
        let last = bytecode.insts.len_idx() - 1;
        self.save(bytecode, last);
        self.sections
    }
}
