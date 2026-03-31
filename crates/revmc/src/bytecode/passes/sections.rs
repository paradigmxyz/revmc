use crate::bytecode::{Bytecode, Inst};
use core::fmt;

/// A gas section tracks the total base gas cost of a sequence of instructions.
///
/// Gas sections end at instructions that require `gasleft` (e.g. `GAS`, `SSTORE`),
/// branching instructions, or suspending instructions.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GasSection {
    /// The total base gas cost of all instructions in the section.
    pub(crate) gas_cost: u32,
}

impl fmt::Debug for GasSection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str("GasSection::EMPTY")
        } else {
            f.debug_struct("GasSection").field("gas_cost", &self.gas_cost).finish()
        }
    }
}

impl GasSection {
    /// Returns `true` if the section is empty.
    #[inline]
    pub(crate) fn is_empty(self) -> bool {
        self == Self::default()
    }
}

/// A stack section tracks stack height requirements for a sequence of instructions.
///
/// Stack sections end at branching or suspending instructions, but NOT at instructions
/// that merely require `gasleft` — those only end gas sections.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StackSection {
    /// The stack height required to execute the section.
    pub(crate) inputs: u16,
    /// The maximum stack height growth relative to the stack height at section start.
    pub(crate) max_growth: i16,
}

impl fmt::Debug for StackSection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str("StackSection::EMPTY")
        } else {
            f.debug_struct("StackSection")
                .field("inputs", &self.inputs)
                .field("max_growth", &self.max_growth)
                .finish()
        }
    }
}

impl StackSection {
    /// Returns `true` if the section is empty.
    #[inline]
    pub(crate) fn is_empty(self) -> bool {
        self == Self::default()
    }
}

/// Gas section analysis state.
struct GasSectionAnalysis {
    gas_cost: u64,
    start_inst: Inst,
}

impl Default for GasSectionAnalysis {
    fn default() -> Self {
        Self { gas_cost: 0, start_inst: Inst::from_usize(0) }
    }
}

impl GasSectionAnalysis {
    /// Saves the current gas section to the bytecode.
    fn save_to(&self, bytecode: &mut Bytecode<'_>, next_section_inst: Inst) {
        if self.start_inst >= bytecode.insts.len_idx() {
            return;
        }
        let section = self.section();
        if !section.is_empty() {
            trace!(
                inst = %self.start_inst,
                len = next_section_inst.index() - self.start_inst.index(),
                ?section,
                "saving gas"
            );
            let mut insts = bytecode.insts.raw[self.start_inst.index()..].iter_mut();
            if let Some(inst) = insts.find(|inst| !inst.is_dead_code()) {
                inst.gas_section = section;
            }
        }
    }

    /// Resets the analysis for a new section.
    fn reset(&mut self, inst: Inst) {
        *self = Self { start_inst: inst, ..Default::default() };
    }

    /// Returns the current gas section.
    fn section(&self) -> GasSection {
        GasSection { gas_cost: self.gas_cost.try_into().unwrap_or(u32::MAX) }
    }
}

/// Stack section analysis state.
struct StackSectionAnalysis {
    inputs: i32,
    diff: i32,
    max_growth: i32,
    start_inst: Inst,
}

impl Default for StackSectionAnalysis {
    fn default() -> Self {
        Self { inputs: 0, diff: 0, max_growth: 0, start_inst: Inst::from_usize(0) }
    }
}

impl StackSectionAnalysis {
    /// Saves the current stack section to the bytecode.
    fn save_to(&self, bytecode: &mut Bytecode<'_>, next_section_inst: Inst) {
        if self.start_inst >= bytecode.insts.len_idx() {
            return;
        }
        let section = self.section();
        if !section.is_empty() {
            trace!(
                inst = %self.start_inst,
                len = next_section_inst.index() - self.start_inst.index(),
                ?section,
                "saving stack"
            );
            let mut insts = bytecode.insts.raw[self.start_inst.index()..].iter_mut();
            if let Some(inst) = insts.find(|inst| !inst.is_dead_code()) {
                inst.stack_section = section;
            }
        }
    }

    /// Resets the analysis for a new section.
    fn reset(&mut self, inst: Inst) {
        *self = Self { start_inst: inst, ..Default::default() };
    }

    /// Returns the current stack section.
    fn section(&self) -> StackSection {
        StackSection {
            inputs: self.inputs.try_into().unwrap_or(u16::MAX),
            max_growth: self.max_growth.try_into().unwrap_or(i16::MAX),
        }
    }
}

/// Instruction section analysis, tracking gas and stack sections separately.
#[derive(Default)]
pub(crate) struct SectionsAnalysis {
    gas: GasSectionAnalysis,
    stack: StackSectionAnalysis,
}

impl SectionsAnalysis {
    /// Process a single instruction.
    pub(crate) fn process(&mut self, bytecode: &mut Bytecode<'_>, inst: Inst) {
        // JUMPDEST starts both gas and stack sections.
        if bytecode.inst(inst).is_reachable_jumpdest(bytecode.has_dynamic_jumps()) {
            self.gas.save_to(bytecode, inst);
            self.gas.reset(inst);
            self.stack.save_to(bytecode, inst);
            self.stack.reset(inst);
        }

        let data = bytecode.inst(inst);
        let (inp, out) = data.stack_io_raw();
        let stack_diff = out as i32 - inp as i32;
        self.stack.inputs = self.stack.inputs.max(inp as i32 - self.stack.diff);
        self.stack.diff += stack_diff;
        self.stack.max_growth = self.stack.max_growth.max(self.stack.diff);

        self.gas.gas_cost += data.base_gas as u64;

        let data = bytecode.inst(inst);

        // Instructions that require `gasleft` end only the gas section.
        // Branching and suspending instructions end both sections.
        if data.may_suspend() || data.is_branching() {
            let next = inst + 1;
            self.gas.save_to(bytecode, next);
            self.gas.reset(next);
            self.stack.save_to(bytecode, next);
            self.stack.reset(next);
        } else if data.requires_gasleft(bytecode.spec_id) {
            let next = inst + 1;
            self.gas.save_to(bytecode, next);
            self.gas.reset(next);
        }
    }

    /// Finishes the analysis.
    pub(crate) fn finish(self, bytecode: &mut Bytecode<'_>) {
        let last = bytecode.insts.len_idx() - 1;
        self.gas.save_to(bytecode, last);
        self.stack.save_to(bytecode, last);
        if enabled!(tracing::Level::DEBUG) {
            let mut max_len = 0usize;
            let mut current = Inst::from_usize(0);
            let mut count = 0usize;
            for (inst, data) in bytecode.iter_insts() {
                if data.stack_section.is_empty() {
                    continue;
                }
                let len = inst.index() - current.index();
                max_len = max_len.max(len);
                current = inst;
                count += 1;
            }
            debug!(count, max_len, "sections");
        }
    }
}
