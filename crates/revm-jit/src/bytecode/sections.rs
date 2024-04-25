use super::Bytecode;
use core::fmt;
use revm_interpreter::opcode as op;

// TODO: integrate stack checks.

/// A section is a sequence of instructions that are executed sequentially without any jumps or
/// branches.
///
/// This would be better named "block" but it's already used in the context of the basic block
/// analysis.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Section {
    /// The total base gas cost of all instructions in the section.
    pub(crate) gas_cost: u32,
    /// The stack height required to execute the section.
    pub(crate) inputs: u16,
    /// The maximum stack height growth relative to the stack height at section start.
    pub(crate) max_growth: i16,
}

impl fmt::Debug for Section {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str("Section::EMPTY")
        } else {
            f.debug_struct("Section")
                .field("gas_cost", &self.gas_cost)
                .field("stack_req", &self.inputs)
                .field("stack_max_growth", &self.max_growth)
                .finish()
        }
    }
}

impl Section {
    /// Returns `true` if the section is empty.
    #[inline]
    pub(crate) fn is_empty(self) -> bool {
        self == Self::default()
    }
}

/// Instruction section analysis.
#[derive(Default)]
pub(crate) struct SectionAnalysis {
    inputs: i32,
    diff: i32,
    max_growth: i32,

    gas_cost: u64,
    start_inst: usize,
}

impl SectionAnalysis {
    /// Process a single instruction.
    pub(crate) fn process(&mut self, bytecode: &mut Bytecode<'_>, inst: usize) {
        // JUMPDEST starts a section.
        if bytecode.inst(inst).is_jumpdest() {
            self.save_to(bytecode);
            self.reset(inst);
        }

        let data = bytecode.inst(inst);
        let (inp, out) = data.stack_io();
        let stack_diff = out as i32 - inp as i32;
        self.inputs = self.inputs.max(inp as i32 - self.diff);
        self.diff += stack_diff;
        self.max_growth = self.max_growth.max(self.diff);

        self.gas_cost += data.static_gas().unwrap_or(0) as u64;

        // Branching instructions end a section, starting a new one on the next instruction, if any.
        if data.opcode == op::GAS || data.is_branching(bytecode.is_eof()) || data.will_suspend() {
            self.save_to(bytecode);
            self.reset(inst + 1);
        }
    }

    /// Finishes the analysis.
    pub(crate) fn finish(self, bytecode: &mut Bytecode<'_>) {
        self.save_to(bytecode);
    }

    /// Saves the current section to the bytecode.
    fn save_to(&self, bytecode: &mut Bytecode<'_>) {
        if self.start_inst >= bytecode.insts.len() {
            return;
        }
        bytecode.inst_mut(self.start_inst).section = self.section();
    }

    /// Starts a new section.
    fn reset(&mut self, inst: usize) {
        *self = Self { start_inst: inst, ..Default::default() };
    }

    /// Returns the current section.
    fn section(&self) -> Section {
        Section {
            gas_cost: self.gas_cost.try_into().unwrap_or(u32::MAX),
            inputs: self.inputs.try_into().unwrap_or(u16::MAX),
            max_growth: self.max_growth.try_into().unwrap_or(i16::MAX),
        }
    }
}
