use super::Bytecode;
use core::fmt;

// TODO: Separate gas sections from stack length sections.
// E.g. `GAS` should stop only a gas section because it requires `gasleft`, and execution will
// continue with the next instruction.

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
        let is_eof = bytecode.is_eof();

        // JUMPDEST starts a section.
        if bytecode.inst(inst).is_reachable_jumpdest(is_eof, bytecode.has_dynamic_jumps()) {
            self.save_to(bytecode, inst);
            self.reset(inst);
        }

        let data = bytecode.inst(inst);
        let (inp, out) = data.stack_io();
        let stack_diff = out as i32 - inp as i32;
        self.inputs = self.inputs.max(inp as i32 - self.diff);
        self.diff += stack_diff;
        self.max_growth = self.max_growth.max(self.diff);

        self.gas_cost += data.base_gas as u64;

        // Instructions that require `gasleft` and branching instructions end a section, starting a
        // new one on the next instruction, if any.
        if (!is_eof && data.requires_gasleft(bytecode.spec_id))
            || data.may_suspend(is_eof)
            || data.is_branching(is_eof)
        {
            let next = inst + 1;
            self.save_to(bytecode, next);
            self.reset(next);
        }
    }

    /// Finishes the analysis.
    pub(crate) fn finish(self, bytecode: &mut Bytecode<'_>) {
        self.save_to(bytecode, bytecode.insts.len() - 1);
        if enabled!(tracing::Level::DEBUG) {
            let mut max_len = 0;
            let mut current = 0;
            let mut count = 0usize;
            for (inst, data) in bytecode.iter_insts() {
                if data.section.is_empty() {
                    continue;
                }
                let len = inst - current;
                max_len = max_len.max(len);
                current = inst;
                count += 1;
            }
            debug!(count, max_len, "sections");
        }
    }

    /// Saves the current section to the bytecode.
    fn save_to(&self, bytecode: &mut Bytecode<'_>, next_section_inst: usize) {
        if self.start_inst >= bytecode.insts.len() {
            return;
        }
        let section = self.section();
        if !section.is_empty() {
            trace!(
                inst = self.start_inst,
                len = next_section_inst - self.start_inst,
                ?section,
                "saving"
            );
            let mut insts = bytecode.insts[self.start_inst..].iter_mut();
            if let Some(inst) = insts.find(|inst| !inst.is_dead_code()) {
                inst.section = section;
            }
        }
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
