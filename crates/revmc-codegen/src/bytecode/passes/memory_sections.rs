use crate::bytecode::{Block, Bytecode, Inst, InstFlags};
use core::fmt;
use oxc_index::{IndexVec, index_vec};
use revm_bytecode::opcode as op;
use std::collections::VecDeque;

/// A memory section tracks known memory-size facts for a basic block.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MemorySection {
    /// The memory size known at section entry.
    pub(crate) min_memory_size: u64,
    /// The maximum constant memory size required by the section.
    pub(crate) memory_size: u64,
}

impl fmt::Debug for MemorySection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str("MemorySection::EMPTY")
        } else {
            f.debug_struct("MemorySection")
                .field("min_memory_size", &self.min_memory_size)
                .field("memory_size", &self.memory_size)
                .finish()
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
    block_memory_sizes: IndexVec<Block, u64>,
    block_entry_sizes: IndexVec<Block, Option<u64>>,
    dynamic_jump_blocks: Vec<Block>,
    dynamic_jump_targets: Vec<Block>,
    sections: IndexVec<Inst, MemorySection>,
}

impl MemorySectionAnalysis {
    pub(crate) fn new(bytecode: &Bytecode<'_>) -> Self {
        let mut dynamic_jump_blocks = Vec::new();
        let mut dynamic_jump_targets = Vec::new();
        if bytecode.has_dynamic_jumps() {
            for bid in bytecode.cfg.blocks.indices() {
                let block = &bytecode.cfg.blocks[bid];
                let head = bytecode.inst(block.insts.start);
                if head.is_reachable_jumpdest(true) {
                    dynamic_jump_targets.push(bid);
                }

                let term = bytecode.inst(block.terminator());
                if term.is_jump()
                    && !term.is_static_jump()
                    && !term.flags.contains(InstFlags::INVALID_JUMP)
                {
                    dynamic_jump_blocks.push(bid);
                }
            }
        }

        Self {
            block_memory_sizes: index_vec![0; bytecode.cfg.blocks.len()],
            block_entry_sizes: index_vec![None; bytecode.cfg.blocks.len()],
            dynamic_jump_blocks,
            dynamic_jump_targets,
            sections: index_vec![MemorySection::default(); bytecode.insts.len()],
        }
    }

    /// Runs memory section analysis.
    pub(crate) fn run(mut self, bytecode: &Bytecode<'_>) -> IndexVec<Inst, MemorySection> {
        self.compute_block_memory_sizes(bytecode);
        self.compute_block_entry_sizes(bytecode);
        self.save_sections(bytecode);
        self.sections
    }

    fn compute_block_memory_sizes(&mut self, bytecode: &Bytecode<'_>) {
        for bid in bytecode.cfg.blocks.indices() {
            let block = &bytecode.cfg.blocks[bid];
            for inst in block.insts() {
                if let Some((offset, len)) = bytecode.const_memory_access(inst) {
                    self.block_memory_sizes[bid] =
                        self.block_memory_sizes[bid].max(memory_size_for_access(offset, len));
                }
            }
        }
    }

    fn compute_block_entry_sizes(&mut self, bytecode: &Bytecode<'_>) {
        let entry = Block::from_usize(0);
        self.block_entry_sizes[entry] = Some(0);

        let mut queue = VecDeque::new();
        queue.push_back(entry);

        while let Some(bid) = queue.pop_front() {
            if self.block_entry_sizes[bid].is_none() {
                continue;
            }

            for &succ in &bytecode.cfg.blocks[bid].succs {
                self.update_entry_size(bytecode, succ, &mut queue);
            }
            if self.dynamic_jump_blocks.contains(&bid) {
                let targets = self.dynamic_jump_targets.clone();
                for succ in targets {
                    self.update_entry_size(bytecode, succ, &mut queue);
                }
            }
        }
    }

    fn update_entry_size(
        &mut self,
        bytecode: &Bytecode<'_>,
        bid: Block,
        queue: &mut VecDeque<Block>,
    ) {
        let new_entry_size = self.join_entry_size(bytecode, bid);
        if self.block_entry_sizes[bid] != new_entry_size {
            self.block_entry_sizes[bid] = new_entry_size;
            queue.push_back(bid);
        }
    }

    fn join_entry_size(&self, bytecode: &Bytecode<'_>, bid: Block) -> Option<u64> {
        let block = &bytecode.cfg.blocks[bid];
        let mut entry_size = None;

        for &pred in &block.preds {
            entry_size = join_size(entry_size, self.block_exit_size(pred));
        }

        if bytecode.inst(block.insts.start).is_reachable_jumpdest(true) {
            for &pred in &self.dynamic_jump_blocks {
                entry_size = join_size(entry_size, self.block_exit_size(pred));
            }
        }

        entry_size
    }

    fn block_exit_size(&self, bid: Block) -> Option<u64> {
        self.block_entry_sizes[bid].map(|entry_size| entry_size.max(self.block_memory_sizes[bid]))
    }

    fn save_sections(&mut self, bytecode: &Bytecode<'_>) {
        for bid in bytecode.cfg.blocks.indices() {
            let Some(min_memory_size) = self.block_entry_sizes[bid] else { continue };
            let memory_size = self.block_memory_sizes[bid];
            let block = &bytecode.cfg.blocks[bid];

            for inst in block.insts() {
                let Some((offset, len)) = bytecode.const_memory_access(inst) else { continue };
                trace!(
                    %bid,
                    %inst,
                    pc = bytecode.pc(inst),
                    opcode = opcode_name(bytecode.inst(inst).opcode),
                    ?offset,
                    ?len,
                    min_memory_size,
                    block_memory_size = memory_size,
                    required_memory_size = memory_size_for_access(offset, len),
                    "memory access"
                );
            }

            if min_memory_size == 0 && memory_size == 0 {
                continue;
            }
            self.sections[block.insts.start] = MemorySection { min_memory_size, memory_size };
        }
    }
}

fn join_size(entry_size: Option<u64>, pred_exit_size: Option<u64>) -> Option<u64> {
    let Some(pred_exit_size) = pred_exit_size else { return entry_size };
    Some(entry_size.map_or(pred_exit_size, |entry_size| entry_size.min(pred_exit_size)))
}

#[inline]
fn memory_size_for_access(offset: Option<u64>, len: Option<u64>) -> u64 {
    offset.unwrap_or(0).saturating_add(len.unwrap_or(0)).saturating_add(31) / 32 * 32
}

fn opcode_name(opcode: u8) -> &'static str {
    match opcode {
        op::KECCAK256 => "KECCAK256",
        op::MLOAD => "MLOAD",
        op::MSTORE => "MSTORE",
        op::MSTORE8 => "MSTORE8",
        op::CALLDATACOPY => "CALLDATACOPY",
        op::CODECOPY => "CODECOPY",
        op::EXTCODECOPY => "EXTCODECOPY",
        op::RETURNDATACOPY => "RETURNDATACOPY",
        op::MCOPY => "MCOPY",
        op::LOG0 => "LOG0",
        op::LOG1 => "LOG1",
        op::LOG2 => "LOG2",
        op::LOG3 => "LOG3",
        op::LOG4 => "LOG4",
        op::RETURN => "RETURN",
        op::REVERT => "REVERT",
        _ => "UNKNOWN",
    }
}
