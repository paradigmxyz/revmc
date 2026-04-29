use crate::bytecode::{Block, Bytecode, Inst};
use core::fmt;
use oxc_index::{IndexVec, index_vec};
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
    sections: IndexVec<Inst, MemorySection>,
}

impl MemorySectionAnalysis {
    pub(crate) fn new(bytecode: &Bytecode<'_>) -> Self {
        Self {
            block_memory_sizes: index_vec![0; bytecode.cfg.blocks.len()],
            block_entry_sizes: index_vec![None; bytecode.cfg.blocks.len()],
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
                        self.block_memory_sizes[bid].max(memory_size(offset, len));
                }
            }
        }
    }

    fn compute_block_entry_sizes(&mut self, bytecode: &Bytecode<'_>) {
        let entry = Block::from_usize(0);
        self.block_entry_sizes[entry] = Some(0);

        let mut queue = VecDeque::new();
        queue.push_back(entry);

        if bytecode.has_dynamic_jumps() {
            for bid in bytecode.cfg.blocks.indices() {
                let block = &bytecode.cfg.blocks[bid];
                if bytecode.inst(block.insts.start).is_reachable_jumpdest(true)
                    && self.block_entry_sizes[bid].is_none()
                {
                    self.block_entry_sizes[bid] = Some(0);
                    queue.push_back(bid);
                }
            }
        }

        while let Some(bid) = queue.pop_front() {
            if self.block_entry_sizes[bid].is_none() {
                continue;
            }

            for &succ in &bytecode.cfg.blocks[bid].succs {
                let new_entry_size = self.join_entry_size(bytecode, succ);
                if self.block_entry_sizes[succ] != new_entry_size {
                    self.block_entry_sizes[succ] = new_entry_size;
                    queue.push_back(succ);
                }
            }
        }
    }

    fn join_entry_size(&self, bytecode: &Bytecode<'_>, bid: Block) -> Option<u64> {
        let block = &bytecode.cfg.blocks[bid];
        let mut entry_size = if bytecode.has_dynamic_jumps()
            && bytecode.inst(block.insts.start).is_reachable_jumpdest(true)
        {
            Some(0)
        } else {
            None
        };

        for &pred in &block.preds {
            if let Some(pred_exit_size) = self.block_entry_sizes[pred]
                .map(|entry_size| entry_size.max(self.block_memory_sizes[pred]))
            {
                entry_size = Some(
                    entry_size.map_or(pred_exit_size, |entry_size| entry_size.min(pred_exit_size)),
                );
            }
        }

        entry_size
    }

    fn save_sections(&mut self, bytecode: &Bytecode<'_>) {
        for bid in bytecode.cfg.blocks.indices() {
            let Some(min_memory_size) = self.block_entry_sizes[bid] else { continue };
            let memory_size = self.block_memory_sizes[bid];
            if min_memory_size == 0 && memory_size == 0 {
                continue;
            }
            let inst = bytecode.cfg.blocks[bid].insts.start;
            self.sections[inst] = MemorySection { min_memory_size, memory_size };
        }
    }
}

#[inline]
fn memory_size(offset: Option<u64>, len: Option<u64>) -> u64 {
    offset.unwrap_or(0).saturating_add(len.unwrap_or(0)).saturating_add(31) / 32 * 32
}
