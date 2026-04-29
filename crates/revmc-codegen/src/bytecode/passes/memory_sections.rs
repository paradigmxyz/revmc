use crate::bytecode::{Block, Bytecode, Inst, InstFlags};
use core::fmt;
use oxc_index::{IndexVec, index_vec};
use std::collections::VecDeque;

/// Known memory-size facts before a memory access.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MemorySection {
    /// The memory size known before the instruction executes.
    pub(crate) known_size: u64,
    /// The memory size required by this instruction, if known.
    pub(crate) required_size: u64,
}

impl fmt::Debug for MemorySection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str("MemorySection::EMPTY")
        } else {
            f.debug_struct("MemorySection")
                .field("known_size", &self.known_size)
                .field("required_size", &self.required_size)
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
    block_min_required_sizes: IndexVec<Block, u64>,
    block_entry_known_sizes: IndexVec<Block, Option<u64>>,
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
            block_min_required_sizes: index_vec![0; bytecode.cfg.blocks.len()],
            block_entry_known_sizes: index_vec![None; bytecode.cfg.blocks.len()],
            dynamic_jump_blocks,
            dynamic_jump_targets,
            sections: index_vec![MemorySection::default(); bytecode.insts.len()],
        }
    }

    /// Runs memory section analysis.
    pub(crate) fn run(mut self, bytecode: &Bytecode<'_>) -> IndexVec<Inst, MemorySection> {
        self.compute_block_min_required_sizes(bytecode);
        self.compute_block_entry_known_sizes(bytecode);
        self.save_sections(bytecode);
        self.sections
    }

    fn compute_block_min_required_sizes(&mut self, bytecode: &Bytecode<'_>) {
        for bid in bytecode.cfg.blocks.indices() {
            let block = &bytecode.cfg.blocks[bid];
            for inst in block.insts() {
                for (offset, len) in bytecode.const_memory_accesses(inst).into_iter().flatten() {
                    self.block_min_required_sizes[bid] = self.block_min_required_sizes[bid]
                        .max(min_memory_size_for_access(offset, len));
                }
            }
        }
    }

    fn compute_block_entry_known_sizes(&mut self, bytecode: &Bytecode<'_>) {
        let entry = Block::from_usize(0);
        self.block_entry_known_sizes[entry] = Some(0);

        let mut queue = VecDeque::new();
        queue.push_back(entry);

        while let Some(bid) = queue.pop_front() {
            if self.block_entry_known_sizes[bid].is_none() {
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
        if self.block_entry_known_sizes[bid] != new_entry_size {
            self.block_entry_known_sizes[bid] = new_entry_size;
            queue.push_back(bid);
        }
    }

    fn join_entry_size(&self, bytecode: &Bytecode<'_>, bid: Block) -> Option<u64> {
        let block = &bytecode.cfg.blocks[bid];
        let mut entry_size = (bid == Block::from_usize(0)).then_some(0);

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
        self.block_entry_known_sizes[bid]
            .map(|known_size| known_size.max(self.block_min_required_sizes[bid]))
    }

    fn save_sections(&mut self, bytecode: &Bytecode<'_>) {
        for bid in bytecode.cfg.blocks.indices() {
            let Some(mut known_size) = self.block_entry_known_sizes[bid] else { continue };
            let block = &bytecode.cfg.blocks[bid];

            for inst in block.insts() {
                for (offset, len) in bytecode.const_memory_accesses(inst).into_iter().flatten() {
                    let exact_required_size = exact_memory_size_for_access(offset, len);
                    let min_required_size = min_memory_size_for_access(offset, len);
                    trace!(
                        %bid,
                        %inst,
                        pc = bytecode.pc(inst),
                        opcode = %bytecode.inst(inst).to_op(),
                        ?offset,
                        ?len,
                        known_size,
                        min_required_size,
                        ?exact_required_size,
                        "memory access"
                    );
                    if known_size != 0 || exact_required_size.is_some_and(|size| size != 0) {
                        let section = &mut self.sections[inst];
                        if section.known_size == 0 && section.required_size == 0 {
                            section.known_size = known_size;
                        }
                        if let Some(exact_required_size) = exact_required_size {
                            section.required_size = section.required_size.max(exact_required_size);
                        }
                    }
                    known_size = known_size.max(min_required_size);
                }
            }
        }
    }
}

fn join_size(entry_size: Option<u64>, pred_exit_size: Option<u64>) -> Option<u64> {
    let Some(pred_exit_size) = pred_exit_size else { return entry_size };
    Some(entry_size.map_or(pred_exit_size, |entry_size| entry_size.min(pred_exit_size)))
}

#[inline]
fn exact_memory_size_for_access(offset: Option<u64>, len: Option<u64>) -> Option<u64> {
    if matches!(len, Some(0)) {
        return Some(0);
    }
    let size = offset?.saturating_add(len?);
    Some(round_memory_size(size))
}

fn min_memory_size_for_access(offset: Option<u64>, len: Option<u64>) -> u64 {
    match (offset, len) {
        (_, Some(0) | None) => 0,
        (Some(offset), Some(len)) => round_memory_size(offset.saturating_add(len)),
        (None, Some(len)) => round_memory_size(len),
    }
}

#[inline]
fn round_memory_size(size: u64) -> u64 {
    size.saturating_add(31) / 32 * 32
}

#[cfg(test)]
mod tests {
    use super::{MemorySection, exact_memory_size_for_access, min_memory_size_for_access};
    use crate::bytecode::{
        Bytecode, Inst,
        passes::block_analysis::tests::{analyze_asm, analyze_asm_spec},
    };
    use revm_bytecode::opcode as op;
    use revm_primitives::hardfork::SpecId;

    fn nth_inst(bytecode: &Bytecode<'_>, opcode: u8, n: usize) -> Inst {
        bytecode
            .iter_all_insts()
            .filter_map(|(inst, data)| (data.opcode == opcode).then_some(inst))
            .nth(n)
            .unwrap()
    }

    fn section(bytecode: &Bytecode<'_>, inst: Inst) -> MemorySection {
        bytecode.memory_section(inst)
    }

    fn assert_section(bytecode: &Bytecode<'_>, inst: Inst, known_size: u64, required_size: u64) {
        assert_eq!(section(bytecode, inst), MemorySection { known_size, required_size });
    }

    #[test]
    fn exact_memory_size_rounds_up_to_words() {
        assert_eq!(exact_memory_size_for_access(Some(0), Some(0)), Some(0));
        assert_eq!(exact_memory_size_for_access(Some(0), Some(1)), Some(32));
        assert_eq!(exact_memory_size_for_access(Some(1), Some(32)), Some(64));
        assert_eq!(exact_memory_size_for_access(Some(64), Some(32)), Some(96));
    }

    #[test]
    fn unknown_offset_or_len_has_no_exact_required_size() {
        assert_eq!(exact_memory_size_for_access(None, Some(32)), None);
        assert_eq!(exact_memory_size_for_access(Some(32), None), None);
        assert_eq!(exact_memory_size_for_access(None, None), None);
        assert_eq!(exact_memory_size_for_access(None, Some(0)), Some(0));
    }

    #[test]
    fn unknown_accesses_have_conservative_min_required_size() {
        assert_eq!(min_memory_size_for_access(None, Some(32)), 32);
        assert_eq!(min_memory_size_for_access(Some(32), None), 0);
        assert_eq!(min_memory_size_for_access(None, None), 0);
        assert_eq!(min_memory_size_for_access(None, Some(0)), 0);
    }

    #[test]
    fn same_block_accesses_use_previous_known_size() {
        let bytecode = analyze_asm("PUSH0 PUSH 64 MSTORE PUSH0 MLOAD STOP");
        let mstore = nth_inst(&bytecode, op::MSTORE, 0);
        let mload = nth_inst(&bytecode, op::MLOAD, 0);

        assert_section(&bytecode, mstore, 0, 96);
        assert_section(&bytecode, mload, 96, 32);
    }

    #[test]
    fn known_size_propagates_across_blocks() {
        let bytecode = analyze_asm(
            "
            PUSH0
            PUSH 64
            MSTORE
            PUSH %target
            JUMP
        target:
            JUMPDEST
            PUSH0
            MLOAD
            STOP
        ",
        );
        let mload = nth_inst(&bytecode, op::MLOAD, 0);
        assert_section(&bytecode, mload, 96, 32);
    }

    #[test]
    fn branch_join_uses_minimum_predecessor_size() {
        let bytecode = analyze_asm(
            "
            PUSH0
            PUSH 64
            MSTORE
            PUSH 1
            PUSH %large
            JUMPI
            PUSH %join
            JUMP
        large:
            JUMPDEST
            PUSH0
            PUSH 128
            MSTORE
        join:
            JUMPDEST
            PUSH0
            MLOAD
            STOP
        ",
        );
        let mload = nth_inst(&bytecode, op::MLOAD, 0);
        assert_section(&bytecode, mload, 96, 32);
    }

    #[test]
    fn loop_backedge_does_not_inflate_entry_known_size() {
        let bytecode = analyze_asm(
            "
        loop:
            JUMPDEST
            PUSH 128
            MLOAD
            PUSH0
            PUSH 49984
            MSTORE
            PUSH %loop
            JUMP
        ",
        );
        let mload = nth_inst(&bytecode, op::MLOAD, 0);
        let mstore = nth_inst(&bytecode, op::MSTORE, 0);

        assert_section(&bytecode, mload, 0, 160);
        assert_section(&bytecode, mstore, 160, 50016);
    }

    #[test]
    fn unknown_nonzero_access_does_not_create_exact_section() {
        let bytecode = analyze_asm(
            "
            PUSH0
            PUSH 64
            MSTORE
            PUSH0
            CALLDATALOAD
            PUSH0
            SWAP1
            MSTORE
            STOP
        ",
        );
        let dynamic_mstore = nth_inst(&bytecode, op::MSTORE, 1);

        assert_eq!(bytecode.const_memory_access(dynamic_mstore), Some((None, Some(32))));
        assert_section(&bytecode, dynamic_mstore, 96, 0);
    }

    #[test]
    fn unknown_offset_known_len_propagates_minimum_size() {
        let bytecode = analyze_asm(
            "
            PUSH0
            CALLDATALOAD
            PUSH0
            SWAP1
            MSTORE
            PUSH0
            MLOAD
            STOP
        ",
        );
        let mload = nth_inst(&bytecode, op::MLOAD, 0);

        assert_section(&bytecode, mload, 32, 32);
    }

    #[test]
    fn zero_len_access_is_exact_even_with_unknown_offset() {
        let bytecode = analyze_asm("PUSH0 PUSH0 PUSH0 CALLDATALOAD CALLDATACOPY STOP");
        let copy = nth_inst(&bytecode, op::CALLDATACOPY, 0);

        assert_eq!(bytecode.const_memory_access(copy), Some((None, Some(0))));
        assert_eq!(section(&bytecode, copy), MemorySection::default());
    }

    #[test]
    fn log0_uses_memory_offset_and_len() {
        let bytecode = analyze_asm("PUSH 32 PUSH 64 LOG0 STOP");
        let log0 = nth_inst(&bytecode, op::LOG0, 0);

        assert_eq!(bytecode.const_memory_access(log0), Some((Some(64), Some(32))));
        assert_section(&bytecode, log0, 0, 96);
    }

    #[test]
    fn call_tracks_input_and_output_memory_ranges() {
        let bytecode = analyze_asm(
            "
            PUSH 64     ; out len
            PUSH 96     ; out offset
            PUSH 32     ; in len
            PUSH 0      ; in offset
            PUSH 0      ; value
            PUSH 1      ; to
            PUSH 50000  ; gas
            CALL
            STOP
        ",
        );
        let call = nth_inst(&bytecode, op::CALL, 0);

        assert_eq!(
            bytecode.const_memory_accesses(call),
            [Some((Some(0), Some(32))), Some((Some(96), Some(64)))]
        );
        assert_section(&bytecode, call, 0, 160);
    }

    #[test]
    fn staticcall_tracks_input_and_output_memory_ranges() {
        let bytecode = analyze_asm(
            "
            PUSH 32     ; out len
            PUSH 128    ; out offset
            PUSH 64     ; in len
            PUSH 0      ; in offset
            PUSH 1      ; to
            PUSH 50000  ; gas
            STATICCALL
            STOP
        ",
        );
        let call = nth_inst(&bytecode, op::STATICCALL, 0);

        assert_eq!(
            bytecode.const_memory_accesses(call),
            [Some((Some(0), Some(64))), Some((Some(128), Some(32)))]
        );
        assert_section(&bytecode, call, 0, 160);
    }

    #[test]
    fn create_tracks_initcode_memory_range() {
        let bytecode = analyze_asm("PUSH 32 PUSH 64 PUSH0 CREATE STOP");
        let create = nth_inst(&bytecode, op::CREATE, 0);

        assert_eq!(bytecode.const_memory_access(create), Some((Some(64), Some(32))));
        assert_section(&bytecode, create, 0, 96);
    }

    #[test]
    fn disabled_memory_opcode_is_ignored() {
        let bytecode = analyze_asm_spec("PUSH 32 PUSH0 PUSH0 MCOPY STOP", SpecId::SHANGHAI);
        let mcopy = nth_inst(&bytecode, op::MCOPY, 0);

        assert_eq!(bytecode.const_memory_access(mcopy), None);
        assert_eq!(section(&bytecode, mcopy), MemorySection::default());
    }
}
