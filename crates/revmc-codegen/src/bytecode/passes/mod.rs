//! Analysis and optimization passes over EVM bytecode.

pub(crate) mod block_analysis;
pub(crate) use block_analysis::{Block, Cfg, Snapshots};

mod const_fold;

mod dead_store_elim;

mod dedup;

mod sections;
pub(crate) use sections::{
    GasSection, MemorySection, SectionsAnalysis, StackSection, StackSectionAnalysis,
};
