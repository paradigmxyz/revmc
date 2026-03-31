//! Analysis and optimization passes over EVM bytecode.

pub(crate) mod block_analysis;
pub(crate) use block_analysis::{Cfg, Snapshots};

mod const_fold;

mod dedup;

mod sections;
pub(crate) use sections::{GasSection, SectionsAnalysis, StackSection};
