//! This module provides the [`CommentWriter`] which makes it possible to add comments to the
//! written cranelift IR.
//!
//! Modified from [`rustc_codegen_cranelift`](https://github.com/rust-lang/rustc_codegen_cranelift/blob/07633821ed63360d4d7464998c29f4f588930a03/src/pretty_clif.rs).

#![allow(dead_code)]

use cranelift::codegen::{
    entity::SecondaryMap,
    ir::{entities::AnyEntity, Block, Fact, Function, Inst, Value},
    isa::TargetIsa,
    write::{FuncWriter, PlainWriter},
};
use std::{
    collections::{hash_map::Entry, HashMap},
    fmt,
    io::Write,
    path::Path,
};

#[derive(Clone, Debug)]
pub(crate) struct CommentWriter {
    enabled: bool,
    global_comments: Vec<String>,
    entity_comments: HashMap<AnyEntity, String>,
}

impl Default for CommentWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl CommentWriter {
    pub(crate) fn new() -> Self {
        Self {
            enabled: should_write_ir(),
            global_comments: Vec::new(),
            entity_comments: HashMap::new(),
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn clear(&mut self) {
        self.global_comments.clear();
        self.entity_comments.clear();
    }

    pub(crate) fn add_global_comment<S: Into<String>>(&mut self, comment: S) {
        debug_assert!(self.enabled);
        self.global_comments.push(comment.into());
    }

    pub(crate) fn add_comment<S: Into<String> + AsRef<str>, E: Into<AnyEntity>>(
        &mut self,
        entity: E,
        comment: S,
    ) {
        debug_assert!(self.enabled);

        match self.entity_comments.entry(entity.into()) {
            Entry::Occupied(mut occ) => {
                occ.get_mut().push('\n');
                occ.get_mut().push_str(comment.as_ref());
            }
            Entry::Vacant(vac) => {
                vac.insert(comment.into());
            }
        }
    }
}

impl FuncWriter for &'_ CommentWriter {
    fn write_preamble(
        &mut self,
        w: &mut dyn fmt::Write,
        func: &Function,
    ) -> Result<bool, fmt::Error> {
        for comment in &self.global_comments {
            if !comment.is_empty() {
                writeln!(w, "; {comment}")?;
            } else {
                writeln!(w)?;
            }
        }
        if !self.global_comments.is_empty() {
            writeln!(w)?;
        }

        self.super_preamble(w, func)
    }

    fn write_entity_definition(
        &mut self,
        w: &mut dyn fmt::Write,
        _func: &Function,
        entity: AnyEntity,
        value: &dyn fmt::Display,
        maybe_fact: Option<&Fact>,
    ) -> fmt::Result {
        if let Some(fact) = maybe_fact {
            write!(w, "    {entity} ! {fact} = {value}")?;
        } else {
            write!(w, "    {entity} = {value}")?;
        }

        if let Some(comment) = self.entity_comments.get(&entity) {
            writeln!(w, " ; {}", comment.replace('\n', "\n; "))
        } else {
            writeln!(w)
        }
    }

    fn write_block_header(
        &mut self,
        w: &mut dyn fmt::Write,
        func: &Function,
        block: Block,
        indent: usize,
    ) -> fmt::Result {
        PlainWriter.write_block_header(w, func, block, indent)
    }

    fn write_instruction(
        &mut self,
        w: &mut dyn fmt::Write,
        func: &Function,
        aliases: &SecondaryMap<Value, Vec<Value>>,
        inst: Inst,
        indent: usize,
    ) -> fmt::Result {
        PlainWriter.write_instruction(w, func, aliases, inst, indent)?;
        if let Some(comment) = self.entity_comments.get(&inst.into()) {
            writeln!(w, "; {}", comment.replace('\n', "\n; "))?;
        }
        Ok(())
    }
}

pub(crate) fn should_write_ir() -> bool {
    cfg!(debug_assertions)
}

pub(crate) fn write_ir_file(
    path: &Path,
    write: impl FnOnce(&mut std::fs::File) -> std::io::Result<()>,
) {
    let res = std::fs::File::create(path).and_then(|mut file| write(&mut file));
    if let Err(err) = res {
        panic!("{err}")
    }
}

pub(crate) fn write_clif_file(
    path: &Path,
    isa: &dyn TargetIsa,
    func: &Function,
    mut clif_comments: &CommentWriter,
) {
    write_ir_file(path, |file| {
        let mut clif = String::new();
        cranelift::codegen::write::decorate_function(&mut clif_comments, &mut clif, func).unwrap();

        for flag in isa.flags().iter() {
            writeln!(file, "set {flag}")?;
        }
        write!(file, "target {}", isa.triple().architecture)?;
        for isa_flag in isa.isa_flags().iter() {
            write!(file, " {isa_flag}")?;
        }
        writeln!(file, "\n")?;
        writeln!(file)?;
        file.write_all(clif.as_bytes())?;
        Ok(())
    });
}
