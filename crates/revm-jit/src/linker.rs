use std::path::{Path, PathBuf};

/// EVM bytecode compiler linker.
#[derive(Debug)]
pub struct Linker {
    cc: Option<PathBuf>,
    linker: Option<PathBuf>,
    cflags: Vec<String>,
}

impl Default for Linker {
    fn default() -> Self {
        Self::new()
    }
}

impl Linker {
    /// Creates a new linker.
    pub fn new() -> Self {
        Self { cc: None, linker: None, cflags: vec![] }
    }

    /// Sets the C compiler to use for linking. Default: "cc".
    pub fn cc(&mut self, cc: Option<PathBuf>) {
        self.cc = cc;
    }

    /// Sets the linker to use for linking. Default: `lld`.
    pub fn linker(&mut self, linker: Option<PathBuf>) {
        self.linker = linker;
    }

    /// Sets the C compiler flags to use for linking.
    pub fn cflags(&mut self, cflags: impl IntoIterator<Item = impl Into<String>>) {
        self.cflags.extend(cflags.into_iter().map(Into::into));
    }

    /// Links the given object files into a shared library at the given path.
    pub fn link(
        &self,
        out: &Path,
        objects: impl IntoIterator<Item = impl AsRef<std::ffi::OsStr>>,
    ) -> std::io::Result<()> {
        debug_time!("link", || self.link_inner(out, objects))
    }

    fn link_inner(
        &self,
        out: &Path,
        objects: impl IntoIterator<Item = impl AsRef<std::ffi::OsStr>>,
    ) -> std::io::Result<()> {
        let storage;
        let cc = match &self.cc {
            Some(cc) => cc,
            None => {
                let str = match std::env::var_os("CC") {
                    Some(cc) => {
                        storage = cc;
                        storage.as_os_str()
                    }
                    None => "cc".as_ref(),
                };
                Path::new(str)
            }
        };

        let mut cmd = std::process::Command::new(cc);
        cmd.arg("-o").arg(out);
        cmd.arg("-shared");
        cmd.arg("-O3");
        if cfg!(target_vendor = "apple") {
            // TODO:
            cmd.arg("-Wl,-undefined");
        } else {
            cmd.arg("-Wl,--gc-sections,--strip-all,--undefined");
        }
        if let Some(linker) = &self.linker {
            cmd.arg(format!("-fuse-ld={}", linker.display()));
        } else {
            cmd.arg("-fuse-ld=lld");
        }
        cmd.args(&self.cflags);
        cmd.args(objects);
        debug!(cmd=?cmd.get_program(), "linking");
        trace!(?cmd, "full linking command");
        let output = cmd.output()?;
        if !output.status.success() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("cc failed with {output:#?}"),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm_primitives::SpecId;
    use std::fs::File;

    #[test]
    fn basic() {
        let tmp = tempfile::tempdir().expect("could not create temp dir");
        let obj = tmp.path().join("out.o");
        let so = tmp.path().join("out.so");

        // Compile and build object file.
        let cx = crate::llvm::inkwell::context::Context::create();
        let opt_level = revm_jit_backend::OptimizationLevel::Aggressive;
        let backend = crate::new_llvm_backend(&cx, true, opt_level).unwrap();
        let mut compiler = crate::EvmCompiler::new(backend);
        if let Err(e) = compiler.translate(Some("link_test_basic"), &[], SpecId::CANCUN) {
            panic!("failed to compile: {e}");
        }

        {
            let mut f = File::create(&obj).unwrap();
            if let Err(e) = compiler.write_object(&mut f) {
                panic!("failed to write object: {e}");
            }
        }
        assert!(obj.exists());

        // Link object to shared library.
        let mut linker = Linker::new();
        let mut n = 0;
        for driver in ["cc", "gcc", "clang"] {
            if !command_v(driver) {
                continue;
            }
            n += 1;

            let _ = std::fs::remove_file(&so);
            linker.cc = Some(driver.into());
            if let Err(e) = linker.link(&so, [&obj]) {
                panic!("failed to link with {driver}: {e}");
            }
            assert!(so.exists());
        }
        assert!(n > 0, "no C compiler found");
    }

    fn command_v(cmd: &str) -> bool {
        std::process::Command::new(cmd).arg("--version").status().is_ok_and(|s| s.success())
    }
}
