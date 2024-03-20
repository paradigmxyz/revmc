use cranelift_module::ModuleError;
use std::fmt;

/// Compiler result.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Compiler error.
#[derive(Debug)]
pub enum Error {
    /// Bytecode error.
    Bytecode(BytecodeError),
    /// Cranelift module error.
    Module(ModuleError),
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bytecode(err) => Some(err),
            Self::Module(err) => Some(err),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytecode(err) => err.fmt(f),
            Self::Module(err) => err.fmt(f),
        }
    }
}

impl From<BytecodeError> for Error {
    fn from(value: BytecodeError) -> Self {
        Self::Bytecode(value)
    }
}

impl From<ModuleError> for Error {
    fn from(value: ModuleError) -> Self {
        Self::Module(value)
    }
}

/// Bytecode error.
#[derive(Debug)]
pub enum BytecodeError {
    /// Something.
    InvalidLen,
}

impl std::error::Error for BytecodeError {}

impl fmt::Display for BytecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLen => f.write_str("invalid bytecode length"),
        }
    }
}
