use thiserror::Error;

#[derive(Error, Debug)]
pub(crate) enum CompilerError {
    #[error("Backend init error, err: {err}")]
    BackendInit { err: String },

    #[error("File I/O error, err: {err}")]
    FileIO { err: String },

    #[error("Bytecode translation error, err: {err}")]
    BytecodeTranslation { err: String },

    #[error("Link error, err: {err}")]
    Link { err: String },
}
