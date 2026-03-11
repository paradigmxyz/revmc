//! Runtime error types.

use std::fmt;

/// Runtime error.
#[derive(Debug)]
pub enum RuntimeError {
    /// The coordinator has shut down.
    Shutdown,
    /// An artifact storage error.
    Storage(StorageError),
    /// An artifact failed to load.
    ArtifactLoad(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shutdown => f.write_str("coordinator has shut down"),
            Self::Storage(e) => write!(f, "storage error: {e}"),
            Self::ArtifactLoad(e) => write!(f, "artifact load error: {e}"),
        }
    }
}

impl std::error::Error for RuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(e) => Some(e),
            _ => None,
        }
    }
}

impl From<StorageError> for RuntimeError {
    fn from(e: StorageError) -> Self {
        Self::Storage(e)
    }
}

/// Artifact storage error.
#[derive(Debug)]
pub struct StorageError(pub Box<dyn std::error::Error + Send + Sync>);

impl StorageError {
    /// Creates a new storage error.
    pub fn new(e: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self(Box::new(e))
    }
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for StorageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.0)
    }
}
