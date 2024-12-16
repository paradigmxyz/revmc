use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExtError {
    #[error("Lib loading error: {err}")]
    LibLoadingError { err: String },

    #[error("Get symbol error: {err}")]
    GetSymbolError { err: String },

    #[error("StateDB account info fetch error: {err}")]
    StateDBAccInfoFetchError { err: String },

    #[error("Illegal access list access")]
    IllegalAccessListAccess,

    #[error("Code none exists error: {err}")]
    CodeNoneExists { err: String },

    #[error("RwLock poison error: {err}")]
    RwLockPoison { err: String },
}
