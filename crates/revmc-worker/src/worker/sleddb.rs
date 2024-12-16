use color_eyre::{self, Result};
use sled::IVec;
use std::{marker::PhantomData, path::Path, sync::Arc};

use super::path::sleddb_path;

/// Embedded Database for local storage of compiled multiple ExternalFns
#[derive(Debug)]
pub(crate) struct SledDB<K>
where
    K: AsRef<[u8]>,
{
    pub db: Arc<sled::Db>,
    _marker: std::marker::PhantomData<K>,
}

impl<K> SledDB<K>
where
    K: AsRef<[u8]>,
{
    pub(crate) fn init() -> Self {
        let db = Self::connect(sleddb_path().to_str().unwrap()).unwrap();

        Self { db: Arc::new(db), _marker: std::marker::PhantomData }
    }

    fn connect(path: &str) -> Result<sled::Db> {
        sled::open(Path::new(path)).map_err(|e| color_eyre::Report::new(e))
    }

    pub(crate) fn put(&self, key: K, value: &[u8]) -> Result<()> {
        self.db.insert(key, value).map_err(|e| color_eyre::Report::new(e))?;

        self.db.flush().map_err(|e| color_eyre::Report::new(e))?;

        Ok(())
    }

    pub(crate) fn get(&self, key: K) -> Result<Option<IVec>> {
        self.db.get(key).map_err(|e| color_eyre::Report::new(e))
    }
}

impl<K> Clone for SledDB<K>
where
    K: AsRef<[u8]>,
{
    fn clone(&self) -> Self {
        Self { db: Arc::clone(&self.db), _marker: PhantomData }
    }
}
