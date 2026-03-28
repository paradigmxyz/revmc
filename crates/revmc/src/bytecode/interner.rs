use indexmap::IndexSet;
use oxc_index::Idx;
use std::hash::{BuildHasher, Hash};

/// A generic interner that deduplicates values and returns compact indices.
///
/// Backed by an [`IndexSet`] so that insertion is O(1) amortized and
/// index-to-value lookup is O(1).
pub(crate) struct Interner<I: Idx, T, S = std::hash::RandomState> {
    set: IndexSet<T, S>,
    _marker: std::marker::PhantomData<fn() -> I>,
}

impl<I: Idx, T: Hash + Eq, S: BuildHasher + Default> Default for Interner<I, T, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Idx, T: Hash + Eq, S: BuildHasher + Default> Interner<I, T, S> {
    pub(crate) fn new() -> Self {
        Self { set: IndexSet::default(), _marker: std::marker::PhantomData }
    }

    #[allow(dead_code)]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            set: IndexSet::with_capacity_and_hasher(capacity, S::default()),
            _marker: std::marker::PhantomData,
        }
    }

    /// Interns a value, returning its deduplicated index.
    pub(crate) fn intern(&mut self, value: T) -> I {
        let (idx, _) = self.set.insert_full(value);
        I::from_usize(idx)
    }
}

impl<I: Idx, T, S> Interner<I, T, S> {
    /// Returns the value at the given index.
    #[inline]
    pub(crate) fn get(&self, idx: I) -> &T {
        &self.set[idx.index()]
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.set.len()
    }
}

impl<I: Idx, T, S> std::ops::Index<I> for Interner<I, T, S> {
    type Output = T;

    #[inline]
    fn index(&self, idx: I) -> &T {
        &self.set[idx.index()]
    }
}

impl<I: Idx, T: std::fmt::Debug, S> std::fmt::Debug for Interner<I, T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.set.iter()).finish()
    }
}
