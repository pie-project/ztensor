//! What a foreign format hands back.
//!
//! A projection reads a file's own metadata and says three things: what
//! tensors are in it and where ([`Catalog`]), which byte ranges the file
//! occupies (so page exclusivity can be decided rather than guessed), and,
//! for formats whose bytes are not simply lying there, how to produce the
//! ones that have no address.
//!
//! It never builds a [`Manifest`](ztensor::schema::Manifest). A safetensors
//! file has no manifest; saying otherwise would be inventing a document
//! nobody wrote.

use std::sync::Arc;

use ztensor::{Catalog, Opaque, Result, Source, Store, Vocabulary};

pub(crate) struct Projection {
    pub catalog: Catalog,
    /// Every byte range the file is known to use, including its header and
    /// any index. Left empty when the format cannot say, and then the store
    /// never claims page exclusivity.
    pub occupied: Vec<(u64, u64)>,
    pub opaque: Option<Box<dyn Opaque>>,
}

impl Projection {
    pub fn new(catalog: Catalog) -> Self {
        Self {
            catalog,
            occupied: Vec::new(),
            opaque: None,
        }
    }

    pub fn occupying(mut self, ranges: Vec<(u64, u64)>) -> Self {
        self.occupied = ranges;
        self
    }

    pub fn with_opaque(mut self, opaque: Box<dyn Opaque>) -> Self {
        self.opaque = Some(opaque);
        self
    }

    pub fn into_source(self, store: Store, vocab: Option<&Vocabulary>) -> Result<Source> {
        let mut store = store.with_occupied(self.occupied);
        if let Some(opaque) = self.opaque {
            store = store.with_opaque(opaque);
        }
        match vocab {
            None => Source::from_parts(vec![store], self.catalog),
            Some(v) => Source::from_parts_with(vec![store], self.catalog, Arc::new(v.clone())),
        }
    }
}
