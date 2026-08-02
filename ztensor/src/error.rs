use std::fmt;

/// The spec rule a rejected file violated.
///
/// Tags correspond to sections of `spec/ztensor-v2-spec.md` so the
/// conformance corpus can assert exact rejection reasons. The set grows with
/// the spec, so it is `#[non_exhaustive]`: matching on one rule stays
/// source-compatible when the next is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Rule {
    FileTooSmall,
    HeaderMagic,
    FooterMagic,
    Version,
    ManifestBounds,
    ManifestTooLarge,
    ManifestHash,
    CborSyntax,
    CborDeterminism,
    CborDuplicateKey,
    CborDepth,
    Schema,
    Name,
    Shape,
    BlobAlignment,
    BlobBounds,
    BlobOverlap,
    /// A part names a shard the manifest's table does not declare.
    ShardRef,
    /// A shard name is outside the character set of spec §7.1.
    ShardName,
    LayoutRule,
    /// Data-level violation of a layout profile (e.g., CSR indptr rules).
    LayoutData,
    DenseSize,
    /// Stored bytes violate their declared encoding profile.
    Encoding,
    Digest,
    /// A resolved shard does not match the identity in the root's table.
    ShardIdentity,
    /// Two sources of one composite claim the same tensor name.
    NameCollision,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The file violates a MUST of the spec and was rejected.
    Reject {
        rule: Rule,
        detail: String,
    },
    /// The named tensor or part does not exist.
    NotFound(String),
    /// The file is valid, but uses vocabulary or requires a capability this
    /// implementation does not support (an unregistered layout or encoding,
    /// a payload that cannot be addressed, ...). Refusal, never
    /// reinterpretation.
    Unsupported(String),
    /// Caller error on the write path.
    InvalidInput(String),
    Io(std::io::Error),
}

impl Error {
    /// Rejects a file under a spec rule.
    ///
    /// Public because a profile registered from another crate has to be able
    /// to refuse a file exactly as a built-in one does — a validator that can
    /// only say `InvalidInput` is a second-class validator.
    pub fn reject(rule: Rule, detail: impl Into<String>) -> Self {
        Error::Reject {
            rule,
            detail: detail.into(),
        }
    }

    /// The rule this error rejected under, if it is a rejection. Lets a
    /// consumer ask the question without matching the struct variant.
    pub fn rule(&self) -> Option<Rule> {
        match self {
            Error::Reject { rule, .. } => Some(*rule),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Reject { rule, detail } => write!(f, "rejected ({rule:?}): {detail}"),
            Error::NotFound(what) => write!(f, "not found: {what}"),
            Error::Unsupported(what) => write!(f, "unsupported: {what}"),
            Error::InvalidInput(what) => write!(f, "invalid input: {what}"),
            Error::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
