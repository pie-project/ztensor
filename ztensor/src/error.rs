use std::fmt;

/// The spec rule a rejected file violated.
///
/// Tags correspond to sections of `spec/ztensor-v2-spec.md` so the
/// conformance corpus can assert exact rejection reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    ShardIndex,
    LayoutRule,
    DenseSize,
    Digest,
}

#[derive(Debug)]
pub enum Error {
    /// The file violates a MUST of the spec and was rejected.
    Reject { rule: Rule, detail: String },
    /// The named object or part does not exist.
    NotFound(String),
    /// The file is valid, but uses vocabulary or requires a capability this
    /// implementation does not support (unknown layout/encoding, foreign
    /// shard reads before M5, ...). Refusal, never reinterpretation.
    Unsupported(String),
    /// Caller error on the write path.
    InvalidInput(String),
    Io(std::io::Error),
}

impl Error {
    pub(crate) fn reject(rule: Rule, detail: impl Into<String>) -> Self {
        Error::Reject {
            rule,
            detail: detail.into(),
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
