use thiserror::Error;

/// Everything that can go wrong decoding a HEIF/HEIC file.
///
/// No variant here is reachable by a panic: every parser in this crate
/// returns `Result` instead of indexing, unwrapping or asserting on
/// attacker-controlled input (see `AGENTS.md` Rule 9 in the parent
/// workspace). A malformed or truncated file is always `Err`, never a
/// crash.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum HeifError {
    #[error("truncated input: needed {needed} more byte(s) at offset {at}")]
    Truncated { at: usize, needed: usize },

    #[error("not a HEIF/HEIC file: {0}")]
    NotHeif(&'static str),

    #[error("malformed ISOBMFF box: {0}")]
    MalformedBox(&'static str),

    #[error("malformed HEVC bitstream: {0}")]
    MalformedHevc(&'static str),

    #[error("unsupported: {0}")]
    Unsupported(&'static str),

    #[error("CABAC decode desynchronized: {0}")]
    CabacDesync(&'static str),

    #[error("limit exceeded: {0}")]
    LimitExceeded(&'static str),
}

pub type Result<T> = core::result::Result<T, HeifError>;
