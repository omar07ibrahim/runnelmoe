use std::io;

use runnel_format::FormatError;
use serde::Serialize;
use thiserror::Error;

/// Stable, bounded-cardinality classification for storage failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorCategory {
    InvalidInput,
    ResourceExhausted,
    Cancelled,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    UnsafeFilesystem,
    Integrity,
    Io,
    Shutdown,
    DurabilityUnconfirmed,
    Internal,
}

/// Sanitized failures at the verified-storage trust boundary.
///
/// Display strings deliberately contain no filesystem path or artifact
/// payload. Callers may classify errors without parsing their text.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum StoreError {
    #[error("invalid configuration for {field}: {problem}")]
    InvalidConfig {
        field: &'static str,
        problem: &'static str,
    },

    #[error("resource limit exceeded for {resource}: required {required}, limit {limit}")]
    ResourceExhausted {
        resource: &'static str,
        required: u64,
        limit: u64,
    },

    #[error("the bounded work queue is full")]
    QueueFull,

    #[error("the storage service is shut down")]
    Shutdown,

    #[error("operation cancelled")]
    Cancelled,

    #[error("operation deadline exceeded")]
    DeadlineExceeded,

    #[error("unsafe filesystem entry for {kind}")]
    UnsafeLayout { kind: &'static str },

    #[error("missing filesystem entry for {kind}")]
    Missing { kind: &'static str },

    #[error("filesystem entry already exists for {kind}")]
    AlreadyExists { kind: &'static str },

    #[error("{kind} length is {actual} bytes, expected {expected}")]
    LengthMismatch {
        kind: &'static str,
        expected: u64,
        actual: u64,
    },

    #[error("integrity verification failed for {kind}")]
    Integrity { kind: &'static str },

    #[error("page index {index} is outside the page count {count}")]
    PageOutOfRange { index: u64, count: u64 },

    #[error("invalid resume token")]
    InvalidResumeToken,

    #[error("resume token does not match {field}")]
    ResumeMismatch { field: &'static str },

    #[error("CAS budget exceeded: required {required} bytes, limit {limit}")]
    BudgetExceeded { required: u64, limit: u64 },

    #[error(
        "filesystem reserve would be violated: required {required} bytes, available {available}, reserve {reserve}"
    )]
    FilesystemReserve {
        required: u64,
        available: u64,
        reserve: u64,
    },

    #[error("{kind} was published but parent-directory durability is unconfirmed")]
    PublishedButDurabilityUnconfirmed { kind: &'static str },

    #[error("checked arithmetic overflow while {context}")]
    ArithmeticOverflow { context: &'static str },

    #[error("I/O failure while {operation}: {kind:?}")]
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
    },

    #[error("artifact format validation failed")]
    Format(#[source] FormatError),

    #[error("storage invariant failed: {problem}")]
    Invariant { problem: &'static str },
}

impl StoreError {
    #[must_use]
    pub const fn invalid_config(field: &'static str, problem: &'static str) -> Self {
        Self::InvalidConfig { field, problem }
    }

    #[must_use]
    pub const fn resource_exhausted(resource: &'static str, required: u64, limit: u64) -> Self {
        Self::ResourceExhausted {
            resource,
            required,
            limit,
        }
    }

    #[must_use]
    pub const fn queue_full() -> Self {
        Self::QueueFull
    }

    #[must_use]
    pub const fn shutdown() -> Self {
        Self::Shutdown
    }

    #[must_use]
    pub const fn cancelled() -> Self {
        Self::Cancelled
    }

    #[must_use]
    pub const fn invariant(problem: &'static str) -> Self {
        Self::Invariant { problem }
    }

    #[must_use]
    pub fn io(operation: &'static str, error: &impl IntoIoErrorKind) -> Self {
        Self::Io {
            operation,
            kind: error.io_error_kind(),
        }
    }

    #[must_use]
    pub const fn category(&self) -> ErrorCategory {
        match self {
            Self::InvalidConfig { .. }
            | Self::PageOutOfRange { .. }
            | Self::InvalidResumeToken
            | Self::ResumeMismatch { .. } => ErrorCategory::InvalidInput,
            Self::ResourceExhausted { .. }
            | Self::QueueFull
            | Self::BudgetExceeded { .. }
            | Self::FilesystemReserve { .. } => ErrorCategory::ResourceExhausted,
            Self::Cancelled => ErrorCategory::Cancelled,
            Self::DeadlineExceeded => ErrorCategory::DeadlineExceeded,
            Self::Missing { .. } => ErrorCategory::NotFound,
            Self::AlreadyExists { .. } => ErrorCategory::AlreadyExists,
            Self::UnsafeLayout { .. } => ErrorCategory::UnsafeFilesystem,
            Self::LengthMismatch { .. } | Self::Integrity { .. } => ErrorCategory::Integrity,
            Self::Io { .. } => ErrorCategory::Io,
            Self::Shutdown => ErrorCategory::Shutdown,
            Self::PublishedButDurabilityUnconfirmed { .. } => ErrorCategory::DurabilityUnconfirmed,
            Self::ArithmeticOverflow { .. } | Self::Invariant { .. } => ErrorCategory::Internal,
            Self::Format(source) => match source {
                FormatError::ArtifactIdMismatch { .. }
                | FormatError::BlobLengthMismatch { .. }
                | FormatError::BlobDigestMismatch { .. }
                | FormatError::PageTable { .. }
                | FormatError::PageHashMismatch { .. } => ErrorCategory::Integrity,
                FormatError::MissingBlob { .. } => ErrorCategory::NotFound,
                FormatError::NotRegularFile { .. } => ErrorCategory::UnsafeFilesystem,
                FormatError::Io { .. } => ErrorCategory::Io,
                FormatError::MemoryBudgetExceeded { .. } => ErrorCategory::ResourceExhausted,
                _ => ErrorCategory::InvalidInput,
            },
        }
    }
}

impl From<FormatError> for StoreError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

/// Converts supported operating-system errors into a sanitized error kind.
pub trait IntoIoErrorKind {
    fn io_error_kind(&self) -> io::ErrorKind;
}

impl IntoIoErrorKind for io::Error {
    fn io_error_kind(&self) -> io::ErrorKind {
        self.kind()
    }
}

impl IntoIoErrorKind for rustix::io::Errno {
    fn io_error_kind(&self) -> io::ErrorKind {
        io::Error::from_raw_os_error(self.raw_os_error()).kind()
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::{ErrorCategory, StoreError};

    #[test]
    fn io_error_is_sanitized_and_classified() {
        let source = io::Error::new(io::ErrorKind::PermissionDenied, "/secret/path");
        let error = StoreError::io("opening retained entry", &source);
        assert_eq!(error.category(), ErrorCategory::Io);
        assert!(!error.to_string().contains("secret"));
    }
}
