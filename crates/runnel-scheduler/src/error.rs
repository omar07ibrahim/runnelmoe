//! Stable, sanitized scheduler error categories.

use std::fmt;

use runnel_runtime::{RuntimeError, SamplingError};
use thiserror::Error;

/// The six bounded-cardinality failure categories exposed by the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCategory {
    InvalidRequest,
    Unsupported,
    ResourceExhausted,
    Cancelled,
    DeadlineExceeded,
    Internal,
}

impl ErrorCategory {
    /// Returns the stable wire spelling of this category.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::Unsupported => "unsupported",
            Self::ResourceExhausted => "resource_exhausted",
            Self::Cancelled => "cancelled",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::Internal => "internal",
        }
    }
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub type SchedulerResult<T> = std::result::Result<T, SchedulerError>;

/// Sanitized failures at the deterministic scheduler boundary.
///
/// Static field names and bounded byte counts are safe to report. Prompt
/// contents, tokens, seeds, logits, adapter payloads, and request identities
/// are deliberately absent. Adapter and sampler failures are classified at
/// this boundary and their potentially sensitive source values are discarded.
#[derive(Clone, Error)]
#[non_exhaustive]
pub enum SchedulerError {
    #[error("invalid scheduler request field {field}: {problem}")]
    InvalidRequest {
        field: &'static str,
        problem: &'static str,
    },

    #[error("unsupported scheduler capability: {feature}")]
    Unsupported { feature: &'static str },

    #[error("scheduler resource exhausted for {resource}: required {required}, limit {limit}")]
    ResourceExhausted {
        resource: &'static str,
        required: u64,
        limit: u64,
    },

    #[error("scheduler allocation failed for {resource}: required {required} bytes")]
    AllocationFailure {
        resource: &'static str,
        required: u64,
    },

    #[error("scheduler request was cancelled")]
    Cancelled,

    #[error("scheduler request deadline was exceeded")]
    DeadlineExceeded,

    #[error("scheduler request identity is unknown or no longer retained")]
    RequestNotFound,

    #[error("scheduler is closed")]
    SchedulerClosed,

    #[error("adapter operation failed during {operation}")]
    Adapter {
        operation: &'static str,
        category: ErrorCategory,
    },

    #[error("sampling operation failed during {operation}")]
    Sampling {
        operation: &'static str,
        category: ErrorCategory,
    },

    #[error("scheduler invariant failed: {problem}")]
    Internal { problem: &'static str },
}

impl SchedulerError {
    #[must_use]
    pub const fn invalid_request(field: &'static str, problem: &'static str) -> Self {
        Self::InvalidRequest { field, problem }
    }

    #[must_use]
    pub const fn unsupported(feature: &'static str) -> Self {
        Self::Unsupported { feature }
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
    pub const fn allocation_failure(resource: &'static str, required: u64) -> Self {
        Self::AllocationFailure { resource, required }
    }

    #[must_use]
    pub const fn cancelled() -> Self {
        Self::Cancelled
    }

    #[must_use]
    pub const fn deadline_exceeded() -> Self {
        Self::DeadlineExceeded
    }

    #[must_use]
    pub const fn request_not_found() -> Self {
        Self::RequestNotFound
    }

    #[must_use]
    pub const fn scheduler_closed() -> Self {
        Self::SchedulerClosed
    }

    #[must_use]
    pub const fn internal(problem: &'static str) -> Self {
        Self::Internal { problem }
    }

    /// Wraps a runtime failure using its stable scheduler classification.
    #[must_use]
    pub fn adapter(operation: &'static str, source: RuntimeError) -> Self {
        let category = classify_runtime_error(&source);
        Self::Adapter {
            operation,
            category,
        }
    }

    /// Wraps a runtime failure whose call site has stronger classification
    /// context than the generic runtime mapping.
    #[must_use]
    pub fn adapter_with_category(
        operation: &'static str,
        category: ErrorCategory,
        _source: RuntimeError,
    ) -> Self {
        Self::Adapter {
            operation,
            category,
        }
    }

    /// Wraps a sampling failure using its stable scheduler classification.
    #[must_use]
    pub fn sampling(operation: &'static str, source: SamplingError) -> Self {
        let category = classify_sampling_error(&source);
        Self::Sampling {
            operation,
            category,
        }
    }

    #[must_use]
    pub const fn category(&self) -> ErrorCategory {
        match self {
            Self::InvalidRequest { .. } | Self::RequestNotFound => ErrorCategory::InvalidRequest,
            Self::Unsupported { .. } => ErrorCategory::Unsupported,
            Self::ResourceExhausted { .. } | Self::AllocationFailure { .. } => {
                ErrorCategory::ResourceExhausted
            }
            Self::Cancelled | Self::SchedulerClosed => ErrorCategory::Cancelled,
            Self::DeadlineExceeded => ErrorCategory::DeadlineExceeded,
            Self::Adapter { category, .. } | Self::Sampling { category, .. } => *category,
            Self::Internal { .. } => ErrorCategory::Internal,
        }
    }
}

impl fmt::Debug for SchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest { field, problem } => formatter
                .debug_struct("InvalidRequest")
                .field("field", field)
                .field("problem", problem)
                .finish(),
            Self::Unsupported { feature } => formatter
                .debug_struct("Unsupported")
                .field("feature", feature)
                .finish(),
            Self::ResourceExhausted {
                resource,
                required,
                limit,
            } => formatter
                .debug_struct("ResourceExhausted")
                .field("resource", resource)
                .field("required", required)
                .field("limit", limit)
                .finish(),
            Self::AllocationFailure { resource, required } => formatter
                .debug_struct("AllocationFailure")
                .field("resource", resource)
                .field("required", required)
                .finish(),
            Self::Cancelled => formatter.write_str("Cancelled"),
            Self::DeadlineExceeded => formatter.write_str("DeadlineExceeded"),
            Self::RequestNotFound => formatter.write_str("RequestNotFound"),
            Self::SchedulerClosed => formatter.write_str("SchedulerClosed"),
            Self::Adapter {
                operation,
                category,
            } => formatter
                .debug_struct("Adapter")
                .field("operation", operation)
                .field("category", category)
                .finish(),
            Self::Sampling {
                operation,
                category,
            } => formatter
                .debug_struct("Sampling")
                .field("operation", operation)
                .field("category", category)
                .finish(),
            Self::Internal { problem } => formatter
                .debug_struct("Internal")
                .field("problem", problem)
                .finish(),
        }
    }
}

fn classify_runtime_error(error: &RuntimeError) -> ErrorCategory {
    match error {
        RuntimeError::InvalidToken { .. }
        | RuntimeError::EmptySequence
        | RuntimeError::ContextLimit { .. } => ErrorCategory::InvalidRequest,
        RuntimeError::UnsupportedCharacter { .. }
        | RuntimeError::BackendIncompatibleWithAdapter { .. } => ErrorCategory::Unsupported,
        RuntimeError::ResourceExhausted { .. } => ErrorCategory::ResourceExhausted,
        RuntimeError::InvalidConfig(_)
        | RuntimeError::InvalidArtifact(_)
        | RuntimeError::MissingTensor(_)
        | RuntimeError::UnexpectedTensor(_)
        | RuntimeError::InvalidTensor { .. }
        | RuntimeError::StateMismatch
        | RuntimeError::ModelIdentityExhausted
        | RuntimeError::InvalidStateLayout(_)
        | RuntimeError::StateIdentityExhausted
        | RuntimeError::StateRevisionExhausted
        | RuntimeError::AdapterTransactionIdentityExhausted
        | RuntimeError::InvalidAdapterWork(_)
        | RuntimeError::InvalidExpertContribution(_)
        | RuntimeError::StateRevisionMismatch { .. }
        | RuntimeError::InvalidState(_)
        | RuntimeError::ResourceSizeOverflow { .. }
        | RuntimeError::NonFinite(_)
        | RuntimeError::InvalidBf16Tensor { .. }
        | RuntimeError::ExpertKernel { .. } => ErrorCategory::Internal,
    }
}

fn classify_sampling_error(error: &SamplingError) -> ErrorCategory {
    match error {
        SamplingError::EmptyVocabulary
        | SamplingError::VocabularyTooLarge { .. }
        | SamplingError::InvalidTemperature
        | SamplingError::InvalidTopK { .. }
        | SamplingError::InvalidTopP
        | SamplingError::WorkspaceSizeOverflow { .. } => ErrorCategory::InvalidRequest,
        SamplingError::WorkspaceAllocation { .. } => ErrorCategory::ResourceExhausted,
        SamplingError::EmptyLogits
        | SamplingError::WorkspaceSizeMismatch { .. }
        | SamplingError::NonFiniteLogit
        | SamplingError::UnexpectedGreedyRngState
        | SamplingError::InvalidArithmetic { .. } => ErrorCategory::Internal,
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use super::*;

    #[test]
    fn categories_have_stable_wire_spellings() {
        let cases = [
            (ErrorCategory::InvalidRequest, "invalid_request"),
            (ErrorCategory::Unsupported, "unsupported"),
            (ErrorCategory::ResourceExhausted, "resource_exhausted"),
            (ErrorCategory::Cancelled, "cancelled"),
            (ErrorCategory::DeadlineExceeded, "deadline_exceeded"),
            (ErrorCategory::Internal, "internal"),
        ];
        for (category, spelling) in cases {
            assert_eq!(category.as_str(), spelling);
            assert_eq!(category.to_string(), spelling);
        }
    }

    #[test]
    fn adapter_source_is_discarded_at_the_public_boundary() {
        let sentinel = "payload-sentinel-78291";
        let error = SchedulerError::adapter(
            "preparing model work",
            RuntimeError::InvalidConfig(sentinel.into()),
        );

        assert_eq!(error.category(), ErrorCategory::Internal);
        assert!(error.source().is_none());
        assert!(!error.to_string().contains(sentinel));
        let debug = format!("{error:?}");
        assert!(!debug.contains(sentinel));
    }

    #[test]
    fn request_and_resource_categories_do_not_require_text_parsing() {
        assert_eq!(
            SchedulerError::request_not_found().category(),
            ErrorCategory::InvalidRequest
        );
        assert_eq!(
            SchedulerError::resource_exhausted("test capacity", 65, 64).category(),
            ErrorCategory::ResourceExhausted
        );
        let allocation = SchedulerError::allocation_failure("bounded test buffer", 64);
        assert_eq!(allocation.category(), ErrorCategory::ResourceExhausted);
        assert_eq!(
            allocation.to_string(),
            "scheduler allocation failed for bounded test buffer: required 64 bytes"
        );
        assert_eq!(
            SchedulerError::scheduler_closed().category(),
            ErrorCategory::Cancelled
        );
    }
}
