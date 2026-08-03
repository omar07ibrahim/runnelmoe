//! Typed simulator failures.

use thiserror::Error;

/// A bounded, stable error returned by trace validation or simulation.
#[derive(Debug, Error)]
pub enum SimError {
    /// The input cannot be read.
    #[error("trace I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// A trace leaf was a symbolic link and was not followed.
    #[error("trace input must not be a symbolic link")]
    TraceInputSymlink,
    /// The retained trace descriptor is not a regular file.
    #[error("trace input must be a regular file")]
    TraceInputNotRegular,
    /// A retained trace descriptor exceeded the hard byte ceiling.
    #[error("trace input is at least {observed_at_least} bytes, limit is {limit}")]
    TraceInputTooLarge {
        /// Smallest size observed without reading beyond the ceiling.
        observed_at_least: u64,
        /// Configured trace byte ceiling.
        limit: u64,
    },
    /// A regular trace changed length while its retained descriptor was read.
    #[error(
        "trace input changed while reading (initial {initial}, read {bytes_read}, final {final_length} bytes)"
    )]
    TraceInputChanged {
        /// Length reported before reading.
        initial: u64,
        /// Bytes returned by the bounded read.
        bytes_read: u64,
        /// Length reported after reading.
        final_length: u64,
    },
    /// JSON syntax or a closed-schema constraint failed.
    #[error("invalid trace JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A trace-level invariant failed.
    #[error("invalid trace: {0}")]
    InvalidTrace(String),
    /// A simulation setting is invalid.
    #[error("invalid simulation configuration: {0}")]
    InvalidConfig(String),
    /// Checked accounting overflowed.
    #[error("simulation counter overflow: {0}")]
    CounterOverflow(&'static str),
    /// An oracle was requested outside its proved geometry.
    #[error("unsupported oracle geometry: {0}")]
    UnsupportedOracleGeometry(String),
    /// A deliberately bounded exact computation exceeded its limit.
    #[error("exact oracle limit exceeded: {0}")]
    ExactOracleLimit(String),
}

impl SimError {
    pub(crate) fn invalid_trace(message: impl Into<String>) -> Self {
        Self::InvalidTrace(message.into())
    }

    pub(crate) fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig(message.into())
    }
}
