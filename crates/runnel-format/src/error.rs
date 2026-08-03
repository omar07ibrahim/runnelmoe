use crate::Digest;
use thiserror::Error;

/// Fail-closed errors produced while parsing or verifying an RMOA artifact.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatError {
    #[error("configured {field} limit {value} exceeds the format ceiling {ceiling}")]
    LimitExceedsFormat {
        field: &'static str,
        value: u64,
        ceiling: u64,
    },

    #[error("manifest is {actual} bytes, exceeding the configured limit of {limit}")]
    ManifestTooLarge { actual: usize, limit: usize },

    #[error(
        "eager M1 artifact requires {required} bytes, exceeding the configured memory budget of {limit}"
    )]
    MemoryBudgetExceeded { required: u64, limit: u64 },

    #[error("invalid JSON at line {line}, column {column}: {message}")]
    JsonSyntax {
        line: usize,
        column: usize,
        message: String,
    },

    #[error("duplicate JSON object key {key:?}")]
    DuplicateKey { key: String },

    #[error("JSON nesting exceeds the format ceiling of {limit}")]
    NestingTooDeep { limit: usize },

    #[error("null is not allowed in an RMOA manifest")]
    NullNotAllowed,

    #[error("floating-point JSON numbers are not allowed in an RMOA manifest")]
    FloatingPointNotAllowed,

    #[error("JSON integer is outside the inclusive range 0..={max}")]
    IntegerOutOfRange { max: u64 },

    #[error("schema string is {actual} bytes, exceeding the format ceiling of {limit}")]
    StringTooLong { actual: usize, limit: usize },

    #[error("the M1 schema accepts ASCII strings only")]
    NonAsciiString,

    #[error("schema violation at {path}: {problem}")]
    Schema { path: String, problem: String },

    #[error("{field} count {actual} exceeds the configured limit of {limit}")]
    CountLimit {
        field: &'static str,
        actual: usize,
        limit: usize,
    },

    #[error("checked arithmetic overflow while validating {context}")]
    ArithmeticOverflow { context: &'static str },

    #[error("manifest is not the RFC 8785 canonical ASCII representation followed by one LF")]
    NonCanonicalManifest,

    #[error("expected artifact ID {expected}, computed {actual}")]
    ArtifactIdMismatch { expected: Digest, actual: Digest },

    #[error("I/O failure while {operation}: {message}")]
    Io {
        operation: &'static str,
        message: String,
    },

    #[error("{kind} is not a regular file")]
    NotRegularFile { kind: &'static str },

    #[error("missing {kind} bytes for {digest}")]
    MissingBlob { kind: &'static str, digest: Digest },

    #[error("{kind} {digest} has length {actual}, expected {expected}")]
    BlobLengthMismatch {
        kind: &'static str,
        digest: Digest,
        expected: u64,
        actual: u64,
    },

    #[error("{kind} digest mismatch: expected {expected}, computed {actual}")]
    BlobDigestMismatch {
        kind: &'static str,
        expected: Digest,
        actual: Digest,
    },

    #[error("invalid page table for object {object}: {problem}")]
    PageTable { object: Digest, problem: String },

    #[error("page {page_index} of object {object} failed SHA-256 verification")]
    PageHashMismatch { object: Digest, page_index: u64 },

    #[error("no tensor has role {role:?}")]
    UnknownTensorRole { role: String },

    #[error(
        "tensor coverage gap in object {object}: expected the next range at {expected_offset}, found {actual_offset}"
    )]
    TensorGap {
        object: Digest,
        expected_offset: u64,
        actual_offset: u64,
    },

    #[error(
        "overlapping tensor range in object {object}: previous range ends at {previous_end}, next starts at {next_offset}"
    )]
    TensorOverlap {
        object: Digest,
        previous_end: u64,
        next_offset: u64,
    },

    #[error("tensor ranges cover {covered} bytes of object {object}, whose length is {length}")]
    TensorCoverage {
        object: Digest,
        covered: u64,
        length: u64,
    },
}
