//! Strict RMOA v1 parsing and byte verification for the tiny M1 adapter.
//!
//! The crate accepts only the ASCII schema subset used by M1, for which its
//! canonical writer is exactly the RFC 8785 representation. It rejects
//! duplicate keys, null, floating point numbers, non-canonical bytes, unsafe
//! integer arithmetic, malformed tables, and every unverified object/page.
//!
//! M1 intentionally uses an independently budgeted eager loader for tiny CI
//! artifacts. M2 will add bounded asynchronous reads and Linux descriptor-
//! relative `openat2`/portable `openat` traversal, no-replace CAS publication,
//! cancellation, deadlines, reserve-aware disk preflight, and orphan GC. Those
//! filesystem controls are not claimed here; byte-level validation is complete
//! before this crate exposes tensor slices.

#![forbid(unsafe_code)]

mod artifact;
mod error;
mod json;
mod manifest;

pub use artifact::{Artifact, ArtifactBytes, PageTable, VerifiedTensor};
pub use error::FormatError;
pub use manifest::{
    Adapter, DType, Digest, DigestParseError, FORMAT_AGGREGATE_OBJECT_BYTES,
    FORMAT_AGGREGATE_PAGE_TABLE_BYTES, FORMAT_MANIFEST_BYTES, FORMAT_OBJECT_BYTES, FORMAT_OBJECTS,
    FORMAT_PAGE_TABLE_BYTES, FORMAT_TENSORS, Limits, Manifest, ObjectRecord, TensorRecord,
    TinyModel, Tokenizer,
};
