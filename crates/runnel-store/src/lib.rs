//! Descriptor-safe, bounded storage for immutable RMOA artifacts.
//!
//! This crate owns the M2 trust boundary: CAS publication, retained filesystem
//! handles, authenticated page reads, asynchronous buffer ownership, and
//! byte-accounted cache leases. `runnel-format` remains the canonical parser;
//! no unverified payload byte is exposed through this crate.

#![forbid(unsafe_code)]

mod async_io;
mod budget;
mod cache;
mod cas;
mod control;
mod error;
mod fs;
mod layout;
mod metrics;
mod page;
mod source;
mod stage;
mod trace;

pub use async_io::{AsyncReader, AsyncReaderConfig};
pub use budget::{DiskBudget, DiskUsage};
pub use cache::{AccessReason, CacheConfig, PageCache, PageLease, PrefetchOutcome};
pub use cas::{Cas, CasConfig, GcPolicy, GcReport, ImportReceipt};
pub use control::{CancellationToken, Control};
pub use error::{ErrorCategory, StoreError};
pub use metrics::{MetricsSnapshot, RssSample};
pub use page::{PageKey, PageSpec, ReadStats, StoredArtifact, SyncReader, VerifiedPage};
pub use source::ArtifactSource;
pub use stage::ResumeToken;
pub use trace::{TraceEvent, TraceOutcome, TraceSink};
