//! Bounded deterministic scheduling for model-agnostic decoder adapters.
//!
//! The synchronous core owns request policy, exact semantic accounting, DRR
//! fairness, coalesced expert work, and transactional token publication. Model
//! execution remains synchronous; endpoints use Tokio only for allocation-free
//! wake notifications consumed by the thin actor layer.

#![forbid(unsafe_code)]

mod accounting;
mod config;
mod control;
mod endpoint;
mod engine;
#[cfg(test)]
mod engine_adversarial_tests;
mod error;
mod id;
mod ledger;
mod request;
mod ring;
mod wave;

pub use config::{
    MAX_BATCH_WIDTH, MAX_WAVES_PER_STEP, SchedulerConfig, SchedulerLimits, SharedStaticCharges,
    StateLayoutSummary,
};
pub use engine::SchedulerEngine;
pub use error::{ErrorCategory, SchedulerError, SchedulerResult};
pub use id::RequestId;
pub use ledger::{
    CategorySnapshot, LEDGER_QUANTUM_BYTES, LedgerCategory, LedgerOwnership, LedgerSnapshot,
};
pub use request::{
    CancelDisposition, EngineSnapshot, OutputEvent, RequestPhase, RequestSpec, ShutdownReport,
    StepReport, TerminalOutcome, TerminalResult,
};
