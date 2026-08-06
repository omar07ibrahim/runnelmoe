//! Bounded deterministic scheduling for model-agnostic decoder adapters.
//!
//! The synchronous core owns immutable FIFO/DRR policy selection, exact
//! semantic accounting, coalesced expert work, and transactional token
//! publication. Model execution remains synchronous; endpoints use Tokio only
//! for allocation-free wake notifications consumed by the thin actor layer.

#![forbid(unsafe_code)]

mod accounting;
mod actor;
mod checkpoint;
mod config;
mod control;
mod endpoint;
mod engine;
#[cfg(test)]
mod engine_adversarial_tests;
mod error;
mod id;
mod ledger;
mod ledger_trace;
mod request;
mod ring;
mod run_observer;
mod trace;
mod wave;

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
#[doc(hidden)]
pub use actor::{
    ActorAcceptedRequestWitness, ActorDisconnectDisposition, ActorProbe, ActorProbeSnapshot,
    ActorPumpHold, ActorReceiveWitness, ActorRequestDropWitness, ActorRequestDropWitnessSink,
    ActorWakeWitness, SubmitCommandWitness,
};
pub use actor::{
    ActorShutdownReport, RequestCancellation, RequestHandle, SchedulerActor, SchedulerClient,
    Submission, TryRecvOutput,
};
#[cfg(any(test, feature = "deterministic-checkpoint-instrumentation"))]
#[doc(hidden)]
pub use checkpoint::{
    CheckpointAction, CheckpointDirective, CheckpointEffect, CheckpointPlan, CheckpointPoint,
    CheckpointRecord, DeadlineExpirationDisposition, MAX_CHECKPOINT_PLAN_ENTRIES,
};
pub use config::{
    MAX_BATCH_WIDTH, MAX_WAVES_PER_STEP, SchedulerConfig, SchedulerLimits, SchedulingPolicy,
    SharedStaticCharges, StateLayoutSummary,
};
#[cfg(any(test, feature = "actor-stress-instrumentation"))]
#[doc(hidden)]
pub use control::ActorControlCasWitness;
#[cfg(any(test, feature = "actor-stress-instrumentation"))]
#[doc(hidden)]
pub use endpoint::{
    ActorSemanticObservation, ActorSemanticObservationKind, ActorStressRecorder,
    ActorStressRecorderStatus, ActorStressRecording, ActorTryPopKind, ActorTryPopWitness,
};
pub use engine::{PreparedAdmission, SchedulerEngine};
pub use error::{ErrorCategory, SchedulerError, SchedulerResult};
pub use id::RequestId;
pub use ledger::{
    CategorySnapshot, LEDGER_QUANTUM_BYTES, LedgerCategory, LedgerOwnership, LedgerSnapshot,
};
pub use ledger_trace::{
    LedgerMutationKind, LedgerTraceCursor, LedgerTraceEvent, LedgerTraceInitialSnapshot,
    LedgerTraceOwner, LedgerTraceRead, LedgerTraceStatus,
};
pub use request::{
    AcceptedAdmission, BatchAdmission, BatchDeadline, BatchRequestSpec, CancelDisposition,
    EngineSnapshot, OutputEvent, RejectedAdmission, RequestPhase, RequestSpec, ShutdownReport,
    StepReport, TerminalOutcome, TerminalResult,
};
#[cfg(any(test, feature = "m5-run-observer-instrumentation"))]
#[doc(hidden)]
pub use run_observer::{
    MAX_RUN_OBSERVER_OUTPUT_TIMESTAMPS, MAX_RUN_OBSERVER_REQUESTS, RUN_OCCUPANCY_BIN_COUNT,
    RunObservationRead, RunObserver, RunObserverAllocationFingerprint, RunObserverFailure,
    RunObserverStatus, RunObserverTotals, RunRequestObservation,
};
pub use trace::{
    ServicePhase, ServiceTraceCursor, ServiceTraceEvent, ServiceTraceRead, ServiceTraceStatus,
    TRACE_SLOT_CHARGE_BYTES,
};
