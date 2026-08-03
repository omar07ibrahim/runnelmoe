//! Deterministic, bounded cache-policy simulation with byte-accurate accounting.
//!
//! This crate is intentionally independent from the production data plane. It
//! replays policy-neutral demand and causal router-signal traces so policy
//! results can be checked against separate references.

mod engine;
mod error;
mod generator;
mod metrics;
mod model;
mod oracle;
mod policy;
mod trace;

pub use engine::simulate;
pub use error::SimError;
pub use generator::{DEFAULT_MEASURED_STEPS, GeneratedTrace, TraceFamily, generate_trace};
pub use metrics::SimulationMetrics;
pub use model::{
    ExpertPrediction, PageClass, PageDescriptor, PageId, PolicySpec, RouterPolicyConfig, SimLimits,
    SimulationConfig, SimulationResult, TinyLfuConfig, TraceEvent, TraceHeader, ValidatedTrace,
};
pub use oracle::exact_variable_byte_cost;
pub use trace::{parse_trace, serialize_trace};
