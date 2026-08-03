//! Closed data model shared by trace parsing, generation, and replay.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::SimError;

/// Trace-format identifier.
pub const TRACE_SCHEMA: &str = "runnel.cache-trace/1";
/// Trace prefetch timing model.
pub const PREFETCH_MODEL: &str = "instant-between-events-v1";
/// Result-format identifier.
pub const RESULT_SCHEMA: &str = "runnel.cache-result/1";

/// Stable compact identifier in a trace-local page catalog.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct PageId(pub u32);

/// Semantic page class used to expand router predictions.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PageClass {
    /// A page outside sparse-expert prediction.
    Shared,
    /// One page of one sparse expert.
    Expert {
        /// Model layer index.
        layer: u32,
        /// Expert index in the layer.
        expert: u32,
        /// Stable page order within the expert.
        ordinal: u32,
    },
}

/// One immutable catalog entry.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PageDescriptor {
    /// Trace-local page identifier.
    pub id: PageId,
    /// Bytes read by a successful miss or prefetch fill.
    pub logical_bytes: u64,
    /// Bytes charged against page-pool capacity.
    pub charge_bytes: u64,
    /// Semantic role.
    pub class: PageClass,
}

/// One causal prediction supplied before its target route is revealed.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertPrediction {
    /// Expert index in the signaled layer.
    pub expert: u32,
    /// Integer score in parts per million. It is not claimed to be calibrated.
    pub score_ppm: u32,
}

/// Policy-neutral event stream.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TraceEvent {
    /// Causal scores targeting a later request step.
    RouterSignal {
        /// Dense stream sequence.
        sequence: u64,
        /// Request identity.
        request: u64,
        /// Target token or route step.
        target_step: u64,
        /// Target layer.
        layer: u32,
        /// Bounded expert scores.
        predictions: Vec<ExpertPrediction>,
    },
    /// A realized page demand.
    Demand {
        /// Dense stream sequence.
        sequence: u64,
        /// Request identity.
        request: u64,
        /// Realized token or route step.
        step: u64,
        /// Demanded page.
        page: PageId,
    },
}

impl TraceEvent {
    /// Dense sequence number shared by all event variants.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        match self {
            Self::RouterSignal { sequence, .. } | Self::Demand { sequence, .. } => *sequence,
        }
    }
}

/// First canonical JSONL record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TraceHeader {
    /// Literal `header` record kind.
    pub kind: String,
    /// Closed schema identifier.
    pub schema: String,
    /// Short public identifier.
    pub trace_id: String,
    /// Number of catalog records that follow.
    pub page_count: usize,
    /// Number of event records that follow the catalog.
    pub event_count: usize,
    /// Smallest valid cache-allocation quantum.
    pub charge_quantum: u64,
    /// Explicit timing abstraction.
    pub prefetch_model: String,
}

/// Validated, immutable trace ready for replay.
#[derive(Clone, Debug)]
pub struct ValidatedTrace {
    pub(crate) header: TraceHeader,
    pub(crate) pages: Vec<PageDescriptor>,
    pub(crate) page_index: BTreeMap<PageId, usize>,
    pub(crate) expert_pages: BTreeMap<(u32, u32), Vec<PageId>>,
    pub(crate) events: Vec<TraceEvent>,
    pub(crate) sha256: String,
}

/// Read-only catalog projection exposed to online policy decisions.
///
/// Keeping this view separate from [`ValidatedTrace`] makes it impossible for
/// an online policy API to inspect future events while still allowing exact
/// page geometry and expert-group lookups.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PageCatalog<'a> {
    pages: &'a [PageDescriptor],
    page_index: &'a BTreeMap<PageId, usize>,
    expert_pages: &'a BTreeMap<(u32, u32), Vec<PageId>>,
}

impl<'a> PageCatalog<'a> {
    pub(crate) fn page(self, id: PageId) -> &'a PageDescriptor {
        &self.pages[self.page_index[&id]]
    }

    pub(crate) fn expert_pages(self, layer: u32, expert: u32) -> &'a [PageId] {
        self.expert_pages
            .get(&(layer, expert))
            .map_or(&[], Vec::as_slice)
    }
}

impl ValidatedTrace {
    /// Header and declared cost model.
    #[must_use]
    pub const fn header(&self) -> &TraceHeader {
        &self.header
    }

    /// Canonically ordered page catalog.
    #[must_use]
    pub fn pages(&self) -> &[PageDescriptor] {
        &self.pages
    }

    /// Dense, policy-neutral event sequence.
    #[must_use]
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// SHA-256 of the canonical JSONL bytes.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub(crate) fn catalog(&self) -> PageCatalog<'_> {
        PageCatalog {
            pages: &self.pages,
            page_index: &self.page_index,
            expert_pages: &self.expert_pages,
        }
    }

    pub(crate) fn page(&self, id: PageId) -> &PageDescriptor {
        self.catalog().page(id)
    }

    pub(crate) fn expert_pages(&self, layer: u32, expert: u32) -> &[PageId] {
        self.catalog().expert_pages(layer, expert)
    }
}

/// Hard parser and replay ceilings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimLimits {
    /// Maximum canonical trace bytes.
    pub max_trace_bytes: usize,
    /// Maximum bytes in one JSONL record, excluding LF.
    pub max_line_bytes: usize,
    /// Maximum catalog entries.
    pub max_pages: usize,
    /// Maximum replay events.
    pub max_events: usize,
    /// Maximum predictions in one signal.
    pub max_predictions_per_signal: usize,
    /// Maximum states in the tiny exact oracle.
    pub max_exact_oracle_states: usize,
}

impl Default for SimLimits {
    fn default() -> Self {
        Self {
            max_trace_bytes: 5 * 1024 * 1024,
            max_line_bytes: 16 * 1024,
            max_pages: 4_096,
            max_events: 100_000,
            max_predictions_per_signal: 16,
            max_exact_oracle_states: 1_000_000,
        }
    }
}

/// Deterministic TinyLFU configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TinyLfuConfig {
    /// Count-min rows.
    pub sketch_depth: usize,
    /// Counters per row and doorkeeper bits.
    pub sketch_width: usize,
    /// Demand observations between aging operations.
    pub sample_accesses: u64,
}

impl TinyLfuConfig {
    /// Frozen M3 configuration for a given estimated cache entry count.
    ///
    /// Construction is fallible so an adversarial capacity cannot wrap the
    /// preregistered ten-observation-per-entry aging interval.
    pub fn m3(estimated_entries: u64) -> Result<Self, SimError> {
        let sample_accesses = estimated_entries
            .max(1)
            .checked_mul(10)
            .ok_or_else(|| SimError::invalid_config("TinyLFU M3 sample_accesses overflows u64"))?;
        Ok(Self {
            sketch_depth: 4,
            sketch_width: 2_048,
            sample_accesses,
        })
    }
}

/// Bounded router-policy configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouterPolicyConfig {
    /// Protected SLRU byte fraction.
    pub protected_fraction_ppm: u32,
    /// Smallest admitted router score.
    pub minimum_score_ppm: u32,
    /// Maximum expert groups considered per signal.
    pub max_experts_per_signal: usize,
    /// Maximum absent pages admitted per signal.
    pub max_pages_per_signal: usize,
    /// Maximum physical fill bytes admitted per signal.
    pub max_prefetch_bytes_per_signal: u64,
}

/// Complete policy setting recorded in every result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "name", rename_all = "kebab-case", deny_unknown_fields)]
pub enum PolicySpec {
    /// Serve every access without retaining a page.
    NoCache,
    /// Byte-capacity least recently used baseline.
    Lru,
    /// Segmented LRU.
    Slru {
        /// Protected byte fraction.
        protected_fraction_ppm: u32,
    },
    /// TinyLFU admission over byte LRU.
    TinyLfu {
        /// Sketch configuration.
        config: TinyLfuConfig,
    },
    /// Predictive SLRU admission without speculative fills.
    RouterAdmit {
        /// Frozen router configuration.
        config: RouterPolicyConfig,
    },
    /// Predictive SLRU admission with bounded speculative fills.
    RouterPrefetch {
        /// Frozen router configuration.
        config: RouterPolicyConfig,
    },
    /// Exact Bélády/MIN oracle for uniform page geometry only.
    Belady,
}

impl PolicySpec {
    /// Stable short name used in evidence matrices.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::NoCache => "no-cache",
            Self::Lru => "lru",
            Self::Slru { .. } => "slru",
            Self::TinyLfu { .. } => "tiny-lfu",
            Self::RouterAdmit { .. } => "router-admit",
            Self::RouterPrefetch { .. } => "router-prefetch",
            Self::Belady => "belady",
        }
    }
}

/// One replay configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimulationConfig {
    /// Page-pool byte capacity.
    pub capacity_bytes: u64,
    /// Replacement/admission policy.
    pub policy: PolicySpec,
}

/// One deterministic normalized simulation result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationResult {
    /// Closed result schema.
    pub schema: String,
    /// Input trace identifier.
    pub trace_id: String,
    /// Canonical input SHA-256.
    pub trace_sha256: String,
    /// Stable policy name.
    pub policy: String,
    /// Full policy configuration.
    pub policy_spec: PolicySpec,
    /// Page-pool capacity.
    pub capacity_bytes: u64,
    /// Whether the result is an exact uniform-geometry optimum.
    pub oracle_optimal: bool,
    /// Byte and event counters.
    pub metrics: crate::SimulationMetrics,
    /// SHA-256 over normalized decisions.
    pub decision_sha256: String,
}
