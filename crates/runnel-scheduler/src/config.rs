//! Allocation-free validation of scheduler capacities and adapter geometry.

use std::fmt;

use runnel_runtime::{
    AdapterExecutionLayout, DecoderAdapter, SamplingWorkspace, SamplingWorkspaceLayout,
    StateLayoutAccounting,
};

use crate::{
    error::{ErrorCategory, SchedulerError, SchedulerResult},
    trace::SERVICE_TRACE_EVENT_CHARGE_BYTES,
};

pub const LEDGER_ALIGNMENT_BYTES: u64 = 64;
pub const MAX_WORKERS: u64 = 64;
pub const MAX_COMMAND_CAPACITY: u64 = 65_536;
pub const MAX_OUTSTANDING_REQUESTS: u64 = 65_536;
pub const MAX_ACTIVE_REQUESTS: u64 = 4_096;
pub const MAX_QUEUED_REQUESTS: u64 = 65_536;
pub const MAX_RETAINED_TERMINAL_RESULTS: u64 = 65_536;
pub const MAX_REQUEST_TOKENS: u64 = u32::MAX as u64;
pub const MAX_STATE_PAGE_TOKENS: u64 = 65_536;
pub const MAX_OUTPUT_EVENTS_PER_REQUEST: u64 = 65_536;
pub const MAX_BATCH_WIDTH: u64 = 8;
pub const MAX_WAVES_PER_STEP: u64 = 4;
pub const MAX_TRACE_EVENTS: u64 = 1_048_576;
pub const MAX_EXPERT_TASKS_PER_WAVE: u64 = 262_144;
pub const MAX_LOGICAL_MEMORY_BYTES: u64 = 1_u64 << 40;

const COMMAND_SLOT_BYTES: u64 = 64;
const CONTROL_WAKE_BYTES: u64 = 64;
const ACCEPTED_CONTROL_SLOT_BYTES: u64 = 64;
const REQUEST_RECORD_BYTES: u64 = 512;
const REQUEST_SLOT_BYTES: u64 = 64;
const OUTPUT_EVENT_BYTES: u64 = 64;
const TERMINAL_SLOT_BYTES: u64 = 64;
const EXPERT_TASK_ENVELOPE_BYTES: u64 = 64;
const PROMPT_TOKEN_BYTES: u64 = 4;

/// Deterministic scheduler policy selected when an engine is constructed.
///
/// The exact evidence names are frozen by ADR 0007. The continuous policy is
/// the runtime default; the FIFO policy is the deliberately serial M5
/// comparison baseline and selects at most one active request per round.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// Accepted FIFO order, at most one model position per wave from the
    /// oldest request. A blocked head is retried and never bypassed; the next
    /// member becomes eligible only after head removal.
    FifoRunToCompletion,
    /// Equal-weight one-position DRR with stable cross-request expert
    /// coalescing.
    #[default]
    DeficitContinuousExpertCoalesce,
}

impl SchedulingPolicy {
    /// Returns the preregistered stable evidence identifier.
    #[must_use]
    pub const fn evidence_id(self) -> &'static str {
        match self {
            Self::FifoRunToCompletion => "fifo-single-request-run-to-completion-v1",
            Self::DeficitContinuousExpertCoalesce => "deficit-continuous-expert-coalesce-v1",
        }
    }
}

/// User-selected scheduler ceilings before adapter-specific validation.
///
/// Values are `u64` so hostile serialized configurations are checked before
/// host-`usize` conversion. Use [`SchedulerConfig::new`] to bind these limits
/// to one adapter; a `SchedulerLimits` value alone is not validated.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SchedulerLimits {
    pub worker_count: u64,
    pub command_capacity: u64,
    pub max_outstanding_requests: u64,
    pub max_active_requests: u64,
    pub max_queued_requests: u64,
    /// Must cover every outstanding request so terminal publication never
    /// blocks while completed results await an explicit reap.
    pub max_retained_terminal_results: u64,
    pub max_prompt_tokens: u64,
    pub max_new_tokens: u64,
    pub max_context_tokens: u64,
    pub state_page_tokens: u64,
    pub output_capacity_per_request: u64,
    pub batch_width: u64,
    pub waves_per_step: u64,
    pub trace_capacity: u64,
    pub logical_memory_limit_bytes: u64,
    /// Must be zero until a typed constructor can authenticate an attached
    /// cache snapshot; caller-provided cache-size estimates are not trusted.
    pub page_pool_partition_bytes: u64,
    /// Shared semantic capacity for unpublished direct batch-admission
    /// metadata. The adapter-typed minimum is available from
    /// `SchedulerEngine::<A>::required_batch_admission_reserve_bytes`.
    pub admission_reserve_bytes: u64,
}

impl SchedulerLimits {
    /// Small deterministic limits suitable for unit tests and local demos.
    ///
    /// The returned value still requires adapter binding through
    /// [`SchedulerConfig::new`].
    #[must_use]
    pub const fn tiny() -> Self {
        Self {
            worker_count: 1,
            command_capacity: 8,
            max_outstanding_requests: 8,
            max_active_requests: 4,
            max_queued_requests: 8,
            max_retained_terminal_results: 8,
            max_prompt_tokens: 8,
            max_new_tokens: 4,
            max_context_tokens: 16,
            state_page_tokens: 4,
            output_capacity_per_request: 4,
            batch_width: 2,
            waves_per_step: 2,
            trace_capacity: 256,
            logical_memory_limit_bytes: 4 * 1024 * 1024,
            page_pool_partition_bytes: 0,
            admission_reserve_bytes: 64 * 1024,
        }
    }

    /// Frozen M5 evidence limits from ADR 0007.
    ///
    /// Prompt and generation ceilings cover the longest registered evidence
    /// case. Model-resident bytes are intentionally absent: the validated
    /// adapter supplies that exact partition.
    #[must_use]
    pub const fn evidence() -> Self {
        Self {
            worker_count: 1,
            command_capacity: 32,
            max_outstanding_requests: 32,
            max_active_requests: 16,
            max_queued_requests: 32,
            max_retained_terminal_results: 32,
            max_prompt_tokens: 896,
            max_new_tokens: 8,
            max_context_tokens: 1_024,
            state_page_tokens: 16,
            output_capacity_per_request: 64,
            batch_width: 8,
            waves_per_step: 4,
            trace_capacity: 8_192,
            logical_memory_limit_bytes: 8 * 1024 * 1024,
            page_pool_partition_bytes: 0,
            admission_reserve_bytes: 1024 * 1024,
        }
    }
}

impl fmt::Debug for SchedulerLimits {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SchedulerLimits")
            .field("bounded_configuration", &Redacted)
            .finish()
    }
}

/// Adapter state accounting captured without allocating the state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateLayoutSummary {
    max_tokens: usize,
    payload_bytes: u64,
    charge_bytes: u64,
}

impl StateLayoutSummary {
    #[must_use]
    pub const fn max_tokens(self) -> usize {
        self.max_tokens
    }

    #[must_use]
    pub const fn payload_bytes(self) -> u64 {
        self.payload_bytes
    }

    #[must_use]
    pub const fn charge_bytes(self) -> u64 {
        self.charge_bytes
    }
}

/// Exact construction-time charges for scheduler-owned shared partitions.
///
/// Each value is already rounded to the 64-byte ledger quantum. The model
/// partition comes only from the adapter; callers cannot provide an estimate.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SharedStaticCharges {
    actor_command_bytes: u64,
    actor_control_bytes: u64,
    worker_scratch_bytes: u64,
    sampling_scratch_bytes: u64,
    coalesced_batch_bytes: u64,
    model_resident_bytes: u64,
    page_pool_bytes: u64,
    trace_bytes: u64,
    admission_reserve_bytes: u64,
    total_bytes: u64,
}

impl SharedStaticCharges {
    #[must_use]
    pub const fn actor_command_bytes(self) -> u64 {
        self.actor_command_bytes
    }

    #[must_use]
    pub const fn actor_control_bytes(self) -> u64 {
        self.actor_control_bytes
    }

    #[must_use]
    pub const fn worker_scratch_bytes(self) -> u64 {
        self.worker_scratch_bytes
    }

    #[must_use]
    pub const fn sampling_scratch_bytes(self) -> u64 {
        self.sampling_scratch_bytes
    }

    #[must_use]
    pub const fn coalesced_batch_bytes(self) -> u64 {
        self.coalesced_batch_bytes
    }

    #[must_use]
    pub const fn model_resident_bytes(self) -> u64 {
        self.model_resident_bytes
    }

    #[must_use]
    pub const fn page_pool_bytes(self) -> u64 {
        self.page_pool_bytes
    }

    #[must_use]
    pub const fn trace_bytes(self) -> u64 {
        self.trace_bytes
    }

    #[must_use]
    pub const fn admission_reserve_bytes(self) -> u64 {
        self.admission_reserve_bytes
    }

    #[must_use]
    pub const fn total_bytes(self) -> u64 {
        self.total_bytes
    }
}

impl fmt::Debug for SharedStaticCharges {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SharedStaticCharges")
            .field("validated_partitions", &Redacted)
            .finish()
    }
}

/// Fully validated, adapter-bound scheduler configuration.
///
/// Construction performs no scheduler allocation. It validates every count,
/// product, host allocation bound, adapter layout, and minimum executable
/// request before this value can be observed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    limits: SchedulerLimits,
    scheduling_policy: SchedulingPolicy,
    worker_count: usize,
    command_capacity: usize,
    max_outstanding_requests: usize,
    max_active_requests: usize,
    max_queued_requests: usize,
    max_retained_terminal_results: usize,
    max_prompt_tokens: usize,
    max_new_tokens: usize,
    max_context_tokens: usize,
    state_page_tokens: usize,
    output_capacity_per_request: usize,
    batch_width: usize,
    waves_per_step: usize,
    trace_capacity: usize,
    vocabulary_size: usize,
    max_tasks_per_token: usize,
    tasks_per_wave: usize,
    execution_layout: AdapterExecutionLayout,
    sampling_layout: SamplingWorkspaceLayout,
    minimum_state_layout: StateLayoutSummary,
    worst_case_state_layout: StateLayoutSummary,
    shared_static_charges: SharedStaticCharges,
    pending_transaction_charge_bytes: u64,
    output_queue_charge_bytes: u64,
    minimum_request_charge_bytes: u64,
    worst_case_request_charge_bytes: u64,
    minimum_total_charge_bytes: u64,
}

impl SchedulerConfig {
    /// Binds hostile external limits to an adapter and validates all geometry
    /// without allocating request, state, task, trace, or sampling storage.
    pub fn new<A: DecoderAdapter>(adapter: &A, limits: SchedulerLimits) -> SchedulerResult<Self> {
        let scalar = ScalarLimits::validate(limits)?;

        let execution_layout = adapter
            .execution_layout()
            .map_err(|source| SchedulerError::adapter("validating execution layout", source))?;
        let vocabulary_size = adapter.vocabulary_size();
        let sampling_layout = SamplingWorkspace::layout(vocabulary_size)
            .map_err(|source| SchedulerError::sampling("validating workspace layout", source))?;

        let minimum_layout =
            adapter
                .state_layout(1, scalar.state_page_tokens)
                .map_err(|source| {
                    SchedulerError::adapter_with_category(
                        "validating minimum state layout",
                        ErrorCategory::InvalidRequest,
                        source,
                    )
                })?;
        let worst_layout = adapter
            .state_layout(scalar.max_context_tokens, scalar.state_page_tokens)
            .map_err(|source| {
                SchedulerError::adapter_with_category(
                    "validating maximum state layout",
                    ErrorCategory::InvalidRequest,
                    source,
                )
            })?;

        let geometry = AdapterGeometry {
            execution_layout,
            sampling_layout,
            vocabulary_size,
            minimum_state_layout: summarize_state_layout(1, minimum_layout)?,
            worst_case_state_layout: summarize_state_layout(
                scalar.max_context_tokens,
                worst_layout,
            )?,
        };
        Self::from_geometry(limits, scalar, geometry)
    }

    fn from_geometry(
        limits: SchedulerLimits,
        scalar: ScalarLimits,
        geometry: AdapterGeometry,
    ) -> SchedulerResult<Self> {
        validate_adapter_geometry(&geometry)?;

        let max_tasks_per_token = geometry.execution_layout.max_tasks_per_token();
        let max_tasks_u64 = to_u64(max_tasks_per_token, "adapter task count")?;
        let tasks_per_wave_u64 =
            checked_mul(limits.batch_width, max_tasks_u64, "expert tasks per wave")?;
        if tasks_per_wave_u64 > MAX_EXPERT_TASKS_PER_WAVE {
            return Err(SchedulerError::invalid_request(
                "expert tasks per wave",
                "exceeds implementation ceiling",
            ));
        }
        let tasks_per_wave = host_usize(tasks_per_wave_u64, "expert tasks per wave")?;

        let shared_static_charges = shared_static_charges(&limits, &geometry, tasks_per_wave_u64)?;
        let pending_transaction_charge = pending_transaction_charge(geometry.execution_layout)?;
        let output_charge = checked_mul(
            limits.output_capacity_per_request,
            OUTPUT_EVENT_BYTES,
            "output queue capacity",
        )?;
        ensure_host_allocation(output_charge, "output queue capacity")?;

        let minimum_request_charge_bytes = request_charge(
            1,
            geometry.minimum_state_layout.charge_bytes,
            pending_transaction_charge,
            output_charge,
        )?;
        let worst_case_request_charge_bytes = request_charge(
            limits.max_prompt_tokens,
            geometry.worst_case_state_layout.charge_bytes,
            pending_transaction_charge,
            output_charge,
        )?;
        let minimum_total_charge_bytes = checked_add(
            shared_static_charges.total_bytes(),
            minimum_request_charge_bytes,
            "minimum executable scheduler charge",
        )?;
        if minimum_total_charge_bytes > limits.logical_memory_limit_bytes {
            return Err(SchedulerError::resource_exhausted(
                "logical memory minimum operation",
                minimum_total_charge_bytes,
                limits.logical_memory_limit_bytes,
            ));
        }

        Ok(Self {
            limits,
            scheduling_policy: SchedulingPolicy::default(),
            worker_count: scalar.worker_count,
            command_capacity: scalar.command_capacity,
            max_outstanding_requests: scalar.max_outstanding_requests,
            max_active_requests: scalar.max_active_requests,
            max_queued_requests: scalar.max_queued_requests,
            max_retained_terminal_results: scalar.max_retained_terminal_results,
            max_prompt_tokens: scalar.max_prompt_tokens,
            max_new_tokens: scalar.max_new_tokens,
            max_context_tokens: scalar.max_context_tokens,
            state_page_tokens: scalar.state_page_tokens,
            output_capacity_per_request: scalar.output_capacity_per_request,
            batch_width: scalar.batch_width,
            waves_per_step: scalar.waves_per_step,
            trace_capacity: scalar.trace_capacity,
            vocabulary_size: geometry.vocabulary_size,
            max_tasks_per_token,
            tasks_per_wave,
            execution_layout: geometry.execution_layout,
            sampling_layout: geometry.sampling_layout,
            minimum_state_layout: geometry.minimum_state_layout,
            worst_case_state_layout: geometry.worst_case_state_layout,
            shared_static_charges,
            pending_transaction_charge_bytes: pending_transaction_charge,
            output_queue_charge_bytes: output_charge,
            minimum_request_charge_bytes,
            worst_case_request_charge_bytes,
            minimum_total_charge_bytes,
        })
    }

    #[must_use]
    pub const fn limits(&self) -> &SchedulerLimits {
        &self.limits
    }

    /// Returns a copy with an immutable scheduler policy selected.
    ///
    /// Policy selection changes no validated geometry or logical charge.
    #[must_use]
    pub const fn with_scheduling_policy(mut self, policy: SchedulingPolicy) -> Self {
        self.scheduling_policy = policy;
        self
    }

    /// Returns the deterministic scheduler policy bound to this config.
    #[must_use]
    pub const fn scheduling_policy(&self) -> SchedulingPolicy {
        self.scheduling_policy
    }

    #[must_use]
    pub const fn worker_count(&self) -> usize {
        self.worker_count
    }

    #[must_use]
    pub const fn command_capacity(&self) -> usize {
        self.command_capacity
    }

    #[must_use]
    pub const fn max_outstanding_requests(&self) -> usize {
        self.max_outstanding_requests
    }

    #[must_use]
    pub const fn max_active_requests(&self) -> usize {
        self.max_active_requests
    }

    #[must_use]
    pub const fn max_queued_requests(&self) -> usize {
        self.max_queued_requests
    }

    #[must_use]
    pub const fn max_retained_terminal_results(&self) -> usize {
        self.max_retained_terminal_results
    }

    #[must_use]
    pub const fn max_prompt_tokens(&self) -> usize {
        self.max_prompt_tokens
    }

    #[must_use]
    pub const fn max_new_tokens(&self) -> usize {
        self.max_new_tokens
    }

    #[must_use]
    pub const fn max_context_tokens(&self) -> usize {
        self.max_context_tokens
    }

    #[must_use]
    pub const fn state_page_tokens(&self) -> usize {
        self.state_page_tokens
    }

    #[must_use]
    pub const fn output_capacity_per_request(&self) -> usize {
        self.output_capacity_per_request
    }

    #[must_use]
    pub const fn batch_width(&self) -> usize {
        self.batch_width
    }

    #[must_use]
    pub const fn waves_per_step(&self) -> usize {
        self.waves_per_step
    }

    #[must_use]
    pub const fn trace_capacity(&self) -> usize {
        self.trace_capacity
    }

    #[must_use]
    pub const fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    #[must_use]
    pub const fn max_tasks_per_token(&self) -> usize {
        self.max_tasks_per_token
    }

    #[must_use]
    pub const fn tasks_per_wave(&self) -> usize {
        self.tasks_per_wave
    }

    #[must_use]
    pub const fn execution_layout(&self) -> AdapterExecutionLayout {
        self.execution_layout
    }

    #[must_use]
    pub const fn sampling_layout(&self) -> SamplingWorkspaceLayout {
        self.sampling_layout
    }

    #[must_use]
    pub const fn minimum_state_layout(&self) -> StateLayoutSummary {
        self.minimum_state_layout
    }

    #[must_use]
    pub const fn worst_case_state_layout(&self) -> StateLayoutSummary {
        self.worst_case_state_layout
    }

    #[must_use]
    pub const fn logical_memory_limit_bytes(&self) -> u64 {
        self.limits.logical_memory_limit_bytes
    }

    #[must_use]
    pub const fn page_pool_partition_bytes(&self) -> u64 {
        self.limits.page_pool_partition_bytes
    }

    #[must_use]
    pub const fn admission_reserve_bytes(&self) -> u64 {
        self.limits.admission_reserve_bytes
    }

    #[must_use]
    pub const fn model_resident_partition_bytes(&self) -> u64 {
        self.shared_static_charges.model_resident_bytes()
    }

    #[must_use]
    pub const fn shared_static_charges(&self) -> SharedStaticCharges {
        self.shared_static_charges
    }

    #[must_use]
    pub const fn shared_static_charge_bytes(&self) -> u64 {
        self.shared_static_charges.total_bytes()
    }

    #[must_use]
    pub const fn pending_transaction_charge_bytes(&self) -> u64 {
        self.pending_transaction_charge_bytes
    }

    #[must_use]
    pub const fn output_queue_charge_bytes(&self) -> u64 {
        self.output_queue_charge_bytes
    }

    #[must_use]
    pub const fn request_record_charge_bytes(&self) -> u64 {
        REQUEST_RECORD_BYTES
    }

    #[must_use]
    pub const fn request_slot_charge_bytes(&self) -> u64 {
        REQUEST_SLOT_BYTES
    }

    #[must_use]
    pub const fn terminal_slot_charge_bytes(&self) -> u64 {
        TERMINAL_SLOT_BYTES
    }

    #[must_use]
    pub const fn minimum_request_charge_bytes(&self) -> u64 {
        self.minimum_request_charge_bytes
    }

    #[must_use]
    pub const fn worst_case_request_charge_bytes(&self) -> u64 {
        self.worst_case_request_charge_bytes
    }

    #[must_use]
    pub const fn minimum_total_charge_bytes(&self) -> u64 {
        self.minimum_total_charge_bytes
    }
}

impl fmt::Debug for SchedulerConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SchedulerConfig")
            .field("validated_configuration", &Redacted)
            .field("adapter_geometry", &Redacted)
            .finish()
    }
}

#[derive(Clone, Copy)]
struct ScalarLimits {
    worker_count: usize,
    command_capacity: usize,
    max_outstanding_requests: usize,
    max_active_requests: usize,
    max_queued_requests: usize,
    max_retained_terminal_results: usize,
    max_prompt_tokens: usize,
    max_new_tokens: usize,
    max_context_tokens: usize,
    state_page_tokens: usize,
    output_capacity_per_request: usize,
    batch_width: usize,
    waves_per_step: usize,
    trace_capacity: usize,
}

impl ScalarLimits {
    fn validate(limits: SchedulerLimits) -> SchedulerResult<Self> {
        let worker_count = bounded_usize(limits.worker_count, "worker_count", 1, MAX_WORKERS)?;
        let command_capacity = bounded_usize(
            limits.command_capacity,
            "command_capacity",
            1,
            MAX_COMMAND_CAPACITY,
        )?;
        let max_outstanding_requests = bounded_usize(
            limits.max_outstanding_requests,
            "max_outstanding_requests",
            1,
            MAX_OUTSTANDING_REQUESTS,
        )?;
        let max_active_requests = bounded_usize(
            limits.max_active_requests,
            "max_active_requests",
            1,
            MAX_ACTIVE_REQUESTS,
        )?;
        let max_queued_requests = bounded_usize(
            limits.max_queued_requests,
            "max_queued_requests",
            1,
            MAX_QUEUED_REQUESTS,
        )?;
        let max_retained_terminal_results = bounded_usize(
            limits.max_retained_terminal_results,
            "max_retained_terminal_results",
            1,
            MAX_RETAINED_TERMINAL_RESULTS,
        )?;
        let max_prompt_tokens = bounded_usize(
            limits.max_prompt_tokens,
            "max_prompt_tokens",
            1,
            MAX_REQUEST_TOKENS,
        )?;
        let max_new_tokens = bounded_usize(
            limits.max_new_tokens,
            "max_new_tokens",
            1,
            MAX_REQUEST_TOKENS,
        )?;
        let max_context_tokens = bounded_usize(
            limits.max_context_tokens,
            "max_context_tokens",
            1,
            MAX_REQUEST_TOKENS,
        )?;
        let state_page_tokens = bounded_usize(
            limits.state_page_tokens,
            "state_page_tokens",
            1,
            MAX_STATE_PAGE_TOKENS,
        )?;
        let output_capacity_per_request = bounded_usize(
            limits.output_capacity_per_request,
            "output_capacity_per_request",
            1,
            MAX_OUTPUT_EVENTS_PER_REQUEST,
        )?;
        let batch_width = bounded_usize(limits.batch_width, "batch_width", 1, MAX_BATCH_WIDTH)?;
        let waves_per_step = bounded_usize(
            limits.waves_per_step,
            "waves_per_step",
            1,
            MAX_WAVES_PER_STEP,
        )?;
        let trace_capacity =
            bounded_usize(limits.trace_capacity, "trace_capacity", 1, MAX_TRACE_EVENTS)?;

        bounded_u64(
            limits.logical_memory_limit_bytes,
            "logical_memory_limit_bytes",
            1,
            MAX_LOGICAL_MEMORY_BYTES,
        )?;
        ensure_host_allocation(
            limits.logical_memory_limit_bytes,
            "logical_memory_limit_bytes",
        )?;
        bounded_u64(
            limits.page_pool_partition_bytes,
            "page_pool_partition_bytes",
            0,
            MAX_LOGICAL_MEMORY_BYTES,
        )?;
        ensure_host_allocation(
            limits.page_pool_partition_bytes,
            "page_pool_partition_bytes",
        )?;
        if limits.page_pool_partition_bytes != 0 {
            return Err(SchedulerError::unsupported(
                "authenticated page-pool partitions",
            ));
        }
        bounded_u64(
            limits.admission_reserve_bytes,
            "admission_reserve_bytes",
            1,
            MAX_LOGICAL_MEMORY_BYTES,
        )?;
        ensure_host_allocation(limits.admission_reserve_bytes, "admission_reserve_bytes")?;

        if max_active_requests > max_outstanding_requests {
            return Err(SchedulerError::invalid_request(
                "max_active_requests",
                "cannot exceed max_outstanding_requests",
            ));
        }
        if max_queued_requests > max_outstanding_requests {
            return Err(SchedulerError::invalid_request(
                "max_queued_requests",
                "cannot exceed max_outstanding_requests",
            ));
        }
        if max_retained_terminal_results < max_outstanding_requests {
            return Err(SchedulerError::invalid_request(
                "max_retained_terminal_results",
                "must cover every max_outstanding_requests slot",
            ));
        }
        if max_prompt_tokens > max_context_tokens {
            return Err(SchedulerError::invalid_request(
                "max_prompt_tokens",
                "cannot exceed max_context_tokens",
            ));
        }
        if state_page_tokens > max_context_tokens {
            return Err(SchedulerError::invalid_request(
                "state_page_tokens",
                "cannot exceed max_context_tokens",
            ));
        }
        let maximum_model_positions = limits
            .max_prompt_tokens
            .checked_add(limits.max_new_tokens - 1)
            .ok_or_else(|| {
                SchedulerError::invalid_request(
                    "max_prompt_tokens/max_new_tokens",
                    "model-position count overflows",
                )
            })?;
        if maximum_model_positions > limits.max_context_tokens {
            return Err(SchedulerError::invalid_request(
                "max_prompt_tokens/max_new_tokens",
                "maximum model positions exceed max_context_tokens",
            ));
        }

        Ok(Self {
            worker_count,
            command_capacity,
            max_outstanding_requests,
            max_active_requests,
            max_queued_requests,
            max_retained_terminal_results,
            max_prompt_tokens,
            max_new_tokens,
            max_context_tokens,
            state_page_tokens,
            output_capacity_per_request,
            batch_width,
            waves_per_step,
            trace_capacity,
        })
    }
}

#[derive(Clone, Copy)]
struct AdapterGeometry {
    execution_layout: AdapterExecutionLayout,
    sampling_layout: SamplingWorkspaceLayout,
    vocabulary_size: usize,
    minimum_state_layout: StateLayoutSummary,
    worst_case_state_layout: StateLayoutSummary,
}

fn validate_adapter_geometry(geometry: &AdapterGeometry) -> SchedulerResult<()> {
    if geometry.vocabulary_size == 0
        || geometry.sampling_layout.vocab_size() != geometry.vocabulary_size
    {
        return Err(SchedulerError::internal(
            "sampling layout does not match adapter vocabulary",
        ));
    }
    validate_host_usize(
        geometry.vocabulary_size,
        "adapter vocabulary allocation count",
    )?;

    let sampling_payload = to_u64(
        geometry.sampling_layout.payload_bytes(),
        "sampling workspace payload",
    )?;
    let sampling_charge = to_u64(
        geometry.sampling_layout.charge_bytes(),
        "sampling workspace charge",
    )?;
    validate_payload_charge(
        sampling_payload,
        sampling_charge,
        "sampling workspace layout",
    )?;

    let execution = geometry.execution_layout;
    validate_payload_charge(
        to_u64(
            execution.workspace_payload_bytes(),
            "adapter workspace payload",
        )?,
        to_u64(
            execution.workspace_charge_bytes(),
            "adapter workspace charge",
        )?,
        "adapter workspace layout",
    )?;
    validate_payload_charge(
        to_u64(
            execution.model_resident_payload_bytes(),
            "model resident payload",
        )?,
        to_u64(
            execution.model_resident_charge_bytes(),
            "model resident charge",
        )?,
        "model resident layout",
    )?;
    for (payload, resource) in [
        (
            execution.prepared_payload_bytes(),
            "adapter prepared-token payload",
        ),
        (
            execution.task_payload_bytes(),
            "adapter expert-task payload",
        ),
        (
            execution.contribution_payload_bytes(),
            "adapter contribution payload",
        ),
        (
            execution.pending_payload_bytes(),
            "adapter pending-token payload",
        ),
    ] {
        ensure_host_allocation(to_u64(payload, resource)?, resource)?;
    }

    validate_state_summary(
        geometry.minimum_state_layout,
        "minimum adapter state layout",
    )?;
    validate_state_summary(
        geometry.worst_case_state_layout,
        "maximum adapter state layout",
    )?;
    if geometry.minimum_state_layout.charge_bytes > geometry.worst_case_state_layout.charge_bytes
        || geometry.minimum_state_layout.payload_bytes
            > geometry.worst_case_state_layout.payload_bytes
    {
        return Err(SchedulerError::internal(
            "adapter state accounting shrinks at the configured maximum",
        ));
    }
    Ok(())
}

fn validate_state_summary(
    summary: StateLayoutSummary,
    resource: &'static str,
) -> SchedulerResult<()> {
    if summary.max_tokens == 0 {
        return Err(SchedulerError::internal(
            "adapter state layout has zero token capacity",
        ));
    }
    validate_payload_charge(summary.payload_bytes, summary.charge_bytes, resource)
}

fn summarize_state_layout<L: StateLayoutAccounting>(
    max_tokens: usize,
    layout: L,
) -> SchedulerResult<StateLayoutSummary> {
    Ok(StateLayoutSummary {
        max_tokens,
        payload_bytes: to_u64(layout.payload_bytes(), "adapter state payload")?,
        charge_bytes: to_u64(layout.charge_bytes(), "adapter state charge")?,
    })
}

fn shared_static_charges(
    limits: &SchedulerLimits,
    geometry: &AdapterGeometry,
    tasks_per_wave: u64,
) -> SchedulerResult<SharedStaticCharges> {
    let offered_prompt = checked_mul(
        limits.max_prompt_tokens,
        PROMPT_TOKEN_BYTES,
        "maximum offered prompt payload",
    )?;
    let command_slot = round_charge(
        checked_add(
            COMMAND_SLOT_BYTES,
            offered_prompt,
            "ordinary command slot capacity",
        )?,
        "ordinary command slot capacity",
    )?;
    let commands = checked_mul(
        limits.command_capacity,
        command_slot,
        "ordinary command capacity",
    )?;
    ensure_host_allocation(commands, "ordinary command capacity")?;
    let accepted_controls = checked_mul(
        limits.max_outstanding_requests,
        ACCEPTED_CONTROL_SLOT_BYTES,
        "accepted actor control capacity",
    )?;
    let actor_control = checked_add(
        CONTROL_WAKE_BYTES,
        accepted_controls,
        "actor control and wake capacity",
    )?;
    ensure_host_allocation(actor_control, "actor control and wake capacity")?;
    let workspace_per_worker = to_u64(
        geometry.execution_layout.workspace_charge_bytes(),
        "adapter workspace charge",
    )?;
    let worker_scratch = checked_mul(
        limits.worker_count,
        workspace_per_worker,
        "worker scratch capacity",
    )?;
    ensure_host_allocation(worker_scratch, "worker scratch capacity")?;

    let task_and_contribution_bytes = checked_sum(
        &[
            EXPERT_TASK_ENVELOPE_BYTES,
            to_u64(
                geometry.execution_layout.task_payload_bytes(),
                "adapter task payload",
            )?,
            to_u64(
                geometry.execution_layout.contribution_payload_bytes(),
                "adapter contribution payload",
            )?,
        ],
        "coalesced per-task capacity",
    )?;
    let coalesced_batch = round_charge(
        checked_mul(
            tasks_per_wave,
            task_and_contribution_bytes,
            "coalesced batch capacity",
        )?,
        "coalesced batch capacity",
    )?;
    ensure_host_allocation(coalesced_batch, "coalesced batch capacity")?;

    let trace = checked_mul(
        limits.trace_capacity,
        SERVICE_TRACE_EVENT_CHARGE_BYTES,
        "trace capacity",
    )?;
    ensure_host_allocation(trace, "trace capacity")?;
    let admission_reserve =
        round_charge(limits.admission_reserve_bytes, "admission reserve capacity")?;
    let page_pool = round_charge(
        limits.page_pool_partition_bytes,
        "page-pool partition capacity",
    )?;

    let sampling_scratch = to_u64(
        geometry.sampling_layout.charge_bytes(),
        "sampling scratch capacity",
    )?;
    let model_resident = to_u64(
        geometry.execution_layout.model_resident_charge_bytes(),
        "model resident partition",
    )?;
    let total_bytes = checked_sum(
        &[
            commands,
            actor_control,
            worker_scratch,
            sampling_scratch,
            coalesced_batch,
            model_resident,
            page_pool,
            trace,
            admission_reserve,
        ],
        "shared static scheduler charge",
    )?;
    Ok(SharedStaticCharges {
        actor_command_bytes: commands,
        actor_control_bytes: actor_control,
        worker_scratch_bytes: worker_scratch,
        sampling_scratch_bytes: sampling_scratch,
        coalesced_batch_bytes: coalesced_batch,
        model_resident_bytes: model_resident,
        page_pool_bytes: page_pool,
        trace_bytes: trace,
        admission_reserve_bytes: admission_reserve,
        total_bytes,
    })
}

fn pending_transaction_charge(execution: AdapterExecutionLayout) -> SchedulerResult<u64> {
    let prepared = to_u64(
        execution.prepared_payload_bytes(),
        "prepared-token capacity",
    )?;
    let pending = to_u64(execution.pending_payload_bytes(), "pending-token capacity")?;
    round_charge(
        checked_add(prepared, pending, "pending transaction capacity")?,
        "pending transaction capacity",
    )
}

fn request_charge(
    prompt_tokens: u64,
    state_charge: u64,
    pending_transaction_charge: u64,
    output_charge: u64,
) -> SchedulerResult<u64> {
    let prompt = round_charge(
        checked_mul(prompt_tokens, PROMPT_TOKEN_BYTES, "prompt storage")?,
        "prompt storage",
    )?;
    ensure_host_allocation(prompt, "prompt storage")?;
    checked_sum(
        &[
            prompt,
            REQUEST_RECORD_BYTES,
            REQUEST_SLOT_BYTES,
            state_charge,
            pending_transaction_charge,
            output_charge,
            TERMINAL_SLOT_BYTES,
        ],
        "request logical charge",
    )
}

fn validate_payload_charge(
    payload: u64,
    charge: u64,
    resource: &'static str,
) -> SchedulerResult<()> {
    if charge < payload || !charge.is_multiple_of(LEDGER_ALIGNMENT_BYTES) {
        return Err(SchedulerError::internal(
            "adapter layout charge is smaller than payload or is unaligned",
        ));
    }
    ensure_host_allocation(payload, resource)?;
    ensure_host_allocation(charge, resource)
}

fn bounded_usize(
    value: u64,
    field: &'static str,
    minimum: u64,
    maximum: u64,
) -> SchedulerResult<usize> {
    bounded_u64(value, field, minimum, maximum)?;
    host_usize(value, field)
}

fn bounded_u64(value: u64, field: &'static str, minimum: u64, maximum: u64) -> SchedulerResult<()> {
    if value < minimum {
        return Err(SchedulerError::invalid_request(
            field,
            "is below the minimum",
        ));
    }
    if value > maximum {
        return Err(SchedulerError::invalid_request(
            field,
            "exceeds implementation ceiling",
        ));
    }
    Ok(())
}

fn host_usize(value: u64, field: &'static str) -> SchedulerResult<usize> {
    usize::try_from(value)
        .map_err(|_| SchedulerError::invalid_request(field, "does not fit host usize"))
}

fn validate_host_usize(value: usize, field: &'static str) -> SchedulerResult<()> {
    let value = to_u64(value, field)?;
    ensure_host_allocation(value, field)
}

fn ensure_host_allocation(bytes: u64, field: &'static str) -> SchedulerResult<()> {
    if bytes > isize::MAX as u64 {
        return Err(SchedulerError::invalid_request(
            field,
            "exceeds the host isize allocation bound",
        ));
    }
    Ok(())
}

fn to_u64(value: usize, field: &'static str) -> SchedulerResult<u64> {
    u64::try_from(value)
        .map_err(|_| SchedulerError::invalid_request(field, "does not fit scheduler accounting"))
}

fn checked_mul(left: u64, right: u64, field: &'static str) -> SchedulerResult<u64> {
    left.checked_mul(right)
        .ok_or_else(|| SchedulerError::invalid_request(field, "checked multiplication overflows"))
}

fn checked_add(left: u64, right: u64, field: &'static str) -> SchedulerResult<u64> {
    left.checked_add(right)
        .ok_or_else(|| SchedulerError::invalid_request(field, "checked addition overflows"))
}

fn checked_sum(values: &[u64], field: &'static str) -> SchedulerResult<u64> {
    values
        .iter()
        .try_fold(0_u64, |total, value| checked_add(total, *value, field))
}

fn round_charge(payload: u64, field: &'static str) -> SchedulerResult<u64> {
    let rounded = payload
        .checked_add(LEDGER_ALIGNMENT_BYTES - 1)
        .map(|value| value / LEDGER_ALIGNMENT_BYTES * LEDGER_ALIGNMENT_BYTES)
        .ok_or_else(|| SchedulerError::invalid_request(field, "ledger rounding overflows"))?;
    ensure_host_allocation(rounded, field)?;
    Ok(rounded)
}

struct Redacted;

impl fmt::Debug for Redacted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted>")
    }
}

#[cfg(test)]
mod tests {
    use runnel_fixture::FixtureArtifact;
    use runnel_format::{Artifact, Limits};
    use runnel_runtime::{BackendRequest, TinyModel};

    use super::*;

    fn test_geometry() -> AdapterGeometry {
        AdapterGeometry {
            execution_layout: AdapterExecutionLayout::new(2, 152, 32, 32, 216, 192, 5_600).unwrap(),
            sampling_layout: SamplingWorkspace::layout(32).unwrap(),
            vocabulary_size: 32,
            minimum_state_layout: StateLayoutSummary {
                max_tokens: 1,
                payload_bytes: 1_024,
                charge_bytes: 1_088,
            },
            worst_case_state_layout: StateLayoutSummary {
                max_tokens: 1_024,
                payload_bytes: 65_536,
                charge_bytes: 69_632,
            },
        }
    }

    fn build(limits: SchedulerLimits) -> SchedulerResult<SchedulerConfig> {
        let scalar = ScalarLimits::validate(limits)?;
        SchedulerConfig::from_geometry(limits, scalar, test_geometry())
    }

    #[test]
    fn frozen_evidence_configuration_has_exact_derived_geometry() {
        let config = build(SchedulerLimits::evidence()).unwrap();
        assert_eq!(config.worker_count(), 1);
        assert_eq!(config.batch_width(), 8);
        assert_eq!(config.waves_per_step(), 4);
        assert_eq!(config.max_tasks_per_token(), 2);
        assert_eq!(config.tasks_per_wave(), 16);
        assert_eq!(config.vocabulary_size(), 32);
        assert_eq!(config.model_resident_partition_bytes(), 5_632);
        assert_eq!(config.sampling_layout().charge_bytes(), 1_024);
        assert_eq!(config.minimum_state_layout().charge_bytes(), 1_088);
        assert_eq!(config.worst_case_state_layout().charge_bytes(), 69_632);
        let shared = config.shared_static_charges();
        let reconstructed = [
            shared.actor_command_bytes(),
            shared.actor_control_bytes(),
            shared.worker_scratch_bytes(),
            shared.sampling_scratch_bytes(),
            shared.coalesced_batch_bytes(),
            shared.model_resident_bytes(),
            shared.page_pool_bytes(),
            shared.trace_bytes(),
            shared.admission_reserve_bytes(),
        ]
        .into_iter()
        .sum::<u64>();
        assert_eq!(reconstructed, shared.total_bytes());
        assert_eq!(shared.total_bytes(), config.shared_static_charge_bytes());
        assert_eq!(shared.actor_command_bytes(), 116_736);
        assert_eq!(shared.actor_control_bytes(), 2_112);
        assert_eq!(config.shared_static_charge_bytes(), 2_224_896);
        assert_eq!(config.pending_transaction_charge_bytes(), 384);
        assert_eq!(config.output_queue_charge_bytes(), 4_096);
        assert_eq!(config.minimum_request_charge_bytes(), 6_272);
        assert_eq!(config.worst_case_request_charge_bytes(), 78_336);
        assert_eq!(config.minimum_total_charge_bytes(), 2_231_168);
        assert!(config.minimum_total_charge_bytes() <= config.logical_memory_limit_bytes());
    }

    #[test]
    fn public_constructor_binds_the_generated_v3_adapter_without_state_allocation() {
        let fixture = FixtureArtifact::build_v3();
        let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default()).unwrap();
        let model =
            TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar).unwrap();

        let config = SchedulerConfig::new(&model, SchedulerLimits::evidence()).unwrap();
        assert_eq!(config.vocabulary_size(), 32);
        assert_eq!(config.execution_layout().max_tasks_per_token(), 2);
        assert_eq!(config.minimum_state_layout().charge_bytes(), 1_088);
        assert_eq!(config.worst_case_state_layout().charge_bytes(), 69_632);
        assert_eq!(config.model_resident_partition_bytes(), 5_632);
    }

    #[test]
    fn zero_and_frozen_policy_ceiling_boundaries_are_checked() {
        for mutate in [
            |limits: &mut SchedulerLimits| limits.worker_count = 0,
            |limits: &mut SchedulerLimits| limits.command_capacity = 0,
            |limits: &mut SchedulerLimits| limits.max_outstanding_requests = 0,
            |limits: &mut SchedulerLimits| limits.max_active_requests = 0,
            |limits: &mut SchedulerLimits| limits.max_queued_requests = 0,
            |limits: &mut SchedulerLimits| limits.max_retained_terminal_results = 0,
            |limits: &mut SchedulerLimits| limits.max_prompt_tokens = 0,
            |limits: &mut SchedulerLimits| limits.max_new_tokens = 0,
            |limits: &mut SchedulerLimits| limits.max_context_tokens = 0,
            |limits: &mut SchedulerLimits| limits.state_page_tokens = 0,
            |limits: &mut SchedulerLimits| limits.output_capacity_per_request = 0,
            |limits: &mut SchedulerLimits| limits.batch_width = 0,
            |limits: &mut SchedulerLimits| limits.waves_per_step = 0,
            |limits: &mut SchedulerLimits| limits.trace_capacity = 0,
            |limits: &mut SchedulerLimits| limits.logical_memory_limit_bytes = 0,
            |limits: &mut SchedulerLimits| limits.admission_reserve_bytes = 0,
        ] {
            let mut limits = SchedulerLimits::evidence();
            mutate(&mut limits);
            assert_eq!(
                build(limits).unwrap_err().category(),
                ErrorCategory::InvalidRequest
            );
        }

        let mut exact = SchedulerLimits::evidence();
        exact.worker_count = MAX_WORKERS;
        exact.command_capacity = MAX_COMMAND_CAPACITY;
        exact.max_outstanding_requests = MAX_OUTSTANDING_REQUESTS;
        exact.max_active_requests = MAX_ACTIVE_REQUESTS;
        exact.max_queued_requests = MAX_OUTSTANDING_REQUESTS;
        exact.max_retained_terminal_results = MAX_OUTSTANDING_REQUESTS;
        exact.output_capacity_per_request = MAX_OUTPUT_EVENTS_PER_REQUEST;
        exact.batch_width = MAX_BATCH_WIDTH;
        exact.waves_per_step = MAX_WAVES_PER_STEP;
        exact.trace_capacity = MAX_TRACE_EVENTS;
        exact.logical_memory_limit_bytes = MAX_LOGICAL_MEMORY_BYTES;
        assert!(build(exact).is_ok());

        for mutate in [
            |limits: &mut SchedulerLimits| limits.worker_count = MAX_WORKERS + 1,
            |limits: &mut SchedulerLimits| {
                limits.command_capacity = MAX_COMMAND_CAPACITY + 1;
            },
            |limits: &mut SchedulerLimits| {
                limits.max_outstanding_requests = MAX_OUTSTANDING_REQUESTS + 1;
            },
            |limits: &mut SchedulerLimits| limits.max_active_requests = MAX_ACTIVE_REQUESTS + 1,
            |limits: &mut SchedulerLimits| limits.max_queued_requests = MAX_QUEUED_REQUESTS + 1,
            |limits: &mut SchedulerLimits| {
                limits.max_retained_terminal_results = MAX_RETAINED_TERMINAL_RESULTS + 1;
            },
            |limits: &mut SchedulerLimits| {
                limits.output_capacity_per_request = MAX_OUTPUT_EVENTS_PER_REQUEST + 1;
            },
            |limits: &mut SchedulerLimits| limits.batch_width = MAX_BATCH_WIDTH + 1,
            |limits: &mut SchedulerLimits| limits.waves_per_step = MAX_WAVES_PER_STEP + 1,
            |limits: &mut SchedulerLimits| limits.trace_capacity = MAX_TRACE_EVENTS + 1,
            |limits: &mut SchedulerLimits| {
                limits.logical_memory_limit_bytes = MAX_LOGICAL_MEMORY_BYTES + 1;
            },
        ] {
            let mut limits = SchedulerLimits::evidence();
            mutate(&mut limits);
            assert_eq!(
                build(limits).unwrap_err().category(),
                ErrorCategory::InvalidRequest
            );
        }
    }

    #[test]
    fn output_capacity_one_is_valid_and_independent_of_generation_limit() {
        let mut limits = SchedulerLimits::evidence();
        limits.output_capacity_per_request = 1;
        limits.max_new_tokens = 65;
        limits.max_prompt_tokens = 896;
        limits.max_context_tokens = 1_024;
        assert!(build(limits).is_ok());
    }

    #[test]
    fn terminal_retention_covers_every_outstanding_request() {
        let mut below = SchedulerLimits::evidence();
        below.max_retained_terminal_results = below.max_outstanding_requests - 1;
        assert!(matches!(
            build(below),
            Err(SchedulerError::InvalidRequest {
                field: "max_retained_terminal_results",
                problem: "must cover every max_outstanding_requests slot",
            })
        ));

        let mut equal = SchedulerLimits::evidence();
        equal.max_retained_terminal_results = equal.max_outstanding_requests;
        let config = build(equal).unwrap();
        assert_eq!(
            config.max_retained_terminal_results(),
            config.max_outstanding_requests()
        );
    }

    #[test]
    fn caller_supplied_page_pool_estimate_is_not_a_trusted_partition() {
        let mut limits = SchedulerLimits::evidence();
        limits.page_pool_partition_bytes = LEDGER_ALIGNMENT_BYTES;
        let error = build(limits).unwrap_err();
        assert_eq!(error.category(), ErrorCategory::Unsupported);
        assert!(!error.to_string().contains("cache path"));
    }

    #[test]
    fn token_page_partition_and_reserve_ceiling_helpers_are_exact() {
        for field in ["max_prompt_tokens", "max_new_tokens", "max_context_tokens"] {
            assert_eq!(
                bounded_usize(MAX_REQUEST_TOKENS, field, 1, MAX_REQUEST_TOKENS).unwrap(),
                MAX_REQUEST_TOKENS as usize
            );
            assert!(bounded_usize(MAX_REQUEST_TOKENS + 1, field, 1, MAX_REQUEST_TOKENS).is_err());
        }
        assert!(
            bounded_usize(
                MAX_STATE_PAGE_TOKENS,
                "state_page_tokens",
                1,
                MAX_STATE_PAGE_TOKENS,
            )
            .is_ok()
        );
        assert!(
            bounded_usize(
                MAX_STATE_PAGE_TOKENS + 1,
                "state_page_tokens",
                1,
                MAX_STATE_PAGE_TOKENS,
            )
            .is_err()
        );
        for (field, minimum) in [
            ("page_pool_partition_bytes", 0),
            ("admission_reserve_bytes", 1),
        ] {
            assert!(
                bounded_u64(
                    MAX_LOGICAL_MEMORY_BYTES,
                    field,
                    minimum,
                    MAX_LOGICAL_MEMORY_BYTES
                )
                .is_ok()
            );
            assert!(
                bounded_u64(
                    MAX_LOGICAL_MEMORY_BYTES + 1,
                    field,
                    minimum,
                    MAX_LOGICAL_MEMORY_BYTES,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn adapter_task_product_exact_ceiling_and_plus_one_are_rejected_deterministically() {
        let mut limits = SchedulerLimits::evidence();
        limits.logical_memory_limit_bytes = MAX_LOGICAL_MEMORY_BYTES;
        let scalar = ScalarLimits::validate(limits).unwrap();

        let mut geometry = test_geometry();
        geometry.execution_layout = AdapterExecutionLayout::new(32_768, 0, 0, 0, 0, 0, 0).unwrap();
        let exact = SchedulerConfig::from_geometry(limits, scalar, geometry).unwrap();
        assert_eq!(exact.tasks_per_wave() as u64, MAX_EXPERT_TASKS_PER_WAVE);

        geometry.execution_layout = AdapterExecutionLayout::new(32_769, 0, 0, 0, 0, 0, 0).unwrap();
        let error = SchedulerConfig::from_geometry(limits, scalar, geometry).unwrap_err();
        assert_eq!(error.category(), ErrorCategory::InvalidRequest);
    }

    #[test]
    fn exact_fit_and_one_byte_short_minimum_operation_are_distinct() {
        let initial = build(SchedulerLimits::evidence()).unwrap();
        let exact_bytes = initial.minimum_total_charge_bytes();

        let mut exact = SchedulerLimits::evidence();
        exact.logical_memory_limit_bytes = exact_bytes;
        assert_eq!(
            build(exact).unwrap().minimum_total_charge_bytes(),
            exact_bytes
        );

        let mut short = exact;
        short.logical_memory_limit_bytes = exact_bytes - 1;
        let error = build(short).unwrap_err();
        assert_eq!(error.category(), ErrorCategory::ResourceExhausted);
    }

    #[test]
    fn count_products_rounding_and_host_bounds_reject_before_allocation() {
        assert_eq!(
            checked_mul(u64::MAX, 2, "hostile product")
                .unwrap_err()
                .category(),
            ErrorCategory::InvalidRequest
        );
        assert_eq!(
            round_charge(u64::MAX, "hostile charge")
                .unwrap_err()
                .category(),
            ErrorCategory::InvalidRequest
        );
        assert_eq!(
            ensure_host_allocation(isize::MAX as u64 + 1, "hostile allocation")
                .unwrap_err()
                .category(),
            ErrorCategory::InvalidRequest
        );
    }

    #[test]
    fn context_and_relationship_checks_precede_adapter_or_allocation_work() {
        let mut limits = SchedulerLimits::evidence();
        limits.max_active_requests = limits.max_outstanding_requests + 1;
        assert!(build(limits).is_err());

        let mut limits = SchedulerLimits::evidence();
        limits.max_prompt_tokens = 1_020;
        limits.max_new_tokens = 8;
        assert!(build(limits).is_err());

        let mut limits = SchedulerLimits::evidence();
        limits.state_page_tokens = limits.max_context_tokens + 1;
        assert!(build(limits).is_err());
    }

    #[test]
    fn adapter_layout_must_be_aligned_monotone_and_vocabulary_consistent() {
        let limits = SchedulerLimits::evidence();
        let scalar = ScalarLimits::validate(limits).unwrap();
        let mut geometry = test_geometry();
        geometry.minimum_state_layout.charge_bytes -= 1;
        assert_eq!(
            SchedulerConfig::from_geometry(limits, scalar, geometry)
                .unwrap_err()
                .category(),
            ErrorCategory::Internal
        );

        let mut geometry = test_geometry();
        geometry.sampling_layout = SamplingWorkspace::layout(31).unwrap();
        assert_eq!(
            SchedulerConfig::from_geometry(limits, scalar, geometry)
                .unwrap_err()
                .category(),
            ErrorCategory::Internal
        );

        let mut geometry = test_geometry();
        geometry.minimum_state_layout.charge_bytes =
            geometry.worst_case_state_layout.charge_bytes + LEDGER_ALIGNMENT_BYTES;
        assert_eq!(
            SchedulerConfig::from_geometry(limits, scalar, geometry)
                .unwrap_err()
                .category(),
            ErrorCategory::Internal
        );
    }

    #[test]
    fn debug_does_not_disclose_hostile_limit_values_or_geometry() {
        let sentinel = 7_829_113_u64;
        let mut limits = SchedulerLimits::evidence();
        limits.logical_memory_limit_bytes = sentinel;
        assert!(!format!("{limits:?}").contains(&sentinel.to_string()));

        let config = build(limits).unwrap();
        let debug = format!("{config:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains(&sentinel.to_string()));
    }
}
