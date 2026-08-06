//! Sealed, allocation-stable M5 run instrumentation.
//!
//! The observer is harness-owned and weakly bound to one scheduler control
//! domain. The engine calls only the private [`StepObserver`] interface; no
//! callback or trait object crosses the public boundary. The ordinary engine
//! path uses [`NullObserver`], which monomorphizes to a zero-sized no-op.

#![cfg_attr(
    not(any(test, feature = "m5-run-observer-instrumentation")),
    allow(
        dead_code,
        reason = "the concrete observer is an opt-in evidence surface"
    )
)]

use std::{fmt, mem::size_of, time::Instant};

use crate::{
    RequestId, SchedulerConfig, StepReport, TerminalOutcome, TerminalResult,
    config::MAX_BATCH_WIDTH,
    control::{ControlDomain, ControlRegistry},
    error::{SchedulerError, SchedulerResult},
    request::CancelDisposition,
    trace::ServicePhase,
};

/// Closed accepted-request ceiling of the preregistered M5 observer.
pub const MAX_RUN_OBSERVER_REQUESTS: usize = 32;
/// Closed output-timestamp ceiling of the preregistered M5 observer.
pub const MAX_RUN_OBSERVER_OUTPUT_TIMESTAMPS: usize = 256;
/// Wave-width histogram bins `0..=8`.
pub const RUN_OCCUPANCY_BIN_COUNT: usize = MAX_BATCH_WIDTH as usize + 1;

const RUN_CLOCK_ID: &str = "std-instant-origin-nanoseconds-v1";

/// Observer-owned monotonic clock used by the sealed evidence path.
#[derive(Clone, Copy)]
pub(crate) struct RunClock {
    origin: Instant,
}

impl RunClock {
    /// Starts one child-local origin before preparation and release.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }

    /// Returns a saturating unsigned nanosecond offset from this clock origin.
    ///
    /// Saturation is independently detected by the observer and poisons the
    /// run. It cannot alter scheduler execution.
    #[must_use]
    pub(crate) fn now_ns(self) -> u64 {
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// Stable evidence identifier for this clock implementation.
    #[must_use]
    pub(crate) const fn clock_id(self) -> &'static str {
        RUN_CLOCK_ID
    }
}

impl fmt::Debug for RunClock {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunClock")
            .field("clock_id", &RUN_CLOCK_ID)
            .field("origin", &"<redacted>")
            .finish()
    }
}

/// First sticky reason an observed run became ineligible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RunObserverFailure {
    DuplicateAdmission,
    AdmissionCapacity,
    UnknownRequest,
    ClockRegression,
    ClockSaturated,
    CounterOverflow,
    InvalidWaveOccupancy,
    DuplicateMilestone,
    NoncontiguousPosition,
    NoncontiguousOutput,
    InvalidPhaseBoundary,
    InvalidTerminal,
    MissingCancellationBoundary,
    CallbackAfterFinish,
    IncompleteRun,
    InconsistentSummary,
}

/// Checked run-level counters retained without per-step allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunObserverTotals {
    pub step_count: u64,
    pub wave_count: u64,
    pub selected_positions: u64,
    pub expert_contributions: u64,
    pub expert_group_calls: u64,
    pub committed_positions: u64,
    pub terminal_decisions: u64,
    pub preempted_survivors: u64,
    pub resumed_requests: u64,
    /// Number of timestamp-bearing milestones retained or validated.
    ///
    /// One physical clock sample may populate more than one milestone; this is
    /// deliberately not represented as a hardware/OS clock-read count.
    pub timestamp_observations: u64,
    pub state_live_token_sample_sum: u64,
    pub state_allocated_page_slot_sample_sum: u64,
    pub state_sample_count: u64,
}

/// One accepted synthetic request and its exact internal timing milestones.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RunRequestObservation {
    offered_index: usize,
    request_id: RequestId,
    prompt_len: usize,
    max_new_tokens: usize,
    total_positions: usize,
    resolved_deadline_ns: Option<u64>,
    admitted_ns: u64,
    first_work_start_ns: Option<u64>,
    first_decode_start_ns: Option<u64>,
    prefill_complete_ns: Option<u64>,
    cancel_linearized_ns: Option<u64>,
    terminal_decided_ns: Option<u64>,
    terminal_outcome: Option<TerminalOutcome>,
    request_owned_zero_ns: Option<u64>,
    worker_quiescent_ns: Option<u64>,
    committed_positions: usize,
    emitted_tokens: usize,
    preemption_count: u64,
    resume_count: u64,
    output_offset: usize,
    allocated_page_slots: usize,
    state_active: bool,
    pending_work_position: Option<usize>,
}

impl RunRequestObservation {
    #[must_use]
    pub const fn offered_index(&self) -> usize {
        self.offered_index
    }

    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    #[must_use]
    pub const fn max_new_tokens(&self) -> usize {
        self.max_new_tokens
    }

    #[must_use]
    pub const fn total_positions(&self) -> usize {
        self.total_positions
    }

    #[must_use]
    pub const fn resolved_deadline_ns(&self) -> Option<u64> {
        self.resolved_deadline_ns
    }

    #[must_use]
    pub const fn admitted_ns(&self) -> u64 {
        self.admitted_ns
    }

    #[must_use]
    pub const fn first_work_start_ns(&self) -> Option<u64> {
        self.first_work_start_ns
    }

    #[must_use]
    pub const fn first_decode_start_ns(&self) -> Option<u64> {
        self.first_decode_start_ns
    }

    #[must_use]
    pub const fn prefill_complete_ns(&self) -> Option<u64> {
        self.prefill_complete_ns
    }

    #[must_use]
    pub const fn cancel_linearized_ns(&self) -> Option<u64> {
        self.cancel_linearized_ns
    }

    #[must_use]
    pub const fn terminal_decided_ns(&self) -> Option<u64> {
        self.terminal_decided_ns
    }

    #[must_use]
    pub const fn terminal_outcome(&self) -> Option<TerminalOutcome> {
        self.terminal_outcome
    }

    #[must_use]
    pub const fn request_owned_zero_ns(&self) -> Option<u64> {
        self.request_owned_zero_ns
    }

    #[must_use]
    pub const fn worker_quiescent_ns(&self) -> Option<u64> {
        self.worker_quiescent_ns
    }

    #[must_use]
    pub const fn committed_positions(&self) -> usize {
        self.committed_positions
    }

    #[must_use]
    pub const fn emitted_tokens(&self) -> usize {
        self.emitted_tokens
    }

    #[must_use]
    pub const fn preemption_count(&self) -> u64 {
        self.preemption_count
    }

    #[must_use]
    pub const fn resume_count(&self) -> u64 {
        self.resume_count
    }
}

impl fmt::Debug for RunRequestObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunRequestObservation")
            .field("offered_index", &self.offered_index)
            .field("request_id", &"<redacted>")
            .field("shape", &"<redacted>")
            .field("timing", &"<redacted>")
            .field("terminal_outcome", &self.terminal_outcome)
            .field("committed_positions", &self.committed_positions)
            .field("emitted_tokens", &self.emitted_tokens)
            .finish()
    }
}

/// Capacity, lifecycle, and sticky-health metadata for one observer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunObserverStatus {
    request_capacity: usize,
    output_timestamp_capacity: usize,
    request_count: usize,
    output_timestamp_count: usize,
    requested_bytes: usize,
    started: bool,
    finished: bool,
    failure: Option<RunObserverFailure>,
}

/// Opaque identity of the observer's two preallocated backing stores.
///
/// Equality across an interval proves neither store moved or changed capacity.
/// The addresses are intentionally unavailable and redacted from diagnostics.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RunObserverAllocationFingerprint {
    request_storage: usize,
    request_capacity: usize,
    output_storage: usize,
    output_capacity: usize,
}

impl fmt::Debug for RunObserverAllocationFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunObserverAllocationFingerprint")
            .field("storage", &"<redacted>")
            .field("request_capacity", &self.request_capacity)
            .field("output_capacity", &self.output_capacity)
            .finish()
    }
}

impl RunObserverStatus {
    #[must_use]
    pub const fn request_capacity(self) -> usize {
        self.request_capacity
    }

    #[must_use]
    pub const fn output_timestamp_capacity(self) -> usize {
        self.output_timestamp_capacity
    }

    #[must_use]
    pub const fn request_count(self) -> usize {
        self.request_count
    }

    #[must_use]
    pub const fn output_timestamp_count(self) -> usize {
        self.output_timestamp_count
    }

    #[must_use]
    pub const fn requested_bytes(self) -> usize {
        self.requested_bytes
    }

    #[must_use]
    pub const fn started(self) -> bool {
        self.started
    }

    #[must_use]
    pub const fn finished(self) -> bool {
        self.finished
    }

    #[must_use]
    pub const fn failure(self) -> Option<RunObserverFailure> {
        self.failure
    }

    #[must_use]
    pub const fn healthy(self) -> bool {
        self.failure.is_none()
    }
}

/// Borrowed, allocation-free observer result.
#[must_use = "observer reads carry sticky health and exact capacity metadata"]
pub struct RunObservationRead<'observer> {
    requests: &'observer [RunRequestObservation],
    output_commit_ns: &'observer [Option<u64>],
    occupancy: &'observer [u64; RUN_OCCUPANCY_BIN_COUNT],
    totals: RunObserverTotals,
    release_ns: Option<u64>,
    last_emitted_commit_ns: Option<u64>,
    last_terminal_decided_ns: Option<u64>,
    status: RunObserverStatus,
}

impl<'observer> RunObservationRead<'observer> {
    #[must_use]
    pub const fn requests(&self) -> &'observer [RunRequestObservation] {
        self.requests
    }

    /// Returns only the populated output timestamp prefix for one local row.
    ///
    /// The index is resolved against this read rather than trusting offsets in
    /// a copied observation from another engine or run.
    #[must_use]
    pub fn output_commit_ns(&self, request_index: usize) -> Option<&'observer [Option<u64>]> {
        let request = self.requests.get(request_index)?;
        let start = request.output_offset;
        let end = start.checked_add(request.emitted_tokens)?;
        self.output_commit_ns.get(start..end)
    }

    #[must_use]
    pub const fn wave_occupancy_histogram(&self) -> &[u64; RUN_OCCUPANCY_BIN_COUNT] {
        self.occupancy
    }

    #[must_use]
    pub const fn totals(&self) -> RunObserverTotals {
        self.totals
    }

    #[must_use]
    pub const fn release_ns(&self) -> Option<u64> {
        self.release_ns
    }

    #[must_use]
    pub const fn last_emitted_commit_ns(&self) -> Option<u64> {
        self.last_emitted_commit_ns
    }

    #[must_use]
    pub const fn last_terminal_decided_ns(&self) -> Option<u64> {
        self.last_terminal_decided_ns
    }

    #[must_use]
    pub const fn status(&self) -> RunObserverStatus {
        self.status
    }
}

impl fmt::Debug for RunObservationRead<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunObservationRead")
            .field("request_count", &self.requests.len())
            .field("totals", &self.totals)
            .field("status", &self.status)
            .field("requests", &"<redacted>")
            .field("output_commit_ns", &"<redacted>")
            .finish()
    }
}

pub(crate) struct AdmissionObservation {
    pub(crate) offered_index: usize,
    pub(crate) request_id: RequestId,
    pub(crate) prompt_len: usize,
    pub(crate) max_new_tokens: usize,
    pub(crate) total_positions: usize,
    pub(crate) resolved_deadline_ns: Option<u64>,
}

/// Harness-owned observer state; engine callbacks are private and infallible.
pub struct RunObserver {
    domain: ControlDomain,
    clock: RunClock,
    request_capacity: usize,
    max_new_tokens: usize,
    state_page_tokens: usize,
    requests: Vec<RunRequestObservation>,
    output_commit_ns: Vec<Option<u64>>,
    occupancy: [u64; RUN_OCCUPANCY_BIN_COUNT],
    totals: RunObserverTotals,
    release_ns: Option<u64>,
    last_emitted_commit_ns: Option<u64>,
    last_terminal_decided_ns: Option<u64>,
    last_seen_ns: Option<u64>,
    first_request_id: Option<u64>,
    active_live_tokens: u64,
    active_page_slots: u64,
    requested_bytes: usize,
    finished: bool,
    failure: Option<RunObserverFailure>,
}

impl RunObserver {
    pub(crate) fn try_bound(
        config: &SchedulerConfig,
        domain: ControlDomain,
    ) -> SchedulerResult<Self> {
        // Capture the run origin before any observer allocation so preparation
        // latency is inside the same child-local clock domain as release.
        let clock = RunClock::new();
        let request_capacity = config.max_outstanding_requests();
        if request_capacity > MAX_RUN_OBSERVER_REQUESTS || config.max_new_tokens() > 8 {
            return Err(SchedulerError::unsupported(
                "M5 run observer requires at most 32 requests and 8 outputs per request",
            ));
        }
        let output_capacity = request_capacity
            .checked_mul(config.max_new_tokens())
            .ok_or_else(|| SchedulerError::invalid_request("run observer", "capacity overflows"))?;
        if output_capacity > MAX_RUN_OBSERVER_OUTPUT_TIMESTAMPS {
            return Err(SchedulerError::invalid_request(
                "run observer",
                "output timestamp capacity exceeds the evidence ceiling",
            ));
        }
        let mut requests = Vec::new();
        reserve_exact::<RunRequestObservation>(
            &mut requests,
            request_capacity,
            "run observer requests",
        )?;
        let mut output_commit_ns = Vec::new();
        reserve_exact::<Option<u64>>(
            &mut output_commit_ns,
            output_capacity,
            "run observer output timestamps",
        )?;
        output_commit_ns.resize(output_capacity, None);
        let requested_bytes = request_capacity
            .checked_mul(size_of::<RunRequestObservation>())
            .and_then(|bytes| {
                output_capacity
                    .checked_mul(size_of::<Option<u64>>())
                    .and_then(|output| bytes.checked_add(output))
            })
            .ok_or_else(|| {
                SchedulerError::invalid_request("run observer", "byte size overflows")
            })?;
        Ok(Self {
            domain,
            clock,
            request_capacity,
            max_new_tokens: config.max_new_tokens(),
            state_page_tokens: config.state_page_tokens(),
            requests,
            output_commit_ns,
            occupancy: [0; RUN_OCCUPANCY_BIN_COUNT],
            totals: RunObserverTotals::default(),
            release_ns: None,
            last_emitted_commit_ns: None,
            last_terminal_decided_ns: None,
            last_seen_ns: None,
            first_request_id: None,
            active_live_tokens: 0,
            active_page_slots: 0,
            requested_bytes,
            finished: false,
            failure: None,
        })
    }

    pub(crate) fn belongs_to(&self, domain: &ControlDomain) -> bool {
        self.domain.same_table(domain)
    }

    pub(crate) fn belongs_to_registry(&self, registry: &ControlRegistry) -> bool {
        registry.matches_domain(&self.domain)
    }

    pub(crate) const fn clock(&self) -> RunClock {
        self.clock
    }

    /// Stable evidence identifier for the observer-owned clock.
    #[must_use]
    pub const fn clock_id(&self) -> &'static str {
        self.clock.clock_id()
    }

    pub(crate) fn preflight_admission(
        &self,
        domain: &ControlDomain,
        accepted_count: usize,
    ) -> SchedulerResult<()> {
        if !self.belongs_to(domain) {
            return Err(SchedulerError::invalid_request(
                "run observer",
                "belongs to another scheduler engine",
            ));
        }
        if self.release_ns.is_some()
            || !self.requests.is_empty()
            || self.output_commit_ns.iter().any(Option::is_some)
            || self.occupancy.iter().any(|count| *count != 0)
            || self.totals != RunObserverTotals::default()
            || self.last_emitted_commit_ns.is_some()
            || self.last_terminal_decided_ns.is_some()
            || self.last_seen_ns.is_some()
            || self.first_request_id.is_some()
            || self.active_live_tokens != 0
            || self.active_page_slots != 0
            || self.finished
            || self.failure.is_some()
        {
            return Err(SchedulerError::invalid_request(
                "run observer",
                "is not pristine",
            ));
        }
        if accepted_count > self.request_capacity {
            return Err(SchedulerError::resource_exhausted(
                "run observer request capacity",
                u64::try_from(accepted_count).unwrap_or(u64::MAX),
                u64::try_from(self.request_capacity).unwrap_or(u64::MAX),
            ));
        }
        Ok(())
    }

    /// Returns a borrowed result without changing observer lifecycle.
    pub fn read(&self) -> RunObservationRead<'_> {
        RunObservationRead {
            requests: &self.requests,
            output_commit_ns: &self.output_commit_ns,
            occupancy: &self.occupancy,
            totals: self.totals,
            release_ns: self.release_ns,
            last_emitted_commit_ns: self.last_emitted_commit_ns,
            last_terminal_decided_ns: self.last_terminal_decided_ns,
            status: self.status(),
        }
    }

    /// Seals the observer after cleanup; a failed invariant remains sticky.
    pub fn finish(&mut self) -> RunObserverStatus {
        if self.finished {
            self.poison(RunObserverFailure::CallbackAfterFinish);
            return self.status();
        }
        if self.failure.is_none() {
            if self.release_ns.is_none()
                || self.requests.iter().any(|request| {
                    request.terminal_decided_ns.is_none()
                        || request.request_owned_zero_ns.is_none()
                        || (request.terminal_outcome == Some(TerminalOutcome::Cancelled)
                            && request.cancel_linearized_ns.is_none())
                        || (request.cancel_linearized_ns.is_some()
                            && request.worker_quiescent_ns.is_none())
                })
            {
                self.poison(RunObserverFailure::IncompleteRun);
            } else if !self.summary_consistent() {
                self.poison(RunObserverFailure::InconsistentSummary);
            }
        }
        self.finished = true;
        self.status()
    }

    #[must_use]
    pub fn status(&self) -> RunObserverStatus {
        let output_timestamp_count = self
            .requests
            .iter()
            .map(|request| request.emitted_tokens)
            .sum();
        RunObserverStatus {
            request_capacity: self.request_capacity,
            output_timestamp_capacity: self.output_commit_ns.len(),
            request_count: self.requests.len(),
            output_timestamp_count,
            requested_bytes: self.requested_bytes,
            started: self.release_ns.is_some(),
            finished: self.finished,
            failure: self.failure,
        }
    }

    /// Captures a non-serializable no-growth witness for the observed interval.
    #[must_use]
    pub fn allocation_fingerprint(&self) -> RunObserverAllocationFingerprint {
        RunObserverAllocationFingerprint {
            request_storage: self.requests.as_ptr() as usize,
            request_capacity: self.requests.capacity(),
            output_storage: self.output_commit_ns.as_ptr() as usize,
            output_capacity: self.output_commit_ns.capacity(),
        }
    }

    fn summary_consistent(&self) -> bool {
        (|| {
            let release_ns = self.release_ns?;
            if self.active_live_tokens != 0 || self.active_page_slots != 0 || self.occupancy[0] != 0
            {
                return None;
            }

            let mut committed_positions = 0_u64;
            let mut emitted_tokens = 0_u64;
            let mut preemptions = 0_u64;
            let mut resumes = 0_u64;
            let mut last_emission = None;
            let mut last_terminal = None;
            let first_request_id = self.first_request_id;
            if self.requests.is_empty() != first_request_id.is_none() {
                return None;
            }

            for (index, request) in self.requests.iter().enumerate() {
                let expected_id = first_request_id?.checked_add(u64::try_from(index).ok()?)?;
                let expected_total_positions = request
                    .prompt_len
                    .checked_add(request.max_new_tokens.saturating_sub(1))?;
                let expected_emitted_tokens = if request.max_new_tokens == 0
                    || request.committed_positions < request.prompt_len
                {
                    0
                } else {
                    request
                        .committed_positions
                        .checked_sub(request.prompt_len)?
                        .checked_add(1)?
                };
                if request.offered_index != index
                    || request.request_id.get() != expected_id
                    || request.prompt_len == 0
                    || request.total_positions != expected_total_positions
                    || request.admitted_ns != release_ns
                    || request.output_offset != index.checked_mul(self.max_new_tokens)?
                    || request.state_active
                    || request.pending_work_position.is_some()
                    || request.committed_positions > request.total_positions
                    || request.emitted_tokens > request.max_new_tokens
                    || request.emitted_tokens != expected_emitted_tokens
                    || request.max_new_tokens > self.max_new_tokens
                    || request.terminal_outcome.is_none()
                    || request
                        .resolved_deadline_ns
                        .is_some_and(|deadline| deadline <= release_ns)
                {
                    return None;
                }

                let terminal_ns = request.terminal_decided_ns?;
                let zero_ns = request.request_owned_zero_ns?;
                if terminal_ns < release_ns
                    || zero_ns < terminal_ns
                    || !terminal_outcome_consistent(
                        request,
                        request.terminal_outcome.as_ref()?,
                        terminal_ns,
                    )
                {
                    return None;
                }
                if request
                    .first_work_start_ns
                    .is_some_and(|timestamp| timestamp < release_ns || timestamp > terminal_ns)
                    || request
                        .prefill_complete_ns
                        .is_some_and(|timestamp| timestamp < release_ns || timestamp > terminal_ns)
                    || request
                        .first_decode_start_ns
                        .is_some_and(|timestamp| timestamp < release_ns || timestamp > terminal_ns)
                {
                    return None;
                }
                if !ordered(request.first_work_start_ns, request.prefill_complete_ns)
                    || !ordered(request.prefill_complete_ns, request.first_decode_start_ns)
                {
                    return None;
                }
                if request.committed_positions > 0 && request.first_work_start_ns.is_none() {
                    return None;
                }
                if request.committed_positions >= request.prompt_len {
                    request.prefill_complete_ns?;
                } else if request.prefill_complete_ns.is_some() {
                    return None;
                }
                if request.committed_positions > request.prompt_len {
                    request.first_decode_start_ns?;
                } else if request
                    .first_decode_start_ns
                    .is_some_and(|_| request.committed_positions < request.prompt_len)
                {
                    return None;
                }
                if request.terminal_outcome == Some(TerminalOutcome::Cancelled)
                    && request.cancel_linearized_ns.is_none()
                {
                    return None;
                }
                if let Some(cancel_ns) = request.cancel_linearized_ns {
                    let quiescent_ns = request.worker_quiescent_ns?;
                    if cancel_ns < release_ns
                        || cancel_ns > terminal_ns
                        || quiescent_ns < terminal_ns
                        || quiescent_ns > zero_ns
                    {
                        return None;
                    }
                } else if request.worker_quiescent_ns.is_some() {
                    return None;
                }

                let used_end = request.output_offset.checked_add(request.emitted_tokens)?;
                let reserved_end = request.output_offset.checked_add(self.max_new_tokens)?;
                if reserved_end > self.output_commit_ns.len() {
                    return None;
                }
                let mut previous_output = None;
                for timestamp in self.output_commit_ns[request.output_offset..used_end]
                    .iter()
                    .copied()
                {
                    let timestamp = timestamp?;
                    if timestamp < release_ns
                        || timestamp > terminal_ns
                        || previous_output.is_some_and(|previous| timestamp < previous)
                    {
                        return None;
                    }
                    previous_output = Some(timestamp);
                    last_emission =
                        Some(last_emission.map_or(timestamp, |last: u64| last.max(timestamp)));
                }
                if self.output_commit_ns[used_end..reserved_end]
                    .iter()
                    .any(Option::is_some)
                {
                    return None;
                }

                committed_positions = committed_positions
                    .checked_add(u64::try_from(request.committed_positions).ok()?)?;
                emitted_tokens =
                    emitted_tokens.checked_add(u64::try_from(request.emitted_tokens).ok()?)?;
                preemptions = preemptions.checked_add(request.preemption_count)?;
                resumes = resumes.checked_add(request.resume_count)?;
                last_terminal =
                    Some(last_terminal.map_or(terminal_ns, |last: u64| last.max(terminal_ns)));
            }

            let admitted_capacity = self.requests.len().checked_mul(self.max_new_tokens)?;
            if self.output_commit_ns[admitted_capacity..]
                .iter()
                .any(Option::is_some)
            {
                return None;
            }

            let mut histogram_waves = 0_u64;
            let mut histogram_selected = 0_u64;
            for (selected, waves) in self.occupancy.iter().copied().enumerate().skip(1) {
                histogram_waves = histogram_waves.checked_add(waves)?;
                histogram_selected = histogram_selected
                    .checked_add(waves.checked_mul(u64::try_from(selected).ok()?)?)?;
            }
            if committed_positions != self.totals.committed_positions
                || emitted_tokens != u64::try_from(self.status().output_timestamp_count()).ok()?
                || self.totals.state_sample_count != committed_positions
                || preemptions != self.totals.preempted_survivors
                || resumes != self.totals.resumed_requests
                || histogram_waves != self.totals.wave_count
                || histogram_selected != self.totals.selected_positions
                || self.totals.terminal_decisions != u64::try_from(self.requests.len()).ok()?
                || last_emission != self.last_emitted_commit_ns
                || last_terminal != self.last_terminal_decided_ns
            {
                return None;
            }
            Some(())
        })()
        .is_some()
    }

    fn poison(&mut self, failure: RunObserverFailure) {
        if self.failure.is_none() {
            self.failure = Some(failure);
        }
    }

    fn callback_open(&mut self) -> bool {
        if self.finished {
            self.poison(RunObserverFailure::CallbackAfterFinish);
            return false;
        }
        self.failure.is_none()
    }

    fn runtime_callback_open(&mut self) -> bool {
        if !self.callback_open() {
            return false;
        }
        if self.release_ns.is_none() {
            self.poison(RunObserverFailure::InvalidPhaseBoundary);
            return false;
        }
        true
    }

    fn validate_time(&mut self, now_ns: u64) -> Option<u64> {
        if !self.callback_open() {
            return None;
        }
        if now_ns == u64::MAX {
            self.poison(RunObserverFailure::ClockSaturated);
            return None;
        }
        if self.last_seen_ns.is_some_and(|last| now_ns < last) {
            self.poison(RunObserverFailure::ClockRegression);
            return None;
        }
        let Some(next_observations) = self.totals.timestamp_observations.checked_add(1) else {
            self.poison(RunObserverFailure::CounterOverflow);
            return None;
        };
        Some(next_observations)
    }

    fn commit_time(&mut self, now_ns: u64, next_observations: u64) {
        self.last_seen_ns = Some(now_ns);
        self.totals.timestamp_observations = next_observations;
    }

    fn record_release(&mut self, release_ns: u64) {
        if !self.callback_open() {
            return;
        }
        if self.release_ns.is_some() || !self.requests.is_empty() {
            self.poison(RunObserverFailure::DuplicateAdmission);
            return;
        }
        let Some(next_observations) = self.validate_time(release_ns) else {
            return;
        };
        self.release_ns = Some(release_ns);
        self.commit_time(release_ns, next_observations);
    }

    fn request_index(&mut self, request_id: RequestId) -> Option<usize> {
        let Some(first) = self.first_request_id else {
            self.poison(RunObserverFailure::UnknownRequest);
            return None;
        };
        let Some(offset) = request_id.get().checked_sub(first) else {
            self.poison(RunObserverFailure::UnknownRequest);
            return None;
        };
        let Ok(index) = usize::try_from(offset) else {
            self.poison(RunObserverFailure::UnknownRequest);
            return None;
        };
        if self
            .requests
            .get(index)
            .is_none_or(|request| request.request_id != request_id)
        {
            self.poison(RunObserverFailure::UnknownRequest);
            return None;
        }
        Some(index)
    }

    fn record_admission(&mut self, admission: AdmissionObservation, release_ns: u64) {
        if !self.runtime_callback_open() {
            return;
        }
        if self.release_ns != Some(release_ns) {
            self.poison(RunObserverFailure::DuplicateAdmission);
            return;
        }
        if self.requests.len() >= self.request_capacity {
            self.poison(RunObserverFailure::AdmissionCapacity);
            return;
        }
        if admission.offered_index != self.requests.len() {
            self.poison(RunObserverFailure::DuplicateAdmission);
            return;
        }
        let next_first_request_id = if let Some(first) = self.first_request_id {
            let Some(expected) = first.checked_add(self.requests.len() as u64) else {
                self.poison(RunObserverFailure::CounterOverflow);
                return;
            };
            if admission.request_id.get() != expected {
                self.poison(RunObserverFailure::UnknownRequest);
                return;
            }
            Some(first)
        } else {
            Some(admission.request_id.get())
        };
        if admission.max_new_tokens > self.max_new_tokens {
            self.poison(RunObserverFailure::AdmissionCapacity);
            return;
        }
        let expected_total_positions = admission
            .prompt_len
            .checked_add(admission.max_new_tokens.saturating_sub(1));
        if admission.prompt_len == 0
            || expected_total_positions != Some(admission.total_positions)
            || admission
                .resolved_deadline_ns
                .is_some_and(|deadline| deadline <= release_ns)
        {
            self.poison(RunObserverFailure::InvalidPhaseBoundary);
            return;
        }
        let Some(output_offset) = self.requests.len().checked_mul(self.max_new_tokens) else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        let allocated_page_slots = admission
            .total_positions
            .checked_add(self.state_page_tokens - 1)
            .map(|tokens| tokens / self.state_page_tokens * self.state_page_tokens);
        let Some(allocated_page_slots) = allocated_page_slots else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        let request = RunRequestObservation {
            offered_index: admission.offered_index,
            request_id: admission.request_id,
            prompt_len: admission.prompt_len,
            max_new_tokens: admission.max_new_tokens,
            total_positions: admission.total_positions,
            resolved_deadline_ns: admission.resolved_deadline_ns,
            admitted_ns: release_ns,
            first_work_start_ns: None,
            first_decode_start_ns: None,
            prefill_complete_ns: None,
            cancel_linearized_ns: None,
            terminal_decided_ns: None,
            terminal_outcome: None,
            request_owned_zero_ns: None,
            worker_quiescent_ns: None,
            committed_positions: 0,
            emitted_tokens: 0,
            preemption_count: 0,
            resume_count: 0,
            output_offset,
            allocated_page_slots,
            state_active: false,
            pending_work_position: None,
        };
        self.first_request_id = next_first_request_id;
        self.requests.push(request);
    }

    fn record_promotion(&mut self, request_id: RequestId) {
        if !self.runtime_callback_open() {
            return;
        }
        let Some(index) = self.request_index(request_id) else {
            return;
        };
        let request = &self.requests[index];
        if request.state_active
            || request.terminal_decided_ns.is_some()
            || request.pending_work_position.is_some()
        {
            self.poison(RunObserverFailure::DuplicateMilestone);
            return;
        }
        let Ok(slots) = u64::try_from(request.allocated_page_slots) else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        let Some(next_slots) = self.active_page_slots.checked_add(slots) else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        self.requests[index].state_active = true;
        self.active_page_slots = next_slots;
    }

    fn record_work_start(&mut self, request_id: RequestId, position: usize, now_ns: u64) {
        if !self.runtime_callback_open() {
            return;
        }
        let Some(index) = self.request_index(request_id) else {
            return;
        };
        let request = &self.requests[index];
        if position != request.committed_positions || position >= request.total_positions {
            self.poison(RunObserverFailure::NoncontiguousPosition);
            return;
        }
        if !request.state_active || request.terminal_decided_ns.is_some() {
            self.poison(RunObserverFailure::InvalidPhaseBoundary);
            return;
        }
        if request.pending_work_position.is_some() {
            self.poison(RunObserverFailure::DuplicateMilestone);
            return;
        }
        if position > request.prompt_len && request.first_decode_start_ns.is_none() {
            self.poison(RunObserverFailure::InvalidPhaseBoundary);
            return;
        }
        let Some(next_observations) = self.validate_time(now_ns) else {
            return;
        };
        let request = &mut self.requests[index];
        request.first_work_start_ns.get_or_insert(now_ns);
        if position == request.prompt_len {
            request.first_decode_start_ns.get_or_insert(now_ns);
        }
        request.pending_work_position = Some(position);
        self.commit_time(now_ns, next_observations);
    }

    fn record_work_abandoned(&mut self, request_id: RequestId, position: usize) {
        if !self.runtime_callback_open() {
            return;
        }
        let Some(index) = self.request_index(request_id) else {
            return;
        };
        if self.requests[index].pending_work_position != Some(position)
            || self.requests[index].terminal_decided_ns.is_some()
        {
            self.poison(RunObserverFailure::InvalidPhaseBoundary);
            return;
        }
        self.requests[index].pending_work_position = None;
    }

    fn record_commit(
        &mut self,
        request_id: RequestId,
        position: usize,
        phase: ServicePhase,
        emitted: bool,
        now_ns: u64,
    ) {
        if !self.runtime_callback_open() {
            return;
        }
        let Some(index) = self.request_index(request_id) else {
            return;
        };
        let mut request = self.requests[index];
        if position != request.committed_positions {
            self.poison(RunObserverFailure::NoncontiguousPosition);
            return;
        }
        if request.pending_work_position != Some(position) {
            self.poison(RunObserverFailure::InvalidPhaseBoundary);
            return;
        }
        let expected_phase = if position < request.prompt_len {
            ServicePhase::Prefill
        } else {
            ServicePhase::Decode
        };
        if phase != expected_phase {
            self.poison(RunObserverFailure::InvalidPhaseBoundary);
            return;
        }
        if !request.state_active {
            self.poison(RunObserverFailure::InvalidPhaseBoundary);
            return;
        }
        let expected_emitted = request.max_new_tokens != 0
            && position
                .checked_add(1)
                .is_some_and(|next| next >= request.prompt_len);
        if emitted != expected_emitted {
            self.poison(RunObserverFailure::InvalidPhaseBoundary);
            return;
        }
        let completes_prefill =
            phase == ServicePhase::Prefill && position.checked_add(1) == Some(request.prompt_len);
        if completes_prefill && request.prefill_complete_ns.is_some() {
            self.poison(RunObserverFailure::DuplicateMilestone);
            return;
        }
        let next_emitted_tokens = if emitted {
            match request.emitted_tokens.checked_add(1) {
                Some(value) if value <= request.max_new_tokens => value,
                _ => {
                    self.poison(RunObserverFailure::NoncontiguousOutput);
                    return;
                }
            }
        } else {
            request.emitted_tokens
        };
        let Some(next_committed_positions) = request.committed_positions.checked_add(1) else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        if next_committed_positions > request.total_positions {
            self.poison(RunObserverFailure::NoncontiguousPosition);
            return;
        }
        let output_slot = if emitted {
            let Some(slot_index) = request.output_offset.checked_add(request.emitted_tokens) else {
                self.poison(RunObserverFailure::CounterOverflow);
                return;
            };
            match self.output_commit_ns.get(slot_index) {
                Some(None) => Some(slot_index),
                Some(Some(_)) => {
                    self.poison(RunObserverFailure::NoncontiguousOutput);
                    return;
                }
                None => {
                    self.poison(RunObserverFailure::AdmissionCapacity);
                    return;
                }
            }
        } else {
            None
        };
        let Some(next_live_tokens) = self.active_live_tokens.checked_add(1) else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        let mut totals = self.totals;
        let Some(committed_positions) = totals.committed_positions.checked_add(1) else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        let Some(live_sum) = totals
            .state_live_token_sample_sum
            .checked_add(next_live_tokens)
        else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        let Some(slot_sum) = totals
            .state_allocated_page_slot_sample_sum
            .checked_add(self.active_page_slots)
        else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        let Some(sample_count) = totals.state_sample_count.checked_add(1) else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        let Some(next_observations) = self.validate_time(now_ns) else {
            return;
        };

        request.committed_positions = next_committed_positions;
        request.emitted_tokens = next_emitted_tokens;
        request.pending_work_position = None;
        if completes_prefill {
            request.prefill_complete_ns = Some(now_ns);
        }
        self.requests[index] = request;
        if let Some(slot_index) = output_slot {
            self.output_commit_ns[slot_index] = Some(now_ns);
            self.last_emitted_commit_ns = Some(now_ns);
        }
        self.active_live_tokens = next_live_tokens;
        totals.committed_positions = committed_positions;
        totals.state_live_token_sample_sum = live_sum;
        totals.state_allocated_page_slot_sample_sum = slot_sum;
        totals.state_sample_count = sample_count;
        self.totals = totals;
        self.commit_time(now_ns, next_observations);
    }

    fn record_cancel(
        &mut self,
        request_id: RequestId,
        disposition: CancelDisposition,
        now_ns: u64,
    ) {
        if !self.runtime_callback_open() {
            return;
        }
        if disposition != CancelDisposition::Requested {
            return;
        }
        let Some(index) = self.request_index(request_id) else {
            return;
        };
        if self.requests[index].cancel_linearized_ns.is_some() {
            self.poison(RunObserverFailure::DuplicateMilestone);
            return;
        }
        let Some(next_observations) = self.validate_time(now_ns) else {
            return;
        };
        self.requests[index].cancel_linearized_ns = Some(now_ns);
        self.commit_time(now_ns, next_observations);
    }

    fn record_terminal(&mut self, result: TerminalResult, now_ns: u64) {
        if !self.runtime_callback_open() {
            return;
        }
        let Some(index) = self.request_index(result.request_id()) else {
            return;
        };
        let request = &self.requests[index];
        if request.terminal_decided_ns.is_some()
            || result.committed_positions() != request.committed_positions
            || result.emitted_tokens() != request.emitted_tokens
        {
            self.poison(RunObserverFailure::InvalidTerminal);
            return;
        }
        if !terminal_outcome_consistent(request, &result.outcome(), now_ns) {
            self.poison(if result.outcome() == TerminalOutcome::Cancelled {
                RunObserverFailure::MissingCancellationBoundary
            } else {
                RunObserverFailure::InvalidTerminal
            });
            return;
        }
        let next_active = if request.state_active {
            let (Ok(live), Ok(slots)) = (
                u64::try_from(request.committed_positions),
                u64::try_from(request.allocated_page_slots),
            ) else {
                self.poison(RunObserverFailure::CounterOverflow);
                return;
            };
            let (Some(live), Some(slots)) = (
                self.active_live_tokens.checked_sub(live),
                self.active_page_slots.checked_sub(slots),
            ) else {
                self.poison(RunObserverFailure::InvalidTerminal);
                return;
            };
            Some((live, slots))
        } else {
            None
        };
        let Some(next_observations) = self.validate_time(now_ns) else {
            return;
        };
        let request = &mut self.requests[index];
        request.terminal_decided_ns = Some(now_ns);
        request.terminal_outcome = Some(result.outcome());
        request.pending_work_position = None;
        if let Some((live, slots)) = next_active {
            request.state_active = false;
            self.active_live_tokens = live;
            self.active_page_slots = slots;
        }
        self.last_terminal_decided_ns = Some(now_ns);
        self.commit_time(now_ns, next_observations);
    }

    fn record_request_zero(&mut self, request_id: RequestId, now_ns: u64) {
        if !self.runtime_callback_open() {
            return;
        }
        let Some(index) = self.request_index(request_id) else {
            return;
        };
        let request = &self.requests[index];
        if request.terminal_decided_ns.is_none() || request.request_owned_zero_ns.is_some() {
            self.poison(RunObserverFailure::DuplicateMilestone);
            return;
        }
        let Some(next_observations) = self.validate_time(now_ns) else {
            return;
        };
        self.requests[index].request_owned_zero_ns = Some(now_ns);
        self.commit_time(now_ns, next_observations);
    }

    fn pending_worker_quiescence(&self) -> bool {
        !self.finished
            && self.failure.is_none()
            && self.requests.iter().any(|request| {
                request.cancel_linearized_ns.is_some()
                    && request.terminal_decided_ns.is_some()
                    && request.worker_quiescent_ns.is_none()
            })
    }

    fn record_worker_quiescent(&mut self, request_id: RequestId, now_ns: u64) {
        if !self.runtime_callback_open() {
            return;
        }
        let Some(index) = self.request_index(request_id) else {
            return;
        };
        let request = &self.requests[index];
        if request.cancel_linearized_ns.is_none()
            || request.terminal_decided_ns.is_none()
            || request.worker_quiescent_ns.is_some()
        {
            self.poison(RunObserverFailure::DuplicateMilestone);
            return;
        }
        let Some(next_observations) = self.validate_time(now_ns) else {
            return;
        };
        self.requests[index].worker_quiescent_ns = Some(now_ns);
        self.commit_time(now_ns, next_observations);
    }

    fn record_pending_worker_quiescence(&mut self, now_ns: u64) {
        if !self.runtime_callback_open() {
            return;
        }
        if !self.pending_worker_quiescence() {
            self.poison(RunObserverFailure::DuplicateMilestone);
            return;
        }
        let Some(next_observations) = self.validate_time(now_ns) else {
            return;
        };
        for request in &mut self.requests {
            if request.cancel_linearized_ns.is_some()
                && request.terminal_decided_ns.is_some()
                && request.worker_quiescent_ns.is_none()
            {
                request.worker_quiescent_ns = Some(now_ns);
            }
        }
        self.commit_time(now_ns, next_observations);
    }

    fn record_preempted(&mut self, request_id: RequestId) {
        if !self.runtime_callback_open() {
            return;
        }
        let Some(index) = self.request_index(request_id) else {
            return;
        };
        let request = &self.requests[index];
        if !request.state_active
            || request.terminal_decided_ns.is_some()
            || request.pending_work_position.is_some()
            || request.preemption_count != request.resume_count
        {
            self.poison(RunObserverFailure::InvalidPhaseBoundary);
            return;
        }
        let Some(next) = self.requests[index].preemption_count.checked_add(1) else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        self.requests[index].preemption_count = next;
    }

    fn record_resumed(&mut self, request_id: RequestId) {
        if !self.runtime_callback_open() {
            return;
        }
        let Some(index) = self.request_index(request_id) else {
            return;
        };
        let request = &self.requests[index];
        if !request.state_active
            || request.terminal_decided_ns.is_some()
            || request.pending_work_position.is_some()
            || request.preemption_count != request.resume_count.saturating_add(1)
        {
            self.poison(RunObserverFailure::InvalidPhaseBoundary);
            return;
        }
        let Some(next) = self.requests[index].resume_count.checked_add(1) else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        self.requests[index].resume_count = next;
    }

    fn record_wave(&mut self, selected: usize) {
        if !self.runtime_callback_open() {
            return;
        }
        if selected == 0 || selected >= RUN_OCCUPANCY_BIN_COUNT {
            self.poison(RunObserverFailure::InvalidWaveOccupancy);
            return;
        }
        let Some(next) = self.occupancy[selected].checked_add(1) else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        self.occupancy[selected] = next;
    }

    fn record_step(&mut self, report: StepReport) {
        if !self.runtime_callback_open() {
            return;
        }
        let values = [
            report.waves,
            report.selected_positions,
            report.expert_tasks,
            report.expert_groups,
            report.terminal_decisions,
            report.preempted_requests,
            report.resumed_requests,
        ];
        let [
            Ok(waves),
            Ok(selected),
            Ok(expert_tasks),
            Ok(expert_groups),
            Ok(terminals),
            Ok(preempted),
            Ok(resumed),
        ] = values.map(u64::try_from)
        else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        let mut totals = self.totals;
        let next = [
            totals.step_count.checked_add(1),
            totals.wave_count.checked_add(waves),
            totals.selected_positions.checked_add(selected),
            totals.expert_contributions.checked_add(expert_tasks),
            totals.expert_group_calls.checked_add(expert_groups),
            totals.terminal_decisions.checked_add(terminals),
            totals.preempted_survivors.checked_add(preempted),
            totals.resumed_requests.checked_add(resumed),
        ];
        let [
            Some(step_count),
            Some(wave_count),
            Some(selected_positions),
            Some(expert_contributions),
            Some(expert_group_calls),
            Some(terminal_decisions),
            Some(preempted_survivors),
            Some(resumed_requests),
        ] = next
        else {
            self.poison(RunObserverFailure::CounterOverflow);
            return;
        };
        totals.step_count = step_count;
        totals.wave_count = wave_count;
        totals.selected_positions = selected_positions;
        totals.expert_contributions = expert_contributions;
        totals.expert_group_calls = expert_group_calls;
        totals.terminal_decisions = terminal_decisions;
        totals.preempted_survivors = preempted_survivors;
        totals.resumed_requests = resumed_requests;
        self.totals = totals;
    }
}

impl fmt::Debug for RunObserver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunObserver")
            .field("status", &self.status())
            .field("domain", &"<redacted>")
            .field("clock", &"<redacted>")
            .field("requests", &"<redacted>")
            .field("timestamps", &"<redacted>")
            .finish()
    }
}

pub(crate) trait StepObserver {
    fn enabled(&self) -> bool;
    fn release(&mut self, release_ns: u64);
    fn admission(&mut self, admission: AdmissionObservation, release_ns: u64);
    fn promotion(&mut self, request_id: RequestId);
    fn work_start(&mut self, request_id: RequestId, position: usize, now_ns: u64);
    fn work_abandoned(&mut self, request_id: RequestId, position: usize);
    fn commit(
        &mut self,
        request_id: RequestId,
        position: usize,
        phase: ServicePhase,
        emitted: bool,
        now_ns: u64,
    );
    fn cancellation(&mut self, request_id: RequestId, disposition: CancelDisposition, now_ns: u64);
    fn terminal(&mut self, result: TerminalResult, now_ns: u64);
    fn request_zero(&mut self, request_id: RequestId, now_ns: u64);
    fn has_pending_worker_quiescence(&self) -> bool;
    fn worker_quiescent(&mut self, request_id: RequestId, now_ns: u64);
    fn flush_worker_quiescence(&mut self, now_ns: u64);
    fn preempted(&mut self, request_id: RequestId);
    fn resumed(&mut self, request_id: RequestId);
    fn wave(&mut self, selected: usize);
    fn step(&mut self, report: StepReport);
}

impl StepObserver for RunObserver {
    #[inline(always)]
    fn enabled(&self) -> bool {
        true
    }

    fn release(&mut self, release_ns: u64) {
        self.record_release(release_ns);
    }

    fn admission(&mut self, admission: AdmissionObservation, release_ns: u64) {
        self.record_admission(admission, release_ns);
    }

    fn promotion(&mut self, request_id: RequestId) {
        self.record_promotion(request_id);
    }

    fn work_start(&mut self, request_id: RequestId, position: usize, now_ns: u64) {
        self.record_work_start(request_id, position, now_ns);
    }

    fn work_abandoned(&mut self, request_id: RequestId, position: usize) {
        self.record_work_abandoned(request_id, position);
    }

    fn commit(
        &mut self,
        request_id: RequestId,
        position: usize,
        phase: ServicePhase,
        emitted: bool,
        now_ns: u64,
    ) {
        self.record_commit(request_id, position, phase, emitted, now_ns);
    }

    fn cancellation(&mut self, request_id: RequestId, disposition: CancelDisposition, now_ns: u64) {
        self.record_cancel(request_id, disposition, now_ns);
    }

    fn terminal(&mut self, result: TerminalResult, now_ns: u64) {
        self.record_terminal(result, now_ns);
    }

    fn request_zero(&mut self, request_id: RequestId, now_ns: u64) {
        self.record_request_zero(request_id, now_ns);
    }

    fn has_pending_worker_quiescence(&self) -> bool {
        self.pending_worker_quiescence()
    }

    fn worker_quiescent(&mut self, request_id: RequestId, now_ns: u64) {
        self.record_worker_quiescent(request_id, now_ns);
    }

    fn flush_worker_quiescence(&mut self, now_ns: u64) {
        self.record_pending_worker_quiescence(now_ns);
    }

    fn preempted(&mut self, request_id: RequestId) {
        self.record_preempted(request_id);
    }

    fn resumed(&mut self, request_id: RequestId) {
        self.record_resumed(request_id);
    }

    fn wave(&mut self, selected: usize) {
        self.record_wave(selected);
    }

    fn step(&mut self, report: StepReport) {
        self.record_step(report);
    }
}

pub(crate) struct NullObserver;

impl StepObserver for NullObserver {
    #[inline(always)]
    fn enabled(&self) -> bool {
        false
    }
    #[inline(always)]
    fn release(&mut self, _: u64) {}
    #[inline(always)]
    fn admission(&mut self, _: AdmissionObservation, _: u64) {}
    #[inline(always)]
    fn promotion(&mut self, _: RequestId) {}
    #[inline(always)]
    fn work_start(&mut self, _: RequestId, _: usize, _: u64) {}
    #[inline(always)]
    fn work_abandoned(&mut self, _: RequestId, _: usize) {}
    #[inline(always)]
    fn commit(&mut self, _: RequestId, _: usize, _: ServicePhase, _: bool, _: u64) {}
    #[inline(always)]
    fn cancellation(&mut self, _: RequestId, _: CancelDisposition, _: u64) {}
    #[inline(always)]
    fn terminal(&mut self, _: TerminalResult, _: u64) {}
    #[inline(always)]
    fn request_zero(&mut self, _: RequestId, _: u64) {}
    #[inline(always)]
    fn has_pending_worker_quiescence(&self) -> bool {
        false
    }
    #[inline(always)]
    fn worker_quiescent(&mut self, _: RequestId, _: u64) {}
    #[inline(always)]
    fn flush_worker_quiescence(&mut self, _: u64) {}
    #[inline(always)]
    fn preempted(&mut self, _: RequestId) {}
    #[inline(always)]
    fn resumed(&mut self, _: RequestId) {}
    #[inline(always)]
    fn wave(&mut self, _: usize) {}
    #[inline(always)]
    fn step(&mut self, _: StepReport) {}
}

fn ordered(before: Option<u64>, after: Option<u64>) -> bool {
    match (before, after) {
        (Some(before), Some(after)) => before <= after,
        _ => true,
    }
}

fn terminal_outcome_consistent(
    request: &RunRequestObservation,
    outcome: &TerminalOutcome,
    terminal_ns: u64,
) -> bool {
    match outcome {
        TerminalOutcome::Completed if request.max_new_tokens == 0 => {
            request.committed_positions == request.total_positions && request.emitted_tokens == 0
        }
        // A generated stop token may complete before the reserved maximum, so
        // exact per-commit emission legality is the strongest retained proof.
        TerminalOutcome::Completed => {
            request.committed_positions >= request.prompt_len && request.emitted_tokens > 0
        }
        TerminalOutcome::Cancelled => request.cancel_linearized_ns.is_some(),
        TerminalOutcome::DeadlineExceeded => {
            request.cancel_linearized_ns.is_none()
                && request
                    .resolved_deadline_ns
                    .is_some_and(|deadline| terminal_ns >= deadline)
        }
        TerminalOutcome::Failed { .. } => true,
    }
}

fn reserve_exact<T>(
    values: &mut Vec<T>,
    count: usize,
    resource: &'static str,
) -> SchedulerResult<()> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| SchedulerError::allocation_failure(resource, u64::MAX))?;
    if bytes > isize::MAX as usize {
        return Err(SchedulerError::allocation_failure(
            resource,
            u64::try_from(bytes).unwrap_or(u64::MAX),
        ));
    }
    values.try_reserve_exact(count).map_err(|_| {
        SchedulerError::allocation_failure(resource, u64::try_from(bytes).unwrap_or(u64::MAX))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::TerminalOutcome;

    #[test]
    fn clock_debug_and_observer_debug_redact_timing_and_identity() {
        let clock = RunClock::new();
        assert!(format!("{clock:?}").contains("<redacted>"));
    }

    #[test]
    fn null_observer_is_zero_sized() {
        assert_eq!(size_of::<NullObserver>(), 0);
    }

    #[test]
    fn failure_enum_and_terminal_outcome_remain_copy() {
        let failure = RunObserverFailure::ClockRegression;
        assert_eq!(failure, failure);
        let outcome = TerminalOutcome::Cancelled;
        assert_eq!(outcome, outcome);
    }
}
