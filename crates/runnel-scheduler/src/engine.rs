use std::{collections::VecDeque, fmt, mem::size_of};

#[cfg(test)]
use std::cell::Cell;

use runnel_runtime::{AdapterWorkIdentity, DecoderAdapter, SamplingWorkspace};

use crate::{
    accounting::{
        ADMISSION_BASE_PLAN_LEN, ChargePlan, active_plan, admission_base_plan,
        aggregate_admission_base_plans, map_ledger_error, shared_static_plan,
    },
    checkpoint::{
        CheckpointContext, CheckpointDriver, CheckpointFire, CheckpointPoint, NoCheckpoints,
    },
    config::{MAX_BATCH_WIDTH, SchedulerConfig},
    control::{
        ControlBatchBindPermit, ControlBinding, ControlRegistry, ControlSnapshot,
        available_batch_control_bytes, batch_control_permit_bytes,
    },
    endpoint::{
        EndpointBatchBindPermit, EndpointDiscardReport, EndpointProducer, EndpointReapReport,
        EndpointReceiver, EndpointRegistry, EndpointSnapshot, PreparedEndpointBinding,
        TerminalCommitGuard, TryPop, ValidatedOutputCommitGuard, batch_endpoint_permit_bytes,
        prepared_batch_endpoint_bytes,
    },
    error::{ErrorCategory, SchedulerError, SchedulerResult},
    id::{
        EngineTransactionIdIssuer, IdentityExhausted, RequestIdIssuer, SlotGeneration,
        SlotGenerationIssuer, SlotKey,
    },
    ledger::{
        CapacityLedger, ExactReservationPartitions, LEDGER_CATEGORY_COUNT, LedgerCategory,
        LedgerReservation, ProvisionalLedgerPermit, ReservationPartition,
    },
    ledger_trace::{LedgerTraceCursor, LedgerTraceRead},
    request::{
        BatchAdmission, BatchDeadline, BatchRequestSpec, CancelDisposition, EngineSnapshot,
        OutputEvent, RequestPhase, RequestSpec, ShutdownReport, StepReport, TerminalOutcome,
        TerminalResult,
    },
    ring::{DrrRing, RingError},
    run_observer::{AdmissionObservation, NullObserver, StepObserver},
    trace::{
        ServicePhase, ServiceTraceCursor, ServiceTraceEvent, ServiceTraceRead, ServiceTraceStatus,
    },
    wave::{TaskEnvelope, WaveScratch, WaveScratchError, WaveSelection},
};

#[cfg(any(test, feature = "m5-run-observer-instrumentation"))]
use crate::run_observer::RunObserver;

#[cfg(any(test, feature = "deterministic-checkpoint-instrumentation"))]
use crate::checkpoint::{CheckpointDirective, CheckpointPlan};

const ACTIVE_PLAN_LEN: usize = 2;

#[cfg(test)]
thread_local! {
    static BATCH_ALLOCATION_FAILURE_AFTER: Cell<Option<usize>> = const { Cell::new(None) };
    static SHUTDOWN_RELEASE_FAILURE: Cell<bool> = const { Cell::new(false) };
    static CORRUPT_NEXT_WAVE_TASK_TRANSACTION: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn fail_batch_allocation_after_for_test(successful_checkpoints: usize) {
    BATCH_ALLOCATION_FAILURE_AFTER.with(|remaining| remaining.set(Some(successful_checkpoints)));
}

#[cfg(test)]
fn batch_allocation_checkpoint() -> SchedulerResult<()> {
    BATCH_ALLOCATION_FAILURE_AFTER.with(|remaining| match remaining.get() {
        None => Ok(()),
        Some(0) => {
            remaining.set(None);
            Err(SchedulerError::allocation_failure(
                "prepared batch request payload",
                0,
            ))
        }
        Some(value) => {
            remaining.set(Some(value - 1));
            Ok(())
        }
    })
}

#[cfg(not(test))]
fn batch_allocation_checkpoint() -> SchedulerResult<()> {
    Ok(())
}

#[cfg(test)]
pub(crate) fn fail_next_shutdown_release_for_test() {
    SHUTDOWN_RELEASE_FAILURE.with(|failure| failure.set(true));
}

#[cfg(test)]
pub(crate) fn corrupt_next_wave_task_transaction_for_test() {
    CORRUPT_NEXT_WAVE_TASK_TRANSACTION.with(|corrupt| corrupt.set(true));
}

#[cfg(test)]
fn take_wave_task_transaction_corruption_for_test() -> bool {
    CORRUPT_NEXT_WAVE_TASK_TRANSACTION.with(|corrupt| corrupt.replace(false))
}

#[cfg(test)]
fn shutdown_release_checkpoint() -> SchedulerResult<()> {
    SHUTDOWN_RELEASE_FAILURE.with(|failure| {
        if failure.replace(false) {
            Err(SchedulerError::allocation_failure(
                "shutdown release checkpoint",
                0,
            ))
        } else {
            Ok(())
        }
    })
}

#[cfg(not(test))]
fn shutdown_release_checkpoint() -> SchedulerResult<()> {
    Ok(())
}

/// Unpublished two-phase direct-engine admission transaction.
///
/// Accepted offers are one strict FIFO prefix; the first capacity failure
/// rejects the complete suffix, so a smaller later request cannot bypass it.
/// The guard holds exclusive engine fields and all accepted endpoint locks,
/// preventing scheduler progress or lifecycle reuse until commit or drop.
/// Dropping it destroys charged prompts and endpoint queues before rolling the
/// aggregate provisional ledger reservation and its peaks back exactly.
#[must_use = "prepared admissions must be committed or deliberately dropped"]
pub struct PreparedAdmission<'engine, A: DecoderAdapter> {
    slots: &'engine mut Vec<RequestSlot<A>>,
    free_slots: &'engine mut Vec<usize>,
    controls: ControlBatchBindPermit<'engine>,
    endpoints: EndpointBatchBindPermit<'engine>,
    queued: &'engine mut VecDeque<SlotKey>,
    request_ids: &'engine mut RequestIdIssuer,
    monotonic_ns: &'engine mut u64,
    partitions: ExactReservationPartitions,
    prepared: Vec<PreparedBatchRequest<A>>,
    prospective_ids: Vec<crate::RequestId>,
    rejection: Option<SchedulerError>,
    offered_deadlines: Vec<BatchDeadline>,
    offered_count: usize,
    #[cfg(any(test, feature = "m5-run-observer-instrumentation"))]
    observation_domain: crate::control::ControlDomain,
    // Keep the provisional charge last: Rust drops fields in declaration
    // order, so every charged payload and endpoint queue is destroyed before
    // rollback can make that capacity available again.
    ledger: Option<ProvisionalLedgerPermit<'engine>>,
}

impl<A: DecoderAdapter> fmt::Debug for PreparedAdmission<'_, A> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedAdmission")
            .field("offered_count", &self.offered_count)
            .field("publication", &"<unpublished>")
            .finish()
    }
}

impl<A: DecoderAdapter> PreparedAdmission<'_, A> {
    /// Publishes all accepted requests at one monotonic release boundary.
    ///
    /// Absolute deadlines expire inclusively; release-relative deadlines are
    /// resolved as `release_ns + offset` with checked arithmetic. Every offered
    /// deadline is checked, including the capacity-rejected suffix. Timestamp
    /// and deadline failures happen before mutation. After that boundary the
    /// method allocates nothing, executes no model work, and has no error path;
    /// transaction scratch is reclaimed after publication.
    pub fn commit_prepared_batch(self, release_ns: u64) -> SchedulerResult<BatchAdmission> {
        let mut observer = NullObserver;
        self.commit_prepared_batch_with_observer(release_ns, &mut observer)
    }

    /// Publishes one preregistered batch and binds its accepted identities to
    /// a preallocated M5 run observer at the same release boundary.
    #[cfg(any(test, feature = "m5-run-observer-instrumentation"))]
    #[doc(hidden)]
    pub fn commit_prepared_batch_observed(
        self,
        observer: &mut RunObserver,
    ) -> SchedulerResult<BatchAdmission> {
        observer.preflight_admission(&self.observation_domain, self.prepared.len())?;
        let clock = observer.clock();
        let release_ns = clock.now_ns();
        self.commit_prepared_batch_with_observer(release_ns, observer)
    }

    #[cfg(test)]
    pub(crate) fn commit_prepared_batch_observed_at_for_test(
        self,
        release_ns: u64,
        observer: &mut RunObserver,
    ) -> SchedulerResult<BatchAdmission> {
        observer.preflight_admission(&self.observation_domain, self.prepared.len())?;
        self.commit_prepared_batch_with_observer(release_ns, observer)
    }

    fn commit_prepared_batch_with_observer<O: StepObserver>(
        mut self,
        release_ns: u64,
        observer: &mut O,
    ) -> SchedulerResult<BatchAdmission> {
        if release_ns < *self.monotonic_ns {
            return Err(SchedulerError::invalid_request(
                "release_ns",
                "cannot move backwards",
            ));
        }
        for deadline in self.offered_deadlines.iter().copied() {
            match deadline {
                BatchDeadline::None => {}
                BatchDeadline::Absolute(deadline) => {
                    if release_ns >= deadline {
                        return Err(SchedulerError::deadline_exceeded());
                    }
                }
                BatchDeadline::AfterRelease(offset) => {
                    let deadline = release_ns.checked_add(offset).ok_or_else(|| {
                        SchedulerError::invalid_request(
                            "deadline",
                            "release-relative deadline overflows",
                        )
                    })?;
                    if release_ns >= deadline {
                        return Err(SchedulerError::deadline_exceeded());
                    }
                }
            }
        }
        for request in &mut self.prepared {
            request.deadline_ns = match request.deadline {
                BatchDeadline::None => None,
                BatchDeadline::Absolute(deadline) => {
                    if release_ns >= deadline {
                        return Err(SchedulerError::deadline_exceeded());
                    }
                    Some(deadline)
                }
                BatchDeadline::AfterRelease(offset) => {
                    let deadline = release_ns.checked_add(offset).ok_or_else(|| {
                        SchedulerError::invalid_request(
                            "deadline",
                            "release-relative deadline overflows",
                        )
                    })?;
                    if release_ns >= deadline {
                        return Err(SchedulerError::deadline_exceeded());
                    }
                    Some(deadline)
                }
            };
        }

        // No fallible call is permitted after this publication boundary.
        *self.monotonic_ns = release_ns;
        observer.release(release_ns);
        let mut aggregate = match self.ledger.take() {
            Some(permit) => permit.commit_for_requests(&self.prospective_ids, &self.partitions),
            None => LedgerReservation::empty(),
        };
        self.request_ids.commit_prevalidated(&self.prospective_ids);
        for request in &self.prepared {
            let issued = self.slots[request.slot_index]
                .generations
                .issue_prevalidated(request.generation);
            debug_assert_eq!(issued, request.generation);
        }
        let controls = self.controls.commit();
        let endpoints = self.endpoints.commit(controls, &self.prospective_ids);
        debug_assert_eq!(endpoints.len(), self.prepared.len());
        let first_accepted_id = self.prospective_ids.first().copied();
        let accepted_count = self.prepared.len();

        for ((request, endpoint), request_id) in self
            .prepared
            .into_iter()
            .zip(endpoints)
            .zip(self.prospective_ids.iter().copied())
        {
            let observation = AdmissionObservation {
                offered_index: request.offered_index,
                request_id,
                prompt_len: request.prompt.len(),
                max_new_tokens: request.max_new_tokens,
                total_positions: request.total_positions,
                resolved_deadline_ns: request.deadline_ns,
            };
            let (base_reservation, remainder) = self.partitions.split_next(aggregate);
            aggregate = remainder;
            let (prompt_reservation, retained_reservation) =
                base_reservation.split_category(LedgerCategory::PromptStorage);
            let (control, endpoint, receiver) = endpoint.into_parts();
            let key = request.key;
            let record = RequestRecord {
                request_id,
                key,
                phase: RequestPhase::Queued,
                prompt: Some(request.prompt),
                max_new_tokens: request.max_new_tokens,
                total_positions: request.total_positions,
                sampling: request.sampling,
                deadline_ns: request.deadline_ns,
                admitted_ns: release_ns,
                control,
                state_layout: request.state_layout,
                state: None,
                active_plan: request.active_plan,
                active_reservation: None,
                prompt_reservation: Some(prompt_reservation),
                retained_reservation: Some(retained_reservation),
                committed_positions: 0,
                emitted_tokens: 0,
                next_decode_token: None,
                rng_state: None,
                adapter_binding: None,
                endpoint,
                direct_receiver: Some(receiver),
            };
            self.slots[request.slot_index].key = Some(key);
            self.slots[request.slot_index].record = Some(record);
            self.queued.push_back(key);
            debug_assert_eq!(
                self.free_slots.get(request.free_position),
                Some(&request.slot_index)
            );
            self.free_slots.swap_remove(request.free_position);
            observer.admission(observation, release_ns);
        }
        debug_assert!(aggregate.is_empty());
        debug_assert!(self.partitions.is_complete());
        Ok(BatchAdmission::new(
            release_ns,
            self.offered_count,
            first_accepted_id,
            accepted_count,
            self.rejection,
        ))
    }
}

/// Pure, synchronously step-able owner of bounded decoder scheduling state.
pub struct SchedulerEngine<A: DecoderAdapter> {
    adapter: Option<A>,
    config: SchedulerConfig,
    ledger: CapacityLedger,
    shared_reservation: Option<LedgerReservation>,
    slots: Vec<RequestSlot<A>>,
    free_slots: Vec<usize>,
    controls: Option<ControlRegistry>,
    endpoints: Option<EndpointRegistry>,
    queued: VecDeque<SlotKey>,
    ring: Option<DrrRing>,
    request_ids: RequestIdIssuer,
    transaction_ids: EngineTransactionIdIssuer,
    last_adapter_transaction_id: Option<u64>,
    workspace: Option<A::Workspace>,
    sampling: Option<SamplingWorkspace>,
    wave: Option<WaveScratch<A::PreparedToken, A::ExpertTask, A::ExpertContribution>>,
    service_trace: Option<Vec<ServiceTraceEvent>>,
    trace_overflowed: bool,
    monotonic_ns: u64,
    closed: bool,
}

impl<A: DecoderAdapter> fmt::Debug for SchedulerEngine<A> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let snapshot = self.snapshot();
        formatter
            .debug_struct("SchedulerEngine")
            .field("configuration", &"<redacted>")
            .field("policy", &self.config.scheduling_policy())
            .field("closed", &snapshot.closed)
            .field("queued_requests", &snapshot.queued_requests)
            .field("active_requests", &snapshot.active_requests)
            .field(
                "retained_terminal_results",
                &snapshot.retained_terminal_results,
            )
            .field("ledger_used_bytes", &snapshot.ledger_used_bytes)
            .field("trace_overflowed", &self.trace_overflowed)
            .finish()
    }
}

impl<A: DecoderAdapter> SchedulerEngine<A> {
    /// Preallocates and engine-binds the sealed M5 timing observer.
    #[cfg(any(test, feature = "m5-run-observer-instrumentation"))]
    #[doc(hidden)]
    pub fn prepare_run_observer(&self) -> SchedulerResult<RunObserver> {
        self.ensure_open()?;
        let domain = self
            .controls
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("request controls are unavailable"))?
            .checkpoint_domain();
        RunObserver::try_bound(&self.config, domain)
    }

    #[cfg(any(test, feature = "m5-run-observer-instrumentation"))]
    fn validate_run_observer_domain(&self, observer: &RunObserver) -> SchedulerResult<()> {
        let controls = self
            .controls
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("request controls are unavailable"))?;
        if observer.belongs_to_registry(controls) {
            Ok(())
        } else {
            Err(SchedulerError::invalid_request(
                "run observer",
                "belongs to another scheduler engine",
            ))
        }
    }

    /// Fallibly preallocates every shared synchronous-core capacity.
    pub fn new(adapter: A, config: SchedulerConfig) -> SchedulerResult<Self> {
        if config.worker_count() != 1 {
            return Err(SchedulerError::unsupported(
                "the synchronous core currently requires exactly one worker",
            ));
        }
        let observed_layout = adapter
            .execution_layout()
            .map_err(|source| SchedulerError::adapter("revalidating execution layout", source))?;
        if observed_layout != config.execution_layout()
            || adapter.vocabulary_size() != config.vocabulary_size()
        {
            return Err(SchedulerError::invalid_request(
                "config",
                "was validated for different adapter geometry",
            ));
        }
        let required_batch_scratch = Self::required_batch_admission_reserve_bytes(&config)?;
        if config.admission_reserve_bytes() < required_batch_scratch {
            return Err(SchedulerError::resource_exhausted(
                "batch admission scratch",
                required_batch_scratch,
                config.admission_reserve_bytes(),
            ));
        }

        let category_limits = [config.logical_memory_limit_bytes(); LEDGER_CATEGORY_COUNT];
        let mut ledger = CapacityLedger::new(config.logical_memory_limit_bytes(), category_limits);
        let shared_plan = shared_static_plan(&config)?;
        let provisional = ledger
            .acquire_provisional(shared_plan.as_slice())
            .map_err(map_ledger_error)?;

        // Preallocate the second half of every paired trace slot before
        // allocating any other scheduler-owned payload. If this fails, destroy
        // the transferred adapter before rolling back its model-resident and
        // other static logical ownership.
        let ledger_trace = match ledger.prepare_trace_recorder(config.trace_capacity()) {
            Ok(trace) => trace,
            Err(_) => {
                drop(adapter);
                ledger
                    .rollback_provisional(provisional)
                    .map_err(map_ledger_error)?;
                return Err(SchedulerError::allocation_failure(
                    "ledger trace capacity",
                    u64::try_from(config.trace_capacity())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(64),
                ));
            }
        };

        let allocation = Self::try_allocate_shared(&adapter, &config);
        let (
            slots,
            free_slots,
            controls,
            endpoints,
            queued,
            ring,
            workspace,
            sampling,
            wave,
            service_trace,
        ) = match allocation {
            Ok(parts) => parts,
            Err(error) => {
                drop(ledger_trace);
                drop(adapter);
                ledger
                    .rollback_provisional(provisional)
                    .map_err(map_ledger_error)?;
                return Err(error);
            }
        };
        let shared_reservation = ledger
            .commit_provisional(provisional)
            .map_err(map_ledger_error)?;
        ledger.install_preallocated_trace(ledger_trace);

        Ok(Self {
            adapter: Some(adapter),
            config,
            ledger,
            shared_reservation: Some(shared_reservation),
            slots,
            free_slots,
            controls: Some(controls),
            endpoints: Some(endpoints),
            queued,
            ring: Some(ring),
            request_ids: RequestIdIssuer::new(),
            transaction_ids: EngineTransactionIdIssuer::new(),
            last_adapter_transaction_id: None,
            workspace: Some(workspace),
            sampling: Some(sampling),
            wave: Some(wave),
            service_trace: Some(service_trace),
            trace_overflowed: false,
            monotonic_ns: 0,
            closed: false,
        })
    }

    /// Returns the immutable validated configuration bound to this engine.
    #[must_use]
    pub const fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    #[cfg(test)]
    pub(crate) fn exhaust_slot_generation_for_test(&mut self, slot_index: usize) {
        assert!(self.free_slots.contains(&slot_index));
        assert!(self.slots[slot_index].record.is_none());
        self.slots[slot_index].generations = SlotGenerationIssuer::from_next(0);
    }

    /// Returns the maximum checked semantic metadata footprint across payload,
    /// control-permit, and endpoint-permit preparation phases.
    ///
    /// This is a logical `requested_count * size_of::<T>()` capacity bound, not
    /// an allocator-overhead or RSS estimate. Prompt and output buffers are
    /// covered separately by the provisional request-owned ledger. Engine
    /// construction rejects a smaller raw `admission_reserve_bytes` value.
    #[must_use = "the typed batch-admission reserve requirement must be checked"]
    pub fn required_batch_admission_reserve_bytes(
        config: &SchedulerConfig,
    ) -> SchedulerResult<u64> {
        let maximum_offers = config.command_capacity();
        let maximum_accepted = maximum_offers
            .min(config.max_queued_requests())
            .min(config.max_outstanding_requests());
        let deadlines = batch_vec_bytes::<BatchDeadline>(maximum_offers);
        let controls = available_batch_control_bytes(maximum_offers);
        let partitions = batch_vec_bytes::<ReservationPartition>(maximum_accepted);
        let identities = batch_vec_bytes::<crate::RequestId>(maximum_accepted);
        let requests = batch_vec_bytes::<PreparedBatchRequest<A>>(maximum_accepted);
        let prepared_endpoints = prepared_batch_endpoint_bytes(maximum_accepted);
        let control_permit = batch_control_permit_bytes(maximum_accepted);
        let endpoint_permit = batch_endpoint_permit_bytes(maximum_accepted);
        let payload_phase = checked_batch_scratch_sum(&[
            batch_vec_bytes::<ValidatedBatchOffer<'static, A>>(maximum_offers),
            deadlines,
            controls,
            batch_vec_bytes::<PreparedSlot>(maximum_offers),
            partitions,
            identities,
            requests,
            prepared_endpoints,
        ])?;
        let control_phase = checked_batch_scratch_sum(&[
            deadlines,
            controls,
            partitions,
            identities,
            requests,
            prepared_endpoints,
            control_permit,
        ])?;
        let endpoint_phase = checked_batch_scratch_sum(&[
            deadlines,
            partitions,
            identities,
            requests,
            prepared_endpoints,
            control_permit,
            endpoint_permit,
        ])?;
        let bytes = payload_phase.max(control_phase).max(endpoint_phase);
        if bytes > isize::MAX as usize {
            return Err(SchedulerError::allocation_failure(
                "batch admission scratch",
                usize_to_u64(bytes)?,
            ));
        }
        usize_to_u64(bytes)
    }

    /// Installs a bounded endpoint semantic probe before the first admission.
    ///
    /// The explicit observation capacity is independent of request-table
    /// geometry. Ordinary construction does not allocate or install a probe.
    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    #[doc(hidden)]
    pub fn install_actor_stress_instrumentation(
        &mut self,
        observation_capacity: usize,
    ) -> SchedulerResult<crate::endpoint::ActorStressRecorder> {
        if self.closed
            || self.free_slots.len() != self.slots.len()
            || !self.queued.is_empty()
            || self.slots.iter().any(|slot| slot.record.is_some())
            || self.request_ids.peek().map(crate::RequestId::get) != Ok(1)
        {
            return Err(SchedulerError::internal(
                "actor stress instrumentation must be installed before admission",
            ));
        }
        let endpoints = self
            .endpoints
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("request endpoints are unavailable"))?;
        if endpoints.actor_stress_recorder().is_some() {
            return Err(SchedulerError::internal(
                "actor stress instrumentation was already installed",
            ));
        }
        let recorder =
            crate::endpoint::ActorStressRecorder::try_with_capacity(observation_capacity)?;
        endpoints.install_actor_stress_recorder(recorder.clone())?;
        Ok(recorder)
    }

    /// Returns the installed semantic probe without exposing endpoint tables.
    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    #[doc(hidden)]
    #[must_use]
    pub fn actor_stress_instrumentation(&self) -> Option<crate::endpoint::ActorStressRecorder> {
        self.endpoints
            .as_ref()
            .and_then(crate::endpoint::EndpointRegistry::actor_stress_recorder)
    }

    /// Counts every live request record, including a terminal whose result was
    /// consumed while its output or receiver acknowledgement remains pending.
    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    pub(crate) fn actor_stress_live_request_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.record.is_some())
            .count()
    }

    fn try_allocate_shared(
        adapter: &A,
        config: &SchedulerConfig,
    ) -> SchedulerResult<SharedAllocation<A>> {
        let mut slots = Vec::new();
        try_reserve_vec(
            &mut slots,
            config.max_outstanding_requests(),
            "request slot table",
        )?;
        for _ in 0..config.max_outstanding_requests() {
            slots.push(RequestSlot::new());
        }

        let mut free_slots = Vec::new();
        try_reserve_vec(
            &mut free_slots,
            config.max_outstanding_requests(),
            "free request slots",
        )?;
        for index in (0..config.max_outstanding_requests()).rev() {
            free_slots.push(index);
        }

        let controls = ControlRegistry::try_with_capacity(config.max_outstanding_requests())?;
        let endpoints = EndpointRegistry::try_with_capacity(config.max_outstanding_requests())?;

        let mut queued = VecDeque::new();
        try_reserve_deque(
            &mut queued,
            config.max_queued_requests(),
            "queued request FIFO",
        )?;
        let ring = DrrRing::try_with_capacity_and_policy(
            config.max_active_requests(),
            config.scheduling_policy(),
        )
        .map_err(map_ring_error)?;
        let workspace = adapter
            .new_workspace()
            .map_err(|source| SchedulerError::adapter("allocating worker workspace", source))?;
        let sampling = SamplingWorkspace::new(config.vocabulary_size())
            .map_err(|source| SchedulerError::sampling("allocating sampling workspace", source))?;
        let wave = WaveScratch::try_new(config.batch_width(), config.max_tasks_per_token())
            .map_err(map_wave_error)?;
        let mut service_trace = Vec::new();
        try_reserve_vec(
            &mut service_trace,
            config.trace_capacity(),
            "service trace capacity",
        )?;

        Ok((
            slots,
            free_slots,
            controls,
            endpoints,
            queued,
            ring,
            workspace,
            sampling,
            wave,
            service_trace,
        ))
    }

    /// Validates every offered description and prepares one unpublished FIFO
    /// admission prefix for the synchronous engine surface.
    ///
    /// The nonempty offer slice is capped by `command_capacity`. Intrinsic
    /// validation covers the complete slice before resource pressure selects
    /// a prefix, so a malformed rejected suffix cannot be hidden. The first
    /// pressure point rejects every remaining offer with the same bounded
    /// error. Preparation assigns no request identity and publishes no slot
    /// generation, endpoint, control, queue entry, or runnable state. The
    /// Tokio actor command API remains a separate single-request ingress.
    pub fn prepare_submit_batch<'engine>(
        &'engine mut self,
        offers: &[BatchRequestSpec<'_>],
    ) -> SchedulerResult<PreparedAdmission<'engine, A>> {
        self.ensure_open()?;
        if offers.is_empty() {
            return Err(SchedulerError::invalid_request(
                "batch offers",
                "must be nonempty",
            ));
        }
        if offers.len() > self.config.command_capacity() {
            return Err(SchedulerError::invalid_request(
                "batch offers",
                "exceeds the configured direct-ingress ceiling",
            ));
        }

        // Description validation is deliberately complete before pressure is
        // considered, so a rejected suffix cannot hide a malformed offer.
        let mut validated = Vec::new();
        try_reserve_vec(
            &mut validated,
            offers.len(),
            "validated batch admission offers",
        )?;
        for offer in offers.iter().copied() {
            validated.push(self.validate_batch_offer(offer)?);
        }
        let mut offered_deadlines = Vec::new();
        try_reserve_vec(
            &mut offered_deadlines,
            offers.len(),
            "validated batch admission deadlines",
        )?;
        offered_deadlines.extend(validated.iter().map(|offer| offer.deadline));

        let controls = self
            .controls
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("request controls are unavailable"))?
            .prepare_available(offers.len())?;

        let mut slot_candidates = Vec::new();
        try_reserve_vec(
            &mut slot_candidates,
            offers.len(),
            "prepared batch request slots",
        )?;
        let mut exhausted_slot_pressure = None;
        for (free_position, slot_index) in self.free_slots.iter().copied().enumerate().rev() {
            if slot_candidates.len() == offers.len() {
                break;
            }
            match self.slots[slot_index].generations.peek() {
                Ok(generation) => slot_candidates.push(PreparedSlot {
                    slot_index,
                    free_position,
                    generation,
                    key: SlotKey::new(slot_index, generation),
                }),
                Err(error) => {
                    if exhausted_slot_pressure.is_none() {
                        exhausted_slot_pressure =
                            Some(ResourcePressure::from_error(map_identity_error(error))?);
                    }
                }
            }
        }

        let mut aggregate_plan = aggregate_admission_base_plans(&[])?;
        let mut accepted_count = 0_usize;
        let mut pressure = None;
        for offer in &validated {
            let queued_after = self
                .queued
                .len()
                .checked_add(accepted_count)
                .and_then(|queued| queued.checked_add(1))
                .ok_or_else(|| SchedulerError::internal("queued request count overflows"))?;
            if queued_after > self.config.max_queued_requests() {
                pressure = Some(ResourcePressure::new(
                    "queued request count",
                    usize_to_u64(queued_after)?,
                    usize_to_u64(self.config.max_queued_requests())?,
                ));
                break;
            }
            if accepted_count >= slot_candidates.len() {
                pressure = Some(if let Some(pressure) = exhausted_slot_pressure {
                    pressure
                } else {
                    let limit = usize_to_u64(self.slots.len())?;
                    ResourcePressure::new("request slot count", limit, limit)
                });
                break;
            }
            if let Err(error) = self.request_ids.ensure_available(accepted_count + 1) {
                pressure = Some(ResourcePressure::from_error(map_identity_error(error))?);
                break;
            }

            let candidate_aggregate =
                aggregate_admission_base_plans(&[aggregate_plan, offer.admission_plan])?;
            match self.ledger.can_acquire(candidate_aggregate.as_slice()) {
                Ok(()) => {}
                Err(error) if error.is_resource_exhausted() => {
                    pressure = Some(ResourcePressure::from_error(map_ledger_error(error))?);
                    break;
                }
                Err(error) => return Err(map_ledger_error(error)),
            }
            if accepted_count >= controls.len() {
                let control_pressure = controls.pressure().ok_or_else(|| {
                    SchedulerError::internal("prepared control capacity is inconsistent")
                })?;
                pressure = Some(ResourcePressure::from_error(control_pressure.error())?);
                break;
            }
            aggregate_plan = candidate_aggregate;
            accepted_count += 1;
        }

        let rejection = pressure.map(ResourcePressure::error);
        debug_assert_eq!(rejection.is_some(), accepted_count != offers.len());

        slot_candidates.truncate(accepted_count);

        let mut partition_descriptors = Vec::new();
        try_reserve_vec(
            &mut partition_descriptors,
            accepted_count,
            "batch ledger reservation partitions",
        )?;
        for offer in validated.iter().take(accepted_count) {
            partition_descriptors.push(
                ReservationPartition::from_charges(offer.admission_plan.as_slice())
                    .map_err(map_ledger_error)?,
            );
        }

        let mut prospective_ids = Vec::new();
        try_reserve_vec(
            &mut prospective_ids,
            accepted_count,
            "prospective batch request identities",
        )?;
        self.request_ids
            .preview_into(accepted_count, &mut prospective_ids)
            .map_err(map_identity_error)?;

        let mut prepared_request_storage = Vec::new();
        try_reserve_vec(
            &mut prepared_request_storage,
            accepted_count,
            "prepared batch request records",
        )?;
        let mut prepared_endpoint_storage = Vec::<PreparedEndpointBinding>::new();
        try_reserve_vec(
            &mut prepared_endpoint_storage,
            accepted_count,
            "prepared batch endpoint offers",
        )?;
        let output_capacity = self.config.output_capacity_per_request();
        let SchedulerEngine {
            ledger,
            slots,
            free_slots,
            controls: control_registry,
            endpoints: endpoint_registry,
            queued,
            request_ids,
            monotonic_ns,
            ..
        } = self;

        let ledger_permit = if accepted_count == 0 {
            None
        } else {
            Some(
                ledger
                    .acquire_provisional_permit(aggregate_plan.as_slice())
                    .map_err(map_ledger_error)?,
            )
        };
        let partitions = match &ledger_permit {
            Some(permit) => permit
                .prove_exact_partitions(partition_descriptors)
                .map_err(map_ledger_error)?,
            None => ExactReservationPartitions::empty(),
        };
        partitions
            .validate_trace_request_owners(&prospective_ids)
            .map_err(map_ledger_error)?;

        // These shadow bindings are deliberately declared after the permit.
        // On every preparation error, Rust drops their charged payloads and
        // endpoint queues before rolling the provisional ledger charge back.
        let mut prepared_requests = prepared_request_storage;
        let mut prepared_endpoints = prepared_endpoint_storage;

        let endpoints = endpoint_registry
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("request endpoints are unavailable"))?;
        for (ordinal, (offer, slot)) in validated
            .into_iter()
            .take(accepted_count)
            .zip(slot_candidates)
            .enumerate()
        {
            batch_allocation_checkpoint()?;
            let prompt = Self::try_allocate_request_payload(offer.request.prompt())?;
            prepared_endpoints.push(endpoints.prepare_batch_binding(
                output_capacity,
                slot.key,
                ordinal,
                slot.free_position,
            )?);
            prepared_requests.push(PreparedBatchRequest {
                offered_index: ordinal,
                slot_index: slot.slot_index,
                free_position: slot.free_position,
                generation: slot.generation,
                key: slot.key,
                prompt,
                max_new_tokens: offer.request.max_new_tokens(),
                total_positions: offer.total_positions,
                sampling: offer.request.sampling(),
                deadline: offer.deadline,
                deadline_ns: None,
                state_layout: offer.state_layout,
                active_plan: offer.active_plan,
            });
        }

        let control_registry = control_registry
            .as_mut()
            .ok_or_else(|| SchedulerError::internal("request controls are unavailable"))?;
        #[cfg(any(test, feature = "m5-run-observer-instrumentation"))]
        let observation_domain = control_registry.checkpoint_domain();
        let controls = control_registry.begin_bind_batch(controls.into_prefix(accepted_count))?;
        let endpoint_registry = endpoint_registry
            .as_mut()
            .ok_or_else(|| SchedulerError::internal("request endpoints are unavailable"))?;
        let endpoints = endpoint_registry.begin_bind_batch(prepared_endpoints)?;

        Ok(PreparedAdmission {
            slots,
            free_slots,
            controls,
            endpoints,
            queued,
            request_ids,
            monotonic_ns,
            partitions,
            prepared: prepared_requests,
            prospective_ids,
            rejection,
            offered_deadlines,
            offered_count: offers.len(),
            #[cfg(any(test, feature = "m5-run-observer-instrumentation"))]
            observation_domain,
            ledger: ledger_permit,
        })
    }

    fn validate_batch_offer<'request>(
        &self,
        offer: BatchRequestSpec<'request>,
    ) -> SchedulerResult<ValidatedBatchOffer<'request, A>> {
        let adapter = self
            .adapter
            .as_ref()
            .ok_or_else(SchedulerError::scheduler_closed)?;
        let request = offer.request();
        let prompt = request.prompt();
        if prompt.is_empty() {
            return Err(SchedulerError::invalid_request(
                "prompt",
                "must be nonempty",
            ));
        }
        if prompt.len() > self.config.max_prompt_tokens() {
            return Err(SchedulerError::invalid_request(
                "prompt",
                "exceeds the configured token ceiling",
            ));
        }
        if request.max_new_tokens() > self.config.max_new_tokens() {
            return Err(SchedulerError::invalid_request(
                "max_new_tokens",
                "exceeds the configured ceiling",
            ));
        }
        if prompt.iter().any(|token| {
            usize::try_from(*token).map_or(true, |token| token >= self.config.vocabulary_size())
        }) {
            return Err(SchedulerError::invalid_request(
                "prompt",
                "contains a token outside the adapter vocabulary",
            ));
        }
        request
            .sampling()
            .validate(self.config.vocabulary_size())
            .map_err(|source| SchedulerError::sampling("validating request policy", source))?;
        match offer.deadline() {
            BatchDeadline::Absolute(deadline) if self.monotonic_ns >= deadline => {
                return Err(SchedulerError::deadline_exceeded());
            }
            BatchDeadline::AfterRelease(0) => {
                return Err(SchedulerError::invalid_request(
                    "deadline",
                    "release-relative deadline must be nonzero",
                ));
            }
            BatchDeadline::None | BatchDeadline::Absolute(_) | BatchDeadline::AfterRelease(_) => {}
        }

        let total_positions = prompt
            .len()
            .checked_add(request.max_new_tokens().saturating_sub(1))
            .ok_or_else(|| {
                SchedulerError::invalid_request(
                    "request length",
                    "prompt plus generation positions overflow",
                )
            })?;
        if total_positions > self.config.max_context_tokens() {
            return Err(SchedulerError::invalid_request(
                "request length",
                "exceeds the configured context ceiling",
            ));
        }
        let state_layout = adapter
            .state_layout(total_positions, self.config.state_page_tokens())
            .map_err(|source| {
                SchedulerError::adapter_with_category(
                    "validating request state layout",
                    ErrorCategory::InvalidRequest,
                    source,
                )
            })?;
        let active_plan = active_plan(&self.config, state_layout)?;
        let admission_plan = admission_base_plan(&self.config, prompt.len())?;
        ensure_request_lifecycle_feasible(&self.config, admission_plan, active_plan)?;
        Ok(ValidatedBatchOffer {
            request,
            deadline: offer.deadline(),
            total_positions,
            state_layout,
            active_plan,
            admission_plan,
        })
    }

    /// Validates and copies one request atomically before assigning its ID.
    pub fn try_submit(&mut self, request: RequestSpec<'_>) -> SchedulerResult<crate::RequestId> {
        self.try_submit_with(request, |request_id, _, receiver| {
            (Some(receiver), request_id)
        })
    }

    /// Admits a request and transfers its sole result receiver to the actor in
    /// the same infallible publication that installs the engine record.
    #[allow(dead_code, reason = "used by the staged Tokio actor integration")]
    pub(crate) fn try_submit_for_actor(
        &mut self,
        request: RequestSpec<'_>,
    ) -> SchedulerResult<ActorAdmission> {
        self.try_submit_with(request, |request_id, control, receiver| {
            (
                None,
                ActorAdmission {
                    request_id,
                    control: control.clone(),
                    receiver,
                },
            )
        })
    }

    fn try_submit_with<R>(
        &mut self,
        request: RequestSpec<'_>,
        finish: impl FnOnce(
            crate::RequestId,
            &ControlBinding,
            EndpointReceiver,
        ) -> (Option<EndpointReceiver>, R),
    ) -> SchedulerResult<R> {
        self.ensure_open()?;
        let adapter = self
            .adapter
            .as_ref()
            .ok_or_else(SchedulerError::scheduler_closed)?;
        let prompt = request.prompt();
        if prompt.is_empty() {
            return Err(SchedulerError::invalid_request(
                "prompt",
                "must be nonempty",
            ));
        }
        if prompt.len() > self.config.max_prompt_tokens() {
            return Err(SchedulerError::invalid_request(
                "prompt",
                "exceeds the configured token ceiling",
            ));
        }
        if request.max_new_tokens() > self.config.max_new_tokens() {
            return Err(SchedulerError::invalid_request(
                "max_new_tokens",
                "exceeds the configured ceiling",
            ));
        }
        if prompt.iter().any(|token| {
            usize::try_from(*token).map_or(true, |token| token >= self.config.vocabulary_size())
        }) {
            return Err(SchedulerError::invalid_request(
                "prompt",
                "contains a token outside the adapter vocabulary",
            ));
        }
        request
            .sampling()
            .validate(self.config.vocabulary_size())
            .map_err(|source| SchedulerError::sampling("validating request policy", source))?;
        if request
            .deadline_ns()
            .is_some_and(|deadline| self.monotonic_ns >= deadline)
        {
            return Err(SchedulerError::deadline_exceeded());
        }

        let total_positions = prompt
            .len()
            .checked_add(request.max_new_tokens().saturating_sub(1))
            .ok_or_else(|| {
                SchedulerError::invalid_request(
                    "request length",
                    "prompt plus generation positions overflow",
                )
            })?;
        if total_positions > self.config.max_context_tokens() {
            return Err(SchedulerError::invalid_request(
                "request length",
                "exceeds the configured context ceiling",
            ));
        }
        let state_layout = adapter
            .state_layout(total_positions, self.config.state_page_tokens())
            .map_err(|source| {
                SchedulerError::adapter_with_category(
                    "validating request state layout",
                    ErrorCategory::InvalidRequest,
                    source,
                )
            })?;
        let active_charge_plan = active_plan(&self.config, state_layout)?;
        let admission_plan = admission_base_plan(&self.config, prompt.len())?;
        ensure_request_lifecycle_feasible(&self.config, admission_plan, active_charge_plan)?;

        if self.queued.len() >= self.config.max_queued_requests() {
            return Err(SchedulerError::resource_exhausted(
                "queued request count",
                usize_to_u64(self.queued.len() + 1)?,
                usize_to_u64(self.config.max_queued_requests())?,
            ));
        }
        let mut exhausted_generation = None;
        let free_position = self
            .free_slots
            .iter()
            .rposition(|index| match self.slots[*index].generations.peek() {
                Ok(_) => true,
                Err(error) => {
                    exhausted_generation = Some(error);
                    false
                }
            })
            .ok_or_else(|| {
                exhausted_generation.map_or_else(
                    || {
                        let limit = u64::try_from(self.slots.len()).unwrap_or(u64::MAX);
                        SchedulerError::resource_exhausted("request slot count", limit, limit)
                    },
                    map_identity_error,
                )
            })?;
        let slot_index = self.free_slots[free_position];
        self.request_ids
            .ensure_available(1)
            .map_err(map_identity_error)?;
        self.slots[slot_index]
            .generations
            .ensure_available(1)
            .map_err(map_identity_error)?;
        self.ledger
            .can_acquire(admission_plan.as_slice())
            .map_err(map_ledger_error)?;
        let prepared_control = self
            .controls
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("request controls are unavailable"))?
            .prepare()?;

        let output_capacity = self.config.output_capacity_per_request();
        let (
            prompt,
            request_id,
            key,
            control,
            endpoint,
            receiver,
            prompt_reservation,
            retained_reservation,
        ) = {
            let SchedulerEngine {
                ledger,
                request_ids,
                slots,
                controls,
                endpoints,
                ..
            } = self;
            // Every fallible ownership transfer stays behind this permit. Its
            // drop path removes the provisional bytes and restores historical
            // peaks exactly, so failed admissions emit no durable evidence.
            let ledger_permit = ledger
                .acquire_provisional_permit(admission_plan.as_slice())
                .map_err(map_ledger_error)?;
            let prompt = Self::try_allocate_request_payload(prompt)?;
            let prepared_endpoint = endpoints
                .as_ref()
                .ok_or_else(|| SchedulerError::internal("request endpoints are unavailable"))?
                .prepare(output_capacity)?;
            let request_id = request_ids.issue().map_err(map_identity_error)?;
            let generation = slots[slot_index]
                .generations
                .issue()
                .map_err(map_identity_error)?;
            let key = SlotKey::new(slot_index, generation);
            let endpoints = endpoints
                .as_mut()
                .ok_or_else(|| SchedulerError::internal("request endpoints are unavailable"))?;
            let controls = controls
                .as_mut()
                .ok_or_else(|| SchedulerError::internal("request controls are unavailable"))?;
            let endpoint = endpoints.begin_bind(prepared_endpoint, key)?;
            let control = controls.bind(prepared_control)?;
            let (producer, receiver) = endpoint.commit(control.clone(), request_id);

            // This is the first infallible boundary at which every accepted
            // owner is public. Recording and disarming the rollback happen as
            // one allocation-free operation behind the exclusive borrow.
            let base_reservation = ledger_permit.commit_for_request(request_id);
            let (prompt_reservation, retained_reservation) =
                base_reservation.split_category(LedgerCategory::PromptStorage);
            (
                prompt,
                request_id,
                key,
                control,
                producer,
                receiver,
                prompt_reservation,
                retained_reservation,
            )
        };
        let (direct_receiver, result) = finish(request_id, &control, receiver);
        let record = RequestRecord {
            request_id,
            key,
            phase: RequestPhase::Queued,
            prompt: Some(prompt),
            max_new_tokens: request.max_new_tokens(),
            total_positions,
            sampling: request.sampling(),
            deadline_ns: request.deadline_ns(),
            admitted_ns: self.monotonic_ns,
            control,
            state_layout,
            state: None,
            active_plan: active_charge_plan,
            active_reservation: None,
            prompt_reservation: Some(prompt_reservation),
            retained_reservation: Some(retained_reservation),
            committed_positions: 0,
            emitted_tokens: 0,
            next_decode_token: None,
            rng_state: None,
            adapter_binding: None,
            endpoint,
            direct_receiver,
        };
        self.free_slots.swap_remove(free_position);
        self.slots[slot_index].key = Some(key);
        self.slots[slot_index].record = Some(record);
        self.queued.push_back(key);
        Ok(result)
    }

    fn try_allocate_request_payload(prompt: &[u32]) -> SchedulerResult<Vec<u32>> {
        let mut owned_prompt = Vec::new();
        try_reserve_vec(&mut owned_prompt, prompt.len(), "request prompt storage")?;
        owned_prompt.extend_from_slice(prompt);
        Ok(owned_prompt)
    }

    pub fn cancel(&self, id: crate::RequestId) -> SchedulerResult<CancelDisposition> {
        self.record_by_id(id)?.control.cancel()
    }

    /// Publishes cancellation and records its successful CAS boundary with
    /// the sealed M5 observer.
    #[cfg(any(test, feature = "m5-run-observer-instrumentation"))]
    #[doc(hidden)]
    pub fn cancel_observed(
        &mut self,
        id: crate::RequestId,
        observer: &mut RunObserver,
    ) -> SchedulerResult<CancelDisposition> {
        self.validate_run_observer_domain(observer)?;
        let clock = observer.clock();
        let disposition = self.cancel(id)?;
        let now_ns = clock.now_ns();
        observer.cancellation(id, disposition, now_ns);
        Ok(disposition)
    }

    #[cfg(test)]
    pub(crate) fn cancel_observed_with_clock_for_test<F>(
        &mut self,
        id: crate::RequestId,
        clock: &F,
        observer: &mut RunObserver,
    ) -> SchedulerResult<CancelDisposition>
    where
        F: Fn() -> u64,
    {
        self.validate_run_observer_domain(observer)?;
        let disposition = self.cancel(id)?;
        observer.cancellation(id, disposition, clock());
        Ok(disposition)
    }

    #[allow(dead_code, reason = "used by the staged Tokio actor integration")]
    pub(crate) fn control_binding(&self, id: crate::RequestId) -> SchedulerResult<ControlBinding> {
        Ok(self.record_by_id(id)?.control.clone())
    }

    pub fn advance_clock(&mut self, monotonic_ns: u64) -> SchedulerResult<()> {
        if monotonic_ns < self.monotonic_ns {
            return Err(SchedulerError::invalid_request(
                "monotonic_ns",
                "cannot move backwards",
            ));
        }
        self.monotonic_ns = monotonic_ns;
        Ok(())
    }

    fn observe_clock(&mut self, clock: &impl Fn() -> u64) -> u64 {
        observe_clock_value(&mut self.monotonic_ns, clock)
    }

    fn fire_checkpoint<D: CheckpointDriver, O: StepObserver>(
        &mut self,
        checkpoints: &mut D,
        key: SlotKey,
        point: CheckpointPoint,
        clock: &impl Fn() -> u64,
        observer: &mut O,
    ) -> SchedulerResult<()> {
        let prior_ns = self.monotonic_ns;
        let fire = {
            let record = self.record_for_key(key)?;
            checkpoints.fire(
                CheckpointContext {
                    point,
                    slot: key,
                    request_id: record.request_id,
                    position: record.committed_positions,
                    control: &record.control,
                    deadline_ns: record.deadline_ns,
                    monotonic_ns: prior_ns,
                },
                clock,
                observer,
            )?
        };
        apply_checkpoint_fire(&mut self.monotonic_ns, prior_ns, fire)
    }

    pub fn request_phase(&self, id: crate::RequestId) -> SchedulerResult<RequestPhase> {
        Ok(self.record_by_id(id)?.phase)
    }

    #[cfg(test)]
    pub(crate) fn rng_state_for_test(&self, id: crate::RequestId) -> SchedulerResult<Option<u64>> {
        Ok(self.record_by_id(id)?.rng_state)
    }

    #[cfg(test)]
    pub(crate) fn remove_preempted_membership_for_test(
        &mut self,
        id: crate::RequestId,
    ) -> SchedulerResult<()> {
        let record = self.record_by_id(id)?;
        if record.phase != RequestPhase::Preempted {
            return Err(SchedulerError::internal(
                "test corruption target is not preempted",
            ));
        }
        let key = record.key;
        self.ring
            .as_mut()
            .ok_or_else(|| SchedulerError::internal("scheduler ring is unavailable"))?
            .remove(key)
            .map_err(map_ring_error)?;
        Ok(())
    }

    /// Returns the exact engine admission boundary recorded for this request.
    pub fn admitted_ns(&self, id: crate::RequestId) -> SchedulerResult<u64> {
        Ok(self.record_by_id(id)?.admitted_ns)
    }

    /// Returns the resolved absolute deadline retained for this request.
    pub fn deadline_ns(&self, id: crate::RequestId) -> SchedulerResult<Option<u64>> {
        Ok(self.record_by_id(id)?.deadline_ns)
    }

    pub fn snapshot(&self) -> EngineSnapshot {
        let ledger = self.ledger.snapshot();
        let mut active = 0_usize;
        let mut preempted = 0_usize;
        let mut output_blocked = 0_usize;
        let terminals = self
            .endpoints
            .as_ref()
            .map_or(0, EndpointRegistry::pending_terminal_count);
        for slot in &self.slots {
            if let Some(record) = &slot.record {
                match record.phase {
                    RequestPhase::Ready
                    | RequestPhase::Preparing
                    | RequestPhase::ExpertOwned
                    | RequestPhase::ReadyToCommit => active += 1,
                    RequestPhase::Preempted => {
                        active += 1;
                        preempted += 1;
                    }
                    RequestPhase::OutputBlocked => {
                        active += 1;
                        output_blocked += 1;
                    }
                    RequestPhase::Terminal => {}
                    RequestPhase::Queued => {}
                }
            }
        }
        EngineSnapshot {
            monotonic_ns: self.monotonic_ns,
            closed: self.closed,
            queued_requests: self.queued.len(),
            active_requests: active,
            preempted_requests: preempted,
            output_blocked_requests: output_blocked,
            retained_terminal_results: terminals,
            ledger_used_bytes: usize::try_from(ledger.total_used()).unwrap_or(usize::MAX),
            ledger_peak_bytes: usize::try_from(ledger.total_peak()).unwrap_or(usize::MAX),
        }
    }

    /// Returns exact per-category logical ownership and historical peaks.
    #[must_use]
    pub fn ledger_snapshot(&self) -> crate::LedgerSnapshot {
        self.ledger.snapshot()
    }

    /// Borrows the append-only logical-ledger suffix beginning at `cursor`.
    ///
    /// The immutable sequence-zero snapshot captures static shared ownership
    /// after construction. Later complete sequences contain only durable
    /// request-owned acquisitions and releases. Reads allocate nothing and do
    /// not affect accounting. Read from origin for a self-contained replay, or
    /// retain each returned cursor and applied state with this same engine for
    /// incremental replay. A whole mutation that cannot fit makes ledger
    /// overflow sticky without affecting the independently bounded service
    /// trace or scheduler behavior. Capture the final request-zero prefix
    /// before successful shutdown destroys both trace allocations.
    pub fn ledger_trace_since(
        &self,
        cursor: LedgerTraceCursor,
    ) -> SchedulerResult<LedgerTraceRead<'_>> {
        if let Some(read) = self.ledger.ledger_trace_since(cursor) {
            return Ok(read);
        }
        if self.ledger.has_ledger_trace() {
            return Err(SchedulerError::invalid_request(
                "ledger_trace_cursor",
                "does not identify a complete retained mutation boundary",
            ));
        }
        Err(SchedulerError::scheduler_closed())
    }

    /// Borrows the append-only service-trace suffix beginning at `cursor`.
    ///
    /// The trace records successful model-position commits in their exact
    /// commit order. Reading never allocates, drains, resets, or changes the
    /// fixed [`crate::LedgerCategory::Trace`] charge. A full trace remains
    /// healthy until a later successful commit cannot be retained; overflow
    /// is sticky and invalidates completeness without affecting scheduling.
    /// Read the final prefix before successful shutdown, which deliberately
    /// drops the allocation and releases its shared ledger ownership.
    pub fn service_trace_since(
        &self,
        cursor: ServiceTraceCursor,
    ) -> SchedulerResult<ServiceTraceRead<'_>> {
        let trace = self
            .service_trace
            .as_ref()
            .ok_or_else(SchedulerError::scheduler_closed)?;
        let event_limit = self.config.trace_capacity();
        if trace.len() > event_limit {
            return Err(SchedulerError::internal(
                "service trace length exceeds its configured limit",
            ));
        }
        if trace.capacity() < event_limit {
            return Err(SchedulerError::internal(
                "service trace allocation is smaller than its configured limit",
            ));
        }
        if self.trace_overflowed && trace.len() != event_limit {
            return Err(SchedulerError::internal(
                "overflowed service trace does not retain its complete capacity prefix",
            ));
        }
        let start = cursor.event_index();
        if start > trace.len() {
            return Err(SchedulerError::invalid_request(
                "service_trace_cursor",
                "is beyond the retained trace prefix",
            ));
        }
        let next_cursor = ServiceTraceCursor::from_retained_len(trace.len());
        let status = ServiceTraceStatus::new(trace.len(), event_limit, self.trace_overflowed);
        Ok(ServiceTraceRead::new(
            &trace[start..],
            cursor,
            next_cursor,
            status,
        ))
    }

    /// Prevalidates and binds a bounded deterministic checkpoint plan.
    ///
    /// The plan owns its sole allocation, retains no request control table,
    /// and can only be executed by this engine generation domain.
    #[cfg(any(test, feature = "deterministic-checkpoint-instrumentation"))]
    #[doc(hidden)]
    pub fn prepare_checkpoint_plan(
        &self,
        directives: &[CheckpointDirective],
    ) -> SchedulerResult<CheckpointPlan> {
        self.ensure_open()?;
        let controls = self
            .controls
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("request controls are unavailable"))?;
        CheckpointPlan::try_bind(
            controls.checkpoint_domain(),
            directives.iter().copied().map(|directive| {
                let record = self.record_by_id(directive.request_id())?;
                if record.phase == RequestPhase::Terminal {
                    return Err(SchedulerError::invalid_request(
                        "checkpoint directives",
                        "target request is already terminal",
                    ));
                }
                if directive.position() < record.committed_positions
                    || directive.position() >= record.total_positions
                {
                    return Err(SchedulerError::invalid_request(
                        "checkpoint directives",
                        "target position is not pending for this request",
                    ));
                }
                if directive.action().expires_deadline() && record.deadline_ns.is_none() {
                    return Err(SchedulerError::invalid_request(
                        "checkpoint directives",
                        "expiry action requires a request deadline",
                    ));
                }
                Ok((directive, record.key, record.deadline_ns))
            }),
        )
    }

    /// Executes one deterministic step with a prevalidated concrete plan.
    #[cfg(any(test, feature = "deterministic-checkpoint-instrumentation"))]
    #[doc(hidden)]
    pub fn step_with_checkpoint_plan(
        &mut self,
        plan: &mut CheckpointPlan,
    ) -> SchedulerResult<StepReport> {
        self.ensure_open()?;
        let controls = self
            .controls
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("request controls are unavailable"))?;
        if !plan.belongs_to_registry(controls) {
            return Err(SchedulerError::invalid_request(
                "checkpoint plan",
                "belongs to another scheduler engine",
            ));
        }
        let fixed_ns = self.monotonic_ns;
        let mut observer = NullObserver;
        self.step_with_clock_and_checkpoints(&|| fixed_ns, plan, &mut observer)
    }

    /// Executes one observed step with prevalidated deterministic barriers.
    ///
    /// This combined surface exists only for the frozen cancellation evidence
    /// cell and internal lifecycle tests. Both instruments remain concrete,
    /// engine-bound, and allocation-stable.
    #[cfg(any(
        test,
        all(
            feature = "deterministic-checkpoint-instrumentation",
            feature = "m5-run-observer-instrumentation"
        )
    ))]
    #[doc(hidden)]
    pub fn step_with_checkpoint_plan_observed(
        &mut self,
        plan: &mut CheckpointPlan,
        observer: &mut RunObserver,
    ) -> SchedulerResult<StepReport> {
        self.ensure_open()?;
        let controls = self
            .controls
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("request controls are unavailable"))?;
        if !plan.belongs_to_registry(controls) {
            return Err(SchedulerError::invalid_request(
                "checkpoint plan",
                "belongs to another scheduler engine",
            ));
        }
        self.validate_run_observer_domain(observer)?;
        let clock = observer.clock();
        self.step_with_clock_and_checkpoints(&|| clock.now_ns(), plan, observer)
    }

    /// Advances bounded admission and executes up to the configured number of
    /// deterministic token waves.
    pub fn step(&mut self) -> SchedulerResult<StepReport> {
        let fixed_ns = self.monotonic_ns;
        self.step_with_clock(&|| fixed_ns)
    }

    /// Runs one step while resampling an origin-relative monotonic clock at
    /// every control boundary. The actor supplies a live source; the public
    /// synchronous path above intentionally observes its manually advanced
    /// value for deterministic tests and embedding.
    pub(crate) fn step_with_clock<F>(&mut self, clock: &F) -> SchedulerResult<StepReport>
    where
        F: Fn() -> u64,
    {
        let mut checkpoints = NoCheckpoints;
        let mut observer = NullObserver;
        self.step_with_clock_and_checkpoints(clock, &mut checkpoints, &mut observer)
    }

    /// Executes one timed direct-engine step with the sealed M5 observer.
    #[cfg(any(test, feature = "m5-run-observer-instrumentation"))]
    #[doc(hidden)]
    pub fn step_observed(&mut self, observer: &mut RunObserver) -> SchedulerResult<StepReport> {
        self.validate_run_observer_domain(observer)?;
        let clock = observer.clock();
        let mut checkpoints = NoCheckpoints;
        self.step_with_clock_and_checkpoints(&|| clock.now_ns(), &mut checkpoints, observer)
    }

    #[cfg(test)]
    pub(crate) fn step_observed_with_clock_for_test<F>(
        &mut self,
        clock: &F,
        observer: &mut RunObserver,
    ) -> SchedulerResult<StepReport>
    where
        F: Fn() -> u64,
    {
        self.validate_run_observer_domain(observer)?;
        let mut checkpoints = NoCheckpoints;
        self.step_with_clock_and_checkpoints(clock, &mut checkpoints, observer)
    }

    fn step_with_clock_and_checkpoints<F, D, O>(
        &mut self,
        clock: &F,
        checkpoints: &mut D,
        observer: &mut O,
    ) -> SchedulerResult<StepReport>
    where
        F: Fn() -> u64,
        D: CheckpointDriver,
        O: StepObserver,
    {
        self.ensure_open()?;
        if self.adapter.is_none()
            || self.ring.is_none()
            || self.workspace.is_none()
            || self.sampling.is_none()
            || self.wave.is_none()
            || self.service_trace.is_none()
        {
            return Err(SchedulerError::internal(
                "scheduler step resources are unavailable",
            ));
        }
        let adapter = self
            .adapter
            .take()
            .ok_or_else(SchedulerError::scheduler_closed)?;
        let mut ring = self
            .ring
            .take()
            .ok_or_else(|| SchedulerError::internal("scheduler ring is unavailable"))?;
        let mut workspace = self
            .workspace
            .take()
            .ok_or_else(|| SchedulerError::internal("scheduler workspace is unavailable"))?;
        let mut sampling = self
            .sampling
            .take()
            .ok_or_else(|| SchedulerError::internal("sampling workspace is unavailable"))?;
        let mut wave = self
            .wave
            .take()
            .ok_or_else(|| SchedulerError::internal("wave scratch is unavailable"))?;
        let mut trace = self
            .service_trace
            .take()
            .ok_or_else(|| SchedulerError::internal("service trace is unavailable"))?;

        let mut result = self.step_with_resources(
            &adapter,
            &mut ring,
            &mut workspace,
            &mut sampling,
            &mut wave,
            &mut trace,
            clock,
            checkpoints,
            observer,
        );
        let mut recovered_after_error = false;
        if result.is_err() {
            match self.recover_wave(&mut ring, &mut wave) {
                Ok(()) => recovered_after_error = true,
                Err(cleanup) => result = Err(cleanup),
            }
        }

        self.adapter = Some(adapter);
        self.ring = Some(ring);
        self.workspace = Some(workspace);
        self.sampling = Some(sampling);
        self.wave = Some(wave);
        self.service_trace = Some(trace);
        if recovered_after_error && observer.has_pending_worker_quiescence() {
            // Error recovery cleared the reusable wave before this sample.
            let quiescent_ns = clock();
            observer.flush_worker_quiescence(quiescent_ns);
        }
        if let Ok(report) = &result {
            observer.step(*report);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn step_with_resources<D: CheckpointDriver, O: StepObserver>(
        &mut self,
        adapter: &A,
        ring: &mut DrrRing,
        workspace: &mut A::Workspace,
        sampling: &mut SamplingWorkspace,
        wave: &mut WaveScratch<A::PreparedToken, A::ExpertTask, A::ExpertContribution>,
        trace: &mut Vec<ServiceTraceEvent>,
        clock: &impl Fn() -> u64,
        checkpoints: &mut D,
        observer: &mut O,
    ) -> SchedulerResult<StepReport> {
        let mut report = StepReport::default();
        self.resolve_visible_controls(ring, &mut report, clock, observer)?;
        report.resumed_requests = self.resume_preempted(ring, observer)?;

        for wave_index in 0..self.config.waves_per_step() {
            self.promote_fifo(adapter, ring, &mut report, clock, observer)?;
            if ring.is_empty() {
                break;
            }
            let final_wave = wave_index + 1 == self.config.waves_per_step();
            let selected = self.execute_wave(
                adapter,
                ring,
                workspace,
                sampling,
                wave,
                trace,
                &mut report,
                clock,
                checkpoints,
                final_wave,
                observer,
            )?;
            if selected == 0 {
                break;
            }
            report.waves += 1;
            self.resolve_visible_controls(ring, &mut report, clock, observer)?;
            if final_wave {
                let mut survivors = 0_usize;
                for record in self.slots.iter().filter_map(|slot| slot.record.as_ref()) {
                    if record.phase == RequestPhase::Preempted {
                        survivors += 1;
                        observer.preempted(record.request_id);
                    }
                }
                report.preempted_requests = survivors;
            }
        }
        Ok(report)
    }

    /// Resumes resident state only after the step's opening control scan.
    /// Ring membership, credit, adapter state, RNG, and ledger ownership never
    /// move across this allocation-free phase transition.
    fn resume_preempted<O: StepObserver>(
        &mut self,
        ring: &DrrRing,
        observer: &mut O,
    ) -> SchedulerResult<usize> {
        let mut preempted_slots = 0_usize;
        for slot in &self.slots {
            let Some(record) = slot.record.as_ref() else {
                continue;
            };
            if record.phase != RequestPhase::Preempted {
                continue;
            }
            if record.state.is_none()
                || record.active_reservation.is_none()
                || record.adapter_binding.is_none()
                || record.prompt.is_none()
                || record.prompt_reservation.is_none()
                || record.retained_reservation.is_none()
            {
                return Err(SchedulerError::internal(
                    "preempted request lost active ownership",
                ));
            }
            preempted_slots += 1;
        }
        for key in &self.queued {
            if self.record_for_key(*key)?.phase == RequestPhase::Preempted {
                return Err(SchedulerError::internal(
                    "preempted request re-entered the admission queue",
                ));
            }
        }
        let mut preempted_members = 0_usize;
        for (key, request_id, has_reservation, deficit) in ring.member_states() {
            let record = self.record_for_key(key)?;
            if record.request_id != request_id {
                return Err(SchedulerError::internal(
                    "resident ring identity does not match request slot",
                ));
            }
            if record.phase == RequestPhase::Preempted {
                if has_reservation || deficit != 0 {
                    return Err(SchedulerError::internal(
                        "preempted request retains unapplied service credit",
                    ));
                }
                preempted_members += 1;
            }
        }
        if preempted_members != preempted_slots {
            return Err(SchedulerError::internal(
                "preempted request lost resident ring membership",
            ));
        }
        for slot in &mut self.slots {
            if let Some(record) = slot.record.as_mut()
                && record.phase == RequestPhase::Preempted
            {
                record.phase = RequestPhase::Ready;
                observer.resumed(record.request_id);
            }
        }
        Ok(preempted_slots)
    }

    fn resolve_visible_controls<O: StepObserver>(
        &mut self,
        ring: &mut DrrRing,
        report: &mut StepReport,
        clock: &impl Fn() -> u64,
        observer: &mut O,
    ) -> SchedulerResult<()> {
        for index in 0..self.slots.len() {
            let now = self.observe_clock(clock);
            let action = if let Some(record) = self.slots[index].record.as_ref() {
                let snapshot = record.control.fresh_snapshot()?;
                if record.phase == RequestPhase::Terminal {
                    Some((record.key, None, snapshot.disconnected()))
                } else {
                    visible_snapshot_outcome(snapshot, record.deadline_ns, now)
                        .map(|outcome| (record.key, Some(outcome), snapshot.disconnected()))
                }
            } else {
                None
            };
            if let Some((key, outcome, disconnected)) = action {
                if let Some(outcome) = outcome {
                    self.terminalize_key_observed(ring, key, outcome, clock, observer)?;
                    report.terminal_decisions += 1;
                } else {
                    let endpoint = self.record_for_key(key)?.endpoint.clone();
                    if disconnected && endpoint.snapshot()?.receiver_connected {
                        let report = endpoint.settle_disconnected()?;
                        let record = self.record_for_key(key)?;
                        validate_endpoint_discard(
                            record.request_id,
                            record.emitted_tokens,
                            endpoint.snapshot()?,
                            report,
                        )?;
                    }
                    if endpoint.snapshot()?.reap_ready {
                        let request_id = self.reap_key(key)?;
                        if observer.enabled() {
                            let zero_ns = clock();
                            observer.request_zero(request_id, zero_ns);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn promote_fifo<O: StepObserver>(
        &mut self,
        adapter: &A,
        ring: &mut DrrRing,
        report: &mut StepReport,
        clock: &impl Fn() -> u64,
        observer: &mut O,
    ) -> SchedulerResult<()> {
        while ring.len() < self.config.max_active_requests() {
            let Some(key) = self.queued.front().copied() else {
                break;
            };
            let now = self.observe_clock(clock);
            let outcome = visible_control_outcome(self.record_for_key(key)?, now)?;
            if let Some(outcome) = outcome {
                self.terminalize_key_observed(ring, key, outcome, clock, observer)?;
                report.terminal_decisions += 1;
                continue;
            }

            let plan = self.record_for_key(key)?.active_plan;
            match self.ledger.can_acquire(plan.as_slice()) {
                Ok(()) => {}
                Err(error) if error.is_resource_exhausted() => break,
                Err(error) => return Err(map_ledger_error(error)),
            }
            let layout = self.record_for_key(key)?.state_layout;
            let request_id = self.record_for_key(key)?.request_id;
            let ledger_permit = self
                .ledger
                .acquire_provisional_permit(plan.as_slice())
                .map_err(map_ledger_error)?;
            let state = match adapter.new_state(layout) {
                Ok(state) => state,
                Err(source) => {
                    drop(ledger_permit);
                    let category =
                        SchedulerError::adapter("allocating request state", source).category();
                    self.terminalize_key_observed(
                        ring,
                        key,
                        TerminalOutcome::Failed { category },
                        clock,
                        observer,
                    )?;
                    report.terminal_decisions += 1;
                    continue;
                }
            };
            let slot = self
                .slots
                .get_mut(key.index())
                .ok_or_else(|| SchedulerError::internal("promotion slot index is out of range"))?;
            if slot.key != Some(key) {
                return Err(SchedulerError::internal(
                    "promotion slot generation is stale",
                ));
            }
            let record = slot
                .record
                .as_mut()
                .ok_or_else(|| SchedulerError::internal("promotion request record is missing"))?;
            if record.request_id != request_id || record.phase != RequestPhase::Queued {
                return Err(SchedulerError::internal(
                    "promotion request identity or phase changed",
                ));
            }
            if let Err(error) = ring.insert(key, request_id) {
                drop(state);
                drop(ledger_permit);
                return Err(map_ring_error(error));
            }
            let reservation = ledger_permit.commit_for_request(request_id);
            record.state = Some(state);
            record.active_reservation = Some(reservation);
            record.phase = RequestPhase::Ready;
            if self.queued.pop_front() != Some(key) {
                return Err(SchedulerError::internal(
                    "FIFO admission head changed during promotion",
                ));
            }
            report.promoted_requests += 1;
            observer.promotion(request_id);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_wave<D: CheckpointDriver, O: StepObserver>(
        &mut self,
        adapter: &A,
        ring: &mut DrrRing,
        workspace: &mut A::Workspace,
        sampling: &mut SamplingWorkspace,
        wave: &mut WaveScratch<A::PreparedToken, A::ExpertTask, A::ExpertContribution>,
        trace: &mut Vec<ServiceTraceEvent>,
        report: &mut StepReport,
        clock: &impl Fn() -> u64,
        checkpoints: &mut D,
        preempt_after_commit: bool,
        observer: &mut O,
    ) -> SchedulerResult<usize> {
        wave.reset().map_err(map_wave_error)?;
        if ring.current_epoch().is_none() && ring.open_round().map_err(map_ring_error)?.is_none() {
            return Ok(0);
        }

        while wave.selections().len() < self.config.batch_width() {
            let visit = ring.next_visit().map_err(map_ring_error)?;
            let key = visit.slot_key();
            if self.record_for_key(key)?.request_id != visit.request_id() {
                let _ = ring.mark_blocked(visit).map_err(map_ring_error)?;
                return Err(SchedulerError::internal(
                    "policy visit identity does not match request slot",
                ));
            }

            let now = self.observe_clock(clock);
            let control = visible_control_outcome(self.record_for_key(key)?, now)?;
            let endpoint = self.record_for_key(key)?.endpoint.snapshot()?;
            if endpoint.output_capacity != self.config.output_capacity_per_request() {
                return Err(SchedulerError::internal(
                    "request endpoint capacity changed while active",
                ));
            }
            let output_available = endpoint.producer_open
                && endpoint.receiver_connected
                && endpoint.buffered_output_events < endpoint.output_capacity;
            if control.is_none() {
                match (self.record_for_key(key)?.phase, output_available) {
                    (RequestPhase::Ready, false) => {
                        self.record_mut_for_key(key)?.phase = RequestPhase::OutputBlocked;
                    }
                    (RequestPhase::OutputBlocked, true) => {
                        self.record_mut_for_key(key)?.phase = RequestPhase::Ready;
                    }
                    _ => {}
                }
            }
            let runnable = control.is_none()
                && self.record_for_key(key)?.phase == RequestPhase::Ready
                && output_available;
            if !runnable {
                let progress = ring.mark_blocked(visit).map_err(map_ring_error)?;
                if let Some(outcome) = control {
                    self.terminalize_key_observed(ring, key, outcome, clock, observer)?;
                    report.terminal_decisions += 1;
                }
                if progress.is_closed() {
                    break;
                }
                continue;
            }

            let engine_transaction = match self.transaction_ids.issue() {
                Ok(identity) => identity,
                Err(error) => {
                    let _ = ring.mark_blocked(visit).map_err(map_ring_error)?;
                    return Err(map_identity_error(error));
                }
            };
            let (reservation, progress) = ring.mark_selected(visit).map_err(map_ring_error)?;
            let input_token = input_token(self.record_for_key(key)?)?;
            let request_id = self.record_for_key(key)?.request_id;
            let position = self.record_for_key(key)?.committed_positions;
            if self.record_for_key(key)?.state.is_none() {
                ring.recover_credit(reservation).map_err(map_ring_error)?;
                return Err(SchedulerError::internal(
                    "ready request has no adapter state",
                ));
            }
            self.record_mut_for_key(key)?.phase = RequestPhase::Preparing;
            let prepared = {
                let record = self.record_for_key(key)?;
                let state = record.state.as_ref().ok_or_else(|| {
                    SchedulerError::internal("ready request has no adapter state")
                })?;
                if observer.enabled() {
                    let work_started_ns = clock();
                    observer.work_start(request_id, position, work_started_ns);
                }
                adapter.prepare_token(state, input_token, workspace)
            };
            let prepared = match prepared {
                Ok(prepared) => prepared,
                Err(source) => {
                    ring.recover_credit(reservation).map_err(map_ring_error)?;
                    let category = SchedulerError::adapter("preparing token", source).category();
                    self.terminalize_key_observed(
                        ring,
                        key,
                        TerminalOutcome::Failed { category },
                        clock,
                        observer,
                    )?;
                    report.terminal_decisions += 1;
                    if progress.is_closed() {
                        break;
                    }
                    continue;
                }
            };
            let adapter_identity = adapter.prepared_identity(&prepared);
            let adapter_transaction_id = adapter_identity.transaction_id().get();
            let fresh_transaction = self
                .last_adapter_transaction_id
                .is_none_or(|previous| adapter_transaction_id > previous);
            if fresh_transaction {
                self.last_adapter_transaction_id = Some(adapter_transaction_id);
            }
            if !fresh_transaction
                || !bind_prepared_identity(self.record_mut_for_key(key)?, adapter_identity)
            {
                ring.recover_credit(reservation).map_err(map_ring_error)?;
                self.terminalize_key_observed(
                    ring,
                    key,
                    TerminalOutcome::Failed {
                        category: ErrorCategory::Internal,
                    },
                    clock,
                    observer,
                )?;
                report.terminal_decisions += 1;
                if progress.is_closed() {
                    break;
                }
                continue;
            }

            let tasks = adapter.expert_tasks(&prepared);
            let task_count = tasks.len();
            if task_count == 0 || task_count > self.config.max_tasks_per_token() {
                ring.recover_credit(reservation).map_err(map_ring_error)?;
                self.terminalize_key_observed(
                    ring,
                    key,
                    TerminalOutcome::Failed {
                        category: ErrorCategory::Internal,
                    },
                    clock,
                    observer,
                )?;
                report.terminal_decisions += 1;
                if progress.is_closed() {
                    break;
                }
                continue;
            }
            let request_id = self.record_for_key(key)?.request_id;
            let mut pending_selection = Some(WaveSelection {
                slot: key,
                request_id,
                transaction_id: engine_transaction,
                adapter_identity,
                reservation: Some(reservation),
                prepared: Some(prepared),
                task_count,
            });
            let selection_index = match wave.push_selection(&mut pending_selection) {
                Ok(index) => index,
                Err(_) => {
                    if let Some(mut selection) = pending_selection
                        && let Some(reservation) = selection.reservation.take()
                    {
                        ring.recover_credit(reservation).map_err(map_ring_error)?;
                    }
                    self.terminalize_key_observed(
                        ring,
                        key,
                        TerminalOutcome::Failed {
                            category: ErrorCategory::Internal,
                        },
                        clock,
                        observer,
                    )?;
                    report.terminal_decisions += 1;
                    if progress.is_closed() {
                        break;
                    }
                    continue;
                }
            };

            let mut malformed_task = false;
            let mut observed_tasks = 0_usize;
            for task in tasks {
                observed_tasks = observed_tasks.saturating_add(1);
                let identity = adapter.task_identity(&task);
                let rank = adapter.task_router_rank(&task);
                let expert_id = adapter.task_expert_id(&task);
                if identity != adapter_identity
                    || usize::from(rank) >= task_count
                    || wave
                        .push_task(TaskEnvelope {
                            selection_index,
                            slot: key,
                            request_id,
                            transaction_id: engine_transaction,
                            adapter_identity: identity,
                            router_rank: rank,
                            expert_id,
                            scatter_index: 0,
                            task,
                        })
                        .is_err()
                {
                    malformed_task = true;
                    break;
                }
            }
            malformed_task |= observed_tasks != task_count;
            if malformed_task {
                let mut selection = wave
                    .rollback_last_selection(selection_index)
                    .map_err(map_wave_error)?;
                if let Some(reservation) = selection.reservation.take() {
                    ring.recover_credit(reservation).map_err(map_ring_error)?;
                }
                self.terminalize_key_observed(
                    ring,
                    key,
                    TerminalOutcome::Failed {
                        category: ErrorCategory::Internal,
                    },
                    clock,
                    observer,
                )?;
                report.terminal_decisions += 1;
            } else {
                self.record_mut_for_key(key)?.phase = RequestPhase::ExpertOwned;
            }
            if progress.is_closed() {
                break;
            }
        }

        let selected_count = wave.selections().len();
        if selected_count == 0 {
            wave.reset().map_err(map_wave_error)?;
            return Ok(0);
        }
        report.selected_positions += selected_count;
        observer.wave(selected_count);
        wave.sort_tasks().map_err(map_wave_error)?;
        #[cfg(test)]
        if take_wave_task_transaction_corruption_for_test() {
            let foreign = self.transaction_ids.issue().map_err(map_identity_error)?;
            wave.corrupt_first_task_transaction_for_test(foreign)
                .map_err(map_wave_error)?;
        }

        let mut failures = [None; MAX_BATCH_WIDTH as usize];
        let mut suppressed_before_expert = [false; MAX_BATCH_WIDTH as usize];
        for (selection_index, selection) in wave.selections().iter().enumerate() {
            let key = selection.slot;
            self.fire_checkpoint(
                checkpoints,
                key,
                CheckpointPoint::PostRouterPreExpert,
                clock,
                observer,
            )?;
            let now = self.observe_clock(clock);
            suppressed_before_expert[selection_index] =
                visible_control_outcome(self.record_for_key(key)?, now)?.is_some();
        }
        let mut previous_expert = None;
        {
            let (tasks, selections, scatter) =
                wave.drain_tasks_and_scatter().map_err(map_wave_error)?;
            for envelope in tasks {
                let (completion, task) = envelope.into_parts();
                let selection_index = completion.selection_index();
                if failures.get(selection_index).copied().flatten().is_some()
                    || suppressed_before_expert
                        .get(selection_index)
                        .copied()
                        .unwrap_or(true)
                {
                    continue;
                }
                let authorized = WaveScratch::<
                    A::PreparedToken,
                    A::ExpertTask,
                    A::ExpertContribution,
                >::completion_is_authorized(
                    selections, scatter, &completion
                );
                let current = authorized
                    && self.record_for_key(completion.slot()).is_ok_and(|record| {
                        record.request_id == completion.request_id()
                            && record.phase == RequestPhase::ExpertOwned
                            && record.committed_positions
                                == completion.adapter_identity().position()
                            && record.adapter_binding.is_some_and(|binding| {
                                binding.matches(completion.adapter_identity())
                            })
                    });
                if !current {
                    if let Some(failure) = failures.get_mut(selection_index) {
                        *failure = Some(ErrorCategory::Internal);
                    }
                    continue;
                }
                if previous_expert != Some(completion.expert_id()) {
                    report.expert_groups += 1;
                    previous_expert = Some(completion.expert_id());
                }
                let contribution = match adapter.execute_expert(task, workspace) {
                    Ok(contribution) => contribution,
                    Err(source) => {
                        if let Some(failure) = failures.get_mut(selection_index) {
                            *failure = Some(
                                SchedulerError::adapter("executing expert", source).category(),
                            );
                        }
                        continue;
                    }
                };
                report.expert_tasks += 1;
                let invalid_contribution = adapter.contribution_identity(&contribution)
                    != completion.adapter_identity()
                    || adapter.contribution_router_rank(&contribution)
                        != completion.router_rank()
                    || adapter.contribution_expert_id(&contribution)
                        != completion.expert_id()
                    || WaveScratch::<
                        A::PreparedToken,
                        A::ExpertTask,
                        A::ExpertContribution,
                    >::put_contribution(scatter, completion, contribution)
                    .is_err();
                if invalid_contribution && let Some(failure) = failures.get_mut(selection_index) {
                    *failure = Some(ErrorCategory::Internal);
                }
            }
        }

        for (selection_index, failure) in failures.iter().copied().enumerate().take(selected_count)
        {
            let (key, expected_identity) = {
                let selection = &wave.selections()[selection_index];
                (selection.slot, selection.adapter_identity)
            };
            let now = self.observe_clock(clock);
            let control = visible_control_outcome(self.record_for_key(key)?, now)?;
            if let Some(category) = failure {
                self.release_wave_selection(ring, wave, selection_index)?;
                self.terminalize_key_observed(
                    ring,
                    key,
                    TerminalOutcome::Failed { category },
                    clock,
                    observer,
                )?;
                report.terminal_decisions += 1;
                continue;
            }
            if let Some(outcome) = control {
                self.release_wave_selection(ring, wave, selection_index)?;
                self.terminalize_wave_key_observed(ring, key, outcome, clock, observer)?;
                report.terminal_decisions += 1;
                continue;
            }

            let prepared = wave.selections_mut()[selection_index]
                .prepared
                .take()
                .ok_or_else(|| SchedulerError::internal("wave prepared token is missing"))?;
            let contributions = wave
                .gather_contributions(selection_index)
                .map_err(map_wave_error)?;
            let pending = match adapter.finish_token(prepared, contributions, workspace) {
                Ok(pending) => pending,
                Err(source) => {
                    self.release_wave_selection(ring, wave, selection_index)?;
                    let category = SchedulerError::adapter("finishing token", source).category();
                    self.terminalize_key_observed(
                        ring,
                        key,
                        TerminalOutcome::Failed { category },
                        clock,
                        observer,
                    )?;
                    report.terminal_decisions += 1;
                    continue;
                }
            };
            let pending_identity = adapter.pending_identity(&pending);
            let logits = adapter.pending_logits(&pending);
            if pending_identity != expected_identity
                || logits.len() != self.config.vocabulary_size()
                || logits.iter().any(|value| !value.is_finite())
            {
                self.release_wave_selection(ring, wave, selection_index)?;
                self.terminalize_key_observed(
                    ring,
                    key,
                    TerminalOutcome::Failed {
                        category: ErrorCategory::Internal,
                    },
                    clock,
                    observer,
                )?;
                report.terminal_decisions += 1;
                continue;
            }
            self.record_mut_for_key(key)?.phase = RequestPhase::ReadyToCommit;
            self.fire_checkpoint(
                checkpoints,
                key,
                CheckpointPoint::ReadyToCommitPrePlan,
                clock,
                observer,
            )?;
            let now = self.observe_clock(clock);
            if let Some(outcome) = visible_control_outcome(self.record_for_key(key)?, now)? {
                self.release_wave_selection(ring, wave, selection_index)?;
                self.terminalize_wave_key_observed(ring, key, outcome, clock, observer)?;
                report.terminal_decisions += 1;
                continue;
            }
            let endpoint = self.record_for_key(key)?.endpoint.clone();
            let publication = match self.plan_publication(
                adapter,
                PublicationContext {
                    key,
                    identity: pending_identity,
                    preempt_after_commit,
                },
                logits,
                sampling,
                &endpoint,
            ) {
                Ok(PublicationPlan::Ready(publication)) => publication,
                Ok(PublicationPlan::OutputBlocked) => {
                    self.release_wave_selection(ring, wave, selection_index)?;
                    self.record_mut_for_key(key)?.phase = RequestPhase::OutputBlocked;
                    if observer.enabled() {
                        let record = self.record_for_key(key)?;
                        observer.work_abandoned(record.request_id, record.committed_positions);
                    }
                    continue;
                }
                Ok(PublicationPlan::Cancelled) => {
                    self.release_wave_selection(ring, wave, selection_index)?;
                    self.terminalize_wave_key_observed(
                        ring,
                        key,
                        TerminalOutcome::Cancelled,
                        clock,
                        observer,
                    )?;
                    report.terminal_decisions += 1;
                    continue;
                }
                Err(error) => {
                    self.release_wave_selection(ring, wave, selection_index)?;
                    let category = error.category();
                    self.terminalize_key_observed(
                        ring,
                        key,
                        TerminalOutcome::Failed { category },
                        clock,
                        observer,
                    )?;
                    report.terminal_decisions += 1;
                    continue;
                }
            };
            let now = self.observe_clock(clock);
            if let Some(outcome) = visible_control_outcome(self.record_for_key(key)?, now)? {
                drop(publication);
                self.release_wave_selection(ring, wave, selection_index)?;
                self.terminalize_wave_key_observed(ring, key, outcome, clock, observer)?;
                report.terminal_decisions += 1;
                continue;
            }
            let reservation = wave.selections_mut()[selection_index]
                .reservation
                .take()
                .ok_or_else(|| SchedulerError::internal("wave service credit is missing"))?;
            let committed = self.commit_publication(
                adapter,
                ring,
                trace,
                key,
                reservation,
                &pending,
                publication,
                clock,
                checkpoints,
                observer,
            );
            match committed {
                Ok(CommitDisposition::Applied {
                    post_commit_failure: _,
                    terminalized,
                }) => {
                    report.committed_positions += 1;
                    if terminalized {
                        report.terminal_decisions += 1;
                    }
                }
                Ok(CommitDisposition::Suppressed(outcome)) => {
                    self.terminalize_wave_key_observed(ring, key, outcome, clock, observer)?;
                    report.terminal_decisions += 1;
                }
                Err(error) => {
                    let category = error.category();
                    self.terminalize_key_observed(
                        ring,
                        key,
                        TerminalOutcome::Failed { category },
                        clock,
                        observer,
                    )?;
                    report.terminal_decisions += 1;
                }
            }
        }
        wave.reset().map_err(map_wave_error)?;
        if observer.has_pending_worker_quiescence() {
            observer.flush_worker_quiescence(clock());
        }
        Ok(selected_count)
    }

    fn plan_publication<'endpoint>(
        &self,
        adapter: &A,
        context: PublicationContext,
        logits: &[f32],
        sampling: &mut SamplingWorkspace,
        endpoint: &'endpoint EndpointProducer,
    ) -> SchedulerResult<PublicationPlan<'endpoint>> {
        let PublicationContext {
            key,
            identity,
            preempt_after_commit,
        } = context;
        let record = self.record_for_key(key)?;
        let next_position = record
            .committed_positions
            .checked_add(1)
            .ok_or_else(|| SchedulerError::internal("committed position count overflows"))?;
        if next_position > record.total_positions {
            return Err(SchedulerError::internal(
                "token commit exceeds reserved context",
            ));
        }
        let prompt_len = record
            .prompt
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("active request prompt is missing"))?
            .len();
        let service_phase = if record.committed_positions < prompt_len {
            ServicePhase::Prefill
        } else {
            ServicePhase::Decode
        };
        let emits = record.max_new_tokens != 0 && next_position >= prompt_len;
        let (output_guard, terminal_guard, output_full_after_publish) = if emits {
            let guard = match endpoint.begin_output_commit() {
                Ok(guard) => guard,
                Err(error) if error.category() == ErrorCategory::ResourceExhausted => {
                    return Ok(PublicationPlan::OutputBlocked);
                }
                Err(error) if error.category() == ErrorCategory::Cancelled => {
                    return Ok(PublicationPlan::Cancelled);
                }
                Err(error) => return Err(error),
            };
            let output_full_after_publish = guard.will_be_full_after_publish();
            (Some(guard), None, output_full_after_publish)
        } else {
            (None, Some(endpoint.begin_terminal_commit()?), false)
        };
        let (event, next_rng_state, next_decode_token, stop) = if emits {
            let preview = sampling
                .preview(logits, record.sampling, record.rng_state)
                .map_err(|source| SchedulerError::sampling("previewing token", source))?;
            if preview.observed_rng_state() != record.rng_state {
                return Err(SchedulerError::internal(
                    "sampling preview observed a stale RNG state",
                ));
            }
            let token = preview.token();
            let event = OutputEvent::new(record.request_id, record.emitted_tokens, token);
            (
                Some(event),
                preview.next_rng_state(),
                Some(token),
                adapter.is_stop_token(token),
            )
        } else {
            (None, record.rng_state, record.next_decode_token, false)
        };
        let emitted = event.is_some();
        let next_emitted_tokens = record
            .emitted_tokens
            .checked_add(usize::from(emitted))
            .ok_or_else(|| SchedulerError::internal("emitted token count overflows"))?;
        let endpoint_guard = if let Some(guard) = output_guard {
            let event = event.ok_or_else(|| {
                SchedulerError::internal("output endpoint guard has no planned event")
            })?;
            PublicationGuard::Output(guard.validate_planned_event(event)?)
        } else {
            let guard = terminal_guard.ok_or_else(|| {
                SchedulerError::internal("terminal endpoint guard is unavailable")
            })?;
            PublicationGuard::Terminal(guard)
        };
        let completed = next_position == record.total_positions || stop;
        let next_phase = if completed {
            RequestPhase::Terminal
        } else if output_full_after_publish {
            RequestPhase::OutputBlocked
        } else if preempt_after_commit {
            RequestPhase::Preempted
        } else {
            RequestPhase::Ready
        };
        let next_adapter_revision = identity
            .state_revision()
            .checked_add(1)
            .ok_or_else(|| SchedulerError::internal("adapter state revision overflows"))?;
        let binding = record.adapter_binding.ok_or_else(|| {
            SchedulerError::internal("adapter state identity was not bound before commit")
        })?;
        if binding.expected_revision != identity.state_revision() {
            return Err(SchedulerError::internal(
                "adapter state revision changed before commit",
            ));
        }
        Ok(PublicationPlan::Ready(TokenPublication {
            position: record.committed_positions,
            next_position,
            next_emitted_tokens,
            next_rng_state,
            next_decode_token,
            next_phase,
            next_adapter_revision,
            service_phase,
            completed,
            endpoint_guard,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_publication<D: CheckpointDriver, O: StepObserver>(
        &mut self,
        adapter: &A,
        ring: &mut DrrRing,
        trace: &mut Vec<ServiceTraceEvent>,
        key: SlotKey,
        reservation: crate::ring::ServiceReservation,
        pending: &A::PendingStateCommit,
        publication: TokenPublication<'_>,
        clock: &impl Fn() -> u64,
        checkpoints: &mut D,
        observer: &mut O,
    ) -> SchedulerResult<CommitDisposition> {
        let TokenPublication {
            position,
            next_position,
            next_emitted_tokens,
            next_rng_state,
            next_decode_token,
            next_phase,
            next_adapter_revision,
            service_phase,
            completed,
            endpoint_guard,
        } = publication;
        if self.queued.contains(&key) {
            return Err(SchedulerError::internal(
                "committing request is still present in the admission FIFO",
            ));
        }
        let slot_index = key.index();
        let trace_capacity = self.config.trace_capacity();
        let (ledger, slots, trace_overflowed, monotonic_ns) = (
            &mut self.ledger,
            &mut self.slots,
            &mut self.trace_overflowed,
            &mut self.monotonic_ns,
        );
        let slot = slots
            .get_mut(slot_index)
            .ok_or_else(|| SchedulerError::internal("commit slot index is out of range"))?;
        if slot.key != Some(key) {
            return Err(SchedulerError::internal("commit slot generation is stale"));
        }
        let record = slot
            .record
            .as_mut()
            .ok_or_else(|| SchedulerError::internal("commit request record is missing"))?;
        if record.phase != RequestPhase::ReadyToCommit || record.committed_positions != position {
            return Err(SchedulerError::internal(
                "commit request phase or position changed",
            ));
        }
        let control_snapshot = record.control.fresh_snapshot()?;
        if control_snapshot.terminal() {
            return Err(SchedulerError::internal(
                "active request control is already terminal",
            ));
        }
        if trace.capacity() < trace_capacity {
            return Err(SchedulerError::internal(
                "service trace capacity changed before commit",
            ));
        }
        let request_id = record.request_id;
        let release = ledger
            .prepare_request_release(
                request_id,
                record
                    .active_reservation
                    .iter()
                    .chain(record.prompt_reservation.iter()),
            )
            .map_err(map_ledger_error)?;
        let mut state = record
            .state
            .take()
            .ok_or_else(|| SchedulerError::internal("commit adapter state is missing"))?;
        let service_event = ServiceTraceEvent::new(request_id, position, service_phase);
        let deadline_ns = record.deadline_ns;
        let control = &record.control;
        let emitted = endpoint_guard.expects_output();
        let mut release = Some(release);
        let mut endpoint_guard = Some(endpoint_guard);
        let (callback, adapter_result, terminal_outcome, checkpoint_error) = ring
            .with_validated_credit_commit(reservation, |mut service_permit| {
                let mut callback = None;
                let mut checkpoint_error = None;
                let adapter_result =
                    adapter.with_validated_state_commit(&mut state, pending, |adapter_permit| {
                        let checkpoint_prior_ns = *monotonic_ns;
                        match checkpoints.fire(
                            CheckpointContext {
                                point: CheckpointPoint::CompositePermitPreFinalSnapshot,
                                slot: key,
                                request_id,
                                position,
                                control,
                                deadline_ns,
                                monotonic_ns: *monotonic_ns,
                            },
                            clock,
                            observer,
                        ) {
                            Ok(fire) => {
                                if let Err(error) =
                                    apply_checkpoint_fire(monotonic_ns, checkpoint_prior_ns, fire)
                                {
                                    checkpoint_error = Some(error);
                                    drop(adapter_permit);
                                    return;
                                }
                            }
                            Err(error) => {
                                checkpoint_error = Some(error);
                                drop(adapter_permit);
                                return;
                            }
                        }
                        let now = observe_clock_value(monotonic_ns, clock);
                        let snapshot = control.fresh_snapshot_prevalidated();
                        let checkpoint_prior_ns = *monotonic_ns;
                        match checkpoints.fire(
                            CheckpointContext {
                                point: CheckpointPoint::PostFinalSnapshot,
                                slot: key,
                                request_id,
                                position,
                                control,
                                deadline_ns,
                                monotonic_ns: *monotonic_ns,
                            },
                            clock,
                            observer,
                        ) {
                            Ok(fire) => {
                                if let Err(error) =
                                    apply_checkpoint_fire(monotonic_ns, checkpoint_prior_ns, fire)
                                {
                                    checkpoint_error = Some(error);
                                    drop(adapter_permit);
                                    return;
                                }
                            }
                            Err(error) => {
                                checkpoint_error = Some(error);
                                drop(adapter_permit);
                                return;
                            }
                        }
                        if let Some(outcome) = visible_snapshot_outcome(snapshot, deadline_ns, now)
                        {
                            drop(adapter_permit);
                            callback = Some(CommitCallback::Suppressed(outcome));
                            return;
                        }
                        A::apply_state_commit(adapter_permit);
                        service_permit.apply();
                        record.committed_positions = next_position;
                        record.emitted_tokens = next_emitted_tokens;
                        record.rng_state = next_rng_state;
                        record.next_decode_token = next_decode_token;
                        record.phase = next_phase;
                        if let Some(binding) = record.adapter_binding.as_mut() {
                            binding.expected_revision = next_adapter_revision;
                        }
                        if trace.len() < trace_capacity {
                            trace.push(service_event);
                        } else {
                            *trace_overflowed = true;
                        }
                        callback = Some(CommitCallback::Applied);
                    });

                let terminal_outcome =
                    if callback == Some(CommitCallback::Applied) && adapter_result.is_err() {
                        Some(TerminalOutcome::Failed {
                            category: ErrorCategory::Internal,
                        })
                    } else if callback == Some(CommitCallback::Applied) && completed {
                        Some(TerminalOutcome::Completed)
                    } else {
                        None
                    };
                let terminal = terminal_outcome.map(|outcome| {
                    TerminalResult::new(
                        record.request_id,
                        outcome,
                        record.committed_positions,
                        record.emitted_tokens,
                    )
                });
                if terminal_outcome.is_some() {
                    service_permit.remove_member();
                    record.phase = RequestPhase::Terminal;
                    control.mark_terminal_prevalidated();
                    let _ = record.prompt.take();
                    let _ = record.active_reservation.take();
                    let _ = record.prompt_reservation.take();
                    // Active-state ownership ends only after its numerical
                    // payload is destroyed; the trace/release follows it.
                    drop(state);
                    if let Some(release) = release.take() {
                        release.apply();
                    }
                } else {
                    record.state = Some(state);
                }
                if callback == Some(CommitCallback::Applied)
                    && let Some(guard) = endpoint_guard.take()
                {
                    guard.publish(terminal);
                    if observer.enabled() {
                        let committed_ns = clock();
                        observer.commit(request_id, position, service_phase, emitted, committed_ns);
                        if let Some(terminal) = terminal {
                            observer.terminal(terminal, committed_ns);
                        }
                    }
                }
                (callback, adapter_result, terminal_outcome, checkpoint_error)
            })
            .map_err(map_ring_error)?;
        if let Some(error) = checkpoint_error {
            return Err(error);
        }
        match (callback, adapter_result) {
            (Some(CommitCallback::Suppressed(outcome)), _) => {
                Ok(CommitDisposition::Suppressed(outcome))
            }
            (Some(CommitCallback::Applied), Ok(())) => Ok(CommitDisposition::Applied {
                post_commit_failure: None,
                terminalized: terminal_outcome.is_some(),
            }),
            (Some(CommitCallback::Applied), Err(_)) => Ok(CommitDisposition::Applied {
                post_commit_failure: Some(ErrorCategory::Internal),
                terminalized: true,
            }),
            (None, Ok(())) => Err(SchedulerError::internal(
                "adapter returned success without invoking the commit callback",
            )),
            (None, Err(source)) => Err(SchedulerError::adapter("committing token", source)),
        }
    }

    fn release_wave_selection(
        &mut self,
        ring: &mut DrrRing,
        wave: &mut WaveScratch<A::PreparedToken, A::ExpertTask, A::ExpertContribution>,
        selection_index: usize,
    ) -> SchedulerResult<()> {
        let selection = wave
            .selections_mut()
            .get_mut(selection_index)
            .ok_or_else(|| SchedulerError::internal("wave selection index is out of range"))?;
        let key = selection.slot;
        let _ = selection.prepared.take();
        if let Some(reservation) = selection.reservation.take() {
            ring.recover_credit(reservation).map_err(map_ring_error)?;
        }
        if let Ok(record) = self.record_mut_for_key(key)
            && record.phase != RequestPhase::Terminal
        {
            record.phase = RequestPhase::Ready;
        }
        Ok(())
    }

    fn recover_wave(
        &mut self,
        ring: &mut DrrRing,
        wave: &mut WaveScratch<A::PreparedToken, A::ExpertTask, A::ExpertContribution>,
    ) -> SchedulerResult<()> {
        for index in 0..wave.selections().len() {
            self.release_wave_selection(ring, wave, index)?;
        }
        wave.reset().map_err(map_wave_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn terminalize_key<O: StepObserver>(
        &mut self,
        ring: &mut DrrRing,
        key: SlotKey,
        outcome: TerminalOutcome,
        clock: &impl Fn() -> u64,
        observer: &mut O,
        defer_worker_quiescence: bool,
    ) -> SchedulerResult<()> {
        if self.record_for_key(key)?.phase == RequestPhase::Terminal {
            return Ok(());
        }
        let endpoint = self.record_for_key(key)?.endpoint.clone();
        let endpoint_guard = endpoint.begin_terminal_commit()?;
        let index = key.index();
        let slot = self
            .slots
            .get_mut(index)
            .ok_or_else(|| SchedulerError::internal("terminal slot index is out of range"))?;
        if slot.key != Some(key) {
            return Err(SchedulerError::internal(
                "terminal slot generation is stale",
            ));
        }
        let record = slot
            .record
            .as_mut()
            .ok_or_else(|| SchedulerError::internal("terminal request record is missing"))?;
        if record.phase == RequestPhase::Terminal {
            return Err(SchedulerError::internal(
                "request became terminal while its endpoint was locked",
            ));
        }
        if ring.contains(key) && !ring.member_matches(key, record.request_id) {
            return Err(SchedulerError::internal(
                "terminal policy member has a foreign request identity",
            ));
        }
        let control_snapshot = record.control.fresh_snapshot()?;
        let release = self
            .ledger
            .prepare_request_release(
                record.request_id,
                record
                    .active_reservation
                    .iter()
                    .chain(record.prompt_reservation.iter()),
            )
            .map_err(map_ledger_error)?;
        if ring.contains(key) {
            let _ = ring.remove(key).map_err(map_ring_error)?;
        }
        self.queued.retain(|queued| *queued != key);
        record.control.mark_terminal_prevalidated();
        let _ = record.state.take();
        let _ = record.prompt.take();
        let _ = record.active_reservation.take();
        let _ = record.prompt_reservation.take();
        record.phase = RequestPhase::Terminal;
        let terminal = TerminalResult::new(
            record.request_id,
            outcome,
            record.committed_positions,
            record.emitted_tokens,
        );
        release.apply();
        endpoint_guard.publish_terminal(terminal);
        if observer.enabled() {
            let decided_ns = clock();
            observer.terminal(terminal, decided_ns);
            if outcome == TerminalOutcome::Cancelled && !defer_worker_quiescence {
                observer.worker_quiescent(record.request_id, decided_ns);
            }
        }
        if control_snapshot.disconnected() {
            let report = endpoint.settle_disconnected()?;
            validate_endpoint_discard(
                record.request_id,
                record.emitted_tokens,
                endpoint.snapshot()?,
                report,
            )?;
        }
        Ok(())
    }

    fn terminalize_key_observed<O: StepObserver>(
        &mut self,
        ring: &mut DrrRing,
        key: SlotKey,
        outcome: TerminalOutcome,
        clock: &impl Fn() -> u64,
        observer: &mut O,
    ) -> SchedulerResult<()> {
        self.terminalize_key_with_observer(ring, key, outcome, clock, observer, false)
    }

    fn terminalize_wave_key_observed<O: StepObserver>(
        &mut self,
        ring: &mut DrrRing,
        key: SlotKey,
        outcome: TerminalOutcome,
        clock: &impl Fn() -> u64,
        observer: &mut O,
    ) -> SchedulerResult<()> {
        self.terminalize_key_with_observer(ring, key, outcome, clock, observer, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn terminalize_key_with_observer<O: StepObserver>(
        &mut self,
        ring: &mut DrrRing,
        key: SlotKey,
        outcome: TerminalOutcome,
        clock: &impl Fn() -> u64,
        observer: &mut O,
        defer_worker_quiescence: bool,
    ) -> SchedulerResult<()> {
        self.terminalize_key(ring, key, outcome, clock, observer, defer_worker_quiescence)
    }

    /// Drains at most `limit` already committed output events.
    pub fn drain_events(
        &mut self,
        id: crate::RequestId,
        limit: usize,
    ) -> SchedulerResult<Vec<OutputEvent>> {
        let mut observer = NullObserver;
        self.drain_events_with_observer(id, limit, &|| 0, &mut observer)
    }

    fn drain_events_with_observer<O: StepObserver>(
        &mut self,
        id: crate::RequestId,
        limit: usize,
        clock: &impl Fn() -> u64,
        observer: &mut O,
    ) -> SchedulerResult<Vec<OutputEvent>> {
        let key = self.record_by_id(id)?.key;
        let count = self
            .record_for_key(key)?
            .endpoint
            .snapshot()?
            .buffered_output_events
            .min(limit);
        let mut drained = Vec::new();
        try_reserve_vec(&mut drained, count, "drained output result")?;

        if count != 0 {
            let record = self.record_mut_for_key(key)?;
            let receiver = record.direct_receiver.as_mut().ok_or_else(|| {
                SchedulerError::internal("request endpoint receiver is externally owned")
            })?;
            for _ in 0..count {
                match receiver.try_pop()? {
                    TryPop::Event(event) => drained.push(event),
                    TryPop::Empty | TryPop::Eof => {
                        return Err(SchedulerError::internal(
                            "prevalidated output event is missing",
                        ));
                    }
                }
            }
            let endpoint = receiver.snapshot()?;
            if !endpoint.producer_open
                && endpoint.buffered_output_events == 0
                && !endpoint.output_eof_acknowledged
            {
                match receiver.try_pop()? {
                    TryPop::Eof => {}
                    TryPop::Empty | TryPop::Event(_) => {
                        return Err(SchedulerError::internal(
                            "closed empty endpoint did not acknowledge EOF",
                        ));
                    }
                }
            }
        }

        let endpoint = self.record_for_key(key)?.endpoint.snapshot()?;
        if self.record_for_key(key)?.phase == RequestPhase::OutputBlocked
            && endpoint.producer_open
            && endpoint.receiver_connected
            && endpoint.buffered_output_events < endpoint.output_capacity
        {
            self.record_mut_for_key(key)?.phase = RequestPhase::Ready;
        }
        if endpoint.reap_ready {
            let request_id = self.reap_key(key)?;
            if observer.enabled() {
                observer.request_zero(request_id, clock());
            }
        }
        Ok(drained)
    }

    /// Drains committed events while timestamping a resulting ownership-zero
    /// reap for the sealed M5 observer.
    #[cfg(any(test, feature = "m5-run-observer-instrumentation"))]
    #[doc(hidden)]
    pub fn drain_events_observed(
        &mut self,
        id: crate::RequestId,
        limit: usize,
        observer: &mut RunObserver,
    ) -> SchedulerResult<Vec<OutputEvent>> {
        self.validate_run_observer_domain(observer)?;
        let clock = observer.clock();
        self.drain_events_with_observer(id, limit, &|| clock.now_ns(), observer)
    }

    #[cfg(test)]
    pub(crate) fn drain_events_observed_with_clock_for_test<F>(
        &mut self,
        id: crate::RequestId,
        limit: usize,
        clock: &F,
        observer: &mut RunObserver,
    ) -> SchedulerResult<Vec<OutputEvent>>
    where
        F: Fn() -> u64,
    {
        self.validate_run_observer_domain(observer)?;
        self.drain_events_with_observer(id, limit, clock, observer)
    }

    /// Takes a terminal result once, independently of buffered output.
    pub fn take_terminal(
        &mut self,
        id: crate::RequestId,
    ) -> SchedulerResult<Option<TerminalResult>> {
        let mut observer = NullObserver;
        self.take_terminal_with_observer(id, &|| 0, &mut observer)
    }

    fn take_terminal_with_observer<O: StepObserver>(
        &mut self,
        id: crate::RequestId,
        clock: &impl Fn() -> u64,
        observer: &mut O,
    ) -> SchedulerResult<Option<TerminalResult>> {
        let key = self.record_by_id(id)?.key;
        let result = {
            let record = self.record_mut_for_key(key)?;
            let receiver = record.direct_receiver.as_mut().ok_or_else(|| {
                SchedulerError::internal("request endpoint receiver is externally owned")
            })?;
            let result = receiver.take_terminal()?;
            if result.is_some() {
                let endpoint = receiver.snapshot()?;
                if !endpoint.producer_open
                    && endpoint.buffered_output_events == 0
                    && !endpoint.output_eof_acknowledged
                {
                    match receiver.try_pop()? {
                        TryPop::Eof => {}
                        TryPop::Empty | TryPop::Event(_) => {
                            return Err(SchedulerError::internal(
                                "closed empty endpoint did not acknowledge EOF",
                            ));
                        }
                    }
                }
            }
            result
        };
        if self.record_for_key(key)?.endpoint.snapshot()?.reap_ready {
            let request_id = self.reap_key(key)?;
            if observer.enabled() {
                observer.request_zero(request_id, clock());
            }
        }
        Ok(result)
    }

    /// Takes a terminal value while timestamping a resulting ownership-zero
    /// reap for the sealed M5 observer.
    #[cfg(any(test, feature = "m5-run-observer-instrumentation"))]
    #[doc(hidden)]
    pub fn take_terminal_observed(
        &mut self,
        id: crate::RequestId,
        observer: &mut RunObserver,
    ) -> SchedulerResult<Option<TerminalResult>> {
        self.validate_run_observer_domain(observer)?;
        let clock = observer.clock();
        self.take_terminal_with_observer(id, &|| clock.now_ns(), observer)
    }

    #[cfg(test)]
    pub(crate) fn take_terminal_observed_with_clock_for_test<F>(
        &mut self,
        id: crate::RequestId,
        clock: &F,
        observer: &mut RunObserver,
    ) -> SchedulerResult<Option<TerminalResult>>
    where
        F: Fn() -> u64,
    {
        self.validate_run_observer_domain(observer)?;
        self.take_terminal_with_observer(id, clock, observer)
    }

    fn reap_key(&mut self, key: SlotKey) -> SchedulerResult<crate::RequestId> {
        let index = key.index();
        let slot = self
            .slots
            .get_mut(index)
            .ok_or_else(|| SchedulerError::internal("reap slot index is out of range"))?;
        if slot.key != Some(key) {
            return Err(SchedulerError::internal("reap slot generation is stale"));
        }
        if self.free_slots.len() >= self.free_slots.capacity() {
            return Err(SchedulerError::internal(
                "free request slot capacity is full",
            ));
        }
        let record = slot
            .record
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("reap request record is missing"))?;
        validate_terminal_resources_released(record)?;
        let request_id = record.request_id;
        let emitted_tokens = record.emitted_tokens;
        let endpoint_snapshot = record.endpoint.snapshot()?;
        let reservation = record
            .retained_reservation
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("retained request reservation is missing"))?;
        let release = self
            .ledger
            .prepare_request_release(request_id, [reservation])
            .map_err(map_ledger_error)?;
        let endpoint = self
            .endpoints
            .as_mut()
            .ok_or_else(|| SchedulerError::internal("request endpoints are unavailable"))?
            .begin_recycle(key)?;
        let endpoint_report = endpoint.report();
        validate_endpoint_reap(emitted_tokens, endpoint_snapshot, endpoint_report)?;
        let controls = self
            .controls
            .as_mut()
            .ok_or_else(|| SchedulerError::internal("request controls are unavailable"))?;
        let record = slot
            .record
            .take()
            .ok_or_else(|| SchedulerError::internal("prevalidated reap record disappeared"))?;
        let recycle = controls.recycle(&record.control);
        if let Err(error) = recycle {
            slot.record = Some(record);
            return Err(error);
        }
        let committed_endpoint_report = endpoint.commit();
        debug_assert_eq!(committed_endpoint_report, endpoint_report);
        slot.key = None;
        self.free_slots.push(index);
        drop(record);
        release.apply();
        Ok(request_id)
    }

    fn discard_and_reap_key(&mut self, key: SlotKey) -> SchedulerResult<()> {
        self.reap_key(key).map(|_| ())
    }

    /// Closes the engine, resolves all requests, and releases all declared
    /// logical ownership. Repeated shutdown is idempotent.
    pub fn shutdown(&mut self) -> SchedulerResult<ShutdownReport> {
        if self.closed && self.shared_reservation.is_none() {
            if self.ledger.snapshot().current_is_zero()
                && self.adapter.is_none()
                && self.ring.is_none()
                && self.workspace.is_none()
                && self.sampling.is_none()
                && self.wave.is_none()
                && self.service_trace.is_none()
                && !self.ledger.has_ledger_trace()
                && self.controls.is_none()
                && self.endpoints.is_none()
            {
                return Ok(ShutdownReport::default());
            }
            return Err(SchedulerError::internal(
                "closed scheduler retains unowned resources",
            ));
        }
        self.closed = true;
        let request_bytes_before = self.ledger.snapshot().request_used();
        let mut terminated_requests = 0_usize;
        let mut discarded_output_events = 0_usize;
        let mut ring = self
            .ring
            .take()
            .ok_or_else(|| SchedulerError::internal("scheduler ring is unavailable"))?;
        let fixed_ns = self.monotonic_ns;
        let mut observer = NullObserver;
        let cleanup = (|| -> SchedulerResult<()> {
            for index in 0..self.slots.len() {
                let decision = self.slots[index]
                    .record
                    .as_ref()
                    .filter(|record| record.phase != RequestPhase::Terminal)
                    .map(|record| record.key);
                if let Some(key) = decision {
                    self.terminalize_key(
                        &mut ring,
                        key,
                        TerminalOutcome::Cancelled,
                        &|| fixed_ns,
                        &mut observer,
                        false,
                    )?;
                    terminated_requests += 1;
                }
            }
            self.queued.clear();
            for index in 0..self.slots.len() {
                let Some(key) = self.slots[index].key else {
                    continue;
                };
                if let Some(receiver) = self.slots[index]
                    .record
                    .as_mut()
                    .and_then(|record| record.direct_receiver.as_mut())
                {
                    receiver.disconnect()?;
                }
                let record = self.slots[index].record.as_ref().ok_or_else(|| {
                    SchedulerError::internal("shutdown request record is missing")
                })?;
                let request_id = record.request_id;
                let emitted_tokens = record.emitted_tokens;
                let discarded = self
                    .endpoints
                    .as_ref()
                    .ok_or_else(|| SchedulerError::internal("request endpoints are unavailable"))?
                    .shutdown_discard(key)?;
                let endpoint = self.slots[index]
                    .record
                    .as_ref()
                    .ok_or_else(|| SchedulerError::internal("shutdown request record disappeared"))?
                    .endpoint
                    .snapshot()?;
                validate_endpoint_discard(request_id, emitted_tokens, endpoint, discarded)?;
                discarded_output_events = discarded_output_events
                    .checked_add(discarded.output_events())
                    .ok_or_else(|| SchedulerError::internal("discarded output count overflows"))?;
            }
            for index in 0..self.slots.len() {
                if let Some(key) = self.slots[index].key {
                    self.discard_and_reap_key(key)?;
                }
            }
            Ok(())
        })();
        if let Err(error) = cleanup {
            self.ring = Some(ring);
            return Err(error);
        }

        if let Err(error) = shutdown_release_checkpoint() {
            self.ring = Some(ring);
            return Err(error);
        }

        let snapshot = self.ledger.snapshot();
        let released_request_bytes = match request_bytes_before.checked_sub(snapshot.request_used())
        {
            Some(bytes) => bytes,
            None => {
                self.ring = Some(ring);
                return Err(SchedulerError::internal(
                    "released request byte count underflows",
                ));
            }
        };
        let shared = match self.shared_reservation.as_ref() {
            Some(shared) => shared,
            None => {
                self.ring = Some(ring);
                return Err(SchedulerError::internal(
                    "shared scheduler reservation is missing",
                ));
            }
        };
        let release = match self.ledger.prepare_release([shared]) {
            Ok(release) => release,
            Err(error) => {
                self.ring = Some(ring);
                return Err(map_ledger_error(error));
            }
        };

        drop(ring);
        let _ = self.wave.take();
        let _ = self.workspace.take();
        let _ = self.sampling.take();
        let _ = self.service_trace.take();
        let _ = self.adapter.take();
        let _ = self.controls.take();
        let _ = self.endpoints.take();
        self.slots = Vec::new();
        self.free_slots = Vec::new();
        self.queued = VecDeque::new();
        let _ = self.shared_reservation.take();
        // The final observable trace prefix proves request ownership reached
        // zero. Shared teardown is proven by the post-shutdown ledger snapshot:
        // destroy the recorder before releasing its own static shared charge.
        release.drop_trace_and_apply();
        let snapshot = self.ledger.snapshot();
        Ok(ShutdownReport {
            terminated_requests,
            discarded_output_events,
            released_request_bytes: usize::try_from(released_request_bytes).unwrap_or(usize::MAX),
            remaining_shared_bytes: usize::try_from(snapshot.shared_used()).unwrap_or(usize::MAX),
        })
    }

    fn record_for_key(&self, key: SlotKey) -> SchedulerResult<&RequestRecord<A>> {
        let slot = self
            .slots
            .get(key.index())
            .ok_or_else(|| SchedulerError::internal("request slot index is out of range"))?;
        if slot.key != Some(key) {
            return Err(SchedulerError::internal("request slot generation is stale"));
        }
        slot.record
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("request slot record is missing"))
    }

    fn record_mut_for_key(&mut self, key: SlotKey) -> SchedulerResult<&mut RequestRecord<A>> {
        let slot = self
            .slots
            .get_mut(key.index())
            .ok_or_else(|| SchedulerError::internal("request slot index is out of range"))?;
        if slot.key != Some(key) {
            return Err(SchedulerError::internal("request slot generation is stale"));
        }
        slot.record
            .as_mut()
            .ok_or_else(|| SchedulerError::internal("request slot record is missing"))
    }

    fn ensure_open(&self) -> SchedulerResult<()> {
        if self.closed {
            Err(SchedulerError::scheduler_closed())
        } else {
            Ok(())
        }
    }

    fn record_by_id(&self, id: crate::RequestId) -> SchedulerResult<&RequestRecord<A>> {
        self.slots
            .iter()
            .filter_map(|slot| slot.record.as_ref())
            .find(|record| record.request_id == id)
            .ok_or_else(SchedulerError::request_not_found)
    }
}

type SharedAllocation<A> = (
    Vec<RequestSlot<A>>,
    Vec<usize>,
    ControlRegistry,
    EndpointRegistry,
    VecDeque<SlotKey>,
    DrrRing,
    <A as DecoderAdapter>::Workspace,
    SamplingWorkspace,
    WaveScratch<
        <A as DecoderAdapter>::PreparedToken,
        <A as DecoderAdapter>::ExpertTask,
        <A as DecoderAdapter>::ExpertContribution,
    >,
    Vec<ServiceTraceEvent>,
);

#[allow(dead_code, reason = "used by the staged Tokio actor integration")]
pub(crate) struct ActorAdmission {
    request_id: crate::RequestId,
    control: ControlBinding,
    receiver: EndpointReceiver,
}

impl fmt::Debug for ActorAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ActorAdmission")
            .field("request_id", &self.request_id)
            .field("control", &"<redacted>")
            .field("receiver", &"<redacted>")
            .finish()
    }
}

impl ActorAdmission {
    #[allow(dead_code, reason = "used by the staged Tokio actor integration")]
    pub(crate) fn into_parts(self) -> (crate::RequestId, ControlBinding, EndpointReceiver) {
        (self.request_id, self.control, self.receiver)
    }
}

#[derive(Clone, Copy)]
struct ResourcePressure {
    resource: &'static str,
    required: u64,
    limit: u64,
}

impl ResourcePressure {
    const fn new(resource: &'static str, required: u64, limit: u64) -> Self {
        Self {
            resource,
            required,
            limit,
        }
    }

    fn from_error(error: SchedulerError) -> SchedulerResult<Self> {
        match error {
            SchedulerError::ResourceExhausted {
                resource,
                required,
                limit,
            } => Ok(Self::new(resource, required, limit)),
            error => Err(error),
        }
    }

    const fn error(self) -> SchedulerError {
        SchedulerError::resource_exhausted(self.resource, self.required, self.limit)
    }
}

struct ValidatedBatchOffer<'request, A: DecoderAdapter> {
    request: RequestSpec<'request>,
    deadline: BatchDeadline,
    total_positions: usize,
    state_layout: A::StateLayout,
    active_plan: ChargePlan<ACTIVE_PLAN_LEN>,
    admission_plan: ChargePlan<ADMISSION_BASE_PLAN_LEN>,
}

#[derive(Clone, Copy)]
struct PreparedSlot {
    slot_index: usize,
    free_position: usize,
    generation: SlotGeneration,
    key: SlotKey,
}

struct PreparedBatchRequest<A: DecoderAdapter> {
    offered_index: usize,
    slot_index: usize,
    free_position: usize,
    generation: SlotGeneration,
    key: SlotKey,
    prompt: Vec<u32>,
    max_new_tokens: usize,
    total_positions: usize,
    sampling: runnel_runtime::SamplingPolicy,
    deadline: BatchDeadline,
    deadline_ns: Option<u64>,
    state_layout: A::StateLayout,
    active_plan: ChargePlan<ACTIVE_PLAN_LEN>,
}

struct RequestSlot<A: DecoderAdapter> {
    generations: SlotGenerationIssuer,
    key: Option<SlotKey>,
    record: Option<RequestRecord<A>>,
}

impl<A: DecoderAdapter> RequestSlot<A> {
    const fn new() -> Self {
        Self {
            generations: SlotGenerationIssuer::new(),
            key: None,
            record: None,
        }
    }
}

struct RequestRecord<A: DecoderAdapter> {
    request_id: crate::RequestId,
    key: SlotKey,
    phase: RequestPhase,
    prompt: Option<Vec<u32>>,
    max_new_tokens: usize,
    total_positions: usize,
    sampling: runnel_runtime::SamplingPolicy,
    deadline_ns: Option<u64>,
    admitted_ns: u64,
    control: ControlBinding,
    state_layout: A::StateLayout,
    state: Option<A::State>,
    active_plan: ChargePlan<ACTIVE_PLAN_LEN>,
    active_reservation: Option<LedgerReservation>,
    prompt_reservation: Option<LedgerReservation>,
    retained_reservation: Option<LedgerReservation>,
    committed_positions: usize,
    emitted_tokens: usize,
    next_decode_token: Option<u32>,
    rng_state: Option<u64>,
    adapter_binding: Option<AdapterBinding>,
    endpoint: EndpointProducer,
    direct_receiver: Option<EndpointReceiver>,
}

struct PublicationContext {
    key: SlotKey,
    identity: AdapterWorkIdentity,
    preempt_after_commit: bool,
}

struct TokenPublication<'endpoint> {
    position: usize,
    next_position: usize,
    next_emitted_tokens: usize,
    next_rng_state: Option<u64>,
    next_decode_token: Option<u32>,
    next_phase: RequestPhase,
    next_adapter_revision: u64,
    service_phase: ServicePhase,
    completed: bool,
    endpoint_guard: PublicationGuard<'endpoint>,
}

impl fmt::Debug for TokenPublication<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TokenPublication")
            .field("position", &self.position)
            .field("next_position", &self.next_position)
            .field("emits", &self.endpoint_guard.expects_output())
            .field("service_phase", &self.service_phase)
            .field("completed", &self.completed)
            .field("endpoint", &"<held>")
            .finish_non_exhaustive()
    }
}

enum PublicationPlan<'endpoint> {
    Ready(TokenPublication<'endpoint>),
    OutputBlocked,
    Cancelled,
}

enum PublicationGuard<'endpoint> {
    Output(ValidatedOutputCommitGuard<'endpoint>),
    Terminal(TerminalCommitGuard<'endpoint>),
}

impl PublicationGuard<'_> {
    const fn expects_output(&self) -> bool {
        matches!(self, Self::Output(_))
    }

    fn publish(self, terminal: Option<TerminalResult>) {
        match (self, terminal) {
            (Self::Output(guard), Some(terminal)) => {
                guard.publish_output_and_terminal(terminal);
            }
            (Self::Output(guard), None) => guard.publish_output(),
            (Self::Terminal(guard), Some(terminal)) => guard.publish_terminal(terminal),
            (Self::Terminal(guard), None) => drop(guard),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommitCallback {
    Applied,
    Suppressed(TerminalOutcome),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommitDisposition {
    Applied {
        post_commit_failure: Option<ErrorCategory>,
        terminalized: bool,
    },
    Suppressed(TerminalOutcome),
}

impl<A: DecoderAdapter> fmt::Debug for RequestRecord<A> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RequestRecord")
            .field("request_id", &self.request_id)
            .field("phase", &self.phase)
            .field("prompt", &"<redacted>")
            .field("sampling", &"<redacted>")
            .field("committed_positions", &self.committed_positions)
            .field("emitted_tokens", &self.emitted_tokens)
            .field("endpoint", &"<endpoint-owned>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdapterBinding {
    model_instance_id: u64,
    state_id: u64,
    expected_revision: u64,
}

impl AdapterBinding {
    fn matches(self, identity: AdapterWorkIdentity) -> bool {
        self.model_instance_id == identity.model_instance_id()
            && self.state_id == identity.state_id().get()
            && self.expected_revision == identity.state_revision()
    }
}

fn visible_control_outcome<A: DecoderAdapter>(
    record: &RequestRecord<A>,
    monotonic_ns: u64,
) -> SchedulerResult<Option<TerminalOutcome>> {
    let snapshot = record.control.fresh_snapshot()?;
    Ok(visible_snapshot_outcome(
        snapshot,
        record.deadline_ns,
        monotonic_ns,
    ))
}

fn visible_snapshot_outcome(
    snapshot: ControlSnapshot,
    deadline_ns: Option<u64>,
    monotonic_ns: u64,
) -> Option<TerminalOutcome> {
    if snapshot.cancelled() || snapshot.disconnected() {
        Some(TerminalOutcome::Cancelled)
    } else if deadline_ns.is_some_and(|deadline| monotonic_ns >= deadline) {
        Some(TerminalOutcome::DeadlineExceeded)
    } else {
        None
    }
}

fn validate_endpoint_discard(
    request_id: crate::RequestId,
    emitted_tokens: usize,
    snapshot: EndpointSnapshot,
    report: EndpointDiscardReport,
) -> SchedulerResult<()> {
    if snapshot.published_output_events != emitted_tokens
        || snapshot.buffered_output_events != 0
        || report.terminal_results > 1
        || (report.terminal_results != 0 && !snapshot.terminal_acknowledged)
    {
        return Err(SchedulerError::internal(
            "request endpoint discard accounting diverged from engine state",
        ));
    }
    if let Some(span) = report.output {
        let prior_discarded = snapshot
            .discarded_output_events
            .checked_sub(span.count())
            .ok_or_else(|| SchedulerError::internal("endpoint discard count underflows"))?;
        let expected_first = snapshot
            .drained_output_events
            .checked_add(prior_discarded)
            .ok_or_else(|| SchedulerError::internal("endpoint discard index overflows"))?;
        let observed_end = span
            .first_output_index()
            .checked_add(span.count())
            .ok_or_else(|| SchedulerError::internal("endpoint discard span overflows"))?;
        if span.request_id() != request_id
            || span.first_output_index() != expected_first
            || observed_end != emitted_tokens
        {
            return Err(SchedulerError::internal(
                "request endpoint discarded a foreign or noncontiguous output span",
            ));
        }
    }
    Ok(())
}

fn validate_endpoint_reap(
    emitted_tokens: usize,
    snapshot: EndpointSnapshot,
    report: EndpointReapReport,
) -> SchedulerResult<()> {
    if !snapshot.reap_ready
        || snapshot.published_output_events != emitted_tokens
        || snapshot.buffered_output_events != 0
        || snapshot.discarded_output_events != report.discarded_output_events
        || snapshot.discarded_terminal_results != report.discarded_terminal_results
        || snapshot.shutdown != report.shutdown
    {
        return Err(SchedulerError::internal(
            "request endpoint reap accounting diverged from engine state",
        ));
    }
    Ok(())
}

fn observe_clock_value(last_seen_ns: &mut u64, clock: &impl Fn() -> u64) -> u64 {
    *last_seen_ns = (*last_seen_ns).max(clock());
    *last_seen_ns
}

fn apply_checkpoint_fire(
    monotonic_ns: &mut u64,
    prior_ns: u64,
    fire: CheckpointFire,
) -> SchedulerResult<()> {
    let next_ns = fire.monotonic_ns();
    if next_ns < prior_ns {
        return Err(SchedulerError::internal(
            "checkpoint instrumentation moved the clock backwards",
        ));
    }
    *monotonic_ns = next_ns;
    Ok(())
}

fn ensure_request_lifecycle_feasible<const BASE: usize, const ACTIVE: usize>(
    config: &SchedulerConfig,
    admission: ChargePlan<BASE>,
    active: ChargePlan<ACTIVE>,
) -> SchedulerResult<()> {
    let required = config
        .shared_static_charge_bytes()
        .checked_add(admission.total_bytes())
        .and_then(|bytes| bytes.checked_add(active.total_bytes()))
        .ok_or_else(|| SchedulerError::internal("request lifecycle charge overflows"))?;
    let limit = config.logical_memory_limit_bytes();
    if required > limit {
        return Err(SchedulerError::resource_exhausted(
            "isolated request lifecycle",
            required,
            limit,
        ));
    }
    Ok(())
}

fn validate_terminal_resources_released<A: DecoderAdapter>(
    record: &RequestRecord<A>,
) -> SchedulerResult<()> {
    if record.phase != RequestPhase::Terminal {
        return Err(SchedulerError::internal(
            "request must be terminal before retained ownership is reaped",
        ));
    }
    if record.active_reservation.is_some()
        || record.prompt_reservation.is_some()
        || record.state.is_some()
        || record.prompt.is_some()
    {
        return Err(SchedulerError::internal(
            "terminal request still owns active resources",
        ));
    }
    Ok(())
}

fn input_token<A: DecoderAdapter>(record: &RequestRecord<A>) -> SchedulerResult<u32> {
    let prompt = record
        .prompt
        .as_ref()
        .ok_or_else(|| SchedulerError::internal("active request prompt is missing"))?;
    if record.committed_positions < prompt.len() {
        prompt
            .get(record.committed_positions)
            .copied()
            .ok_or_else(|| SchedulerError::internal("prompt position is out of range"))
    } else {
        record
            .next_decode_token
            .ok_or_else(|| SchedulerError::internal("decode input token is missing"))
    }
}

fn bind_prepared_identity<A: DecoderAdapter>(
    record: &mut RequestRecord<A>,
    identity: AdapterWorkIdentity,
) -> bool {
    if identity.position() != record.committed_positions
        || identity.state_revision().checked_add(1).is_none()
    {
        return false;
    }
    match record.adapter_binding {
        Some(binding)
            if binding.model_instance_id != identity.model_instance_id()
                || binding.state_id != identity.state_id().get()
                || binding.expected_revision != identity.state_revision() =>
        {
            return false;
        }
        Some(_) => {}
        None => {
            record.adapter_binding = Some(AdapterBinding {
                model_instance_id: identity.model_instance_id(),
                state_id: identity.state_id().get(),
                expected_revision: identity.state_revision(),
            });
        }
    }
    true
}

fn try_reserve_vec<T>(
    values: &mut Vec<T>,
    count: usize,
    resource: &'static str,
) -> SchedulerResult<()> {
    let bytes = batch_vec_bytes::<T>(count)
        .ok_or_else(|| SchedulerError::internal("scheduler allocation size overflows"))?;
    if bytes > isize::MAX as usize {
        return Err(SchedulerError::allocation_failure(
            resource,
            usize_to_u64(bytes)?,
        ));
    }
    values.try_reserve_exact(count).map_err(|_| {
        SchedulerError::allocation_failure(resource, u64::try_from(bytes).unwrap_or(u64::MAX))
    })
}

const fn batch_vec_bytes<T>(count: usize) -> Option<usize> {
    count.checked_mul(size_of::<T>())
}

fn checked_batch_scratch_sum(contributions: &[Option<usize>]) -> SchedulerResult<usize> {
    contributions.iter().try_fold(0_usize, |total, bytes| {
        total
            .checked_add(bytes.ok_or_else(|| {
                SchedulerError::allocation_failure("batch admission scratch", u64::MAX)
            })?)
            .ok_or_else(|| SchedulerError::allocation_failure("batch admission scratch", u64::MAX))
    })
}

fn try_reserve_deque<T>(
    values: &mut VecDeque<T>,
    count: usize,
    resource: &'static str,
) -> SchedulerResult<()> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| SchedulerError::internal("scheduler allocation size overflows"))?;
    if bytes > isize::MAX as usize {
        return Err(SchedulerError::allocation_failure(
            resource,
            usize_to_u64(bytes)?,
        ));
    }
    values.try_reserve_exact(count).map_err(|_| {
        SchedulerError::allocation_failure(resource, u64::try_from(bytes).unwrap_or(u64::MAX))
    })
}

fn map_ring_error(error: RingError) -> SchedulerError {
    match error {
        RingError::AllocationFailure => {
            SchedulerError::allocation_failure("scheduler membership", 0)
        }
        RingError::CapacityExceeded => {
            SchedulerError::internal("scheduler membership capacity was exceeded")
        }
        RingError::EpochExhausted => {
            SchedulerError::resource_exhausted("round epoch identity", u64::MAX, u64::MAX)
        }
        RingError::InvalidCapacity
        | RingError::DuplicateMember
        | RingError::RequestOrderViolation
        | RingError::RoundAlreadyOpen
        | RingError::RoundNotOpen
        | RingError::VisitPending
        | RingError::NoVisitPending
        | RingError::StaleVisit
        | RingError::OutstandingReservation
        | RingError::StaleReservation
        | RingError::InternalInvariant => {
            SchedulerError::internal("scheduler ring invariant failed")
        }
    }
}

fn map_wave_error(error: WaveScratchError) -> SchedulerError {
    match error {
        WaveScratchError::AllocationFailure { resource, bytes } => {
            SchedulerError::allocation_failure(resource, u64::try_from(bytes).unwrap_or(u64::MAX))
        }
        WaveScratchError::SizeOverflow { .. } | WaveScratchError::CapacityInvariant(_) => {
            SchedulerError::internal("wave scratch invariant failed")
        }
    }
}

fn map_identity_error(error: IdentityExhausted) -> SchedulerError {
    let resource = match error.kind() {
        crate::id::IdentityKind::Request => "request identity",
        crate::id::IdentityKind::EngineTransaction => "engine transaction identity",
        crate::id::IdentityKind::SlotGeneration => "slot generation identity",
        crate::id::IdentityKind::RoundEpoch => "round epoch identity",
    };
    SchedulerError::resource_exhausted(resource, u64::MAX, u64::MAX)
}

fn usize_to_u64(value: usize) -> SchedulerResult<u64> {
    u64::try_from(value)
        .map_err(|_| SchedulerError::internal("host usize does not fit scheduler accounting"))
}
