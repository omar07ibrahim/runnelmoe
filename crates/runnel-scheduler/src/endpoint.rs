//! Allocation-stable, generation-tagged request output endpoints.
//!
//! One endpoint slot is aligned with one scheduler request slot. Admission
//! allocates the sole output queue before binding; publication thereafter is
//! a mutex-protected, allocation-free commit that can remain locked across an
//! adapter's validated-state callback.

#![allow(dead_code, reason = "the actor API is integrated in staged slices")]

use std::{
    collections::VecDeque,
    fmt,
    mem::{self, size_of},
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicUsize, Ordering},
    },
};

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
use std::sync::{OnceLock, atomic::AtomicBool};

use tokio::sync::Notify;

use crate::{
    control::ControlBinding,
    error::{SchedulerError, SchedulerResult},
    id::SlotKey,
    request::{OutputEvent, TerminalResult},
};

/// Kind of one semantic endpoint mutation retained by the actor stress probe.
#[cfg(any(test, feature = "actor-stress-instrumentation"))]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActorSemanticObservationKind {
    Output,
    Terminal,
    OutputEof,
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum ActorSemanticPayload {
    Output(OutputEvent),
    Terminal(TerminalResult),
    OutputEof(crate::RequestId),
}

/// Copy-only observation captured at an endpoint mutation's linearization.
///
/// Raw request and token identities are available only through this hidden,
/// nondefault instrumentation API and are intentionally redacted from Debug.
#[cfg(any(test, feature = "actor-stress-instrumentation"))]
#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ActorSemanticObservation {
    payload: ActorSemanticPayload,
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
impl ActorSemanticObservation {
    const fn output(event: OutputEvent) -> Self {
        Self {
            payload: ActorSemanticPayload::Output(event),
        }
    }

    const fn terminal(terminal: TerminalResult) -> Self {
        Self {
            payload: ActorSemanticPayload::Terminal(terminal),
        }
    }

    const fn output_eof(request_id: crate::RequestId) -> Self {
        Self {
            payload: ActorSemanticPayload::OutputEof(request_id),
        }
    }

    #[must_use]
    pub const fn kind(self) -> ActorSemanticObservationKind {
        match self.payload {
            ActorSemanticPayload::Output(_) => ActorSemanticObservationKind::Output,
            ActorSemanticPayload::Terminal(_) => ActorSemanticObservationKind::Terminal,
            ActorSemanticPayload::OutputEof(_) => ActorSemanticObservationKind::OutputEof,
        }
    }

    #[must_use]
    pub const fn request_id(self) -> crate::RequestId {
        match self.payload {
            ActorSemanticPayload::Output(event) => event.request_id(),
            ActorSemanticPayload::Terminal(terminal) => terminal.request_id(),
            ActorSemanticPayload::OutputEof(request_id) => request_id,
        }
    }

    #[must_use]
    pub const fn output_event(self) -> Option<OutputEvent> {
        match self.payload {
            ActorSemanticPayload::Output(event) => Some(event),
            ActorSemanticPayload::Terminal(_) | ActorSemanticPayload::OutputEof(_) => None,
        }
    }

    #[must_use]
    pub const fn terminal_result(self) -> Option<TerminalResult> {
        match self.payload {
            ActorSemanticPayload::Terminal(terminal) => Some(terminal),
            ActorSemanticPayload::Output(_) | ActorSemanticPayload::OutputEof(_) => None,
        }
    }
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
impl fmt::Debug for ActorSemanticObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ActorSemanticObservation")
            .field("kind", &self.kind())
            .field("identity", &"<redacted>")
            .finish()
    }
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
struct ActorStressRecorderState {
    observations: Vec<ActorSemanticObservation>,
    limit: usize,
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
struct ActorStressRecorderInner {
    state: Mutex<ActorStressRecorderState>,
    overflowed: AtomicBool,
    poisoned: AtomicBool,
}

/// Sticky health and allocation state for the bounded semantic recorder.
#[cfg(any(test, feature = "actor-stress-instrumentation"))]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActorStressRecorderStatus {
    observation_count: usize,
    observation_limit: usize,
    allocated_capacity: usize,
    overflowed: bool,
    poisoned: bool,
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
impl ActorStressRecorderStatus {
    #[must_use]
    pub const fn observation_count(self) -> usize {
        self.observation_count
    }

    #[must_use]
    pub const fn observation_limit(self) -> usize {
        self.observation_limit
    }

    #[must_use]
    pub const fn allocated_capacity(self) -> usize {
        self.allocated_capacity
    }

    #[must_use]
    pub const fn overflowed(self) -> bool {
        self.overflowed
    }

    #[must_use]
    pub const fn poisoned(self) -> bool {
        self.poisoned
    }

    #[must_use]
    pub const fn healthy(self) -> bool {
        !self.overflowed && !self.poisoned
    }
}

/// Owned recorder snapshot copied only after endpoint locks are released.
#[cfg(any(test, feature = "actor-stress-instrumentation"))]
#[doc(hidden)]
#[derive(Debug)]
pub struct ActorStressRecording {
    observations: Vec<ActorSemanticObservation>,
    status: ActorStressRecorderStatus,
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
impl ActorStressRecording {
    #[must_use]
    pub fn observations(&self) -> &[ActorSemanticObservation] {
        &self.observations
    }

    #[must_use]
    pub const fn status(&self) -> ActorStressRecorderStatus {
        self.status
    }
}

/// Cloneable handle to a fixed-capacity endpoint semantic recorder.
#[cfg(any(test, feature = "actor-stress-instrumentation"))]
#[doc(hidden)]
#[derive(Clone)]
pub struct ActorStressRecorder {
    inner: Arc<ActorStressRecorderInner>,
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
impl fmt::Debug for ActorStressRecorder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ActorStressRecorder")
            .field("contents", &"<redacted>")
            .finish()
    }
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
impl ActorStressRecorder {
    pub(crate) fn try_with_capacity(capacity: usize) -> SchedulerResult<Self> {
        if capacity == 0 {
            return Err(SchedulerError::invalid_request(
                "actor stress recorder capacity",
                "must be nonzero",
            ));
        }
        let mut observations = Vec::new();
        try_reserve_vec(
            &mut observations,
            capacity,
            "actor stress semantic observations",
        )?;
        Ok(Self {
            inner: Arc::new(ActorStressRecorderInner {
                state: Mutex::new(ActorStressRecorderState {
                    observations,
                    limit: capacity,
                }),
                overflowed: AtomicBool::new(false),
                poisoned: AtomicBool::new(false),
            }),
        })
    }

    fn append_batch(&self, observations: &[Option<ActorSemanticObservation>]) {
        if self.inner.poisoned.load(Ordering::Acquire) {
            return;
        }
        let mut state = match self.inner.state.lock() {
            Ok(state) => state,
            Err(_) => {
                self.inner.poisoned.store(true, Ordering::Release);
                return;
            }
        };
        for observation in observations.iter().flatten().copied() {
            if state.observations.len() >= state.limit {
                self.inner.overflowed.store(true, Ordering::Release);
                continue;
            }
            debug_assert!(state.observations.capacity() >= state.limit);
            state.observations.push(observation);
        }
    }

    #[must_use]
    pub fn status(&self) -> ActorStressRecorderStatus {
        let poisoned =
            self.inner.poisoned.load(Ordering::Acquire) || self.inner.state.is_poisoned();
        if poisoned {
            self.inner.poisoned.store(true, Ordering::Release);
        }
        let state = match self.inner.state.lock() {
            Ok(state) => state,
            Err(error) => {
                self.inner.poisoned.store(true, Ordering::Release);
                error.into_inner()
            }
        };
        ActorStressRecorderStatus {
            observation_count: state.observations.len(),
            observation_limit: state.limit,
            allocated_capacity: state.observations.capacity(),
            overflowed: self.inner.overflowed.load(Ordering::Acquire),
            poisoned: self.inner.poisoned.load(Ordering::Acquire),
        }
    }

    /// Fallibly copies the retained prefix for an independent checker.
    pub fn recording(&self) -> SchedulerResult<ActorStressRecording> {
        let poisoned =
            self.inner.poisoned.load(Ordering::Acquire) || self.inner.state.is_poisoned();
        if poisoned {
            self.inner.poisoned.store(true, Ordering::Release);
        }
        let state = match self.inner.state.lock() {
            Ok(state) => state,
            Err(error) => {
                self.inner.poisoned.store(true, Ordering::Release);
                error.into_inner()
            }
        };
        let mut observations = Vec::new();
        try_reserve_vec(
            &mut observations,
            state.observations.len(),
            "actor stress recording snapshot",
        )?;
        observations.extend_from_slice(&state.observations);
        let status = ActorStressRecorderStatus {
            observation_count: state.observations.len(),
            observation_limit: state.limit,
            allocated_capacity: state.observations.capacity(),
            overflowed: self.inner.overflowed.load(Ordering::Acquire),
            poisoned: self.inner.poisoned.load(Ordering::Acquire),
        };
        Ok(ActorStressRecording {
            observations,
            status,
        })
    }
}

/// Purpose of one observed endpoint receive attempt.
#[cfg(any(test, feature = "actor-stress-instrumentation"))]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActorTryPopKind {
    Primary,
    OpportunisticEof,
}

/// Exact endpoint state transition observed by one receive attempt.
#[cfg(any(test, feature = "actor-stress-instrumentation"))]
#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ActorTryPopWitness {
    kind: ActorTryPopKind,
    boundary_reached: bool,
    slot_index: usize,
    slot_generation: u64,
    drained_before: usize,
    drained_after: usize,
    consumed_output: Option<OutputEvent>,
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
impl ActorTryPopWitness {
    pub(crate) const fn sentinel(kind: ActorTryPopKind) -> Self {
        Self {
            kind,
            boundary_reached: false,
            slot_index: 0,
            slot_generation: 0,
            drained_before: 0,
            drained_after: 0,
            consumed_output: None,
        }
    }

    #[must_use]
    pub const fn kind(self) -> ActorTryPopKind {
        self.kind
    }

    #[must_use]
    pub const fn boundary_reached(self) -> bool {
        self.boundary_reached
    }

    #[must_use]
    pub const fn slot_index(self) -> usize {
        self.slot_index
    }

    #[must_use]
    pub const fn slot_generation(self) -> u64 {
        self.slot_generation
    }

    #[must_use]
    pub const fn drained_before(self) -> usize {
        self.drained_before
    }

    #[must_use]
    pub const fn drained_after(self) -> usize {
        self.drained_after
    }

    #[must_use]
    pub const fn consumed_output(self) -> Option<OutputEvent> {
        self.consumed_output
    }
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
impl fmt::Debug for ActorTryPopWitness {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ActorTryPopWitness")
            .field("kind", &self.kind)
            .field("boundary_reached", &self.boundary_reached)
            .field("identity", &"<redacted>")
            .field("drained_before", &self.drained_before)
            .field("drained_after", &self.drained_after)
            .field("consumed_output", &self.consumed_output.is_some())
            .finish()
    }
}

/// A non-sensitive, allocation-free view of one bound endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EndpointSnapshot {
    pub(crate) buffered_output_events: usize,
    pub(crate) output_capacity: usize,
    pub(crate) published_output_events: usize,
    pub(crate) drained_output_events: usize,
    pub(crate) terminal_published: bool,
    pub(crate) terminal_pending: bool,
    pub(crate) terminal_acknowledged: bool,
    pub(crate) output_eof_acknowledged: bool,
    pub(crate) receiver_connected: bool,
    pub(crate) producer_open: bool,
    pub(crate) shutdown: bool,
    pub(crate) discarded_output_events: usize,
    pub(crate) discarded_terminal_results: usize,
    pub(crate) reap_ready: bool,
}

/// One contiguous FIFO suffix destroyed after its identity was validated.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscardedOutputSpan {
    request_id: crate::RequestId,
    first_output_index: usize,
    count: usize,
}

impl DiscardedOutputSpan {
    pub(crate) const fn request_id(self) -> crate::RequestId {
        self.request_id
    }

    pub(crate) const fn first_output_index(self) -> usize {
        self.first_output_index
    }

    pub(crate) const fn count(self) -> usize {
        self.count
    }
}

impl fmt::Debug for DiscardedOutputSpan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscardedOutputSpan")
            .field("identity", &"<redacted>")
            .field("count", &self.count)
            .finish()
    }
}

/// Payloads discarded by one explicit endpoint lifecycle operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EndpointDiscardReport {
    pub(crate) output: Option<DiscardedOutputSpan>,
    pub(crate) terminal_results: usize,
}

impl EndpointDiscardReport {
    pub(crate) const fn output_events(self) -> usize {
        match self.output {
            Some(span) => span.count(),
            None => 0,
        }
    }
}

/// Final counts returned while recycling one completed generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EndpointReapReport {
    pub(crate) discarded_output_events: usize,
    pub(crate) discarded_terminal_results: usize,
    pub(crate) shutdown: bool,
}

/// Result of a synchronous output receive attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TryPop {
    Event(OutputEvent),
    Empty,
    Eof,
}

struct EndpointTable {
    slots: Box<[EndpointSlot]>,
    pending_terminals: AtomicUsize,
    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    stress_recorder: OnceLock<ActorStressRecorder>,
}

impl fmt::Debug for EndpointTable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EndpointTable")
            .field("slot_count", &self.slots.len())
            .finish_non_exhaustive()
    }
}

struct EndpointSlot {
    state: Mutex<EndpointState>,
    notify: Notify,
}

struct EndpointState {
    key: Option<SlotKey>,
    last_key: Option<SlotKey>,
    request_id: Option<crate::RequestId>,
    output: VecDeque<OutputEvent>,
    logical_capacity: usize,
    terminal: Option<TerminalResult>,
    terminal_published: bool,
    terminal_acknowledged: bool,
    output_eof_acknowledged: bool,
    receiver_connected: bool,
    producer_open: bool,
    shutdown: bool,
    published_output_events: usize,
    drained_output_events: usize,
    discarded_output_events: usize,
    discarded_terminal_results: usize,
}

impl EndpointState {
    const fn vacant() -> Self {
        Self {
            key: None,
            last_key: None,
            request_id: None,
            output: VecDeque::new(),
            logical_capacity: 0,
            terminal: None,
            terminal_published: false,
            terminal_acknowledged: false,
            output_eof_acknowledged: false,
            receiver_connected: false,
            producer_open: false,
            shutdown: false,
            published_output_events: 0,
            drained_output_events: 0,
            discarded_output_events: 0,
            discarded_terminal_results: 0,
        }
    }

    fn snapshot(&self) -> SchedulerResult<EndpointSnapshot> {
        self.validate_conservation()?;
        if let Some(terminal) = self.terminal.as_ref().copied() {
            validate_terminal_identity(self, terminal)?;
        }
        Ok(EndpointSnapshot {
            buffered_output_events: self.output.len(),
            output_capacity: self.logical_capacity,
            published_output_events: self.published_output_events,
            drained_output_events: self.drained_output_events,
            terminal_published: self.terminal_published,
            terminal_pending: self.terminal.is_some(),
            terminal_acknowledged: self.terminal_acknowledged,
            output_eof_acknowledged: self.output_eof_acknowledged,
            receiver_connected: self.receiver_connected,
            producer_open: self.producer_open,
            shutdown: self.shutdown,
            discarded_output_events: self.discarded_output_events,
            discarded_terminal_results: self.discarded_terminal_results,
            reap_ready: self.reap_ready(),
        })
    }

    fn reap_ready(&self) -> bool {
        self.terminal_published
            && self.terminal_acknowledged
            && self.output.is_empty()
            && self.output_eof_acknowledged
            && !self.receiver_connected
    }

    fn validate_conservation(&self) -> SchedulerResult<()> {
        let accounted = self
            .drained_output_events
            .checked_add(self.discarded_output_events)
            .and_then(|count| count.checked_add(self.output.len()))
            .ok_or_else(|| SchedulerError::internal("request output accounting overflows"))?;
        if accounted != self.published_output_events {
            return Err(SchedulerError::internal(
                "request output accounting does not conserve publication",
            ));
        }
        Ok(())
    }

    fn expected_request_id(&self) -> SchedulerResult<crate::RequestId> {
        self.request_id.ok_or_else(|| {
            SchedulerError::internal("bound request endpoint has no expected request identity")
        })
    }

    /// Validates the exact FIFO identity suffix without changing queue state.
    fn discarded_output_span(&self) -> SchedulerResult<Option<DiscardedOutputSpan>> {
        self.validate_conservation()?;
        let Some(first) = self.output.front().copied() else {
            return Ok(None);
        };
        let request_id = self.expected_request_id()?;
        let first_output_index = self
            .drained_output_events
            .checked_add(self.discarded_output_events)
            .ok_or_else(|| SchedulerError::internal("request output identity index overflows"))?;
        if first.output_index() != first_output_index {
            return Err(SchedulerError::internal(
                "discarded request output is not the expected FIFO suffix",
            ));
        }
        for (offset, event) in self.output.iter().copied().enumerate() {
            let expected_index = first_output_index.checked_add(offset).ok_or_else(|| {
                SchedulerError::internal("discarded request output identity overflows")
            })?;
            if event.request_id() != request_id || event.output_index() != expected_index {
                return Err(SchedulerError::internal(
                    "discarded request output identities are not contiguous",
                ));
            }
        }
        let suffix_end = first_output_index
            .checked_add(self.output.len())
            .ok_or_else(|| SchedulerError::internal("request output suffix end overflows"))?;
        if suffix_end != self.published_output_events {
            return Err(SchedulerError::internal(
                "discarded request output does not end at publication frontier",
            ));
        }
        Ok(Some(DiscardedOutputSpan {
            request_id,
            first_output_index,
            count: self.output.len(),
        }))
    }

    fn discard_receiver_payload(
        &mut self,
        output: Option<DiscardedOutputSpan>,
        pending_terminals: &AtomicUsize,
    ) -> SchedulerResult<EndpointDiscardReport> {
        let output_events = output.map_or(0, DiscardedOutputSpan::count);
        debug_assert_eq!(output_events, self.output.len());
        let pending_terminal = self.terminal.as_ref().copied();
        if let Some(terminal) = pending_terminal {
            validate_terminal_identity(self, terminal)?;
        }
        let terminal_results = usize::from(pending_terminal.is_some());
        if terminal_results != 0 {
            decrement_pending_terminal(pending_terminals)?;
        }

        self.output.clear();
        self.discarded_output_events += output_events;

        let removed_terminal = usize::from(self.terminal.take().is_some());
        debug_assert_eq!(removed_terminal, terminal_results);
        self.discarded_terminal_results += terminal_results;
        if self.terminal_published {
            self.terminal_acknowledged = true;
            self.output_eof_acknowledged = true;
        }
        self.receiver_connected = false;

        debug_assert!(self.validate_conservation().is_ok());

        Ok(EndpointDiscardReport {
            output,
            terminal_results,
        })
    }
}

/// Nonmutating admission work holding the only fallibly allocated queue.
pub(crate) struct PreparedEndpoint {
    table: Arc<EndpointTable>,
    output: VecDeque<OutputEvent>,
    logical_capacity: usize,
}

impl fmt::Debug for PreparedEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedEndpoint")
            .field("identity", &"<redacted>")
            .field("output_capacity", &self.logical_capacity)
            .finish()
    }
}

/// A fully validated, still-unpublished endpoint binding.
///
/// The guard holds the vacant endpoint mutex and lifecycle free list. Dropping
/// it releases the prepared queue and leaves both structures byte-for-byte
/// unchanged. Once a control binding succeeds, [`Self::commit`] has no
/// allocation or error path, so the two registries cannot be left partially
/// bound.
pub(crate) struct EndpointBindGuard<'a> {
    prepared: PreparedEndpoint,
    key: SlotKey,
    state: MutexGuard<'a, EndpointState>,
    free_slots: &'a mut Vec<usize>,
    free_position: usize,
}

impl fmt::Debug for EndpointBindGuard<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EndpointBindGuard")
            .field("identity", &"<redacted>")
            .finish()
    }
}

impl EndpointBindGuard<'_> {
    /// Infallibly publishes the endpoint after control binding succeeds.
    pub(crate) fn commit(
        self,
        control: ControlBinding,
        request_id: crate::RequestId,
    ) -> (EndpointProducer, EndpointReceiver) {
        let Self {
            prepared,
            key,
            mut state,
            free_slots,
            free_position,
        } = self;

        debug_assert!(state.key.is_none());
        debug_assert_ne!(state.last_key, Some(key));
        debug_assert!(prepared.output.capacity() >= prepared.logical_capacity);
        debug_assert!(free_position < free_slots.len());
        debug_assert_eq!(free_slots[free_position], key.index());

        state.key = Some(key);
        state.request_id = Some(request_id);
        state.output = prepared.output;
        state.logical_capacity = prepared.logical_capacity;
        state.terminal = None;
        state.terminal_published = false;
        state.terminal_acknowledged = false;
        state.output_eof_acknowledged = false;
        state.receiver_connected = true;
        state.producer_open = true;
        state.shutdown = false;
        state.published_output_events = 0;
        state.drained_output_events = 0;
        state.discarded_output_events = 0;
        state.discarded_terminal_results = 0;
        drop(state);

        free_slots.swap_remove(free_position);
        let producer = EndpointProducer {
            table: Arc::clone(&prepared.table),
            key,
            control: control.clone(),
        };
        let receiver = EndpointReceiver {
            table: prepared.table,
            key,
            control: Some(control),
            locally_connected: true,
        };
        (producer, receiver)
    }
}

/// A fully validated, still-bound endpoint recycle operation.
///
/// Dropping this guard is an exact rollback. [`Self::commit`] only moves
/// already-owned values, resets fixed metadata, and returns one slot to a
/// construction-time reserved free list.
pub(crate) struct EndpointRecycleGuard<'a> {
    state: MutexGuard<'a, EndpointState>,
    notify: &'a Notify,
    free_slots: &'a mut Vec<usize>,
    index: usize,
    report: EndpointReapReport,
}

impl fmt::Debug for EndpointRecycleGuard<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EndpointRecycleGuard")
            .field("identity", &"<redacted>")
            .finish()
    }
}

impl EndpointRecycleGuard<'_> {
    /// Returns the prevalidated final accounting before peer recycle commits.
    pub(crate) const fn report(&self) -> EndpointReapReport {
        self.report
    }

    /// Infallibly retires the endpoint after every peer lifecycle check passes.
    pub(crate) fn commit(self) -> EndpointReapReport {
        let Self {
            mut state,
            notify,
            free_slots,
            index,
            report,
        } = self;

        debug_assert!(state.reap_ready());
        debug_assert!(state.validate_conservation().is_ok());
        debug_assert!(!free_slots.contains(&index));
        debug_assert!(free_slots.len() < free_slots.capacity());

        let allocation = mem::take(&mut state.output);
        state.last_key = state.key.take();
        state.request_id = None;
        state.logical_capacity = 0;
        state.terminal = None;
        state.terminal_published = false;
        state.terminal_acknowledged = false;
        state.output_eof_acknowledged = false;
        state.receiver_connected = false;
        state.producer_open = false;
        state.shutdown = false;
        state.published_output_events = 0;
        state.drained_output_events = 0;
        state.discarded_output_events = 0;
        state.discarded_terminal_results = 0;
        drop(state);

        drop(allocation);
        free_slots.push(index);
        notify.notify_waiters();
        report
    }
}

/// Single-owner lifecycle manager for a fixed endpoint table.
pub(crate) struct EndpointRegistry {
    table: Arc<EndpointTable>,
    free_slots: Vec<usize>,
}

impl fmt::Debug for EndpointRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EndpointRegistry")
            .field("capacity", &self.table.slots.len())
            .field("available", &self.free_slots.len())
            .finish()
    }
}

impl EndpointRegistry {
    /// Fallibly reserves the table and its allocation-stable lifecycle list.
    pub(crate) fn try_with_capacity(capacity: usize) -> SchedulerResult<Self> {
        if capacity == 0 {
            return Err(SchedulerError::invalid_request(
                "request endpoint capacity",
                "must be nonzero",
            ));
        }

        let mut slots = Vec::new();
        try_reserve_vec(&mut slots, capacity, "request endpoint slots")?;
        for _ in 0..capacity {
            slots.push(EndpointSlot {
                state: Mutex::new(EndpointState::vacant()),
                notify: Notify::new(),
            });
        }

        let mut free_slots = Vec::new();
        try_reserve_vec(
            &mut free_slots,
            capacity,
            "request endpoint lifecycle slots",
        )?;
        for index in (0..capacity).rev() {
            free_slots.push(index);
        }

        Ok(Self {
            // Stable Rust has no fallible Arc constructor. All predictable,
            // size-dependent allocations above are completed fallibly first.
            table: Arc::new(EndpointTable {
                slots: slots.into_boxed_slice(),
                pending_terminals: AtomicUsize::new(0),
                #[cfg(any(test, feature = "actor-stress-instrumentation"))]
                stress_recorder: OnceLock::new(),
            }),
            free_slots,
        })
    }

    /// Installs one fixed-capacity semantic recorder before any endpoint bind.
    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    pub(crate) fn install_actor_stress_recorder(
        &self,
        recorder: ActorStressRecorder,
    ) -> SchedulerResult<()> {
        if self.free_slots.len() != self.table.slots.len() {
            return Err(SchedulerError::internal(
                "actor stress recorder must be installed before endpoint admission",
            ));
        }
        self.table
            .stress_recorder
            .set(recorder)
            .map_err(|_| SchedulerError::internal("actor stress recorder was already installed"))
    }

    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    pub(crate) fn actor_stress_recorder(&self) -> Option<ActorStressRecorder> {
        self.table.stress_recorder.get().cloned()
    }

    /// Allocates a request's sole queue without mutating registry state.
    pub(crate) fn prepare(&self, output_capacity: usize) -> SchedulerResult<PreparedEndpoint> {
        let mut output = VecDeque::new();
        try_reserve_deque(&mut output, output_capacity, "request output endpoint")?;
        Ok(PreparedEndpoint {
            table: Arc::clone(&self.table),
            output,
            logical_capacity: output_capacity,
        })
    }

    /// Validates a prepared endpoint without publishing it.
    ///
    /// The returned guard retains the slot mutex and exclusive lifecycle-list
    /// access. Callers may next perform a fallible control bind; dropping this
    /// guard on failure leaves the endpoint registry unchanged.
    pub(crate) fn begin_bind(
        &mut self,
        prepared: PreparedEndpoint,
        key: SlotKey,
    ) -> SchedulerResult<EndpointBindGuard<'_>> {
        if !Arc::ptr_eq(&self.table, &prepared.table) {
            return Err(SchedulerError::internal(
                "prepared request endpoint belongs to another registry",
            ));
        }
        let index = key.index();
        let Self { table, free_slots } = self;
        let free_position = free_slots
            .iter()
            .position(|candidate| *candidate == index)
            .ok_or_else(|| SchedulerError::internal("request endpoint slot is not free"))?;
        let slot = table
            .slots
            .get(index)
            .ok_or_else(|| SchedulerError::internal("request endpoint slot is out of range"))?;
        let state = lock_state(slot)?;
        if state.key.is_some() {
            return Err(SchedulerError::internal(
                "free request endpoint slot remains bound",
            ));
        }
        if state.last_key == Some(key) {
            return Err(SchedulerError::internal(
                "request endpoint generation was reused",
            ));
        }
        if prepared.output.capacity() < prepared.logical_capacity {
            return Err(SchedulerError::internal(
                "prepared request endpoint lost reserved capacity",
            ));
        }

        Ok(EndpointBindGuard {
            prepared,
            key,
            state,
            free_slots,
            free_position,
        })
    }

    pub(crate) fn snapshot(&self, key: SlotKey) -> SchedulerResult<EndpointSnapshot> {
        snapshot_for(&self.table, key)
    }

    /// Returns an allocation-free concurrent hint with exact atomic updates.
    pub(crate) fn pending_terminal_count(&self) -> usize {
        self.table.pending_terminals.load(Ordering::Acquire)
    }

    /// Discards a generation during destructive scheduler shutdown.
    pub(crate) fn shutdown_discard(&self, key: SlotKey) -> SchedulerResult<EndpointDiscardReport> {
        let slot = slot_for(&self.table, key)?;
        let mut state = lock_bound(slot, key)?;
        if !state.terminal_published || state.producer_open {
            return Err(SchedulerError::internal(
                "request endpoint must be terminal before shutdown discard",
            ));
        }
        let output = state.discarded_output_span()?;
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        let eof_before = state.output_eof_acknowledged;
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        let request_id = state.expected_request_id()?;
        let report = state.discard_receiver_payload(output, &self.table.pending_terminals)?;
        state.producer_open = false;
        state.shutdown = true;
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        let eof = (!eof_before && state.output_eof_acknowledged)
            .then_some(ActorSemanticObservation::output_eof(request_id));
        drop(state);
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        append_semantic_observations(&self.table, &[eof]);
        slot.notify.notify_waiters();
        Ok(report)
    }

    /// Validates endpoint recycle eligibility without changing lifecycle state.
    ///
    /// The guard keeps the endpoint locked while a caller performs the final
    /// fallible control-registry recycle. If that operation fails, dropping the
    /// guard leaves this generation bound and retryable.
    pub(crate) fn begin_recycle(
        &mut self,
        key: SlotKey,
    ) -> SchedulerResult<EndpointRecycleGuard<'_>> {
        let index = key.index();
        let Self { table, free_slots } = self;
        let slot = slot_for(table, key)?;
        let state = lock_bound(slot, key)?;
        state.validate_conservation()?;
        if !state.reap_ready() {
            return Err(SchedulerError::internal(
                "request endpoint is not ready to reap",
            ));
        }
        if free_slots.contains(&index)
            || free_slots.len() >= table.slots.len()
            || free_slots.len() >= free_slots.capacity()
        {
            return Err(SchedulerError::internal(
                "request endpoint lifecycle list is inconsistent",
            ));
        }

        let report = EndpointReapReport {
            discarded_output_events: state.discarded_output_events,
            discarded_terminal_results: state.discarded_terminal_results,
            shutdown: state.shutdown,
        };
        Ok(EndpointRecycleGuard {
            state,
            notify: &slot.notify,
            free_slots,
            index,
            report,
        })
    }

    #[cfg(test)]
    fn available(&self) -> usize {
        self.free_slots.len()
    }
}

/// Cloneable allocation-free publication and actor-wake handle.
#[derive(Clone)]
pub(crate) struct EndpointProducer {
    table: Arc<EndpointTable>,
    key: SlotKey,
    control: ControlBinding,
}

impl fmt::Debug for EndpointProducer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EndpointProducer")
            .field("identity", &"<redacted>")
            .finish()
    }
}

impl EndpointProducer {
    pub(crate) fn snapshot(&self) -> SchedulerResult<EndpointSnapshot> {
        snapshot_for(&self.table, self.key)
    }

    /// Returns the table-wide pending-terminal count without taking a mutex.
    pub(crate) fn pending_terminal_count(&self) -> usize {
        self.table.pending_terminals.load(Ordering::Acquire)
    }

    /// Locks and validates output capacity before adapter validation begins.
    pub(crate) fn begin_output_commit(&self) -> SchedulerResult<OutputCommitGuard<'_>> {
        let slot = slot_for(&self.table, self.key)?;
        let state = lock_bound(slot, self.key)?;
        validate_publication(&state, true)?;
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        let held = HeldCommit::new(
            state,
            &slot.notify,
            &self.table.pending_terminals,
            &self.table,
        );
        #[cfg(not(any(test, feature = "actor-stress-instrumentation")))]
        let held = HeldCommit::new(state, &slot.notify, &self.table.pending_terminals);
        Ok(OutputCommitGuard { held })
    }

    /// Locks a terminal-only publication independently of output fullness.
    pub(crate) fn begin_terminal_commit(&self) -> SchedulerResult<TerminalCommitGuard<'_>> {
        let slot = slot_for(&self.table, self.key)?;
        let state = lock_bound(slot, self.key)?;
        validate_publication(&state, false)?;
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        let held = HeldCommit::new(
            state,
            &slot.notify,
            &self.table.pending_terminals,
            &self.table,
        );
        #[cfg(not(any(test, feature = "actor-stress-instrumentation")))]
        let held = HeldCommit::new(state, &slot.notify, &self.table.pending_terminals);
        Ok(TerminalCommitGuard { held })
    }

    pub(crate) fn wake_hook(&self) -> EndpointWake {
        EndpointWake {
            table: Arc::clone(&self.table),
            key: self.key,
            control: self.control.clone(),
        }
    }

    /// Detaches and drains a receiver after its generation-safe atomic
    /// disconnection signal is visible.
    ///
    /// Receiver Drop never waits for this mutex: a commit guard may be holding
    /// it across adapter validation. The engine calls this method after its
    /// current-record control scan observes `disconnected`, so an old receiver
    /// can neither block nor mutate a rebound endpoint generation.
    pub(crate) fn settle_disconnected(&self) -> SchedulerResult<EndpointDiscardReport> {
        if !self.control.fresh_snapshot()?.disconnected() {
            return Err(SchedulerError::internal(
                "request endpoint receiver is not disconnected",
            ));
        }
        let slot = slot_for(&self.table, self.key)?;
        let mut state = lock_bound(slot, self.key)?;
        if !self.control.fresh_snapshot()?.disconnected() {
            return Err(SchedulerError::internal(
                "request endpoint disconnection changed unexpectedly",
            ));
        }
        if !state.terminal_published || state.producer_open {
            return Err(SchedulerError::internal(
                "request endpoint must be terminal before disconnection settlement",
            ));
        }
        let output = state.discarded_output_span()?;
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        let eof_before = state.output_eof_acknowledged;
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        let request_id = state.expected_request_id()?;
        let report = state.discard_receiver_payload(output, &self.table.pending_terminals)?;
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        let eof = (!eof_before && state.output_eof_acknowledged)
            .then_some(ActorSemanticObservation::output_eof(request_id));
        drop(state);
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        append_semantic_observations(&self.table, &[eof]);
        slot.notify.notify_waiters();
        Ok(report)
    }
}

/// A held endpoint lock with one prevalidated output position.
pub(crate) struct OutputCommitGuard<'a> {
    held: HeldCommit<'a>,
}

impl fmt::Debug for OutputCommitGuard<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OutputCommitGuard")
            .field("identity", &"<redacted>")
            .finish()
    }
}

impl<'a> OutputCommitGuard<'a> {
    /// Reports whether publishing the reserved event will exhaust capacity.
    pub(crate) fn will_be_full_after_publish(&self) -> bool {
        let Some(state) = self.held.state.as_ref() else {
            debug_assert!(false, "endpoint commit guard lost its held state");
            return true;
        };
        debug_assert!(state.output.len() < state.logical_capacity);
        state.output.len() + 1 >= state.logical_capacity
    }

    /// Validates and retains the exact event before model state may be applied.
    pub(crate) fn validate_planned_event(
        self,
        event: OutputEvent,
    ) -> SchedulerResult<ValidatedOutputCommitGuard<'a>> {
        let state = self
            .held
            .state
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("endpoint commit guard lost held state"))?;
        validate_next_output_identity(state, event)?;
        Ok(ValidatedOutputCommitGuard {
            held: self.held,
            event,
        })
    }
}

/// An output guard carrying the sole prevalidated event it may publish.
pub(crate) struct ValidatedOutputCommitGuard<'a> {
    held: HeldCommit<'a>,
    event: OutputEvent,
}

impl fmt::Debug for ValidatedOutputCommitGuard<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedOutputCommitGuard")
            .field("identity", &"<redacted>")
            .finish()
    }
}

impl ValidatedOutputCommitGuard<'_> {
    /// Publishes the retained event with no allocation or error path.
    pub(crate) fn publish_output(self) {
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        self.held.publish(Some(self.event), None);
        #[cfg(not(any(test, feature = "actor-stress-instrumentation")))]
        {
            let event = self.event;
            self.held.finish(|state, _pending_terminals| {
                debug_assert!(state.output.len() < state.logical_capacity);
                debug_assert!(state.output.capacity() >= state.logical_capacity);
                debug_assert!(validate_next_output_identity(state, event).is_ok());
                state.output.push_back(event);
                state.published_output_events += 1;
                debug_assert!(state.validate_conservation().is_ok());
            });
        }
    }

    /// Publishes one event and its terminal result in the same locked commit.
    pub(crate) fn publish_output_and_terminal(self, terminal: TerminalResult) {
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        self.held.publish(Some(self.event), Some(terminal));
        #[cfg(not(any(test, feature = "actor-stress-instrumentation")))]
        {
            let event = self.event;
            self.held.finish(|state, pending_terminals| {
                debug_assert!(state.output.len() < state.logical_capacity);
                debug_assert!(state.output.capacity() >= state.logical_capacity);
                debug_assert!(validate_next_output_identity(state, event).is_ok());
                state.output.push_back(event);
                state.published_output_events += 1;
                publish_terminal(state, terminal, pending_terminals);
            });
        }
    }
}

/// A held endpoint lock for a terminal that emits no output event.
pub(crate) struct TerminalCommitGuard<'a> {
    held: HeldCommit<'a>,
}

impl fmt::Debug for TerminalCommitGuard<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TerminalCommitGuard")
            .field("identity", &"<redacted>")
            .finish()
    }
}

impl TerminalCommitGuard<'_> {
    /// Publishes terminal state with no queue-capacity dependency.
    pub(crate) fn publish_terminal(self, terminal: TerminalResult) {
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        self.held.publish(None, Some(terminal));
        #[cfg(not(any(test, feature = "actor-stress-instrumentation")))]
        self.held.finish(|state, pending_terminals| {
            publish_terminal(state, terminal, pending_terminals);
        });
    }
}

struct HeldCommit<'a> {
    state: Option<MutexGuard<'a, EndpointState>>,
    notify: &'a Notify,
    pending_terminals: &'a AtomicUsize,
    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    table: &'a EndpointTable,
}

impl<'a> HeldCommit<'a> {
    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    const fn new(
        state: MutexGuard<'a, EndpointState>,
        notify: &'a Notify,
        pending_terminals: &'a AtomicUsize,
        table: &'a EndpointTable,
    ) -> Self {
        Self {
            state: Some(state),
            notify,
            pending_terminals,
            table,
        }
    }

    #[cfg(not(any(test, feature = "actor-stress-instrumentation")))]
    const fn new(
        state: MutexGuard<'a, EndpointState>,
        notify: &'a Notify,
        pending_terminals: &'a AtomicUsize,
    ) -> Self {
        Self {
            state: Some(state),
            notify,
            pending_terminals,
        }
    }

    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    fn publish(mut self, output: Option<OutputEvent>, terminal: Option<TerminalResult>) {
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        let mut observations = [None, None, None];
        let Some(state) = self.state.as_mut() else {
            debug_assert!(false, "endpoint commit guard lost its held state");
            return;
        };

        if let Some(event) = output {
            debug_assert!(state.output.len() < state.logical_capacity);
            debug_assert!(state.output.capacity() >= state.logical_capacity);
            debug_assert!(validate_next_output_identity(state, event).is_ok());
            state.output.push_back(event);
            state.published_output_events += 1;
            debug_assert!(state.validate_conservation().is_ok());
            #[cfg(any(test, feature = "actor-stress-instrumentation"))]
            {
                observations[0] = Some(ActorSemanticObservation::output(event));
            }
        }

        if let Some(terminal) = terminal {
            #[cfg(any(test, feature = "actor-stress-instrumentation"))]
            let eof_before = state.output_eof_acknowledged;
            publish_terminal(state, terminal, self.pending_terminals);
            #[cfg(any(test, feature = "actor-stress-instrumentation"))]
            {
                let terminal_index = usize::from(output.is_some());
                observations[terminal_index] = Some(ActorSemanticObservation::terminal(terminal));
                if !eof_before && state.output_eof_acknowledged {
                    observations[terminal_index + 1] =
                        Some(ActorSemanticObservation::output_eof(terminal.request_id()));
                }
            }
        }

        // The option is private and initialized exactly once. Taking it makes
        // the endpoint unlock explicit before touching the independent probe.
        drop(self.state.take());
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        append_semantic_observations(self.table, &observations);
        self.notify.notify_waiters();
    }

    #[cfg(not(any(test, feature = "actor-stress-instrumentation")))]
    fn finish(mut self, publish: impl FnOnce(&mut EndpointState, &AtomicUsize)) {
        // These three compile-time call sites are fixed, infallible endpoint
        // mutations. The option permits an explicit unlock before wake-up.
        if let Some(state) = self.state.as_mut() {
            publish(state, self.pending_terminals);
        } else {
            debug_assert!(false, "endpoint commit guard lost its held state");
            return;
        }
        drop(self.state.take());
        self.notify.notify_waiters();
    }
}

fn publish_terminal(
    state: &mut EndpointState,
    terminal: TerminalResult,
    pending_terminals: &AtomicUsize,
) {
    debug_assert!(!state.terminal_published);
    debug_assert!(state.validate_conservation().is_ok());
    debug_assert!(validate_terminal_identity(state, terminal).is_ok());
    state.terminal_published = true;
    state.producer_open = false;
    if state.receiver_connected {
        increment_pending_terminal(pending_terminals);
        state.terminal = Some(terminal);
    } else {
        state.discarded_terminal_results += 1;
        state.terminal_acknowledged = true;
        state.output_eof_acknowledged = true;
    }
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
fn append_semantic_observations(
    table: &EndpointTable,
    observations: &[Option<ActorSemanticObservation>],
) {
    if let Some(recorder) = table.stress_recorder.get() {
        recorder.append_batch(observations);
    }
}

fn increment_pending_terminal(pending_terminals: &AtomicUsize) {
    let previous = pending_terminals.fetch_add(1, Ordering::AcqRel);
    debug_assert!(previous < usize::MAX, "pending terminal count overflowed");
}

fn decrement_pending_terminal(pending_terminals: &AtomicUsize) -> SchedulerResult<()> {
    pending_terminals
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
            count.checked_sub(1)
        })
        .map(|_| ())
        .map_err(|_| SchedulerError::internal("pending terminal count underflows"))
}

/// The sole consumer for one endpoint generation.
pub(crate) struct EndpointReceiver {
    table: Arc<EndpointTable>,
    key: SlotKey,
    control: Option<ControlBinding>,
    locally_connected: bool,
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
struct TryPopMutation {
    result: TryPop,
    drained_before: usize,
    drained_after: usize,
    consumed_output: Option<OutputEvent>,
}

impl fmt::Debug for EndpointReceiver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EndpointReceiver")
            .field("identity", &"<redacted>")
            .field("connected", &self.locally_connected)
            .finish()
    }
}

impl EndpointReceiver {
    pub(crate) fn snapshot(&self) -> SchedulerResult<EndpointSnapshot> {
        snapshot_for(&self.table, self.key)
    }

    /// Pops one event, distinguishes temporary emptiness from acknowledged EOF.
    #[cfg(not(any(test, feature = "actor-stress-instrumentation")))]
    pub(crate) fn try_pop(&mut self) -> SchedulerResult<TryPop> {
        self.ensure_connected()?;
        let slot = slot_for(&self.table, self.key)?;
        let mut state = lock_bound(slot, self.key)?;
        state.validate_conservation()?;
        if let Some(event) = state.output.front().copied() {
            let request_id = state.expected_request_id()?;
            let expected_index = state
                .drained_output_events
                .checked_add(state.discarded_output_events)
                .ok_or_else(|| {
                    SchedulerError::internal("drained request output identity overflows")
                })?;
            if event.request_id() != request_id || event.output_index() != expected_index {
                return Err(SchedulerError::internal(
                    "drained request output has an unexpected identity",
                ));
            }
            let event = state.output.pop_front().ok_or_else(|| {
                SchedulerError::internal("prevalidated request output disappeared")
            })?;
            state.drained_output_events = state
                .drained_output_events
                .checked_add(1)
                .ok_or_else(|| SchedulerError::internal("drained output count overflows"))?;
            debug_assert!(state.validate_conservation().is_ok());
            drop(state);
            slot.notify.notify_waiters();
            return Ok(TryPop::Event(event));
        }
        if !state.producer_open {
            state.output_eof_acknowledged = true;
            if state.terminal_acknowledged {
                state.receiver_connected = false;
                self.locally_connected = false;
            }
            drop(state);
            slot.notify.notify_waiters();
            return Ok(TryPop::Eof);
        }
        Ok(TryPop::Empty)
    }

    /// Pops one event while retaining semantic and structural observations.
    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    pub(crate) fn try_pop(&mut self) -> SchedulerResult<TryPop> {
        self.try_pop_mutation().map(|mutation| mutation.result)
    }

    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    pub(crate) fn try_pop_with_stress_witness(
        &mut self,
        kind: ActorTryPopKind,
    ) -> (SchedulerResult<TryPop>, ActorTryPopWitness) {
        let slot_index = self.key.index();
        let slot_generation = self.key.generation().get();
        match self.try_pop_mutation() {
            Ok(mutation) => (
                Ok(mutation.result),
                ActorTryPopWitness {
                    kind,
                    boundary_reached: true,
                    slot_index,
                    slot_generation,
                    drained_before: mutation.drained_before,
                    drained_after: mutation.drained_after,
                    consumed_output: mutation.consumed_output,
                },
            ),
            Err(error) => (Err(error), ActorTryPopWitness::sentinel(kind)),
        }
    }

    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    fn try_pop_mutation(&mut self) -> SchedulerResult<TryPopMutation> {
        self.ensure_connected()?;
        let slot = slot_for(&self.table, self.key)?;
        let mut state = lock_bound(slot, self.key)?;
        state.validate_conservation()?;
        let drained_before = state.drained_output_events;
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        let mut eof_observation = None;
        let mut notify = false;
        let mut consumed_output = None;
        if let Some(event) = state.output.front().copied() {
            let request_id = state.expected_request_id()?;
            let expected_index = state
                .drained_output_events
                .checked_add(state.discarded_output_events)
                .ok_or_else(|| {
                    SchedulerError::internal("drained request output identity overflows")
                })?;
            if event.request_id() != request_id || event.output_index() != expected_index {
                return Err(SchedulerError::internal(
                    "drained request output has an unexpected identity",
                ));
            }
            let event = state.output.pop_front().ok_or_else(|| {
                SchedulerError::internal("prevalidated request output disappeared")
            })?;
            state.drained_output_events = state
                .drained_output_events
                .checked_add(1)
                .ok_or_else(|| SchedulerError::internal("drained output count overflows"))?;
            debug_assert!(state.validate_conservation().is_ok());
            consumed_output = Some(event);
            notify = true;
        } else if !state.producer_open {
            #[cfg(any(test, feature = "actor-stress-instrumentation"))]
            if !state.output_eof_acknowledged {
                eof_observation = Some(ActorSemanticObservation::output_eof(
                    state.expected_request_id()?,
                ));
            }
            state.output_eof_acknowledged = true;
            if state.terminal_acknowledged {
                state.receiver_connected = false;
                self.locally_connected = false;
            }
            notify = true;
        }
        let drained_after = state.drained_output_events;
        let result = match consumed_output {
            Some(event) => TryPop::Event(event),
            None if !state.producer_open => TryPop::Eof,
            None => TryPop::Empty,
        };
        drop(state);
        #[cfg(any(test, feature = "actor-stress-instrumentation"))]
        append_semantic_observations(&self.table, &[eof_observation]);
        if notify {
            slot.notify.notify_waiters();
        }
        Ok(TryPopMutation {
            result,
            drained_before,
            drained_after,
            consumed_output,
        })
    }

    /// Takes and acknowledges the terminal result independently of output.
    pub(crate) fn take_terminal(&mut self) -> SchedulerResult<Option<TerminalResult>> {
        self.ensure_connected()?;
        let slot = slot_for(&self.table, self.key)?;
        let mut state = lock_bound(slot, self.key)?;
        state.validate_conservation()?;
        let pending_terminal = state.terminal.as_ref().copied();
        if let Some(terminal) = pending_terminal {
            validate_terminal_identity(&state, terminal)?;
            decrement_pending_terminal(&self.table.pending_terminals)?;
        }
        let terminal = state.terminal.take();
        if terminal.is_some() {
            debug_assert!(pending_terminal.is_some());
            state.terminal_acknowledged = true;
            if state.output_eof_acknowledged && state.output.is_empty() {
                state.receiver_connected = false;
                self.locally_connected = false;
            }
        }
        drop(state);
        if terminal.is_some() {
            slot.notify.notify_waiters();
        }
        Ok(terminal)
    }

    /// Makes control disconnection visible before possibly waiting on commit.
    pub(crate) fn disconnect(&mut self) -> SchedulerResult<()> {
        if !self.locally_connected {
            return Ok(());
        }
        self.control
            .as_ref()
            .ok_or_else(|| SchedulerError::internal("request endpoint control is unavailable"))?
            .disconnect()?;
        self.locally_connected = false;
        self.notify_actor();
        Ok(())
    }

    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    pub(crate) fn disconnect_with_stress_witness(
        &mut self,
    ) -> (
        SchedulerResult<crate::control::DisconnectDisposition>,
        Option<crate::control::ActorControlCasWitness>,
    ) {
        if !self.locally_connected {
            return (Err(SchedulerError::request_not_found()), None);
        }
        let (result, witness) = match self.control.as_ref() {
            Some(control) => control.disconnect_with_stress_witness(),
            None => {
                self.locally_connected = false;
                self.notify_actor();
                return (
                    Err(SchedulerError::internal(
                        "request endpoint control is unavailable",
                    )),
                    None,
                );
            }
        };
        // This hidden method represents the sole destructor attempt. Consume
        // the local obligation even on a stale/error result so Drop cannot
        // perform an unobserved second load/CAS attempt.
        self.locally_connected = false;
        self.notify_actor();
        (result, Some(witness))
    }

    /// Waits without losing a wake between predicate inspection and parking.
    pub(crate) async fn wait_until_readable(&self) -> SchedulerResult<EndpointSnapshot> {
        self.ensure_connected()?;
        let slot = slot_for(&self.table, self.key)?;
        loop {
            let notified = slot.notify.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            {
                let state = lock_bound(slot, self.key)?;
                if !state.output.is_empty()
                    || state.terminal.is_some()
                    || !state.producer_open
                    || state.shutdown
                {
                    return state.snapshot();
                }
            }
            notified.await;
        }
    }

    /// Waits specifically for terminal publication, ignoring buffered output.
    ///
    /// This prevents a terminal-only consumer from spinning when output is
    /// already readable but the producer has not published its terminal slot.
    pub(crate) async fn wait_until_terminal(&self) -> SchedulerResult<EndpointSnapshot> {
        self.ensure_connected()?;
        let slot = slot_for(&self.table, self.key)?;
        loop {
            let notified = slot.notify.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            {
                let state = lock_bound(slot, self.key)?;
                if state.terminal_published || !state.producer_open || state.shutdown {
                    return state.snapshot();
                }
            }
            notified.await;
        }
    }

    fn ensure_connected(&self) -> SchedulerResult<()> {
        if self.locally_connected {
            Ok(())
        } else {
            Err(SchedulerError::request_not_found())
        }
    }

    fn publish_control_disconnect(&self) {
        if let Some(control) = self.control.as_ref() {
            let _ = control.disconnect();
        }
    }

    fn notify_actor(&self) {
        if let Some(slot) = self.table.slots.get(self.key.index()) {
            slot.notify.notify_waiters();
        }
    }
}

impl Drop for EndpointReceiver {
    fn drop(&mut self) {
        if !self.locally_connected {
            return;
        }
        // This atomic signal must precede the mutex acquisition: a commit may
        // be holding the endpoint lock while it performs adapter validation.
        // Drop deliberately never acquires that mutex; the actor settles the
        // current generation after observing this tagged control signal.
        self.publish_control_disconnect();
        self.notify_actor();
        self.locally_connected = false;
    }
}

/// Allocation-free actor-side waiter derived from a producer.
#[derive(Clone)]
pub(crate) struct EndpointWake {
    table: Arc<EndpointTable>,
    key: SlotKey,
    control: ControlBinding,
}

impl fmt::Debug for EndpointWake {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EndpointWake")
            .field("identity", &"<redacted>")
            .finish()
    }
}

impl EndpointWake {
    /// Waits until output capacity exists or publication can no longer proceed.
    pub(crate) async fn wait_until_writable(&self) -> SchedulerResult<EndpointSnapshot> {
        let slot = slot_for(&self.table, self.key)?;
        loop {
            let notified = slot.notify.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            {
                let state = lock_bound(slot, self.key)?;
                // ControlBinding is a generation-tagged atomic read. It takes
                // no registry lock, so reading it under the endpoint mutex
                // cannot invert lifecycle lock order.
                let control = self.control.fresh_snapshot()?;
                if control.cancelled()
                    || control.disconnected()
                    || state.output.len() < state.logical_capacity
                    || !state.receiver_connected
                    || !state.producer_open
                    || state.shutdown
                {
                    return state.snapshot();
                }
            }
            notified.await;
        }
    }
}

fn validate_next_output_identity(state: &EndpointState, event: OutputEvent) -> SchedulerResult<()> {
    state.validate_conservation()?;
    if event.request_id() != state.expected_request_id()?
        || event.output_index() != state.published_output_events
    {
        return Err(SchedulerError::internal(
            "planned request output has an unexpected identity",
        ));
    }
    Ok(())
}

fn validate_terminal_identity(
    state: &EndpointState,
    terminal: TerminalResult,
) -> SchedulerResult<()> {
    state.validate_conservation()?;
    if terminal.request_id() != state.expected_request_id()?
        || terminal.emitted_tokens() != state.published_output_events
    {
        return Err(SchedulerError::internal(
            "request terminal has an unexpected identity or output count",
        ));
    }
    Ok(())
}

fn validate_publication(state: &EndpointState, needs_output: bool) -> SchedulerResult<()> {
    state.validate_conservation()?;
    if state.shutdown || !state.producer_open {
        return Err(SchedulerError::scheduler_closed());
    }
    if state.terminal_published {
        return Err(SchedulerError::internal(
            "request endpoint terminal was already published",
        ));
    }
    if needs_output {
        if !state.receiver_connected {
            return Err(SchedulerError::cancelled());
        }
        if state.output.capacity() < state.logical_capacity {
            return Err(SchedulerError::internal(
                "request endpoint reserved capacity changed",
            ));
        }
        if state.output.len() >= state.logical_capacity {
            return Err(SchedulerError::resource_exhausted(
                "request output endpoint",
                usize_to_u64(state.output.len()).saturating_add(1),
                usize_to_u64(state.logical_capacity),
            ));
        }
    }
    Ok(())
}

fn snapshot_for(table: &EndpointTable, key: SlotKey) -> SchedulerResult<EndpointSnapshot> {
    let slot = slot_for(table, key)?;
    lock_bound(slot, key)?.snapshot()
}

fn slot_for(table: &EndpointTable, key: SlotKey) -> SchedulerResult<&EndpointSlot> {
    table
        .slots
        .get(key.index())
        .ok_or_else(SchedulerError::request_not_found)
}

fn lock_state(slot: &EndpointSlot) -> SchedulerResult<MutexGuard<'_, EndpointState>> {
    slot.state
        .lock()
        .map_err(|_| SchedulerError::internal("request endpoint state is unavailable"))
}

fn lock_bound<'a>(
    slot: &'a EndpointSlot,
    key: SlotKey,
) -> SchedulerResult<MutexGuard<'a, EndpointState>> {
    let state = lock_state(slot)?;
    if state.key == Some(key) {
        Ok(state)
    } else {
        Err(SchedulerError::request_not_found())
    }
}

fn try_reserve_vec<T>(
    values: &mut Vec<T>,
    count: usize,
    resource: &'static str,
) -> SchedulerResult<()> {
    let bytes = allocation_bytes::<T>(count, resource)?;
    values
        .try_reserve_exact(count)
        .map_err(|_| SchedulerError::allocation_failure(resource, bytes))
}

fn try_reserve_deque<T>(
    values: &mut VecDeque<T>,
    count: usize,
    resource: &'static str,
) -> SchedulerResult<()> {
    let bytes = allocation_bytes::<T>(count, resource)?;
    values
        .try_reserve_exact(count)
        .map_err(|_| SchedulerError::allocation_failure(resource, bytes))
}

fn allocation_bytes<T>(count: usize, resource: &'static str) -> SchedulerResult<u64> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| SchedulerError::allocation_failure(resource, u64::MAX))?;
    if bytes > isize::MAX as usize {
        return Err(SchedulerError::allocation_failure(
            resource,
            usize_to_u64(bytes),
        ));
    }
    Ok(usize_to_u64(bytes))
}

const fn usize_to_u64(value: usize) -> u64 {
    if size_of::<usize>() > size_of::<u64>() && value > u64::MAX as usize {
        u64::MAX
    } else {
        value as u64
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Barrier, mpsc},
        thread,
        time::Duration,
    };

    use crate::{
        control::ControlRegistry,
        error::ErrorCategory,
        id::{request_id_for_test, slot_generation_for_test},
        request::TerminalOutcome,
    };

    use super::*;

    fn key(index: usize, generation: u64) -> SlotKey {
        SlotKey::new(index, slot_generation_for_test(generation))
    }

    fn output(index: usize) -> OutputEvent {
        OutputEvent::new(request_id_for_test(1), index, 77)
    }

    fn terminal(emitted: usize) -> TerminalResult {
        TerminalResult::new(
            request_id_for_test(1),
            TerminalOutcome::Completed,
            emitted,
            emitted,
        )
    }

    fn publish(producer: &EndpointProducer, event: OutputEvent) {
        producer
            .begin_output_commit()
            .expect("output guard")
            .validate_planned_event(event)
            .expect("validate output identity")
            .publish_output();
    }

    fn publish_and_terminal(
        producer: &EndpointProducer,
        event: OutputEvent,
        terminal: TerminalResult,
    ) {
        producer
            .begin_output_commit()
            .expect("output guard")
            .validate_planned_event(event)
            .expect("validate output identity")
            .publish_output_and_terminal(terminal);
    }

    fn assert_conserved(snapshot: EndpointSnapshot) {
        assert_eq!(
            snapshot.published_output_events,
            snapshot.drained_output_events
                + snapshot.discarded_output_events
                + snapshot.buffered_output_events
        );
    }

    fn bind(
        registry: &mut EndpointRegistry,
        controls: &mut ControlRegistry,
        key: SlotKey,
        capacity: usize,
    ) -> (EndpointProducer, EndpointReceiver) {
        let prepared = registry.prepare(capacity).expect("prepare endpoint");
        let endpoint = registry.begin_bind(prepared, key).expect("begin bind");
        let control = controls
            .bind(controls.prepare().expect("prepare control"))
            .expect("bind control");
        endpoint.commit(control, request_id_for_test(1))
    }

    fn install_recorder(registry: &EndpointRegistry, capacity: usize) -> ActorStressRecorder {
        let recorder = ActorStressRecorder::try_with_capacity(capacity).expect("recorder");
        registry
            .install_actor_stress_recorder(recorder.clone())
            .expect("install recorder");
        recorder
    }

    #[test]
    fn primary_and_opportunistic_pop_witnesses_preserve_the_exact_boundary() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let recorder = install_recorder(&registry, 8);
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, mut receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);
        publish_and_terminal(&producer, output(0), terminal(1));

        let (primary, primary_witness) =
            receiver.try_pop_with_stress_witness(ActorTryPopKind::Primary);
        assert!(matches!(primary, Ok(TryPop::Event(event)) if event.output_index() == 0));
        assert_eq!(primary_witness.kind(), ActorTryPopKind::Primary);
        assert!(primary_witness.boundary_reached());
        assert_eq!(primary_witness.slot_index(), 0);
        assert_eq!(primary_witness.slot_generation(), 1);
        assert_eq!(primary_witness.drained_before(), 0);
        assert_eq!(primary_witness.drained_after(), 1);
        assert_eq!(
            primary_witness
                .consumed_output()
                .map(OutputEvent::output_index),
            Some(0)
        );

        let (opportunistic, opportunistic_witness) =
            receiver.try_pop_with_stress_witness(ActorTryPopKind::OpportunisticEof);
        assert!(matches!(opportunistic, Ok(TryPop::Eof)));
        assert_eq!(
            opportunistic_witness.kind(),
            ActorTryPopKind::OpportunisticEof
        );
        assert_eq!(opportunistic_witness.drained_before(), 1);
        assert_eq!(opportunistic_witness.drained_after(), 1);
        assert_eq!(opportunistic_witness.consumed_output(), None);

        let (repeat, repeat_witness) =
            receiver.try_pop_with_stress_witness(ActorTryPopKind::Primary);
        assert!(matches!(repeat, Ok(TryPop::Eof)));
        assert_eq!(repeat_witness.drained_before(), 1);
        assert_eq!(repeat_witness.drained_after(), 1);

        let recording = recorder.recording().expect("recording");
        let kinds = recording
            .observations()
            .iter()
            .copied()
            .map(ActorSemanticObservation::kind)
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            vec![
                ActorSemanticObservationKind::Output,
                ActorSemanticObservationKind::Terminal,
                ActorSemanticObservationKind::OutputEof,
            ]
        );
        assert!(recording.status().healthy());
    }

    #[test]
    fn pop_error_before_the_endpoint_boundary_uses_only_zero_sentinels() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (_producer, mut receiver) = bind(&mut registry, &mut controls, key(0, 7), 1);
        receiver.disconnect().expect("disconnect receiver");

        let (result, witness) = receiver.try_pop_with_stress_witness(ActorTryPopKind::Primary);
        assert_eq!(
            result
                .expect_err("locally disconnected pop must fail")
                .category(),
            ErrorCategory::InvalidRequest
        );
        assert!(!witness.boundary_reached());
        assert_eq!(witness.slot_index(), 0);
        assert_eq!(witness.slot_generation(), 0);
        assert_eq!(witness.drained_before(), 0);
        assert_eq!(witness.drained_after(), 0);
        assert_eq!(witness.consumed_output(), None);
    }

    #[test]
    fn disconnect_before_terminal_records_terminal_and_discard_eof_once() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let recorder = install_recorder(&registry, 4);
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, mut receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);

        let (disconnect, witness) = receiver.disconnect_with_stress_witness();
        disconnect.expect("disconnect");
        let witness = witness.expect("disconnect boundary");
        assert!(witness.boundary_reached());
        producer
            .begin_terminal_commit()
            .expect("terminal guard")
            .publish_terminal(terminal(0));
        let discarded = producer
            .settle_disconnected()
            .expect("settle disconnected receiver");
        assert_eq!(discarded.output_events(), 0);
        assert_eq!(discarded.terminal_results, 1);

        let recording = recorder.recording().expect("recording");
        let kinds = recording
            .observations()
            .iter()
            .copied()
            .map(ActorSemanticObservation::kind)
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            vec![
                ActorSemanticObservationKind::Terminal,
                ActorSemanticObservationKind::OutputEof,
            ]
        );
        assert!(recording.status().healthy());
    }

    #[test]
    fn witnessed_disconnect_error_consumes_the_drop_obligation_exactly_once() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, mut receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);
        let stale_control = producer.control.clone();
        stale_control
            .mark_terminal()
            .expect("terminal control publication");
        controls.recycle(&stale_control).expect("recycle control");
        let current = controls
            .bind(controls.prepare().expect("prepare current control"))
            .expect("bind current control");

        let (result, witness) = receiver.disconnect_with_stress_witness();
        assert_eq!(
            result
                .expect_err("stale witnessed disconnect must fail")
                .category(),
            ErrorCategory::InvalidRequest
        );
        assert!(witness.expect("stale load witness").boundary_reached());
        assert!(!receiver.locally_connected);
        let before_drop = current.fresh_snapshot().expect("current before drop");
        drop(receiver);
        assert_eq!(
            current.fresh_snapshot().expect("current after drop"),
            before_drop
        );
        assert_eq!(before_drop, crate::control::ControlSnapshot::default());
    }

    #[test]
    fn shutdown_discard_records_the_first_eof_transition() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let recorder = install_recorder(&registry, 4);
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let slot_key = key(0, 1);
        let (producer, _receiver) = bind(&mut registry, &mut controls, slot_key, 1);
        publish_and_terminal(&producer, output(0), terminal(1));

        let discarded = registry
            .shutdown_discard(slot_key)
            .expect("shutdown discard");
        assert_eq!(discarded.output_events(), 1);
        assert_eq!(discarded.terminal_results, 1);
        let recording = recorder.recording().expect("recording");
        let kinds = recording
            .observations()
            .iter()
            .copied()
            .map(ActorSemanticObservation::kind)
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            vec![
                ActorSemanticObservationKind::Output,
                ActorSemanticObservationKind::Terminal,
                ActorSemanticObservationKind::OutputEof,
            ]
        );
    }

    #[test]
    fn recorder_overflow_is_sticky_and_never_grows_its_allocation() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let recorder = install_recorder(&registry, 2);
        let initial = recorder.status();
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, mut receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);
        publish_and_terminal(&producer, output(0), terminal(1));
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Event(_))));
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Eof)));
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Eof)));

        let status = recorder.status();
        assert_eq!(status.observation_count(), 2);
        assert_eq!(status.observation_limit(), 2);
        assert_eq!(status.allocated_capacity(), initial.allocated_capacity());
        assert!(status.overflowed());
        assert!(!status.poisoned());
        assert!(!status.healthy());
        let recording = recorder.recording().expect("overflow prefix");
        assert_eq!(recording.observations().len(), 2);
        assert!(recording.status().overflowed());
    }

    #[test]
    fn recorder_poison_is_sticky_and_visible_without_exposing_contents() {
        let recorder = ActorStressRecorder::try_with_capacity(1).expect("recorder");
        let poison = recorder.clone();
        let join = thread::spawn(move || {
            let _held = poison.inner.state.lock().expect("recorder lock");
            panic!("intentional recorder poison");
        });
        assert!(join.join().is_err());
        let status = recorder.status();
        assert!(status.poisoned());
        assert!(!status.healthy());
        assert!(recorder.status().poisoned());
        assert!(!format!("{recorder:?}").contains("RequestId"));
    }

    #[test]
    fn construction_and_prepare_fail_cleanly_and_prepare_is_nonmutating() {
        let zero = EndpointRegistry::try_with_capacity(0).expect_err("zero must fail");
        assert_eq!(zero.category(), ErrorCategory::InvalidRequest);
        let impossible = EndpointRegistry::try_with_capacity(usize::MAX)
            .expect_err("impossible table must fail");
        assert_eq!(impossible.category(), ErrorCategory::ResourceExhausted);

        let registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let prepared = registry.prepare(3).expect("prepare");
        assert!(prepared.output.capacity() >= 3);
        assert_eq!(registry.available(), 1);
        drop(prepared);
        assert_eq!(registry.available(), 1);
    }

    #[test]
    fn dropped_begin_bind_guard_rolls_back_after_control_bind_failure() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let foreign_controls = ControlRegistry::try_with_capacity(1).expect("foreign controls");
        let slot_key = key(0, 1);

        let prepared = registry.prepare(2).expect("prepare endpoint");
        let endpoint = registry
            .begin_bind(prepared, slot_key)
            .expect("begin endpoint bind");
        let foreign = foreign_controls.prepare().expect("foreign preparation");
        let error = controls
            .bind(foreign)
            .expect_err("foreign control bind must fail");
        assert_eq!(error.category(), ErrorCategory::Internal);
        drop(endpoint);

        assert_eq!(registry.available(), 1);
        let (producer, receiver) = bind(&mut registry, &mut controls, slot_key, 2);
        assert_eq!(registry.available(), 0);
        assert_eq!(
            producer
                .snapshot()
                .expect("bound after rollback")
                .output_capacity,
            2
        );
        drop(receiver);
    }

    #[test]
    fn dropped_begin_recycle_guard_rolls_back_after_control_recycle_failure() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let slot_key = key(0, 1);
        let (producer, mut receiver) = bind(&mut registry, &mut controls, slot_key, 1);
        producer
            .begin_terminal_commit()
            .expect("terminal guard")
            .publish_terminal(terminal(0));
        assert!(receiver.take_terminal().expect("take terminal").is_some());
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Eof)));

        let endpoint = registry
            .begin_recycle(slot_key)
            .expect("begin endpoint recycle");
        let error = controls
            .recycle(&producer.control)
            .expect_err("active control recycle must fail");
        assert_eq!(error.category(), ErrorCategory::Internal);
        drop(endpoint);

        let snapshot = registry.snapshot(slot_key).expect("bound after rollback");
        assert!(snapshot.reap_ready);
        assert_eq!(registry.available(), 0);

        producer
            .control
            .mark_terminal()
            .expect("mark control terminal");
        let endpoint = registry
            .begin_recycle(slot_key)
            .expect("retry endpoint recycle");
        controls
            .recycle(&producer.control)
            .expect("recycle terminal control");
        endpoint.commit();
        assert_eq!(registry.available(), 1);
    }

    #[test]
    fn logical_capacity_is_exact_even_when_allocator_capacity_is_larger() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, mut receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);

        publish(&producer, output(0));
        let full = producer
            .begin_output_commit()
            .expect_err("one over logical capacity must fail");
        assert_eq!(full.category(), ErrorCategory::ResourceExhausted);
        assert_eq!(
            producer
                .snapshot()
                .expect("snapshot")
                .buffered_output_events,
            1
        );
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Event(_))));
    }

    #[test]
    fn dropped_guard_rolls_back_and_committed_guard_publishes_once() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, mut receiver) = bind(&mut registry, &mut controls, key(0, 1), 2);

        drop(producer.begin_output_commit().expect("guard"));
        assert_eq!(
            producer
                .snapshot()
                .expect("snapshot")
                .buffered_output_events,
            0
        );
        publish(&producer, output(0));
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Event(_))));
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Empty)));
    }

    #[test]
    fn output_guard_may_atomically_add_terminal_after_sampling() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, mut receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);

        publish_and_terminal(&producer, output(0), terminal(1));
        let snapshot = producer.snapshot().expect("snapshot");
        assert_eq!(snapshot.buffered_output_events, 1);
        assert!(snapshot.terminal_published);
        assert!(!snapshot.producer_open);
        assert!(receiver.take_terminal().expect("terminal").is_some());
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Event(_))));
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Eof)));
    }

    #[test]
    fn planned_output_identity_is_validated_before_publication() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, _receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);

        let foreign = producer
            .begin_output_commit()
            .expect("foreign guard")
            .validate_planned_event(OutputEvent::new(request_id_for_test(2), 0, 77))
            .expect_err("foreign request identity must fail");
        assert_eq!(foreign.category(), ErrorCategory::Internal);
        let skipped = producer
            .begin_output_commit()
            .expect("skipped-index guard")
            .validate_planned_event(output(1))
            .expect_err("skipped output index must fail");
        assert_eq!(skipped.category(), ErrorCategory::Internal);
        let snapshot = producer.snapshot().expect("unchanged endpoint");
        assert_eq!(snapshot.published_output_events, 0);
        assert_eq!(snapshot.buffered_output_events, 0);
    }

    #[test]
    fn terminal_publication_is_independent_of_a_full_output_queue() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, _receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);
        publish(&producer, output(0));
        assert!(producer.begin_output_commit().is_err());
        producer
            .begin_terminal_commit()
            .expect("terminal guard despite full output")
            .publish_terminal(terminal(1));
        assert!(producer.snapshot().expect("snapshot").terminal_published);
    }

    #[test]
    fn terminal_ack_and_output_eof_ack_are_distinct_reap_gates() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let slot_key = key(0, 1);
        let (producer, mut receiver) = bind(&mut registry, &mut controls, slot_key, 1);
        publish_and_terminal(&producer, output(0), terminal(1));

        assert!(receiver.take_terminal().expect("terminal").is_some());
        assert!(!registry.snapshot(slot_key).expect("snapshot").reap_ready);
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Event(_))));
        assert!(!registry.snapshot(slot_key).expect("snapshot").reap_ready);
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Eof)));
        assert!(registry.snapshot(slot_key).expect("snapshot").reap_ready);
        registry
            .begin_recycle(slot_key)
            .expect("begin recycle")
            .commit();
    }

    #[test]
    fn receiver_signal_bypasses_but_settlement_waits_behind_held_guard() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, mut receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);
        let held = producer.begin_terminal_commit().expect("held guard");
        let settle_producer = producer.clone();
        let (receiver_done_tx, receiver_done_rx) = mpsc::channel();
        let (settle_done_tx, settle_done_rx) = mpsc::channel();

        let receiver_thread = thread::spawn(move || {
            receiver.disconnect().expect("disconnect receiver");
            receiver_done_tx.send(()).expect("report receiver");
        });
        receiver_done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("receiver signal must not wait for held endpoint mutex");
        let settle_thread = thread::spawn(move || {
            let report = settle_producer
                .settle_disconnected()
                .expect("settle receiver");
            settle_done_tx.send(report).expect("report settlement");
        });
        assert!(
            settle_done_rx
                .recv_timeout(Duration::from_millis(20))
                .is_err()
        );

        held.publish_terminal(terminal(0));
        let report = settle_done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("receiver is eventually settled");
        assert_eq!(report.output_events(), 0);
        assert_eq!(report.terminal_results, 1);
        receiver_thread.join().expect("receiver thread");
        settle_thread.join().expect("settle thread");
    }

    #[test]
    fn stale_handles_cannot_observe_mutate_or_disconnect_rebound_generation() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(2).expect("controls");
        let first_key = key(0, 1);
        let (old_producer, mut old_receiver) = bind(&mut registry, &mut controls, first_key, 1);
        old_producer
            .begin_terminal_commit()
            .expect("terminal guard")
            .publish_terminal(terminal(0));
        assert!(old_receiver.take_terminal().expect("terminal").is_some());
        assert!(matches!(old_receiver.try_pop(), Ok(TryPop::Eof)));
        registry
            .begin_recycle(first_key)
            .expect("begin recycle first")
            .commit();

        let second_key = key(0, 2);
        let (new_producer, new_receiver) = bind(&mut registry, &mut controls, second_key, 1);
        assert_eq!(
            old_producer
                .snapshot()
                .expect_err("stale producer")
                .category(),
            ErrorCategory::InvalidRequest
        );
        assert_eq!(
            old_receiver
                .try_pop()
                .expect_err("stale receiver")
                .category(),
            ErrorCategory::InvalidRequest
        );
        drop(old_receiver);
        assert!(
            new_producer
                .snapshot()
                .expect("new snapshot")
                .receiver_connected
        );
        drop(new_receiver);
    }

    #[test]
    fn exact_generation_cannot_be_rebound_after_recycle() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(2).expect("controls");
        let slot_key = key(0, 1);
        let (producer, mut receiver) = bind(&mut registry, &mut controls, slot_key, 1);
        producer
            .begin_terminal_commit()
            .expect("terminal guard")
            .publish_terminal(terminal(0));
        receiver.take_terminal().expect("take terminal");
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Eof)));
        registry
            .begin_recycle(slot_key)
            .expect("begin recycle")
            .commit();

        let prepared = registry.prepare(1).expect("prepare");
        let error = registry
            .begin_bind(prepared, slot_key)
            .expect_err("same generation must fail");
        assert_eq!(error.category(), ErrorCategory::Internal);
    }

    #[test]
    fn receiver_drop_after_terminal_discards_payload_exactly() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let slot_key = key(0, 1);
        let (producer, receiver) = bind(&mut registry, &mut controls, slot_key, 2);
        publish_and_terminal(&producer, output(0), terminal(1));
        drop(receiver);
        let discarded = producer
            .settle_disconnected()
            .expect("settle disconnected receiver");
        assert_eq!(discarded.output_events(), 1);
        let span = discarded.output.expect("discarded output span");
        assert_eq!(span.request_id(), request_id_for_test(1));
        assert_eq!(span.first_output_index(), 0);
        assert_eq!(span.count(), 1);
        assert_eq!(discarded.terminal_results, 1);
        let disconnected = producer.snapshot().expect("snapshot");
        assert!(!disconnected.receiver_connected);
        assert_eq!(disconnected.discarded_output_events, 1);
        assert_eq!(disconnected.discarded_terminal_results, 1);
        assert!(disconnected.reap_ready);
        let report = registry
            .begin_recycle(slot_key)
            .expect("begin recycle")
            .commit();
        assert_eq!(report.discarded_output_events, 1);
        assert_eq!(report.discarded_terminal_results, 1);
    }

    #[test]
    fn disconnection_settlement_requires_terminal_and_preserves_live_output() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);
        publish(&producer, output(0));
        drop(receiver);

        let error = producer
            .settle_disconnected()
            .expect_err("live endpoint settlement must fail");
        assert_eq!(error.category(), ErrorCategory::Internal);
        let snapshot = producer.snapshot().expect("unchanged endpoint");
        assert_eq!(snapshot.buffered_output_events, 1);
        assert_eq!(snapshot.discarded_output_events, 0);
        assert!(snapshot.receiver_connected);
        assert!(snapshot.producer_open);
        assert_conserved(snapshot);
    }

    #[test]
    fn prefix_drain_and_suffix_discard_record_exact_identities_and_conserve() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let slot_key = key(0, 1);
        let (producer, mut receiver) = bind(&mut registry, &mut controls, slot_key, 3);

        for index in 0..3 {
            publish(&producer, output(index));
            assert_conserved(producer.snapshot().expect("published snapshot"));
        }
        assert!(
            matches!(receiver.try_pop(), Ok(TryPop::Event(event)) if event.output_index() == 0)
        );
        assert_conserved(producer.snapshot().expect("drained snapshot"));
        producer
            .begin_terminal_commit()
            .expect("terminal guard")
            .publish_terminal(terminal(3));
        receiver.disconnect().expect("disconnect receiver");

        let discarded = producer
            .settle_disconnected()
            .expect("settle disconnected receiver");
        let span = discarded.output.expect("discarded suffix");
        assert_eq!(span.request_id(), request_id_for_test(1));
        assert_eq!(span.first_output_index(), 1);
        assert_eq!(span.count(), 2);
        assert_eq!(discarded.terminal_results, 1);
        let snapshot = producer.snapshot().expect("settled snapshot");
        assert_eq!(snapshot.published_output_events, 3);
        assert_eq!(snapshot.drained_output_events, 1);
        assert_eq!(snapshot.discarded_output_events, 2);
        assert_eq!(snapshot.buffered_output_events, 0);
        assert_conserved(snapshot);
        assert!(snapshot.reap_ready);

        let repeated = producer
            .settle_disconnected()
            .expect("repeated settlement is a no-op");
        assert_eq!(repeated, EndpointDiscardReport::default());
        assert_conserved(producer.snapshot().expect("repeated snapshot"));
    }

    fn assert_corrupt_suffix_rejected_without_mutation(corrupt_request_id: bool) {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let slot_key = key(0, 1);
        let (producer, mut receiver) = bind(&mut registry, &mut controls, slot_key, 2);
        publish(&producer, output(0));
        publish_and_terminal(&producer, output(1), terminal(2));
        receiver.disconnect().expect("disconnect receiver");

        let slot = registry.table.slots.first().expect("endpoint slot");
        {
            let mut state = lock_bound(slot, slot_key).expect("bound state");
            let second = state.output.get_mut(1).expect("second queued event");
            *second = if corrupt_request_id {
                OutputEvent::new(request_id_for_test(2), 1, 88)
            } else {
                OutputEvent::new(request_id_for_test(1), 9, 88)
            };
        }

        let error = producer
            .settle_disconnected()
            .expect_err("corrupt suffix must fail closed");
        assert_eq!(error.category(), ErrorCategory::Internal);
        let state = lock_bound(slot, slot_key).expect("state remains bound");
        assert_eq!(state.output.len(), 2);
        assert_eq!(state.discarded_output_events, 0);
        assert!(state.receiver_connected);
        assert!(state.terminal.is_some());
    }

    #[test]
    fn corrupt_request_identity_rejects_discard_without_mutation() {
        assert_corrupt_suffix_rejected_without_mutation(true);
    }

    #[test]
    fn uniformly_foreign_contiguous_suffix_rejects_discard_without_mutation() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let slot_key = key(0, 1);
        let (producer, mut receiver) = bind(&mut registry, &mut controls, slot_key, 2);
        publish(&producer, output(0));
        publish_and_terminal(&producer, output(1), terminal(2));
        receiver.disconnect().expect("disconnect receiver");

        let slot = registry.table.slots.first().expect("endpoint slot");
        {
            let mut state = lock_bound(slot, slot_key).expect("bound state");
            for (index, event) in state.output.iter_mut().enumerate() {
                *event = OutputEvent::new(request_id_for_test(2), index, 99);
            }
        }
        let error = producer
            .settle_disconnected()
            .expect_err("uniformly foreign suffix must fail");
        assert_eq!(error.category(), ErrorCategory::Internal);
        let state = lock_bound(slot, slot_key).expect("unchanged state");
        assert_eq!(state.output.len(), 2);
        assert_eq!(state.discarded_output_events, 0);
        assert!(state.terminal.is_some());
        assert!(state.receiver_connected);
        assert_eq!(registry.pending_terminal_count(), 1);
    }

    #[test]
    fn foreign_pop_and_terminal_fail_without_mutating_endpoint_or_count() {
        let mut registry = EndpointRegistry::try_with_capacity(2).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(2).expect("controls");

        let output_key = key(0, 1);
        let (output_producer, mut output_receiver) =
            bind(&mut registry, &mut controls, output_key, 1);
        publish(&output_producer, output(0));
        {
            let slot = registry.table.slots.first().expect("output slot");
            let mut state = lock_bound(slot, output_key).expect("output state");
            state.output[0] = OutputEvent::new(request_id_for_test(2), 0, 99);
        }
        let pop_error = output_receiver
            .try_pop()
            .expect_err("foreign output must not pop");
        assert_eq!(pop_error.category(), ErrorCategory::Internal);
        {
            let slot = registry.table.slots.first().expect("output slot");
            let state = lock_bound(slot, output_key).expect("output state");
            assert_eq!(state.output.len(), 1);
            assert_eq!(state.drained_output_events, 0);
        }

        let terminal_key = key(1, 1);
        let (terminal_producer, mut terminal_receiver) =
            bind(&mut registry, &mut controls, terminal_key, 1);
        terminal_producer
            .begin_terminal_commit()
            .expect("terminal guard")
            .publish_terminal(terminal(0));
        assert_eq!(registry.pending_terminal_count(), 1);
        {
            let slot = registry.table.slots.get(1).expect("terminal slot");
            let mut state = lock_bound(slot, terminal_key).expect("terminal state");
            state.terminal = Some(TerminalResult::new(
                request_id_for_test(2),
                TerminalOutcome::Completed,
                0,
                0,
            ));
        }
        let terminal_error = terminal_receiver
            .take_terminal()
            .expect_err("foreign terminal must not be taken");
        assert_eq!(terminal_error.category(), ErrorCategory::Internal);
        let slot = registry.table.slots.get(1).expect("terminal slot");
        let state = lock_bound(slot, terminal_key).expect("terminal state");
        assert!(state.terminal.is_some());
        assert!(!state.terminal_acknowledged);
        assert_eq!(registry.pending_terminal_count(), 1);
    }

    #[test]
    fn corrupt_output_index_rejects_discard_without_mutation() {
        assert_corrupt_suffix_rejected_without_mutation(false);
    }

    #[test]
    fn shutdown_discards_payload_and_is_immediately_reap_ready() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let slot_key = key(0, 1);
        let (producer, _receiver) = bind(&mut registry, &mut controls, slot_key, 2);
        publish_and_terminal(&producer, output(0), terminal(1));
        assert_eq!(registry.pending_terminal_count(), 1);
        let discarded = registry.shutdown_discard(slot_key).expect("shutdown");
        assert_eq!(registry.pending_terminal_count(), 0);
        assert_eq!(discarded.output_events(), 1);
        let span = discarded.output.expect("shutdown output span");
        assert_eq!(span.request_id(), request_id_for_test(1));
        assert_eq!(span.first_output_index(), 0);
        assert_eq!(span.count(), 1);
        assert_eq!(discarded.terminal_results, 1);
        let snapshot = registry.snapshot(slot_key).expect("snapshot");
        assert!(snapshot.shutdown);
        assert!(snapshot.reap_ready);
        assert_conserved(snapshot);
        let report = registry
            .begin_recycle(slot_key)
            .expect("begin recycle")
            .commit();
        assert!(report.shutdown);
        assert_eq!(registry.pending_terminal_count(), 0);
    }

    #[test]
    fn pending_terminal_count_is_exact_through_take_discard_and_recycle() {
        let mut registry = EndpointRegistry::try_with_capacity(2).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(2).expect("controls");
        let first_key = key(0, 1);
        let second_key = key(1, 1);
        let (first_producer, mut first_receiver) = bind(&mut registry, &mut controls, first_key, 1);
        let (second_producer, mut second_receiver) =
            bind(&mut registry, &mut controls, second_key, 1);
        assert_eq!(registry.pending_terminal_count(), 0);
        assert_eq!(first_producer.pending_terminal_count(), 0);

        first_producer
            .begin_terminal_commit()
            .expect("first terminal guard")
            .publish_terminal(terminal(0));
        publish_and_terminal(&second_producer, output(0), terminal(1));
        assert_eq!(registry.pending_terminal_count(), 2);
        assert_eq!(second_producer.pending_terminal_count(), 2);

        assert!(
            first_receiver
                .take_terminal()
                .expect("take first terminal")
                .is_some()
        );
        assert_eq!(registry.pending_terminal_count(), 1);
        assert!(matches!(first_receiver.try_pop(), Ok(TryPop::Eof)));
        registry
            .begin_recycle(first_key)
            .expect("first recycle")
            .commit();
        assert_eq!(registry.pending_terminal_count(), 1);

        second_receiver.disconnect().expect("disconnect second");
        let discarded = second_producer
            .settle_disconnected()
            .expect("settle second");
        assert_eq!(discarded.terminal_results, 1);
        assert_eq!(registry.pending_terminal_count(), 0);
        registry
            .begin_recycle(second_key)
            .expect("second recycle")
            .commit();
        assert_eq!(registry.pending_terminal_count(), 0);
    }

    #[test]
    fn shutdown_discard_requires_terminal_and_preserves_live_payload_on_error() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let slot_key = key(0, 1);
        let (producer, _receiver) = bind(&mut registry, &mut controls, slot_key, 1);
        publish(&producer, output(0));

        let error = registry
            .shutdown_discard(slot_key)
            .expect_err("live endpoint shutdown discard must fail");
        assert_eq!(error.category(), ErrorCategory::Internal);
        let snapshot = producer.snapshot().expect("unchanged endpoint");
        assert_eq!(snapshot.buffered_output_events, 1);
        assert_eq!(snapshot.discarded_output_events, 0);
        assert!(snapshot.producer_open);
        assert!(!snapshot.shutdown);
        assert!(!snapshot.reap_ready);
        assert_conserved(snapshot);
    }

    #[test]
    fn poisoned_guard_fails_safely_with_redacted_error() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, _receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);
        let poisoner = producer.clone();
        let panic = thread::spawn(move || {
            let _guard = poisoner.begin_output_commit().expect("guard");
            panic!("intentional endpoint poison");
        })
        .join();
        assert!(panic.is_err());
        let error = producer.snapshot().expect_err("poison must remain visible");
        assert_eq!(error.category(), ErrorCategory::Internal);
        let debug = format!("{error:?}");
        assert!(!debug.contains("SlotKey"));
        assert!(!debug.contains("generation"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn current_thread_wait_has_no_lost_publication_wake() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);
        let waiter = tokio::spawn(async move {
            let snapshot = receiver.wait_until_readable().await.expect("readable");
            (receiver, snapshot)
        });
        tokio::task::yield_now().await;
        publish(&producer, output(0));
        let (_receiver, snapshot) = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("wait timeout")
            .expect("wait task");
        assert_eq!(snapshot.buffered_output_events, 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn full_output_writable_wait_returns_promptly_after_receiver_drop() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);
        publish(&producer, output(0));

        let wake = producer.wake_hook();
        let waiter = tokio::spawn(async move { wake.wait_until_writable().await });
        tokio::task::yield_now().await;
        assert!(
            !waiter.is_finished(),
            "a full live endpoint must keep the writable waiter parked"
        );

        drop(receiver);
        let snapshot = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("receiver-drop wake timeout")
            .expect("writable waiter task")
            .expect("current-generation control snapshot");
        assert_eq!(snapshot.buffered_output_events, 1);
        assert!(snapshot.producer_open);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn terminal_waiter_ignores_buffered_output_until_terminal_publication() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);
        publish(&producer, output(0));

        let waiter = tokio::spawn(async move {
            let snapshot = receiver.wait_until_terminal().await.expect("terminal wait");
            (receiver, snapshot)
        });
        tokio::task::yield_now().await;
        assert!(
            !waiter.is_finished(),
            "buffered output must not satisfy a terminal-only waiter"
        );

        producer
            .begin_terminal_commit()
            .expect("terminal guard")
            .publish_terminal(terminal(1));
        let (_receiver, snapshot) = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("terminal wake timeout")
            .expect("terminal waiter task");
        assert_eq!(snapshot.buffered_output_events, 1);
        assert!(snapshot.terminal_pending);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stale_wake_hook_cannot_observe_a_rebound_endpoint_generation() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let first_key = key(0, 1);
        let (first_producer, mut first_receiver) = bind(&mut registry, &mut controls, first_key, 1);
        let stale_wake = first_producer.wake_hook();
        first_producer
            .begin_terminal_commit()
            .expect("first terminal guard")
            .publish_terminal(terminal(0));
        assert!(
            first_receiver
                .take_terminal()
                .expect("first terminal")
                .is_some()
        );
        assert!(matches!(first_receiver.try_pop(), Ok(TryPop::Eof)));
        first_producer
            .control
            .mark_terminal()
            .expect("mark first control terminal");
        let endpoint = registry
            .begin_recycle(first_key)
            .expect("begin first recycle");
        controls
            .recycle(&first_producer.control)
            .expect("recycle first control");
        endpoint.commit();

        let second_key = key(0, 2);
        let (second_producer, _second_receiver) = bind(&mut registry, &mut controls, second_key, 1);
        let error = tokio::time::timeout(Duration::from_secs(1), stale_wake.wait_until_writable())
            .await
            .expect("stale waiter timeout")
            .expect_err("stale wake hook must fail");
        assert_eq!(error.category(), ErrorCategory::InvalidRequest);
        assert!(
            second_producer
                .snapshot()
                .expect("second endpoint")
                .receiver_connected
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn multiple_waiters_are_woken_when_full_output_becomes_writable() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, mut receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);
        publish(&producer, output(0));
        let first = producer.wake_hook();
        let second = first.clone();
        let first_waiter = tokio::spawn(async move { first.wait_until_writable().await });
        let second_waiter = tokio::spawn(async move { second.wait_until_writable().await });
        tokio::task::yield_now().await;
        assert!(matches!(receiver.try_pop(), Ok(TryPop::Event(_))));

        for waiter in [first_waiter, second_waiter] {
            let snapshot = tokio::time::timeout(Duration::from_secs(1), waiter)
                .await
                .expect("wait timeout")
                .expect("wait task")
                .expect("wait result");
            assert_eq!(snapshot.buffered_output_events, 0);
        }
    }

    #[test]
    fn debug_views_redact_generation_and_payload() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, receiver) = bind(&mut registry, &mut controls, key(0, 999_999), 1);
        for debug in [format!("{producer:?}"), format!("{receiver:?}")] {
            assert!(!debug.contains("999999"));
            assert!(!debug.contains("SlotKey"));
        }
    }

    #[test]
    fn concurrent_drop_control_signal_precedes_endpoint_mutex_wait() {
        let mut registry = EndpointRegistry::try_with_capacity(1).expect("registry");
        let mut controls = ControlRegistry::try_with_capacity(1).expect("controls");
        let (producer, receiver) = bind(&mut registry, &mut controls, key(0, 1), 1);
        let barrier = Arc::new(Barrier::new(2));
        let held_barrier = Arc::clone(&barrier);
        let held_producer = producer.clone();
        let held = thread::spawn(move || {
            let guard = held_producer.begin_output_commit().expect("guard");
            held_barrier.wait();
            thread::sleep(Duration::from_millis(200));
            drop(guard);
        });
        barrier.wait();
        let (drop_done_tx, drop_done_rx) = mpsc::channel();
        let dropper = thread::spawn(move || {
            drop(receiver);
            drop_done_tx.send(()).expect("drop report");
        });
        drop_done_rx
            .recv_timeout(Duration::from_millis(50))
            .expect("receiver Drop must not wait for endpoint mutex");
        held.join().expect("held guard thread");
        dropper.join().expect("drop thread");
        producer
            .begin_terminal_commit()
            .expect("terminal after abandoned output guard")
            .publish_terminal(terminal(0));
        producer
            .settle_disconnected()
            .expect("settle disconnected receiver");
        assert!(!producer.snapshot().expect("snapshot").receiver_connected);
    }
}
