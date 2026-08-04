//! Bounded Tokio ownership for the deterministic scheduler engine.
//!
//! Only request submission uses the ordinary command lane. Cancellation,
//! output consumption, receiver disconnection, and shutdown operate through
//! generation-tagged controls and preallocated wake notifications, so they
//! continue to make progress when every command slot is occupied.

use std::{
    collections::VecDeque,
    fmt,
    mem::size_of,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use runnel_runtime::{DecoderAdapter, SamplingPolicy};
use tokio::{runtime::Handle as RuntimeHandle, sync::Notify, task::JoinHandle};

use crate::{
    SchedulerConfig, SchedulerEngine,
    control::ControlBinding,
    endpoint::{EndpointReceiver, TryPop},
    error::{SchedulerError, SchedulerResult},
    request::{
        CancelDisposition, OutputEvent, RequestSpec, ShutdownReport, StepReport, TerminalResult,
    },
};

/// Result of a nonblocking output read from a request handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TryRecvOutput {
    /// One committed output event was consumed.
    Output(OutputEvent),
    /// The producer is still open but no output is currently buffered.
    Empty,
    /// The producer is closed and every committed output event was consumed.
    Eof,
}

/// Cloneable cancellation authority for one accepted request generation.
#[derive(Clone)]
pub struct RequestCancellation {
    control: ControlBinding,
    activity: Arc<Activity>,
}

impl fmt::Debug for RequestCancellation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RequestCancellation")
            .field("identity", &"<redacted>")
            .finish()
    }
}

impl RequestCancellation {
    /// Requests cancellation without entering the bounded submission lane.
    pub fn cancel(&self) -> SchedulerResult<CancelDisposition> {
        let result = self.control.cancel();
        self.activity.signal();
        result
    }
}

/// Sole output and terminal consumer for one accepted request.
pub struct RequestHandle {
    request_id: crate::RequestId,
    cancellation: RequestCancellation,
    receiver: EndpointReceiver,
    terminal_cache: Option<TerminalResult>,
    output_eof: bool,
}

impl fmt::Debug for RequestHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RequestHandle")
            .field("request_id", &self.request_id)
            .field("output_eof", &self.output_eof)
            .field("has_terminal", &self.terminal_cache.is_some())
            .finish()
    }
}

impl RequestHandle {
    fn new(
        request_id: crate::RequestId,
        control: ControlBinding,
        receiver: EndpointReceiver,
        activity: Arc<Activity>,
    ) -> Self {
        Self {
            request_id,
            cancellation: RequestCancellation { control, activity },
            receiver,
            terminal_cache: None,
            output_eof: false,
        }
    }

    /// Returns the stable scheduler identity assigned at admission.
    #[must_use]
    pub const fn request_id(&self) -> crate::RequestId {
        self.request_id
    }

    /// Returns a cancellation handle that may outlive this output receiver.
    #[must_use]
    pub fn cancellation(&self) -> RequestCancellation {
        self.cancellation.clone()
    }

    /// Requests cancellation without consuming this result receiver.
    pub fn cancel(&self) -> SchedulerResult<CancelDisposition> {
        self.cancellation.cancel()
    }

    /// Attempts to consume one output event without waiting.
    pub fn try_recv_output(&mut self) -> SchedulerResult<TryRecvOutput> {
        if self.output_eof {
            return Ok(TryRecvOutput::Eof);
        }
        let pop = self.receiver.try_pop().map_err(|error| {
            if self
                .cancellation
                .activity
                .owner_done
                .load(Ordering::Acquire)
            {
                owner_stopped_error(&self.cancellation.activity)
            } else {
                error
            }
        })?;
        let mut eof_ack = Ok(());
        let result = match pop {
            TryPop::Event(event) => {
                eof_ack = self.acknowledge_closed_eof();
                TryRecvOutput::Output(event)
            }
            TryPop::Empty => TryRecvOutput::Empty,
            TryPop::Eof => {
                self.output_eof = true;
                TryRecvOutput::Eof
            }
        };
        if result != TryRecvOutput::Empty {
            self.cancellation.activity.signal();
        }
        eof_ack.map_err(|error| {
            if self
                .cancellation
                .activity
                .owner_done
                .load(Ordering::Acquire)
            {
                owner_stopped_error(&self.cancellation.activity)
            } else {
                error
            }
        })?;
        Ok(result)
    }

    /// Waits until one output event or output EOF can be consumed.
    pub async fn recv_output(&mut self) -> SchedulerResult<TryRecvOutput> {
        loop {
            let result = self.try_recv_output()?;
            if result != TryRecvOutput::Empty {
                return Ok(result);
            }
            let activity = Arc::clone(&self.cancellation.activity);
            tokio::select! {
                snapshot = self.receiver.wait_until_readable() => {
                    match snapshot {
                        Ok(_) => {}
                        Err(_) if activity.owner_done.load(Ordering::Acquire) => {
                            return Err(owner_stopped_error(&activity));
                        }
                        Err(error) => return Err(error),
                    }
                }
                () = activity.wait_owner_done() => {
                    return Err(owner_stopped_error(&activity));
                }
            }
        }
    }

    /// Returns the terminal result if it has been published.
    ///
    /// Once consumed from the endpoint, the value is cached in the handle so
    /// repeated observations do not re-enter a recycled endpoint generation.
    pub fn try_terminal(&mut self) -> SchedulerResult<Option<TerminalResult>> {
        if let Some(terminal) = self.terminal_cache {
            return Ok(Some(terminal));
        }
        let terminal = self.receiver.take_terminal().map_err(|error| {
            if self
                .cancellation
                .activity
                .owner_done
                .load(Ordering::Acquire)
            {
                owner_stopped_error(&self.cancellation.activity)
            } else {
                error
            }
        })?;
        if let Some(terminal) = terminal {
            self.terminal_cache = Some(terminal);
            let eof = self.acknowledge_closed_eof();
            self.cancellation.activity.signal();
            eof.map_err(|error| {
                if self
                    .cancellation
                    .activity
                    .owner_done
                    .load(Ordering::Acquire)
                {
                    owner_stopped_error(&self.cancellation.activity)
                } else {
                    error
                }
            })?;
        }
        Ok(terminal)
    }

    /// Waits specifically for terminal publication, ignoring readable output.
    pub async fn terminal(&mut self) -> SchedulerResult<TerminalResult> {
        loop {
            if let Some(terminal) = self.try_terminal()? {
                return Ok(terminal);
            }
            let activity = Arc::clone(&self.cancellation.activity);
            tokio::select! {
                snapshot = self.receiver.wait_until_terminal() => {
                    match snapshot {
                        Ok(_) => {}
                        Err(_) if activity.owner_done.load(Ordering::Acquire) => {
                            return Err(owner_stopped_error(&activity));
                        }
                        Err(error) => return Err(error),
                    }
                }
                () = activity.wait_owner_done() => {
                    return Err(owner_stopped_error(&activity));
                }
            }
        }
    }

    /// Explicitly disconnects this receiver without using a command slot.
    pub fn disconnect(&mut self) -> SchedulerResult<()> {
        let result = self.receiver.disconnect();
        self.cancellation.activity.signal();
        result
    }

    /// Reports whether output EOF has already been acknowledged.
    #[must_use]
    pub const fn output_eof(&self) -> bool {
        self.output_eof
    }

    fn acknowledge_closed_eof(&mut self) -> SchedulerResult<()> {
        if self.output_eof {
            return Ok(());
        }
        let snapshot = self.receiver.snapshot()?;
        if !snapshot.producer_open && snapshot.buffered_output_events == 0 {
            match self.receiver.try_pop()? {
                TryPop::Eof => self.output_eof = true,
                TryPop::Empty | TryPop::Event(_) => {
                    return Err(SchedulerError::internal(
                        "closed empty request endpoint did not acknowledge EOF",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Drop for RequestHandle {
    fn drop(&mut self) {
        let _ = self.receiver.disconnect();
        self.cancellation.activity.signal();
    }
}

/// One owned, already copied submission waiting for the actor's admission
/// response.
///
/// Dropping this value abandons the command. If admission has already
/// succeeded, the resulting request receiver is dropped and cancellation is
/// published; otherwise the actor skips the command. In both cases the slot
/// is eventually returned without enlarging the table.
pub struct Submission {
    shared: Arc<Shared>,
    slot: usize,
    ticket: u64,
    pending: bool,
}

impl fmt::Debug for Submission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Submission")
            .field("state", &if self.pending { "pending" } else { "resolved" })
            .finish()
    }
}

impl Submission {
    /// Waits for deterministic engine admission and returns the sole request
    /// receiver. A responded command continues to occupy its slot until this
    /// method or [`Drop`] consumes it.
    pub async fn wait(mut self) -> SchedulerResult<RequestHandle> {
        loop {
            let notified = self.shared.response_notify(self.slot)?.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            if let Some(response) = self.shared.take_response(self.slot, self.ticket)? {
                self.pending = false;
                return response;
            }
            if self.shared.activity.owner_done.load(Ordering::Acquire) {
                self.pending = false;
                return Err(
                    if self.shared.activity.owner_failed.load(Ordering::Acquire) {
                        SchedulerError::internal("scheduler actor owner failed")
                    } else {
                        SchedulerError::internal("scheduler actor stopped before responding")
                    },
                );
            }
            notified.await;
        }
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        if self.pending {
            self.shared.abandon(self.slot, self.ticket);
        }
    }
}

/// Cloneable ingress and control surface for one scheduler actor.
#[derive(Clone)]
pub struct SchedulerClient {
    shared: Arc<Shared>,
}

impl fmt::Debug for SchedulerClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SchedulerClient")
            .field("closed", &self.shared.closed.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl SchedulerClient {
    /// Copies a request into one pre-reserved command slot.
    ///
    /// After request validation, capacity is claimed before prompt allocation.
    /// Therefore a saturated valid request is rejected without copying prompt
    /// bytes, and repeated rejection does not change accepted FIFO order.
    pub fn try_submit(&self, request: RequestSpec<'_>) -> SchedulerResult<Submission> {
        self.shared.validate_submission_shape(request)?;
        let (slot, ticket) = self.shared.reserve()?;
        let command = match SubmitCommand::try_from_request(request) {
            Ok(command) => command,
            Err(error) => {
                self.shared.release_reserved(slot, ticket);
                return Err(error);
            }
        };
        if let Err(error) = self.shared.commit(slot, ticket, command) {
            self.shared.release_reserved(slot, ticket);
            return Err(error);
        }
        Ok(Submission {
            shared: Arc::clone(&self.shared),
            slot,
            ticket,
            pending: true,
        })
    }

    /// Returns nanoseconds elapsed on the actor's monotonic clock.
    #[must_use]
    pub fn monotonic_ns(&self) -> u64 {
        self.shared.monotonic_ns()
    }

    /// Produces an actor-clock deadline after `duration`, rejecting arithmetic
    /// that cannot be represented by the `u64` nanosecond contract.
    pub fn deadline_after(&self, duration: Duration) -> SchedulerResult<u64> {
        let delta = u64::try_from(duration.as_nanos()).map_err(|_| {
            SchedulerError::invalid_request("deadline", "duration exceeds u64 nanoseconds")
        })?;
        self.monotonic_ns().checked_add(delta).ok_or_else(|| {
            SchedulerError::invalid_request("deadline", "absolute nanoseconds overflow")
        })
    }

    /// Closes ingress and wakes the actor independently of command capacity.
    /// Returns `true` only for the caller that linearized the close.
    pub fn request_shutdown(&self) -> bool {
        self.shared.request_shutdown()
    }

    /// Reports whether the actor has stopped accepting submissions.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.shared.closed.load(Ordering::Acquire)
    }
}

/// Completed actor and synchronous-engine shutdown accounting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActorShutdownReport {
    engine: ShutdownReport,
    accepted_submissions: usize,
    rejected_submissions: usize,
    shutdown_cancellations: usize,
    engine_steps: u64,
}

impl ActorShutdownReport {
    /// Returns the synchronous engine's exact reclamation report.
    #[must_use]
    pub const fn engine(self) -> ShutdownReport {
        self.engine
    }

    /// Returns commands accepted by the engine before actor closure.
    #[must_use]
    pub const fn accepted_submissions(self) -> usize {
        self.accepted_submissions
    }

    /// Returns ready commands rejected because admission failed or ingress
    /// closed. Abandoned commands that never reached admission are included.
    #[must_use]
    pub const fn rejected_submissions(self) -> usize {
        self.rejected_submissions
    }

    /// Returns requests whose cancellation bit was first set by shutdown.
    #[must_use]
    pub const fn shutdown_cancellations(self) -> usize {
        self.shutdown_cancellations
    }

    /// Returns the number of bounded synchronous engine steps executed.
    #[must_use]
    pub const fn engine_steps(self) -> u64 {
        self.engine_steps
    }
}

/// Handle for one long-lived scheduler owner with at most one awaited Tokio
/// blocking pump in flight.
pub struct SchedulerActor {
    client: SchedulerClient,
    join: Option<JoinHandle<SchedulerResult<ActorShutdownReport>>>,
}

impl fmt::Debug for SchedulerActor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SchedulerActor")
            .field("closed", &self.client.is_closed())
            .finish_non_exhaustive()
    }
}

impl SchedulerActor {
    /// Builds every bounded actor structure and starts one blocking owner.
    ///
    /// This must be called from a live Tokio runtime. The blocking task is
    /// never aborted by `SchedulerActor::drop`; dropping the actor only begins
    /// cooperative shutdown and detaches the join handle.
    pub fn spawn<A>(adapter: A, config: SchedulerConfig) -> SchedulerResult<Self>
    where
        A: DecoderAdapter + 'static,
    {
        let runtime = RuntimeHandle::try_current().map_err(|_| {
            SchedulerError::unsupported("SchedulerActor requires a live Tokio runtime")
        })?;
        let engine = SchedulerEngine::new(adapter, config)?;
        let shared = Arc::new(Shared::try_new(config)?);
        let client = SchedulerClient {
            shared: Arc::clone(&shared),
        };
        let owner = BlockingOwner::try_new(engine, shared)?;
        let join = runtime.spawn(async move { owner.run().await });
        Ok(Self {
            client,
            join: Some(join),
        })
    }

    /// Returns a cloneable client for submission and out-of-band shutdown.
    #[must_use]
    pub fn client(&self) -> SchedulerClient {
        self.client.clone()
    }

    /// Requests cooperative shutdown and waits for all live request ownership
    /// to be drained, acknowledged, or disconnected before engine teardown.
    pub async fn shutdown(mut self) -> SchedulerResult<ActorShutdownReport> {
        self.client.request_shutdown();
        let join = self
            .join
            .take()
            .ok_or_else(|| SchedulerError::internal("scheduler actor join is unavailable"))?;
        join.await
            .map_err(|_| SchedulerError::internal("scheduler actor blocking owner panicked"))?
    }
}

impl Drop for SchedulerActor {
    fn drop(&mut self) {
        if self.join.is_some() {
            self.client.request_shutdown();
        }
    }
}

struct Activity {
    dirty: AtomicBool,
    notify: Notify,
    owner_failed: AtomicBool,
    owner_done: AtomicBool,
    lifecycle: Notify,
    #[cfg(test)]
    pump_in_flight: AtomicBool,
    #[cfg(test)]
    parked: AtomicBool,
    #[cfg(test)]
    park_epoch: AtomicU64,
    #[cfg(test)]
    park_notify: Notify,
}

impl Activity {
    const fn new() -> Self {
        Self {
            dirty: AtomicBool::new(true),
            notify: Notify::const_new(),
            owner_failed: AtomicBool::new(false),
            owner_done: AtomicBool::new(false),
            lifecycle: Notify::const_new(),
            #[cfg(test)]
            pump_in_flight: AtomicBool::new(false),
            #[cfg(test)]
            parked: AtomicBool::new(false),
            #[cfg(test)]
            park_epoch: AtomicU64::new(0),
            #[cfg(test)]
            park_notify: Notify::const_new(),
        }
    }

    fn signal(&self) {
        self.dirty.store(true, Ordering::Release);
        self.notify.notify_one();
    }

    async fn wait_until(&self, deadline: Option<Instant>) {
        let notified = self.notify.notified();
        let mut notified = std::pin::pin!(notified);
        notified.as_mut().enable();
        if self.dirty.swap(false, Ordering::AcqRel) {
            return;
        }
        #[cfg(test)]
        {
            self.parked.store(true, Ordering::Release);
            self.park_epoch.fetch_add(1, Ordering::AcqRel);
            self.park_notify.notify_waiters();
        }
        if let Some(deadline) = deadline {
            tokio::select! {
                () = &mut notified => {},
                () = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {},
            }
        } else {
            notified.await;
        }
        #[cfg(test)]
        self.parked.store(false, Ordering::Release);
    }

    async fn wait_owner_done(&self) {
        loop {
            let notified = self.lifecycle.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            if self.owner_done.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    fn mark_owner_done(&self, failed: bool) {
        if failed {
            self.owner_failed.store(true, Ordering::Release);
        }
        self.owner_done.store(true, Ordering::Release);
        self.lifecycle.notify_waiters();
        self.signal();
    }
}

struct Shared {
    queue: Mutex<CommandQueue>,
    slots: Box<[CommandSlot]>,
    activity: Arc<Activity>,
    closed: AtomicBool,
    #[cfg(test)]
    observed_engine_steps: AtomicU64,
    #[cfg(test)]
    hold_pumps: AtomicBool,
    #[cfg(test)]
    hold_epoch: AtomicU64,
    #[cfg(test)]
    hold_observed: AtomicU64,
    #[cfg(test)]
    hold_notify: Notify,
    #[cfg(test)]
    panic_next_pump: AtomicBool,
    origin: Instant,
    max_prompt_tokens: usize,
    max_new_tokens: usize,
    max_context_tokens: usize,
    vocabulary_size: usize,
}

impl fmt::Debug for Shared {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ActorShared")
            .field("closed", &self.closed.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl Shared {
    fn try_new(config: SchedulerConfig) -> SchedulerResult<Self> {
        let capacity = config.command_capacity();
        let mut slots = Vec::new();
        try_reserve_vec(&mut slots, capacity, "actor command slot table")?;
        for _ in 0..capacity {
            slots.push(CommandSlot::new());
        }
        Ok(Self {
            queue: Mutex::new(CommandQueue::try_new(capacity)?),
            slots: slots.into_boxed_slice(),
            activity: Arc::new(Activity::new()),
            closed: AtomicBool::new(false),
            #[cfg(test)]
            observed_engine_steps: AtomicU64::new(0),
            #[cfg(test)]
            hold_pumps: AtomicBool::new(false),
            #[cfg(test)]
            hold_epoch: AtomicU64::new(0),
            #[cfg(test)]
            hold_observed: AtomicU64::new(0),
            #[cfg(test)]
            hold_notify: Notify::const_new(),
            #[cfg(test)]
            panic_next_pump: AtomicBool::new(false),
            origin: Instant::now(),
            max_prompt_tokens: config.max_prompt_tokens(),
            max_new_tokens: config.max_new_tokens(),
            max_context_tokens: config.max_context_tokens(),
            vocabulary_size: config.vocabulary_size(),
        })
    }

    fn validate_submission_shape(&self, request: RequestSpec<'_>) -> SchedulerResult<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(SchedulerError::scheduler_closed());
        }
        if request.prompt().is_empty() {
            return Err(SchedulerError::invalid_request(
                "prompt",
                "must be nonempty",
            ));
        }
        if request.prompt().len() > self.max_prompt_tokens {
            return Err(SchedulerError::invalid_request(
                "prompt",
                "exceeds the configured token ceiling",
            ));
        }
        if request.max_new_tokens() > self.max_new_tokens {
            return Err(SchedulerError::invalid_request(
                "max_new_tokens",
                "exceeds the configured ceiling",
            ));
        }
        let total_positions = request
            .prompt()
            .len()
            .checked_add(request.max_new_tokens().saturating_sub(1))
            .ok_or_else(|| {
                SchedulerError::invalid_request(
                    "request length",
                    "prompt plus generation positions overflow",
                )
            })?;
        if total_positions > self.max_context_tokens {
            return Err(SchedulerError::invalid_request(
                "request length",
                "exceeds the configured context ceiling",
            ));
        }
        if request
            .deadline_ns()
            .is_some_and(|deadline| self.monotonic_ns() >= deadline)
        {
            return Err(SchedulerError::deadline_exceeded());
        }
        if request.prompt().iter().any(|token| {
            usize::try_from(*token).map_or(true, |token| token >= self.vocabulary_size)
        }) {
            return Err(SchedulerError::invalid_request(
                "prompt",
                "contains a token outside the adapter vocabulary",
            ));
        }
        request
            .sampling()
            .validate(self.vocabulary_size)
            .map_err(|source| SchedulerError::sampling("validating request policy", source))?;
        Ok(())
    }

    fn reserve(&self) -> SchedulerResult<(usize, u64)> {
        if self.closed.load(Ordering::Acquire) {
            return Err(SchedulerError::scheduler_closed());
        }
        let mut queue = self.lock_queue()?;
        if self.closed.load(Ordering::Acquire) {
            return Err(SchedulerError::scheduler_closed());
        }
        let capacity = self.slots.len();
        while let Some(index) = queue.free.pop() {
            let slot = self
                .slots
                .get(index)
                .ok_or_else(|| SchedulerError::internal("command free index is out of range"))?;
            let ticket =
                slot.next_ticket
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                        value.checked_add(1)
                    });
            let Ok(previous) = ticket else {
                queue.retired = queue.retired.checked_add(1).ok_or_else(|| {
                    SchedulerError::internal("retired command slot count overflows")
                })?;
                queue.debug_assert_conserved();
                continue;
            };
            let ticket = previous + 1;
            let state = queue
                .states
                .get(index)
                .ok_or_else(|| SchedulerError::internal("command state index is out of range"))?;
            if !matches!(state.phase, CommandPhase::Vacant) {
                return Err(SchedulerError::internal("free command slot is not vacant"));
            }
            transition(&mut queue, index, CommandPhase::Reserved)?;
            let state = queue
                .states
                .get_mut(index)
                .ok_or_else(|| SchedulerError::internal("command state index is out of range"))?;
            state.ticket = ticket;
            state.abandoned = false;
            queue.debug_assert_conserved();
            return Ok((index, ticket));
        }
        Err(SchedulerError::resource_exhausted(
            if queue.retired == 0 {
                "actor command slots"
            } else {
                "actor command slot generations"
            },
            u64::try_from(capacity)
                .unwrap_or(u64::MAX)
                .saturating_add(1),
            u64::try_from(capacity).unwrap_or(u64::MAX),
        ))
    }

    fn commit(&self, index: usize, ticket: u64, command: SubmitCommand) -> SchedulerResult<()> {
        let mut queue = self.lock_queue()?;
        if self.closed.load(Ordering::Acquire) {
            return Err(SchedulerError::scheduler_closed());
        }
        if queue.ready.len() >= queue.ready.capacity() {
            return Err(SchedulerError::internal(
                "actor ready FIFO exceeded reserved capacity",
            ));
        }
        let _ = checked_state_mut(&mut queue, index, ticket, CommandPhase::Reserved)?;
        transition(&mut queue, index, CommandPhase::Ready)?;
        queue.states[index].command = Some(command);
        queue.ready.push_back(index);
        queue.debug_assert_conserved();
        drop(queue);
        self.activity.signal();
        Ok(())
    }

    fn release_reserved(&self, index: usize, ticket: u64) {
        let mut queue = lock_recover(&self.queue);
        let should_release = queue.states.get(index).is_some_and(|state| {
            state.ticket == ticket && matches!(state.phase, CommandPhase::Reserved)
        });
        if should_release {
            make_vacant(&mut queue, index);
        }
        drop(queue);
        self.activity.signal();
    }

    fn take_response(
        &self,
        index: usize,
        ticket: u64,
    ) -> SchedulerResult<Option<SchedulerResult<RequestHandle>>> {
        let mut queue = self.lock_queue()?;
        let state = queue
            .states
            .get_mut(index)
            .ok_or_else(|| SchedulerError::internal("command response index is out of range"))?;
        if state.ticket != ticket {
            return Err(SchedulerError::internal(
                "command response generation is stale",
            ));
        }
        if !matches!(state.phase, CommandPhase::Responded) {
            return Ok(None);
        }
        let response = state
            .response
            .take()
            .ok_or_else(|| SchedulerError::internal("responded command has no admission result"))?;
        make_vacant(&mut queue, index);
        Ok(Some(response))
    }

    fn abandon(&self, index: usize, ticket: u64) {
        let mut dropped_response = None;
        let mut queue = lock_recover(&self.queue);
        if let Some(state) = queue.states.get_mut(index)
            && state.ticket == ticket
        {
            match state.phase {
                CommandPhase::Ready | CommandPhase::InFlight => state.abandoned = true,
                CommandPhase::Responded => {
                    dropped_response = state.response.take();
                    make_vacant(&mut queue, index);
                }
                CommandPhase::Reserved => {
                    state.abandoned = true;
                }
                CommandPhase::Vacant => {}
            }
        }
        drop(queue);
        drop(dropped_response);
        self.activity.signal();
    }

    fn response_notify(&self, index: usize) -> SchedulerResult<&Notify> {
        self.slots
            .get(index)
            .map(|slot| &slot.response)
            .ok_or_else(|| SchedulerError::internal("command response index is out of range"))
    }

    fn request_shutdown(&self) -> bool {
        let first = self
            .closed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok();
        self.activity.signal();
        first
    }

    fn fail_without_owner(&self) {
        self.closed.store(true, Ordering::Release);
        self.fail_submissions();
        self.mark_owner_done(true);
    }

    fn fail_submissions(&self) {
        let mut queue = lock_recover(&self.queue);
        queue.ready.clear();
        for index in 0..queue.states.len() {
            let phase = queue.states[index].phase;
            match phase {
                CommandPhase::Vacant | CommandPhase::Reserved => {}
                CommandPhase::Ready | CommandPhase::InFlight | CommandPhase::Responded => {
                    let abandoned = queue.states[index].abandoned;
                    let old_response = queue.states[index].response.take();
                    queue.states[index].command = None;
                    drop(old_response);
                    if abandoned {
                        make_vacant(&mut queue, index);
                    } else if let Err(error) =
                        transition(&mut queue, index, CommandPhase::Responded)
                    {
                        debug_assert!(false, "failed actor-error transition: {error:?}");
                    } else {
                        queue.states[index].response = Some(Err(SchedulerError::internal(
                            "scheduler actor owner failed",
                        )));
                    }
                }
            }
        }
        queue.debug_assert_conserved();
        drop(queue);
        for slot in &self.slots {
            slot.response.notify_waiters();
        }
        self.activity.signal();
    }

    fn mark_owner_done(&self, failed: bool) {
        self.activity.mark_owner_done(failed);
        for slot in &self.slots {
            slot.response.notify_waiters();
        }
    }

    fn monotonic_ns(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn lock_queue(&self) -> SchedulerResult<MutexGuard<'_, CommandQueue>> {
        self.queue
            .lock()
            .map_err(|_| SchedulerError::internal("actor command table is unavailable"))
    }
}

struct CommandSlot {
    next_ticket: AtomicU64,
    response: Notify,
}

impl CommandSlot {
    const fn new() -> Self {
        Self {
            next_ticket: AtomicU64::new(0),
            response: Notify::const_new(),
        }
    }
}

struct SubmitCommand {
    prompt: Vec<u32>,
    max_new_tokens: usize,
    sampling: SamplingPolicy,
    deadline_ns: Option<u64>,
}

struct ClaimedCommand {
    index: usize,
    ticket: u64,
    command: SubmitCommand,
    abandoned: bool,
    closed_at_claim: bool,
}

impl fmt::Debug for SubmitCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SubmitCommand")
            .field("prompt_len", &self.prompt.len())
            .field("max_new_tokens", &self.max_new_tokens)
            .field("has_deadline", &self.deadline_ns.is_some())
            .finish()
    }
}

impl SubmitCommand {
    fn try_from_request(request: RequestSpec<'_>) -> SchedulerResult<Self> {
        let mut prompt = Vec::new();
        try_reserve_vec(&mut prompt, request.prompt().len(), "actor offered prompt")?;
        prompt.extend_from_slice(request.prompt());
        Ok(Self {
            prompt,
            max_new_tokens: request.max_new_tokens(),
            sampling: request.sampling(),
            deadline_ns: request.deadline_ns(),
        })
    }

    fn as_spec(&self) -> RequestSpec<'_> {
        RequestSpec::new(
            &self.prompt,
            self.max_new_tokens,
            self.sampling,
            self.deadline_ns,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandPhase {
    Vacant,
    Reserved,
    Ready,
    InFlight,
    Responded,
}

struct CommandState {
    phase: CommandPhase,
    ticket: u64,
    abandoned: bool,
    command: Option<SubmitCommand>,
    response: Option<SchedulerResult<RequestHandle>>,
}

impl CommandState {
    const fn vacant() -> Self {
        Self {
            phase: CommandPhase::Vacant,
            ticket: 0,
            abandoned: false,
            command: None,
            response: None,
        }
    }
}

struct CommandQueue {
    states: Vec<CommandState>,
    ready: VecDeque<usize>,
    free: Vec<usize>,
    occupancy: QueueOccupancy,
    retired: usize,
}

impl CommandQueue {
    fn try_new(capacity: usize) -> SchedulerResult<Self> {
        let mut states = Vec::new();
        try_reserve_vec(&mut states, capacity, "actor command states")?;
        let mut free = Vec::new();
        try_reserve_vec(&mut free, capacity, "actor command free list")?;
        for index in 0..capacity {
            states.push(CommandState::vacant());
            free.push(capacity - index - 1);
        }
        let mut ready = VecDeque::new();
        try_reserve_deque(&mut ready, capacity, "actor command ready FIFO")?;
        Ok(Self {
            states,
            ready,
            free,
            occupancy: QueueOccupancy::default(),
            retired: 0,
        })
    }

    const fn occupancy(&self) -> QueueOccupancy {
        self.occupancy
    }

    fn debug_assert_conserved(&self) {
        debug_assert_eq!(
            self.occupancy
                .total()
                .saturating_add(self.free.len())
                .saturating_add(self.retired),
            self.states.len()
        );
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct QueueOccupancy {
    reserved: usize,
    ready: usize,
    in_flight: usize,
    responded: usize,
}

struct AcceptedControl {
    control: ControlBinding,
    deadline_ns: Option<u64>,
}

struct BlockingOwner<A: DecoderAdapter> {
    engine: SchedulerEngine<A>,
    shared: Arc<Shared>,
    accepted: Vec<AcceptedControl>,
    accepted_submissions: usize,
    rejected_submissions: usize,
    shutdown_cancellations: usize,
    engine_steps: u64,
    command_batch_limit: usize,
}

impl<A: DecoderAdapter> BlockingOwner<A> {
    fn try_new(engine: SchedulerEngine<A>, shared: Arc<Shared>) -> SchedulerResult<Self> {
        let maximum = engine.config().max_outstanding_requests();
        let command_batch_limit = engine.config().batch_width();
        let mut accepted = Vec::new();
        try_reserve_vec(&mut accepted, maximum, "actor accepted-control table")?;
        Ok(Self {
            engine,
            shared,
            accepted,
            accepted_submissions: 0,
            rejected_submissions: 0,
            shutdown_cancellations: 0,
            engine_steps: 0,
            command_batch_limit,
        })
    }

    async fn run(mut self) -> SchedulerResult<ActorShutdownReport>
    where
        A: 'static,
    {
        loop {
            let shared = Arc::clone(&self.shared);
            #[cfg(test)]
            {
                self.shared.activity.parked.store(false, Ordering::Release);
                self.shared
                    .activity
                    .pump_in_flight
                    .store(true, Ordering::Release);
            }
            let joined = tokio::task::spawn_blocking(move || {
                let result = catch_unwind(AssertUnwindSafe(|| self.pump()));
                (self, result)
            })
            .await;
            #[cfg(test)]
            shared
                .activity
                .pump_in_flight
                .store(false, Ordering::Release);
            let (owner, result) = match joined {
                Ok(result) => result,
                Err(_) => {
                    shared.fail_without_owner();
                    return Err(SchedulerError::internal(
                        "scheduler actor blocking pump was cancelled",
                    ));
                }
            };
            self = owner;
            let state = match result {
                Ok(Ok(state)) => state,
                Ok(Err(error)) => return self.fail(error),
                Err(_) => {
                    return self.fail(SchedulerError::internal(
                        "scheduler actor blocking pump panicked",
                    ));
                }
            };
            match state {
                PumpState::Continue => {}
                PumpState::Wait(deadline) => self.shared.activity.wait_until(deadline).await,
                PumpState::Complete(report) => return Ok(report),
            }
        }
    }

    fn pump(&mut self) -> SchedulerResult<PumpState> {
        #[cfg(test)]
        if self.shared.panic_next_pump.swap(false, Ordering::AcqRel) {
            panic!("injected actor pump panic");
        }
        #[cfg(test)]
        if self.shared.hold_pumps.load(Ordering::Acquire) {
            let epoch = self.shared.hold_epoch.load(Ordering::Acquire);
            self.shared.hold_observed.store(epoch, Ordering::Release);
            self.shared.hold_notify.notify_waiters();
            return Ok(PumpState::Wait(None));
        }
        self.prune_accepted()?;
        let initially_closing = self.shared.closed.load(Ordering::Acquire);
        if initially_closing {
            self.cancel_accepted()?;
        }
        self.process_commands(initially_closing)?;
        let closing = self.shared.closed.load(Ordering::Acquire);
        if closing {
            // A command claimed before close is allowed to finish admission,
            // but it joins shutdown cancellation before any model step.
            self.cancel_accepted()?;
        }

        let request_used = self.engine.ledger_snapshot().request_used();
        let report = if request_used == 0 {
            StepReport::default()
        } else {
            self.engine_steps = self.engine_steps.checked_add(1).ok_or_else(|| {
                SchedulerError::resource_exhausted("actor engine-step count", u64::MAX, u64::MAX)
            })?;
            #[cfg(test)]
            self.shared
                .observed_engine_steps
                .store(self.engine_steps, Ordering::Release);
            self.engine
                .step_with_clock(&|| self.shared.monotonic_ns())?
        };
        self.prune_accepted()?;

        if closing {
            self.cancel_accepted()?;
            let occupancy = self.shared.lock_queue()?.occupancy();
            if occupancy.reserved == 0
                && occupancy.ready == 0
                && occupancy.in_flight == 0
                && self.engine.ledger_snapshot().request_used() == 0
            {
                let engine = self.engine.shutdown()?;
                let report = ActorShutdownReport {
                    engine,
                    accepted_submissions: self.accepted_submissions,
                    rejected_submissions: self.rejected_submissions,
                    shutdown_cancellations: self.shutdown_cancellations,
                    engine_steps: self.engine_steps,
                };
                self.shared.mark_owner_done(false);
                return Ok(PumpState::Complete(report));
            }
        }

        if self.should_continue(report)? {
            return Ok(PumpState::Continue);
        }
        let deadline = if closing { None } else { self.next_deadline()? };
        Ok(PumpState::Wait(deadline))
    }

    fn process_commands(&mut self, reject: bool) -> SchedulerResult<()> {
        for _ in 0..self.command_batch_limit {
            let Some(claimed) = self.take_ready()? else {
                break;
            };
            let ClaimedCommand {
                index,
                ticket,
                command,
                abandoned,
                closed_at_claim,
            } = claimed;
            let response = if reject || closed_at_claim || abandoned {
                self.rejected_submissions = self
                    .rejected_submissions
                    .checked_add(1)
                    .ok_or_else(|| SchedulerError::internal("actor rejection count overflows"))?;
                Err(SchedulerError::scheduler_closed())
            } else {
                self.engine.advance_clock(self.shared.monotonic_ns())?;
                match self.engine.try_submit_for_actor(command.as_spec()) {
                    Ok(admission) => {
                        let (request_id, control, receiver) = admission.into_parts();
                        if self.accepted.len() >= self.accepted.capacity() {
                            return Err(SchedulerError::internal(
                                "actor accepted-control table exceeded reserved capacity",
                            ));
                        }
                        self.accepted.push(AcceptedControl {
                            control: control.clone(),
                            deadline_ns: command.deadline_ns,
                        });
                        self.accepted_submissions =
                            self.accepted_submissions.checked_add(1).ok_or_else(|| {
                                SchedulerError::internal("actor acceptance count overflows")
                            })?;
                        Ok(RequestHandle::new(
                            request_id,
                            control,
                            receiver,
                            Arc::clone(&self.shared.activity),
                        ))
                    }
                    Err(error) => {
                        self.rejected_submissions =
                            self.rejected_submissions.checked_add(1).ok_or_else(|| {
                                SchedulerError::internal("actor rejection count overflows")
                            })?;
                        Err(error)
                    }
                }
            };
            drop(command);
            self.publish_response(index, ticket, response)?;
        }
        Ok(())
    }

    fn take_ready(&self) -> SchedulerResult<Option<ClaimedCommand>> {
        let mut queue = self.shared.lock_queue()?;
        let Some(index) = queue.ready.pop_front() else {
            return Ok(None);
        };
        let state = queue
            .states
            .get_mut(index)
            .ok_or_else(|| SchedulerError::internal("ready command index is out of range"))?;
        if !matches!(state.phase, CommandPhase::Ready) {
            return Err(SchedulerError::internal(
                "ready command FIFO references a non-ready slot",
            ));
        }
        transition(&mut queue, index, CommandPhase::InFlight)?;
        let state = queue
            .states
            .get_mut(index)
            .ok_or_else(|| SchedulerError::internal("ready command index is out of range"))?;
        let command = state
            .command
            .take()
            .ok_or_else(|| SchedulerError::internal("ready command payload is missing"))?;
        Ok(Some(ClaimedCommand {
            index,
            ticket: state.ticket,
            command,
            abandoned: state.abandoned,
            closed_at_claim: self.shared.closed.load(Ordering::Acquire),
        }))
    }

    fn publish_response(
        &self,
        index: usize,
        ticket: u64,
        response: SchedulerResult<RequestHandle>,
    ) -> SchedulerResult<()> {
        let mut dropped = None;
        let abandoned;
        {
            let mut queue = self.shared.lock_queue()?;
            abandoned =
                checked_state_mut(&mut queue, index, ticket, CommandPhase::InFlight)?.abandoned;
            if abandoned {
                dropped = Some(response);
                make_vacant(&mut queue, index);
            } else {
                transition(&mut queue, index, CommandPhase::Responded)?;
                queue.states[index].response = Some(response);
            }
            queue.debug_assert_conserved();
        }
        drop(dropped);
        if !abandoned {
            self.shared.response_notify(index)?.notify_waiters();
        }
        Ok(())
    }

    fn prune_accepted(&mut self) -> SchedulerResult<()> {
        let mut first_error = None;
        self.accepted
            .retain(|entry| match entry.control.fresh_snapshot() {
                Ok(_) => true,
                Err(SchedulerError::RequestNotFound) => false,
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                    true
                }
            });
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(())
        }
    }

    fn cancel_accepted(&mut self) -> SchedulerResult<()> {
        for entry in &self.accepted {
            match entry.control.cancel() {
                Ok(CancelDisposition::Requested) => {
                    self.shutdown_cancellations =
                        self.shutdown_cancellations.checked_add(1).ok_or_else(|| {
                            SchedulerError::internal("shutdown cancellation count overflows")
                        })?;
                }
                Ok(CancelDisposition::AlreadyRequested | CancelDisposition::AlreadyTerminal)
                | Err(SchedulerError::RequestNotFound) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    fn should_continue(&self, report: StepReport) -> SchedulerResult<bool> {
        let occupancy = self.shared.lock_queue()?.occupancy();
        if occupancy.ready != 0 {
            return Ok(true);
        }
        let snapshot = self.engine.snapshot();
        if snapshot.active_requests > snapshot.output_blocked_requests {
            return Ok(true);
        }
        let made_progress = report.promoted_requests != 0
            || report.committed_positions != 0
            || report.terminal_decisions != 0;
        Ok(snapshot.queued_requests != 0 && snapshot.active_requests == 0 && made_progress)
    }

    fn next_deadline(&self) -> SchedulerResult<Option<Instant>> {
        let now_ns = self.shared.monotonic_ns();
        let next_ns = self
            .accepted
            .iter()
            .filter_map(|entry| {
                let snapshot = entry.control.fresh_snapshot().ok()?;
                (!snapshot.terminal())
                    .then_some(entry.deadline_ns)
                    .flatten()
            })
            .min();
        let Some(deadline_ns) = next_ns else {
            return Ok(None);
        };
        if deadline_ns <= now_ns {
            return Ok(Some(Instant::now()));
        }
        Ok(self
            .shared
            .origin
            .checked_add(Duration::from_nanos(deadline_ns)))
    }

    fn fail(mut self, error: SchedulerError) -> SchedulerResult<ActorShutdownReport> {
        self.shared.closed.store(true, Ordering::Release);
        let _ = self.cancel_accepted();
        self.shared.fail_submissions();
        self.shared.mark_owner_done(true);
        let _ = self.engine.shutdown();
        Err(error)
    }
}

impl<A: DecoderAdapter> Drop for BlockingOwner<A> {
    fn drop(&mut self) {
        if self.shared.activity.owner_done.load(Ordering::Acquire) {
            return;
        }
        self.shared.closed.store(true, Ordering::Release);
        let _ = self.cancel_accepted();
        self.shared.fail_submissions();
        self.shared.mark_owner_done(true);
        let _ = self.engine.shutdown();
    }
}

enum PumpState {
    Continue,
    Wait(Option<Instant>),
    Complete(ActorShutdownReport),
}

fn owner_stopped_error(activity: &Activity) -> SchedulerError {
    if activity.owner_failed.load(Ordering::Acquire) {
        SchedulerError::internal("scheduler actor owner failed")
    } else {
        SchedulerError::internal("scheduler actor stopped before request completion")
    }
}

fn checked_state_mut(
    queue: &mut CommandQueue,
    index: usize,
    ticket: u64,
    expected: CommandPhase,
) -> SchedulerResult<&mut CommandState> {
    let state = queue
        .states
        .get_mut(index)
        .ok_or_else(|| SchedulerError::internal("command state index is out of range"))?;
    if state.ticket != ticket || state.phase != expected {
        return Err(SchedulerError::internal(
            "command state or generation changed unexpectedly",
        ));
    }
    Ok(state)
}

fn make_vacant(queue: &mut CommandQueue, index: usize) {
    if let Err(error) = transition(queue, index, CommandPhase::Vacant) {
        debug_assert!(false, "failed to vacate command slot: {error:?}");
        return;
    }
    let state = &mut queue.states[index];
    state.abandoned = false;
    state.command = None;
    state.response = None;
    debug_assert!(queue.free.len() < queue.free.capacity());
    queue.free.push(index);
    queue.debug_assert_conserved();
}

fn transition(queue: &mut CommandQueue, index: usize, next: CommandPhase) -> SchedulerResult<()> {
    let current = queue
        .states
        .get(index)
        .ok_or_else(|| SchedulerError::internal("command transition index is out of range"))?
        .phase;
    decrement_phase(&mut queue.occupancy, current)?;
    if let Err(error) = increment_phase(&mut queue.occupancy, next) {
        increment_phase(&mut queue.occupancy, current)?;
        return Err(error);
    }
    queue.states[index].phase = next;
    Ok(())
}

fn decrement_phase(occupancy: &mut QueueOccupancy, phase: CommandPhase) -> SchedulerResult<()> {
    let count = match phase {
        CommandPhase::Vacant => return Ok(()),
        CommandPhase::Reserved => &mut occupancy.reserved,
        CommandPhase::Ready => &mut occupancy.ready,
        CommandPhase::InFlight => &mut occupancy.in_flight,
        CommandPhase::Responded => &mut occupancy.responded,
    };
    *count = count
        .checked_sub(1)
        .ok_or_else(|| SchedulerError::internal("actor command phase count underflows"))?;
    Ok(())
}

fn increment_phase(occupancy: &mut QueueOccupancy, phase: CommandPhase) -> SchedulerResult<()> {
    let count = match phase {
        CommandPhase::Vacant => return Ok(()),
        CommandPhase::Reserved => &mut occupancy.reserved,
        CommandPhase::Ready => &mut occupancy.ready,
        CommandPhase::InFlight => &mut occupancy.in_flight,
        CommandPhase::Responded => &mut occupancy.responded,
    };
    *count = count
        .checked_add(1)
        .ok_or_else(|| SchedulerError::internal("actor command phase count overflows"))?;
    Ok(())
}

impl QueueOccupancy {
    const fn total(self) -> usize {
        self.reserved
            .saturating_add(self.ready)
            .saturating_add(self.in_flight)
            .saturating_add(self.responded)
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn try_reserve_vec<T>(
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

fn try_reserve_deque<T>(
    values: &mut VecDeque<T>,
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
    use std::{sync::Arc, time::Duration};

    use runnel_fixture::FixtureArtifact;
    use runnel_format::{Artifact, Limits as ArtifactLimits};
    use runnel_runtime::{BackendRequest, SamplingPolicy, TinyModel};
    use tokio::time::timeout;

    use super::*;
    use crate::{ErrorCategory, SchedulerLimits, TerminalOutcome};

    fn model_and_config(mut limits: SchedulerLimits) -> (TinyModel, SchedulerConfig) {
        limits.worker_count = 1;
        let fixture = FixtureArtifact::build_v3();
        let artifact =
            Artifact::from_bytes(fixture.to_parts(), ArtifactLimits::default()).expect("artifact");
        let model = TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar)
            .expect("scalar model");
        let config = SchedulerConfig::new(&model, limits).expect("actor config");
        (model, config)
    }

    fn request<'a>(
        prompt: &'a [u32],
        max_new_tokens: usize,
        deadline_ns: Option<u64>,
    ) -> RequestSpec<'a> {
        RequestSpec::new(prompt, max_new_tokens, SamplingPolicy::Greedy, deadline_ns)
    }

    async fn wait_submission(submission: Submission) -> RequestHandle {
        timeout(Duration::from_secs(5), submission.wait())
            .await
            .expect("submission timeout")
            .expect("submission accepted")
    }

    async fn wait_terminal(handle: &mut RequestHandle) -> TerminalResult {
        timeout(Duration::from_secs(5), handle.terminal())
            .await
            .expect("terminal timeout")
            .expect("terminal result")
    }

    async fn drain_to_eof(handle: &mut RequestHandle) -> Vec<OutputEvent> {
        let mut events = Vec::new();
        loop {
            match timeout(Duration::from_secs(5), handle.recv_output())
                .await
                .expect("output timeout")
                .expect("output result")
            {
                TryRecvOutput::Output(event) => events.push(event),
                TryRecvOutput::Eof => return events,
                TryRecvOutput::Empty => panic!("blocking receive returned empty"),
            }
        }
    }

    async fn wait_until_parked(shared: &Shared) -> u64 {
        timeout(Duration::from_secs(5), async {
            loop {
                let notified = shared.activity.park_notify.notified();
                let mut notified = std::pin::pin!(notified);
                notified.as_mut().enable();
                if shared.activity.parked.load(Ordering::Acquire)
                    && !shared.activity.pump_in_flight.load(Ordering::Acquire)
                    && !shared.activity.dirty.load(Ordering::Acquire)
                {
                    for _ in 0..16 {
                        tokio::task::yield_now().await;
                    }
                    if shared.activity.parked.load(Ordering::Acquire)
                        && !shared.activity.pump_in_flight.load(Ordering::Acquire)
                    {
                        return shared.observed_engine_steps.load(Ordering::Acquire);
                    }
                }
                notified.await;
            }
        })
        .await
        .expect("actor parked acknowledgement timeout")
    }

    async fn hold_owner(shared: &Shared) {
        let epoch = shared.hold_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        shared.hold_pumps.store(true, Ordering::Release);
        shared.activity.signal();
        timeout(Duration::from_secs(5), async {
            loop {
                let notified = shared.hold_notify.notified();
                let mut notified = std::pin::pin!(notified);
                notified.as_mut().enable();
                if shared.hold_observed.load(Ordering::Acquire) >= epoch {
                    return;
                }
                notified.await;
            }
        })
        .await
        .expect("owner hold acknowledgement timeout");
    }

    fn release_owner(shared: &Shared) {
        shared.hold_pumps.store(false, Ordering::Release);
        shared.activity.signal();
    }

    async fn wait_for_buffered_output(handle: &RequestHandle) {
        timeout(Duration::from_secs(5), async {
            loop {
                if handle
                    .receiver
                    .snapshot()
                    .expect("endpoint snapshot")
                    .buffered_output_events
                    != 0
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("buffered output timeout");
    }

    #[test]
    fn command_slots_are_preallocated_generation_safe_and_skip_exhausted_slots() {
        let (_, config) = model_and_config(SchedulerLimits::tiny());
        let shared = Arc::new(Shared::try_new(config).expect("shared actor state"));
        shared.slots[0]
            .next_ticket
            .store(u64::MAX, Ordering::Release);

        let (index, ticket) = shared.reserve().expect("second live command slot");
        assert_eq!(index, 1);
        assert_eq!(ticket, 1);
        shared
            .commit(
                index,
                ticket,
                SubmitCommand::try_from_request(request(&[1], 0, None)).expect("command"),
            )
            .expect("ready command");
        let submission = Submission {
            shared: Arc::clone(&shared),
            slot: index,
            ticket,
            pending: true,
        };
        drop(submission);
        {
            let mut queue = shared.lock_queue().expect("command queue");
            assert_eq!(queue.occupancy().ready, 1);
            assert!(queue.states[index].abandoned);
            assert_eq!(queue.ready.pop_front(), Some(index));
            transition(&mut queue, index, CommandPhase::InFlight).expect("claim");
            make_vacant(&mut queue, index);
        }

        let (reused, in_flight_ticket) = shared.reserve().expect("reused command slot");
        assert_eq!(reused, index);
        shared
            .commit(
                reused,
                in_flight_ticket,
                SubmitCommand::try_from_request(request(&[1], 0, None)).expect("command"),
            )
            .expect("ready command");
        {
            let mut queue = shared.lock_queue().expect("command queue");
            assert_eq!(queue.ready.pop_front(), Some(reused));
            transition(&mut queue, reused, CommandPhase::InFlight).expect("in-flight command");
        }
        drop(Submission {
            shared: Arc::clone(&shared),
            slot: reused,
            ticket: in_flight_ticket,
            pending: true,
        });
        {
            let mut queue = shared.lock_queue().expect("command queue");
            assert!(queue.states[reused].abandoned);
            make_vacant(&mut queue, reused);
        }

        let (responded, responded_ticket) = shared.reserve().expect("responded command slot");
        shared
            .commit(
                responded,
                responded_ticket,
                SubmitCommand::try_from_request(request(&[1], 0, None)).expect("command"),
            )
            .expect("ready command");
        {
            let mut queue = shared.lock_queue().expect("command queue");
            assert_eq!(queue.ready.pop_front(), Some(responded));
            transition(&mut queue, responded, CommandPhase::InFlight).expect("in-flight command");
            queue.states[responded].command = None;
            transition(&mut queue, responded, CommandPhase::Responded).expect("response");
            queue.states[responded].response = Some(Err(SchedulerError::scheduler_closed()));
        }
        drop(Submission {
            shared: Arc::clone(&shared),
            slot: responded,
            ticket: responded_ticket,
            pending: true,
        });
        assert!(matches!(
            shared.lock_queue().unwrap().states[responded].phase,
            CommandPhase::Vacant
        ));

        let (reused, new_ticket) = shared.reserve().expect("generation test slot");
        assert_ne!(new_ticket, ticket);
        let stale = shared.take_response(reused, ticket).unwrap_err();
        assert_eq!(stale.category(), ErrorCategory::Internal);
        assert!(shared.request_shutdown());
        let closed = shared
            .commit(
                reused,
                new_ticket,
                SubmitCommand::try_from_request(request(&[1], 0, None)).expect("command"),
            )
            .unwrap_err();
        assert_eq!(closed.category(), ErrorCategory::Cancelled);
        shared.release_reserved(reused, new_ticket);
        let queue = shared.lock_queue().expect("final command queue");
        assert_eq!(queue.occupancy(), QueueOccupancy::default());
        assert_eq!(queue.retired, 1);
        assert_eq!(queue.free.len() + queue.retired, queue.states.len());
    }

    #[test]
    fn spawn_without_a_tokio_runtime_fails_cleanly() {
        let (model, config) = model_and_config(SchedulerLimits::tiny());
        let error = SchedulerActor::spawn(model, config).unwrap_err();
        assert_eq!(error.category(), ErrorCategory::Unsupported);
    }

    #[test]
    fn actor_static_prompt_and_control_charges_are_exact() {
        let (_, config) = model_and_config(SchedulerLimits::tiny());
        let shared = config.shared_static_charges();
        assert_eq!(shared.actor_command_bytes(), 8 * 128);
        assert_eq!(shared.actor_control_bytes(), 64 + 8 * 64);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn activity_signal_is_not_lost_across_final_scan_and_park() {
        let activity = Arc::new(Activity::new());
        activity.dirty.store(false, Ordering::Release);
        for turn in 0..256 {
            let waiter_activity = Arc::clone(&activity);
            let waiter = tokio::spawn(async move {
                waiter_activity.wait_until(None).await;
            });
            if turn % 2 == 0 {
                tokio::task::yield_now().await;
            }
            activity.signal();
            timeout(Duration::from_secs(1), waiter)
                .await
                .expect("lost actor activity wake")
                .expect("activity waiter task");
            activity.dirty.store(false, Ordering::Release);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn responded_slots_hold_capacity_then_reuse_without_notify_aba() {
        let mut limits = SchedulerLimits::tiny();
        limits.command_capacity = 1;
        limits.batch_width = 1;
        let (model, config) = model_and_config(limits);
        let actor = SchedulerActor::spawn(model, config).expect("actor");
        let client = actor.client();

        let invalid = client.try_submit(request(&[32], 0, None)).unwrap_err();
        assert_eq!(invalid.category(), ErrorCategory::InvalidRequest);
        assert_eq!(
            client.shared.slots[0].next_ticket.load(Ordering::Acquire),
            0
        );

        let first = client
            .try_submit(request(&[1], 0, None))
            .expect("first command");
        for _ in 0..64 {
            let full = client.try_submit(request(&[1], 0, None)).unwrap_err();
            assert_eq!(full.category(), ErrorCategory::ResourceExhausted);
        }
        assert_eq!(
            client.shared.slots[0].next_ticket.load(Ordering::Acquire),
            1
        );

        let mut first_handle = wait_submission(first).await;
        let first_id = first_handle.request_id();
        let terminal = wait_terminal(&mut first_handle).await;
        assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
        assert!(first_handle.output_eof());

        let second = client
            .try_submit(request(&[1], 0, None))
            .expect("reused command slot");
        let mut second_handle = wait_submission(second).await;
        assert!(second_handle.request_id() > first_id);
        let _ = wait_terminal(&mut second_handle).await;
        drop(first_handle);
        drop(second_handle);

        let report = timeout(Duration::from_secs(5), actor.shutdown())
            .await
            .expect("shutdown timeout")
            .expect("shutdown");
        assert_eq!(report.accepted_submissions(), 2);
        assert_eq!(report.engine().remaining_shared_bytes, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ready_commit_fifo_is_preserved_across_one_command_pumps() {
        let mut limits = SchedulerLimits::tiny();
        limits.command_capacity = 4;
        limits.batch_width = 1;
        let (model, config) = model_and_config(limits);
        let actor = SchedulerActor::spawn(model, config).expect("actor");
        let client = actor.client();
        let submissions = [
            client.try_submit(request(&[1], 0, None)).unwrap(),
            client.try_submit(request(&[14], 0, None)).unwrap(),
            client.try_submit(request(&[16], 0, None)).unwrap(),
        ];
        let mut handles = Vec::new();
        for submission in submissions {
            handles.push(wait_submission(submission).await);
        }
        let ids = handles
            .iter()
            .map(|handle| handle.request_id().get())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![1, 2, 3]);
        for handle in &mut handles {
            assert_eq!(
                wait_terminal(handle).await.outcome(),
                TerminalOutcome::Completed
            );
        }
        drop(handles);
        timeout(Duration::from_secs(5), actor.shutdown())
            .await
            .expect("shutdown timeout")
            .expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn saturated_lane_cannot_block_drain_cancel_drop_or_shutdown() {
        let mut limits = SchedulerLimits::tiny();
        limits.command_capacity = 2;
        limits.batch_width = 1;
        limits.output_capacity_per_request = 1;
        let (model, config) = model_and_config(limits);
        let actor = SchedulerActor::spawn(model, config).expect("actor");
        let client = actor.client();
        let prompt = [1, 14, 16, 6];
        let mut handle =
            wait_submission(client.try_submit(request(&prompt, 4, None)).unwrap()).await;
        let dropped_receiver =
            wait_submission(client.try_submit(request(&prompt, 4, None)).unwrap()).await;

        wait_for_buffered_output(&handle).await;
        let second = client.try_submit(request(&[1], 0, None)).unwrap();
        let third = client.try_submit(request(&[14], 0, None)).unwrap();
        assert_eq!(
            client
                .try_submit(request(&[16], 0, None))
                .unwrap_err()
                .category(),
            ErrorCategory::ResourceExhausted
        );

        assert!(matches!(
            handle.try_recv_output().expect("direct drain"),
            TryRecvOutput::Output(_)
        ));
        assert_eq!(
            handle.cancel().expect("direct cancellation"),
            CancelDisposition::Requested
        );
        drop(dropped_receiver);
        assert_eq!(
            wait_terminal(&mut handle).await.outcome(),
            TerminalOutcome::Cancelled
        );
        let _ = drain_to_eof(&mut handle).await;
        assert!(client.request_shutdown());
        drop(second);
        drop(third);
        drop(handle);

        timeout(Duration::from_secs(5), actor.shutdown())
            .await
            .expect("shutdown timeout")
            .expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn output_blocked_actor_parks_until_direct_receiver_progress() {
        let mut limits = SchedulerLimits::tiny();
        limits.output_capacity_per_request = 1;
        let (model, config) = model_and_config(limits);
        let actor = SchedulerActor::spawn(model, config).expect("actor");
        let client = actor.client();
        let shared = Arc::clone(&client.shared);
        let prompt = [1, 14, 16, 6];
        let mut handle =
            wait_submission(client.try_submit(request(&prompt, 4, None)).unwrap()).await;
        wait_for_buffered_output(&handle).await;
        let parked = wait_until_parked(&shared).await;
        for _ in 0..256 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            shared.observed_engine_steps.load(Ordering::Acquire),
            parked,
            "output-blocked actor busy-spun"
        );

        assert!(matches!(
            handle.try_recv_output().unwrap(),
            TryRecvOutput::Output(_)
        ));
        timeout(Duration::from_secs(5), async {
            while shared.observed_engine_steps.load(Ordering::Acquire) <= parked {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("receiver progress did not wake actor pump");
        let _ = handle.cancel();
        let _ = wait_terminal(&mut handle).await;
        let _ = drain_to_eof(&mut handle).await;
        drop(handle);
        timeout(Duration::from_secs(5), actor.shutdown())
            .await
            .expect("shutdown timeout")
            .expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queued_deadline_is_rechecked_before_id_and_max_deadline_is_valid() {
        let mut limits = SchedulerLimits::tiny();
        limits.command_capacity = 2;
        limits.batch_width = 1;
        let (model, config) = model_and_config(limits);
        let actor = SchedulerActor::spawn(model, config).expect("actor");
        let client = actor.client();
        hold_owner(&client.shared).await;

        let deadline = client
            .deadline_after(Duration::from_millis(20))
            .expect("deadline");
        let expired = client
            .try_submit(request(&[1], 0, Some(deadline)))
            .expect("queued expiring command");
        tokio::time::sleep(Duration::from_millis(30)).await;
        release_owner(&client.shared);
        let error = timeout(Duration::from_secs(5), expired.wait())
            .await
            .expect("expired response timeout")
            .unwrap_err();
        assert_eq!(error.category(), ErrorCategory::DeadlineExceeded);

        let maximum = client
            .try_submit(request(&[1], 0, Some(u64::MAX)))
            .expect("maximum deadline command");
        let mut handle = wait_submission(maximum).await;
        assert_eq!(handle.request_id().get(), 1, "expired offer consumed an ID");
        assert_eq!(
            wait_terminal(&mut handle).await.outcome(),
            TerminalOutcome::Completed
        );
        assert!(
            client
                .deadline_after(Duration::from_nanos(u64::MAX))
                .is_err()
        );
        drop(handle);
        timeout(Duration::from_secs(5), actor.shutdown())
            .await
            .expect("shutdown timeout")
            .expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn terminal_first_and_zero_output_paths_acknowledge_eof_and_reap() {
        let mut limits = SchedulerLimits::tiny();
        limits.output_capacity_per_request = 1;
        let (model, config) = model_and_config(limits);
        let actor = SchedulerActor::spawn(model, config).expect("actor");
        let client = actor.client();

        let mut zero = wait_submission(client.try_submit(request(&[1], 0, None)).unwrap()).await;
        assert_eq!(
            wait_terminal(&mut zero).await.outcome(),
            TerminalOutcome::Completed
        );
        assert!(zero.output_eof());
        assert_eq!(zero.try_recv_output().unwrap(), TryRecvOutput::Eof);

        let mut one = wait_submission(client.try_submit(request(&[1], 1, None)).unwrap()).await;
        assert_eq!(
            wait_terminal(&mut one).await.outcome(),
            TerminalOutcome::Completed
        );
        assert!(!one.output_eof());
        assert!(matches!(
            one.try_recv_output().unwrap(),
            TryRecvOutput::Output(_)
        ));
        assert!(one.output_eof());
        assert_eq!(one.try_recv_output().unwrap(), TryRecvOutput::Eof);

        let report = timeout(Duration::from_secs(5), actor.shutdown())
            .await
            .expect("shutdown timeout")
            .expect("shutdown");
        assert_eq!(report.engine().remaining_shared_bytes, 0);
        drop(zero);
        drop(one);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_waits_for_live_receiver_then_drop_releases_it() {
        let (model, config) = model_and_config(SchedulerLimits::tiny());
        let actor = SchedulerActor::spawn(model, config).expect("actor");
        let client = actor.client();
        let handle = wait_submission(client.try_submit(request(&[1], 0, None)).unwrap()).await;
        let shutdown = tokio::spawn(actor.shutdown());
        for _ in 0..256 {
            tokio::task::yield_now().await;
        }
        assert!(
            !shutdown.is_finished(),
            "shutdown ignored live receiver ownership"
        );
        drop(handle);
        let report = timeout(Duration::from_secs(5), shutdown)
            .await
            .expect("shutdown timeout")
            .expect("shutdown task")
            .expect("shutdown result");
        assert!(report.shutdown_cancellations() <= report.accepted_submissions());
        assert_eq!(report.engine().remaining_shared_bytes, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pump_panic_wakes_pending_submission_and_parked_request_handle() {
        let mut limits = SchedulerLimits::tiny();
        limits.output_capacity_per_request = 1;
        limits.command_capacity = 2;
        let (model, config) = model_and_config(limits);
        let actor = SchedulerActor::spawn(model, config).expect("actor");
        let client = actor.client();
        let prompt = [1, 14, 16, 6];
        let mut handle =
            wait_submission(client.try_submit(request(&prompt, 4, None)).unwrap()).await;
        wait_for_buffered_output(&handle).await;

        hold_owner(&client.shared).await;
        let pending = client.try_submit(request(&[1], 0, None)).unwrap();
        assert_eq!(client.shared.lock_queue().unwrap().occupancy().ready, 1);
        let handle_wait = tokio::spawn(async move { handle.terminal().await });
        for _ in 0..32 {
            tokio::task::yield_now().await;
        }
        client.shared.panic_next_pump.store(true, Ordering::Release);
        release_owner(&client.shared);

        let submission_error = timeout(Duration::from_secs(5), pending.wait())
            .await
            .expect("submission failure timeout")
            .unwrap_err();
        assert_eq!(submission_error.category(), ErrorCategory::Internal);
        let handle_error = timeout(Duration::from_secs(5), handle_wait)
            .await
            .expect("handle failure timeout")
            .expect("handle task")
            .unwrap_err();
        assert_eq!(handle_error.category(), ErrorCategory::Internal);
        let actor_error = timeout(Duration::from_secs(5), actor.shutdown())
            .await
            .expect("actor failure timeout")
            .unwrap_err();
        assert_eq!(actor_error.category(), ErrorCategory::Internal);
    }
}
