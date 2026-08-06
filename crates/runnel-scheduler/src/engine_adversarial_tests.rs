use std::{
    mem::size_of,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread,
};

use runnel_runtime::{
    AdapterExecutionLayout, AdapterWorkIdentity, DecoderAdapter, Result as RuntimeResult,
    RuntimeError, SamplingPolicy, StateLayoutAccounting,
};

use crate::control::ControlBinding;
use crate::endpoint::TryPop;
use crate::{
    BatchRequestSpec, CancelDisposition, ErrorCategory, LedgerCategory, LedgerOwnership,
    LedgerTraceCursor, RequestPhase, RequestSpec, SchedulerConfig, SchedulerEngine, SchedulerError,
    SchedulerLimits, ServiceTraceCursor, StepReport, TerminalOutcome,
};

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
struct GateState {
    claimed: bool,
    arrived: bool,
    released: bool,
}

/// A deterministic, no-sleep rendezvous immediately before the adapter's
/// validated commit callback. Only the first callback is held.
#[derive(Default)]
struct CommitGate {
    state: Mutex<GateState>,
    changed: Condvar,
}

impl CommitGate {
    fn passthrough() -> Self {
        Self {
            state: Mutex::new(GateState {
                claimed: true,
                arrived: false,
                released: true,
            }),
            changed: Condvar::new(),
        }
    }

    fn block_first_callback(&self) {
        let mut state = self.state.lock().expect("commit gate lock");
        if state.claimed {
            return;
        }
        state.claimed = true;
        state.arrived = true;
        self.changed.notify_all();
        while !state.released {
            state = self.changed.wait(state).expect("commit gate wait");
        }
    }

    fn wait_until_arrived(&self) {
        let mut state = self.state.lock().expect("commit gate lock");
        while !state.arrived {
            state = self.changed.wait(state).expect("commit gate wait");
        }
    }

    fn release(&self) {
        let mut state = self.state.lock().expect("commit gate lock");
        assert!(state.arrived, "commit gate released before adapter arrived");
        state.released = true;
        self.changed.notify_all();
    }
}

#[derive(Default)]
struct ApplyCancelHook {
    binding: Mutex<Option<ControlBinding>>,
    disposition: Mutex<Option<Result<CancelDisposition, ErrorCategory>>>,
}

impl ApplyCancelHook {
    fn install(&self, binding: ControlBinding) {
        let previous = self
            .binding
            .lock()
            .expect("apply-cancel binding lock")
            .replace(binding);
        assert!(previous.is_none(), "apply-cancel binding installed twice");
    }

    fn cancel_before_state_mutation(&self) {
        let Some(binding) = self
            .binding
            .lock()
            .expect("apply-cancel binding lock")
            .take()
        else {
            return;
        };
        let disposition = binding.cancel().map_err(|error| error.category());
        let previous = self
            .disposition
            .lock()
            .expect("apply-cancel result lock")
            .replace(disposition);
        assert!(previous.is_none(), "apply-cancel result published twice");
    }

    fn disposition(&self) -> Option<Result<CancelDisposition, ErrorCategory>> {
        *self.disposition.lock().expect("apply-cancel result lock")
    }
}

#[derive(Clone, Copy)]
struct GateStateLayout {
    max_tokens: usize,
    payload_bytes: usize,
    charge_bytes: usize,
}

impl StateLayoutAccounting for GateStateLayout {
    fn payload_bytes(self) -> usize {
        self.payload_bytes
    }

    fn charge_bytes(self) -> usize {
        self.charge_bytes
    }
}

struct DecoderState {
    identity: u64,
    revision: u64,
    position: usize,
    tokens: Vec<u32>,
    apply_count: Arc<AtomicUsize>,
    apply_cancel: Option<Arc<ApplyCancelHook>>,
}

struct PreparedToken {
    identity: AdapterWorkIdentity,
    input_token: u32,
}

struct ExpertTask {
    identity: AdapterWorkIdentity,
    input_token: u32,
}

struct ExpertContribution {
    identity: AdapterWorkIdentity,
    input_token: u32,
}

struct PendingCommit {
    identity: AdapterWorkIdentity,
    input_token: u32,
    logits: [f32; 4],
}

struct CommitPermit<'a> {
    state: &'a mut DecoderState,
    input_token: u32,
    next_position: usize,
    next_revision: u64,
}

struct GatedAdapter {
    model_id: u64,
    next_state_id: AtomicU64,
    next_transaction_id: AtomicU64,
    gate: Arc<CommitGate>,
    apply_count: Arc<AtomicUsize>,
    apply_cancel: Option<Arc<ApplyCancelHook>>,
}

impl GatedAdapter {
    fn new(gate: Arc<CommitGate>, apply_count: Arc<AtomicUsize>) -> Self {
        Self {
            model_id: issue(&NEXT_MODEL_ID, RuntimeError::ModelIdentityExhausted)
                .expect("test model identity"),
            next_state_id: AtomicU64::new(1),
            next_transaction_id: AtomicU64::new(1),
            gate,
            apply_count,
            apply_cancel: None,
        }
    }

    fn with_apply_cancel(
        gate: Arc<CommitGate>,
        apply_count: Arc<AtomicUsize>,
        apply_cancel: Arc<ApplyCancelHook>,
    ) -> Self {
        Self {
            apply_cancel: Some(apply_cancel),
            ..Self::new(gate, apply_count)
        }
    }
}

impl DecoderAdapter for GatedAdapter {
    type StateLayout = GateStateLayout;
    type State = DecoderState;
    type Workspace = ();
    type PreparedToken = PreparedToken;
    type ExpertTask = ExpertTask;
    type ExpertTasks = std::array::IntoIter<ExpertTask, 1>;
    type ExpertContribution = ExpertContribution;
    type PendingStateCommit = PendingCommit;
    type StateCommitPermit<'a> = CommitPermit<'a>;

    fn execution_layout(&self) -> RuntimeResult<AdapterExecutionLayout> {
        AdapterExecutionLayout::new(
            1,
            size_of::<PreparedToken>(),
            size_of::<ExpertTask>(),
            size_of::<ExpertContribution>(),
            size_of::<PendingCommit>(),
            0,
            0,
        )
    }

    fn vocabulary_size(&self) -> usize {
        4
    }

    fn is_stop_token(&self, _token: u32) -> bool {
        false
    }

    fn state_layout(
        &self,
        max_tokens: usize,
        page_tokens: usize,
    ) -> RuntimeResult<GateStateLayout> {
        if max_tokens == 0 || page_tokens == 0 {
            return Err(RuntimeError::InvalidStateLayout(
                "gated test state geometry must be nonzero",
            ));
        }
        let payload_bytes =
            max_tokens
                .checked_mul(size_of::<u32>())
                .ok_or(RuntimeError::ResourceSizeOverflow {
                    resource: "gated test state payload",
                })?;
        let charge_bytes = payload_bytes
            .checked_add(63)
            .map(|bytes| bytes / 64 * 64)
            .ok_or(RuntimeError::ResourceSizeOverflow {
                resource: "gated test state charge",
            })?;
        Ok(GateStateLayout {
            max_tokens,
            payload_bytes,
            charge_bytes,
        })
    }

    fn new_state(&self, layout: GateStateLayout) -> RuntimeResult<DecoderState> {
        let mut tokens = Vec::new();
        tokens.try_reserve_exact(layout.max_tokens).map_err(|_| {
            RuntimeError::ResourceExhausted {
                resource: "gated test state",
                bytes: layout.payload_bytes,
            }
        })?;
        tokens.resize(layout.max_tokens, 0);
        Ok(DecoderState {
            identity: issue(&self.next_state_id, RuntimeError::StateIdentityExhausted)?,
            revision: 0,
            position: 0,
            tokens,
            apply_count: Arc::clone(&self.apply_count),
            apply_cancel: self.apply_cancel.as_ref().map(Arc::clone),
        })
    }

    fn new_workspace(&self) -> RuntimeResult<()> {
        Ok(())
    }

    fn prepare_token(
        &self,
        state: &DecoderState,
        token: u32,
        _workspace: &mut (),
    ) -> RuntimeResult<PreparedToken> {
        if usize::try_from(token).map_or(true, |token| token >= self.vocabulary_size()) {
            return Err(RuntimeError::InvalidToken {
                vocab_size: self.vocabulary_size(),
            });
        }
        if state.position >= state.tokens.len() {
            return Err(RuntimeError::ContextLimit {
                limit: state.tokens.len(),
            });
        }
        state
            .revision
            .checked_add(1)
            .ok_or(RuntimeError::StateRevisionExhausted)?;
        let transaction_id = issue(
            &self.next_transaction_id,
            RuntimeError::AdapterTransactionIdentityExhausted,
        )?;
        Ok(PreparedToken {
            identity: AdapterWorkIdentity::try_new(
                transaction_id,
                self.model_id,
                state.identity,
                state.revision,
                state.position,
            )?,
            input_token: token,
        })
    }

    fn prepared_identity(&self, prepared: &PreparedToken) -> AdapterWorkIdentity {
        prepared.identity
    }

    fn expert_tasks(&self, prepared: &PreparedToken) -> Self::ExpertTasks {
        [ExpertTask {
            identity: prepared.identity,
            input_token: prepared.input_token,
        }]
        .into_iter()
    }

    fn task_identity(&self, task: &ExpertTask) -> AdapterWorkIdentity {
        task.identity
    }

    fn task_router_rank(&self, _task: &ExpertTask) -> u16 {
        0
    }

    fn task_expert_id(&self, _task: &ExpertTask) -> u16 {
        0
    }

    fn execute_expert(
        &self,
        task: ExpertTask,
        _workspace: &mut (),
    ) -> RuntimeResult<ExpertContribution> {
        Ok(ExpertContribution {
            identity: task.identity,
            input_token: task.input_token,
        })
    }

    fn contribution_identity(&self, contribution: &ExpertContribution) -> AdapterWorkIdentity {
        contribution.identity
    }

    fn contribution_router_rank(&self, _contribution: &ExpertContribution) -> u16 {
        0
    }

    fn contribution_expert_id(&self, _contribution: &ExpertContribution) -> u16 {
        0
    }

    fn finish_token(
        &self,
        prepared: PreparedToken,
        contributions: &[ExpertContribution],
        _workspace: &mut (),
    ) -> RuntimeResult<PendingCommit> {
        let [contribution] = contributions else {
            return Err(RuntimeError::InvalidExpertContribution(
                "gated test adapter expects one contribution",
            ));
        };
        if contribution.identity != prepared.identity
            || contribution.input_token != prepared.input_token
        {
            return Err(RuntimeError::InvalidExpertContribution(
                "gated test contribution identity mismatch",
            ));
        }
        let sampled_token = (prepared.input_token + 1) % 4;
        let mut logits = [0.0_f32; 4];
        logits[usize::try_from(sampled_token).expect("sampled token index")] = 10.0;
        Ok(PendingCommit {
            identity: prepared.identity,
            input_token: prepared.input_token,
            logits,
        })
    }

    fn pending_identity(&self, pending: &PendingCommit) -> AdapterWorkIdentity {
        pending.identity
    }

    fn pending_logits<'a>(&self, pending: &'a PendingCommit) -> &'a [f32] {
        &pending.logits
    }

    fn with_validated_state_commit<R, F>(
        &self,
        state: &mut DecoderState,
        pending: &PendingCommit,
        apply: F,
    ) -> RuntimeResult<R>
    where
        F: for<'permit> FnOnce(CommitPermit<'permit>) -> R,
    {
        if pending.identity.model_instance_id() != self.model_id
            || pending.identity.state_id().get() != state.identity
            || pending.identity.state_revision() != state.revision
            || pending.identity.position() != state.position
        {
            return Err(RuntimeError::InvalidAdapterWork(
                "gated test pending identity mismatch",
            ));
        }
        let next_position = state
            .position
            .checked_add(1)
            .ok_or(RuntimeError::StateRevisionExhausted)?;
        if next_position > state.tokens.len() {
            return Err(RuntimeError::ContextLimit {
                limit: state.tokens.len(),
            });
        }
        let next_revision = state
            .revision
            .checked_add(1)
            .ok_or(RuntimeError::StateRevisionExhausted)?;

        self.gate.block_first_callback();
        Ok(apply(CommitPermit {
            state,
            input_token: pending.input_token,
            next_position,
            next_revision,
        }))
    }

    fn apply_state_commit(permit: CommitPermit<'_>) {
        if let Some(hook) = &permit.state.apply_cancel {
            hook.cancel_before_state_mutation();
        }
        permit.state.tokens[permit.state.position] = permit.input_token;
        permit.state.position = permit.next_position;
        permit.state.revision = permit.next_revision;
        permit.state.apply_count.fetch_add(1, Ordering::AcqRel);
    }
}

fn issue(counter: &AtomicU64, exhausted: RuntimeError) -> RuntimeResult<u64> {
    let mut candidate = counter.load(Ordering::Relaxed);
    loop {
        if candidate == 0 {
            return Err(exhausted);
        }
        let successor = candidate.checked_add(1).unwrap_or(0);
        match counter.compare_exchange_weak(
            candidate,
            successor,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Ok(candidate),
            Err(observed) => candidate = observed,
        }
    }
}

fn new_engine(
    gate: Arc<CommitGate>,
    apply_count: Arc<AtomicUsize>,
    batch_width: u64,
) -> SchedulerEngine<GatedAdapter> {
    let adapter = GatedAdapter::new(gate, apply_count);
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = batch_width;
    limits.waves_per_step = 1;
    let config = SchedulerConfig::new(&adapter, limits).expect("gated scheduler config");
    SchedulerEngine::new(adapter, config).expect("gated scheduler engine")
}

fn assert_no_active_request_ownership(engine: &SchedulerEngine<GatedAdapter>) {
    let ledger = engine.ledger_snapshot();
    for category in [
        LedgerCategory::PromptStorage,
        LedgerCategory::ActiveState,
        LedgerCategory::PendingTransaction,
    ] {
        assert_eq!(
            ledger.category(category).used(),
            0,
            "{category} ownership leaked after terminalization"
        );
    }
}

fn assert_all_request_ownership_reaped(
    engine: &SchedulerEngine<GatedAdapter>,
    pristine_total_used: u64,
    pristine_shared_used: u64,
) {
    let ledger = engine.ledger_snapshot();
    assert_eq!(ledger.request_used(), 0);
    assert_eq!(ledger.shared_used(), pristine_shared_used);
    assert_eq!(ledger.total_used(), pristine_total_used);
    for category in LedgerCategory::ALL {
        if category.ownership() == LedgerOwnership::Request {
            assert_eq!(
                ledger.category(category).used(),
                0,
                "{category} ownership survived final reap"
            );
        }
    }
}

#[test]
fn externally_owned_endpoint_reaps_after_normal_two_sided_acknowledgement() {
    let gate = Arc::new(CommitGate::passthrough());
    let apply_count = Arc::new(AtomicUsize::new(0));
    let mut engine = new_engine(gate, apply_count, 1);
    let pristine = engine.ledger_snapshot();
    let admission = engine
        .try_submit_for_actor(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("accepted external endpoint request");
    let (request_id, control, mut receiver) = admission.into_parts();

    let report = engine.step().expect("terminal actor-owned request");
    assert_eq!(report.committed_positions, 1);
    assert_eq!(report.terminal_decisions, 1);
    assert!(matches!(
        receiver.try_pop().expect("external output"),
        TryPop::Event(_)
    ));
    assert_eq!(receiver.try_pop().expect("external EOF"), TryPop::Eof);
    assert!(
        receiver
            .take_terminal()
            .expect("external terminal")
            .is_some()
    );

    engine.step().expect("reap acknowledged external endpoint");
    assert_eq!(
        engine
            .request_phase(request_id)
            .expect_err("acknowledged request must be reaped")
            .category(),
        ErrorCategory::InvalidRequest
    );
    assert_eq!(
        control.cancel().expect("terminal control remains stable"),
        CancelDisposition::AlreadyTerminal
    );
    assert_all_request_ownership_reaped(&engine, pristine.total_used(), pristine.shared_used());
}

#[test]
fn cancellation_during_adapter_validation_suppresses_commit_and_recovers_ring_credit() {
    let gate = Arc::new(CommitGate::default());
    let apply_count = Arc::new(AtomicUsize::new(0));
    let mut engine = new_engine(Arc::clone(&gate), Arc::clone(&apply_count), 2);
    let pristine = engine.ledger_snapshot();

    let cancelled = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("accepted cancellable request");
    let sibling = engine
        .try_submit(RequestSpec::new(&[1], 2, SamplingPolicy::Greedy, None))
        .expect("accepted sibling request");
    let control = engine
        .control_binding(cancelled)
        .expect("cancellable control binding");

    let worker = thread::spawn(move || {
        let report = engine
            .step_with_clock(&|| 0)
            .expect("gated cancellation step");
        (engine, report)
    });
    gate.wait_until_arrived();
    assert_eq!(
        control.cancel().expect("boundary cancellation"),
        CancelDisposition::Requested
    );
    gate.release();
    let (mut engine, report) = worker.join().expect("scheduler worker");

    assert_eq!(report.selected_positions, 2);
    assert_eq!(report.committed_positions, 1);
    assert_eq!(report.terminal_decisions, 1);
    assert_eq!(apply_count.load(Ordering::Acquire), 1);
    assert_eq!(
        engine.request_phase(cancelled).expect("cancelled phase"),
        RequestPhase::Terminal
    );
    assert_eq!(
        engine.request_phase(sibling).expect("sibling phase"),
        RequestPhase::Ready
    );

    let recovery = engine.step().expect("ring service after suppression");
    assert_eq!(recovery.selected_positions, 1);
    assert_eq!(recovery.committed_positions, 1);
    assert_eq!(recovery.terminal_decisions, 1);
    assert_eq!(apply_count.load(Ordering::Acquire), 2);
    assert_no_active_request_ownership(&engine);

    assert!(
        engine
            .drain_events(cancelled, usize::MAX)
            .expect("cancelled output query")
            .is_empty()
    );
    let cancelled_terminal = engine
        .take_terminal(cancelled)
        .expect("cancelled terminal query")
        .expect("cancelled terminal result");
    assert_eq!(cancelled_terminal.outcome(), TerminalOutcome::Cancelled);
    assert_eq!(cancelled_terminal.committed_positions(), 0);
    assert_eq!(cancelled_terminal.emitted_tokens(), 0);

    let sibling_events = engine
        .drain_events(sibling, usize::MAX)
        .expect("sibling output drain");
    assert_eq!(sibling_events.len(), 2);
    assert_eq!(
        sibling_events
            .iter()
            .map(|event| event.output_index())
            .collect::<Vec<_>>(),
        [0, 1]
    );
    let sibling_terminal = engine
        .take_terminal(sibling)
        .expect("sibling terminal query")
        .expect("sibling terminal result");
    assert_eq!(sibling_terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(sibling_terminal.committed_positions(), 2);
    assert_eq!(sibling_terminal.emitted_tokens(), 2);
    assert_all_request_ownership_reaped(&engine, pristine.total_used(), pristine.shared_used());
}

#[test]
fn inclusive_deadline_crossing_during_adapter_validation_suppresses_commit() {
    let gate = Arc::new(CommitGate::default());
    let apply_count = Arc::new(AtomicUsize::new(0));
    let clock = Arc::new(AtomicU64::new(9));
    let mut engine = new_engine(Arc::clone(&gate), Arc::clone(&apply_count), 1);
    let pristine = engine.ledger_snapshot();
    let request = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, Some(10)))
        .expect("accepted deadline request");

    let worker_clock = Arc::clone(&clock);
    let worker = thread::spawn(move || {
        let report = engine
            .step_with_clock(&|| worker_clock.load(Ordering::Acquire))
            .expect("gated deadline step");
        (engine, report)
    });
    gate.wait_until_arrived();
    clock.store(10, Ordering::Release);
    gate.release();
    let (mut engine, report) = worker.join().expect("scheduler worker");

    assert_eq!(report.selected_positions, 1);
    assert_eq!(report.committed_positions, 0);
    assert_eq!(report.terminal_decisions, 1);
    assert_eq!(apply_count.load(Ordering::Acquire), 0);
    assert_eq!(engine.snapshot().monotonic_ns, 10);
    assert_no_active_request_ownership(&engine);
    assert!(
        engine
            .drain_events(request, usize::MAX)
            .expect("deadline output query")
            .is_empty()
    );
    let terminal = engine
        .take_terminal(request)
        .expect("deadline terminal query")
        .expect("deadline terminal result");
    assert_eq!(terminal.outcome(), TerminalOutcome::DeadlineExceeded);
    assert_eq!(terminal.committed_positions(), 0);
    assert_eq!(terminal.emitted_tokens(), 0);
    assert_all_request_ownership_reaped(&engine, pristine.total_used(), pristine.shared_used());
}

#[test]
fn cancellation_precedes_an_inclusive_deadline_at_the_final_boundary() {
    let gate = Arc::new(CommitGate::default());
    let apply_count = Arc::new(AtomicUsize::new(0));
    let clock = Arc::new(AtomicU64::new(9));
    let mut engine = new_engine(Arc::clone(&gate), Arc::clone(&apply_count), 1);
    let pristine = engine.ledger_snapshot();
    let request = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, Some(10)))
        .expect("accepted precedence request");
    let control = engine
        .control_binding(request)
        .expect("precedence control binding");

    let worker_clock = Arc::clone(&clock);
    let worker = thread::spawn(move || {
        let report = engine
            .step_with_clock(&|| worker_clock.load(Ordering::Acquire))
            .expect("gated precedence step");
        (engine, report)
    });
    gate.wait_until_arrived();
    assert_eq!(
        control.cancel().expect("boundary cancellation"),
        CancelDisposition::Requested
    );
    clock.store(10, Ordering::Release);
    gate.release();
    let (mut engine, report) = worker.join().expect("scheduler worker");

    assert_eq!(report.selected_positions, 1);
    assert_eq!(report.committed_positions, 0);
    assert_eq!(report.terminal_decisions, 1);
    assert_eq!(apply_count.load(Ordering::Acquire), 0);
    assert_eq!(engine.snapshot().monotonic_ns, 10);
    assert_no_active_request_ownership(&engine);
    assert!(
        engine
            .drain_events(request, usize::MAX)
            .expect("precedence output query")
            .is_empty()
    );
    let terminal = engine
        .take_terminal(request)
        .expect("precedence terminal query")
        .expect("precedence terminal result");
    assert_eq!(terminal.outcome(), TerminalOutcome::Cancelled);
    assert_eq!(terminal.committed_positions(), 0);
    assert_eq!(terminal.emitted_tokens(), 0);
    assert_all_request_ownership_reaped(&engine, pristine.total_used(), pristine.shared_used());
}

#[test]
fn cancellation_after_the_final_snapshot_loses_to_the_committed_position() {
    let gate = Arc::new(CommitGate::passthrough());
    let apply_count = Arc::new(AtomicUsize::new(0));
    let apply_cancel = Arc::new(ApplyCancelHook::default());
    let adapter =
        GatedAdapter::with_apply_cancel(gate, Arc::clone(&apply_count), Arc::clone(&apply_cancel));
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 1;
    limits.waves_per_step = 1;
    let config = SchedulerConfig::new(&adapter, limits).expect("late-cancel scheduler config");
    let mut engine = SchedulerEngine::new(adapter, config).expect("late-cancel scheduler engine");
    let pristine = engine.ledger_snapshot();
    let request = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("accepted late-cancel request");
    apply_cancel.install(
        engine
            .control_binding(request)
            .expect("late-cancel control binding"),
    );

    let report = engine.step().expect("late-cancel step");
    assert_eq!(
        report,
        StepReport {
            promoted_requests: 1,
            waves: 1,
            selected_positions: 1,
            expert_tasks: 1,
            expert_groups: 1,
            committed_positions: 1,
            terminal_decisions: 1,
        }
    );
    assert_eq!(
        apply_cancel.disposition(),
        Some(Ok(CancelDisposition::Requested))
    );
    assert_eq!(apply_count.load(Ordering::Acquire), 1);
    assert_eq!(
        engine.request_phase(request).expect("late-cancel phase"),
        RequestPhase::Terminal
    );
    assert_eq!(
        engine
            .cancel(request)
            .expect("late-cancel terminal control"),
        CancelDisposition::AlreadyTerminal
    );
    assert_no_active_request_ownership(&engine);

    let events = engine
        .drain_events(request, usize::MAX)
        .expect("late-cancel output drain");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].request_id(), request);
    assert_eq!(events[0].output_index(), 0);
    assert_eq!(events[0].token(), 1);
    let terminal = engine
        .take_terminal(request)
        .expect("late-cancel terminal query")
        .expect("late-cancel terminal result");
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(terminal.committed_positions(), 1);
    assert_eq!(terminal.emitted_tokens(), 1);
    assert_all_request_ownership_reaped(&engine, pristine.total_used(), pristine.shared_used());
}

#[test]
fn recycled_engine_slot_rejects_stale_control_without_affecting_its_new_request() {
    let gate = Arc::new(CommitGate::default());
    let apply_count = Arc::new(AtomicUsize::new(0));
    let adapter = GatedAdapter::new(Arc::clone(&gate), Arc::clone(&apply_count));
    let mut limits = SchedulerLimits::tiny();
    limits.max_outstanding_requests = 1;
    limits.max_active_requests = 1;
    limits.max_queued_requests = 1;
    limits.max_retained_terminal_results = 1;
    limits.batch_width = 1;
    limits.waves_per_step = 1;
    let config = SchedulerConfig::new(&adapter, limits).expect("single-slot scheduler config");
    let mut engine = SchedulerEngine::new(adapter, config).expect("single-slot scheduler engine");
    let pristine = engine.ledger_snapshot();

    let first = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("accepted first slot generation");
    let stale = engine
        .control_binding(first)
        .expect("first-generation control binding");
    let worker = thread::spawn(move || {
        let report = engine.step().expect("first-generation step");
        (engine, report)
    });
    gate.wait_until_arrived();
    gate.release();
    let (mut engine, first_report) = worker.join().expect("scheduler worker");
    assert_eq!(first_report.committed_positions, 1);
    assert_eq!(first_report.terminal_decisions, 1);
    assert_eq!(
        engine
            .drain_events(first, usize::MAX)
            .expect("first-generation output")
            .len(),
        1
    );
    assert_eq!(
        engine
            .take_terminal(first)
            .expect("first-generation terminal query")
            .expect("first-generation terminal")
            .outcome(),
        TerminalOutcome::Completed
    );

    let current = engine
        .try_submit(RequestSpec::new(&[1], 1, SamplingPolicy::Greedy, None))
        .expect("accepted recycled slot generation");
    assert_ne!(first, current);
    let stale_error = stale
        .cancel()
        .expect_err("stale generation must not cancel");
    assert_eq!(stale_error.category(), ErrorCategory::InvalidRequest);

    let current_report = engine.step().expect("current-generation step");
    assert_eq!(current_report.committed_positions, 1);
    assert_eq!(current_report.terminal_decisions, 1);
    assert_eq!(apply_count.load(Ordering::Acquire), 2);
    assert_eq!(
        engine
            .drain_events(current, usize::MAX)
            .expect("current-generation output")
            .len(),
        1
    );
    let terminal = engine
        .take_terminal(current)
        .expect("current-generation terminal query")
        .expect("current-generation terminal");
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(terminal.committed_positions(), 1);
    assert_eq!(terminal.emitted_tokens(), 1);
    assert_all_request_ownership_reaped(&engine, pristine.total_used(), pristine.shared_used());
}

#[test]
fn batch_allocation_failure_after_provisional_charge_rolls_back_every_owner_and_identity() {
    let gate = Arc::new(CommitGate::passthrough());
    let apply_count = Arc::new(AtomicUsize::new(0));
    let mut engine = new_engine(gate, apply_count, 2);
    let before_engine = engine.snapshot();
    let before_ledger = engine.ledger_snapshot();
    let prompt = [0_u32; 2];
    let offers =
        [BatchRequestSpec::release_relative(&prompt, 1, SamplingPolicy::Greedy, 20_000_000); 2];

    crate::engine::fail_batch_allocation_after_for_test(1);
    let error = engine
        .prepare_submit_batch(&offers)
        .expect_err("second prepared payload allocation is injected to fail");
    assert!(matches!(
        error,
        SchedulerError::AllocationFailure {
            resource: "prepared batch request payload",
            ..
        }
    ));
    assert_eq!(engine.snapshot(), before_engine);
    assert_eq!(engine.ledger_snapshot(), before_ledger);
    let trace = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("ledger trace after injected allocation rollback");
    assert!(trace.events().is_empty());
    assert!(trace.status().healthy());
    assert_eq!(trace.initial_snapshot().ledger_snapshot(), before_ledger);
    let first = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("failed prepare consumed no request identity");
    assert_eq!(first.get(), 1);
}

#[test]
fn failed_shutdown_retains_both_trace_prefixes_and_can_be_retried() {
    let gate = Arc::new(CommitGate::passthrough());
    let apply_count = Arc::new(AtomicUsize::new(0));
    let mut engine = new_engine(gate, apply_count, 1);
    let request_id = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("shutdown test admission");
    let report = engine.step().expect("shutdown test service");
    assert_eq!(report.committed_positions, 1);
    assert_eq!(report.terminal_decisions, 1);
    assert_eq!(
        engine
            .drain_events(request_id, usize::MAX)
            .expect("shutdown test output")
            .len(),
        1
    );
    assert_eq!(
        engine
            .take_terminal(request_id)
            .expect("shutdown test terminal query")
            .expect("shutdown test terminal")
            .outcome(),
        TerminalOutcome::Completed
    );
    assert_eq!(engine.ledger_snapshot().request_used(), 0);

    let service_before = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("service trace before failed shutdown");
    let service_events = service_before.events().to_vec();
    let service_status = service_before.status();
    drop(service_before);
    let ledger_before = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("ledger trace before failed shutdown");
    let initial = ledger_before.initial_snapshot();
    let ledger_events = ledger_before.events().to_vec();
    let ledger_status = ledger_before.status();
    drop(ledger_before);
    let snapshot = engine.ledger_snapshot();

    crate::engine::fail_next_shutdown_release_for_test();
    let error = engine
        .shutdown()
        .expect_err("injected shutdown release failure");
    assert!(matches!(
        error,
        SchedulerError::AllocationFailure {
            resource: "shutdown release checkpoint",
            ..
        }
    ));
    assert!(engine.snapshot().closed);
    assert_eq!(engine.ledger_snapshot(), snapshot);

    let service_after = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("service trace retained after failed shutdown");
    assert_eq!(service_after.events(), service_events);
    assert_eq!(service_after.status(), service_status);
    drop(service_after);
    let ledger_after = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("ledger trace retained after failed shutdown");
    assert_eq!(ledger_after.initial_snapshot(), initial);
    assert_eq!(ledger_after.events(), ledger_events);
    assert_eq!(ledger_after.status(), ledger_status);
    drop(ledger_after);

    assert_eq!(
        engine
            .shutdown()
            .expect("retry successful shutdown")
            .remaining_shared_bytes,
        0
    );
    assert!(engine.ledger_snapshot().current_is_zero());
}

#[test]
fn batch_admission_reserve_accepts_exact_fit_and_rejects_one_byte_short() {
    let probe_adapter = GatedAdapter::new(
        Arc::new(CommitGate::passthrough()),
        Arc::new(AtomicUsize::new(0)),
    );
    let mut limits = SchedulerLimits::tiny();
    limits.admission_reserve_bytes = 1024 * 1024;
    let probe_config =
        SchedulerConfig::new(&probe_adapter, limits).expect("batch scratch probe config");
    let required =
        SchedulerEngine::<GatedAdapter>::required_batch_admission_reserve_bytes(&probe_config)
            .expect("checked batch scratch bound");
    assert!(required > 1);

    let exact_adapter = GatedAdapter::new(
        Arc::new(CommitGate::passthrough()),
        Arc::new(AtomicUsize::new(0)),
    );
    limits.admission_reserve_bytes = required;
    let exact_config =
        SchedulerConfig::new(&exact_adapter, limits).expect("exact batch scratch config");
    let _engine =
        SchedulerEngine::new(exact_adapter, exact_config).expect("exact scratch reserve succeeds");

    let short_adapter = GatedAdapter::new(
        Arc::new(CommitGate::passthrough()),
        Arc::new(AtomicUsize::new(0)),
    );
    limits.admission_reserve_bytes = required - 1;
    let short_config =
        SchedulerConfig::new(&short_adapter, limits).expect("short batch scratch config");
    let error = SchedulerEngine::new(short_adapter, short_config)
        .expect_err("one-byte-short scratch reserve must fail before allocation");
    assert!(matches!(
        error,
        SchedulerError::ResourceExhausted {
            resource: "batch admission scratch",
            required: observed_required,
            limit,
        } if observed_required == required && limit == required - 1
    ));
}

#[test]
fn descending_prevalidated_removals_match_repeated_value_lookup_with_holes() {
    for length in 1_usize..=10 {
        let free_slots = (0..length)
            .map(|position| position.wrapping_mul(7).wrapping_add(3))
            .collect::<Vec<_>>();
        for selected_mask in 0_usize..(1_usize << length) {
            let selections = (0..length)
                .rev()
                .filter(|position| selected_mask & (1 << position) != 0)
                .map(|position| (position, free_slots[position]))
                .collect::<Vec<_>>();

            let mut lookup_baseline = free_slots.clone();
            for (_, selected) in &selections {
                let position = lookup_baseline
                    .iter()
                    .rposition(|candidate| candidate == selected)
                    .expect("selected free-list value");
                lookup_baseline.swap_remove(position);
            }

            let mut prevalidated = free_slots.clone();
            for (position, selected) in &selections {
                assert_eq!(prevalidated.get(*position), Some(selected));
                prevalidated.swap_remove(*position);
            }
            assert_eq!(prevalidated, lookup_baseline);
        }
    }
}

#[test]
fn batch_and_single_submission_report_identical_slot_pressure() {
    fn single_slot_engine() -> SchedulerEngine<GatedAdapter> {
        let adapter = GatedAdapter::new(
            Arc::new(CommitGate::passthrough()),
            Arc::new(AtomicUsize::new(0)),
        );
        let mut limits = SchedulerLimits::tiny();
        limits.max_outstanding_requests = 1;
        limits.max_active_requests = 1;
        limits.max_queued_requests = 1;
        limits.max_retained_terminal_results = 1;
        limits.batch_width = 1;
        let config = SchedulerConfig::new(&adapter, limits).expect("single-slot config");
        SchedulerEngine::new(adapter, config).expect("single-slot engine")
    }

    let request = RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None);
    let mut ordinary_exhausted = single_slot_engine();
    ordinary_exhausted.exhaust_slot_generation_for_test(0);
    let ordinary_generation_error = ordinary_exhausted
        .try_submit(request)
        .expect_err("ordinary exhausted generation");
    let mut batch_exhausted = single_slot_engine();
    batch_exhausted.exhaust_slot_generation_for_test(0);
    let generation_result = batch_exhausted
        .prepare_submit_batch(&[BatchRequestSpec::absolute(request)])
        .expect("prepare generation-pressure batch")
        .commit_prepared_batch(0)
        .expect("commit generation-pressure batch");
    let batch_generation_error = generation_result
        .rejected()
        .next()
        .expect("generation-pressure rejection");
    assert_eq!(generation_result.accepted_count(), 0);
    assert_eq!(
        format!("{ordinary_generation_error:?}"),
        format!("{:?}", batch_generation_error.error())
    );

    let active_request = RequestSpec::new(&[0], 2, SamplingPolicy::Greedy, None);
    let mut ordinary_occupied = single_slot_engine();
    ordinary_occupied
        .try_submit(active_request)
        .expect("ordinary first request");
    ordinary_occupied.step().expect("ordinary first step");
    let ordinary_slot_error = ordinary_occupied
        .try_submit(request)
        .expect_err("ordinary occupied slot");

    let mut batch_occupied = single_slot_engine();
    batch_occupied
        .try_submit(active_request)
        .expect("batch first request");
    batch_occupied.step().expect("batch first step");
    let slot_result = batch_occupied
        .prepare_submit_batch(&[BatchRequestSpec::absolute(request)])
        .expect("prepare occupied-slot batch")
        .commit_prepared_batch(0)
        .expect("commit occupied-slot batch");
    let batch_slot_error = slot_result
        .rejected()
        .next()
        .expect("occupied-slot rejection");
    assert_eq!(slot_result.accepted_count(), 0);
    assert_eq!(
        format!("{ordinary_slot_error:?}"),
        format!("{:?}", batch_slot_error.error())
    );
    assert!(matches!(
        ordinary_slot_error,
        SchedulerError::ResourceExhausted {
            resource: "request slot count",
            required: 1,
            limit: 1,
        }
    ));
}
