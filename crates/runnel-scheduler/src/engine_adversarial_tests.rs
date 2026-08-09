use std::{
    cell::Cell,
    mem::size_of,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread,
};

use runnel_runtime::{
    AdapterExecutionLayout, AdapterWorkIdentity, DecoderAdapter, Result as RuntimeResult,
    RuntimeError, SampleConfig, SamplingPolicy, StateLayoutAccounting,
};

use crate::control::ControlBinding;
use crate::endpoint::TryPop;
use crate::run_observer::StepObserver;
use crate::{
    BatchRequestSpec, CancelDisposition, CheckpointAction, CheckpointDirective, CheckpointPoint,
    DeadlineExpirationDisposition, ErrorCategory, LedgerCategory, LedgerOwnership,
    LedgerTraceCursor, MAX_CHECKPOINT_PLAN_ENTRIES, RequestPhase, RequestSpec, SchedulerConfig,
    SchedulerEngine, SchedulerError, SchedulerLimits, ServiceTraceCursor, StepReport,
    TerminalOutcome, TerminalResult,
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
    expert_count: Arc<AtomicUsize>,
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
            expert_count: Arc::new(AtomicUsize::new(0)),
            apply_cancel: None,
        }
    }

    fn with_expert_count(
        gate: Arc<CommitGate>,
        apply_count: Arc<AtomicUsize>,
        expert_count: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            expert_count,
            ..Self::new(gate, apply_count)
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
        self.expert_count.fetch_add(1, Ordering::AcqRel);
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

fn new_checkpoint_engine(
    batch_width: u64,
    waves_per_step: u64,
    apply_count: Arc<AtomicUsize>,
    expert_count: Arc<AtomicUsize>,
) -> SchedulerEngine<GatedAdapter> {
    let adapter = GatedAdapter::with_expert_count(
        Arc::new(CommitGate::passthrough()),
        apply_count,
        expert_count,
    );
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = batch_width;
    limits.waves_per_step = waves_per_step;
    let config = SchedulerConfig::new(&adapter, limits).expect("checkpoint scheduler config");
    SchedulerEngine::new(adapter, config).expect("checkpoint scheduler engine")
}

fn seeded_policy(seed: u64) -> SamplingPolicy {
    SamplingPolicy::Sample(SampleConfig {
        seed,
        temperature: 1.0,
        top_k: 4,
        top_p: 1.0,
    })
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

fn admitted_run_observer(
    max_new_tokens: usize,
) -> (
    SchedulerEngine<GatedAdapter>,
    crate::RunObserver,
    crate::RequestId,
) {
    let mut engine = new_engine(
        Arc::new(CommitGate::passthrough()),
        Arc::new(AtomicUsize::new(0)),
        1,
    );
    let mut observer = engine.prepare_run_observer().expect("callback observer");
    let prompt = [0_u32];
    let offers = [BatchRequestSpec::absolute(RequestSpec::new(
        &prompt,
        max_new_tokens,
        SamplingPolicy::Greedy,
        None,
    ))];
    let request = engine
        .prepare_submit_batch(&offers)
        .expect("prepare callback batch")
        .commit_prepared_batch_observed_at_for_test(10, &mut observer)
        .expect("publish callback batch")
        .accepted()
        .next()
        .expect("accepted callback request")
        .request_id();
    (engine, observer, request)
}

fn admitted_deadline_run_observer(
    deadline_ns: u64,
) -> (
    SchedulerEngine<GatedAdapter>,
    crate::RunObserver,
    crate::RequestId,
) {
    let mut engine = new_engine(
        Arc::new(CommitGate::passthrough()),
        Arc::new(AtomicUsize::new(0)),
        1,
    );
    let mut observer = engine
        .prepare_run_observer()
        .expect("deadline callback observer");
    let prompt = [0_u32];
    let offers = [BatchRequestSpec::absolute(RequestSpec::new(
        &prompt,
        2,
        SamplingPolicy::Greedy,
        Some(deadline_ns),
    ))];
    let request = engine
        .prepare_submit_batch(&offers)
        .expect("prepare deadline callback batch")
        .commit_prepared_batch_observed_at_for_test(10, &mut observer)
        .expect("publish deadline callback batch")
        .accepted()
        .next()
        .expect("accepted deadline callback request")
        .request_id();
    (engine, observer, request)
}

#[test]
fn run_observer_callback_state_machine_rejects_unpaired_and_impossible_events() {
    let (_engine, mut observer, request) = admitted_run_observer(2);
    observer.promotion(request);
    observer.commit(request, 0, crate::ServicePhase::Prefill, true, 20);
    assert_eq!(
        observer.read().status().failure(),
        Some(crate::RunObserverFailure::InvalidPhaseBoundary)
    );
    assert_eq!(observer.read().requests()[0].committed_positions(), 0);

    let (_engine, mut observer, request) = admitted_run_observer(2);
    observer.promotion(request);
    observer.work_start(request, 0, 20);
    observer.commit(request, 0, crate::ServicePhase::Prefill, false, 30);
    assert_eq!(
        observer.read().status().failure(),
        Some(crate::RunObserverFailure::InvalidPhaseBoundary)
    );
    assert_eq!(observer.read().requests()[0].emitted_tokens(), 0);

    let (_engine, mut observer, request) = admitted_run_observer(2);
    observer.terminal(
        TerminalResult::new(request, TerminalOutcome::Cancelled, 0, 0),
        20,
    );
    assert_eq!(
        observer.read().status().failure(),
        Some(crate::RunObserverFailure::MissingCancellationBoundary)
    );

    let (_engine, mut observer, request) = admitted_run_observer(2);
    observer.terminal(
        TerminalResult::new(request, TerminalOutcome::DeadlineExceeded, 0, 0),
        20,
    );
    assert_eq!(
        observer.read().status().failure(),
        Some(crate::RunObserverFailure::InvalidTerminal)
    );

    let (_engine, mut observer, request) = admitted_run_observer(2);
    observer.terminal(
        TerminalResult::new(request, TerminalOutcome::Completed, 0, 0),
        20,
    );
    assert_eq!(
        observer.read().status().failure(),
        Some(crate::RunObserverFailure::InvalidTerminal)
    );

    let (_engine, mut observer, request) = admitted_deadline_run_observer(50);
    observer.terminal(
        TerminalResult::new(request, TerminalOutcome::DeadlineExceeded, 0, 0),
        49,
    );
    assert_eq!(
        observer.read().status().failure(),
        Some(crate::RunObserverFailure::InvalidTerminal)
    );

    let (_engine, mut observer, request) = admitted_deadline_run_observer(50);
    observer.cancellation(request, CancelDisposition::Requested, 20);
    observer.terminal(
        TerminalResult::new(request, TerminalOutcome::DeadlineExceeded, 0, 0),
        50,
    );
    assert_eq!(
        observer.read().status().failure(),
        Some(crate::RunObserverFailure::InvalidTerminal)
    );

    let (_engine, mut observer, request) = admitted_deadline_run_observer(50);
    observer.terminal(
        TerminalResult::new(request, TerminalOutcome::DeadlineExceeded, 0, 0),
        50,
    );
    assert!(observer.read().status().healthy());
    assert_eq!(
        observer.read().requests()[0].terminal_outcome(),
        Some(TerminalOutcome::DeadlineExceeded)
    );
}

#[test]
fn run_observer_allows_a_backpressured_work_attempt_to_retry_once_unowned() {
    let (_engine, mut observer, request) = admitted_run_observer(2);
    observer.promotion(request);
    observer.work_start(request, 0, 20);
    observer.work_abandoned(request, 0);
    observer.work_start(request, 0, 30);
    observer.commit(request, 0, crate::ServicePhase::Prefill, true, 40);

    let read = observer.read();
    assert!(read.status().healthy());
    assert_eq!(read.requests()[0].first_work_start_ns(), Some(20));
    assert_eq!(read.requests()[0].committed_positions(), 1);
    assert_eq!(
        read.output_commit_ns(0).expect("retry output timestamps"),
        &[Some(40)]
    );
}

#[test]
fn run_observer_manual_clock_freezes_exact_phase_and_lifecycle_boundaries() {
    let mut engine = new_engine(
        Arc::new(CommitGate::passthrough()),
        Arc::new(AtomicUsize::new(0)),
        1,
    );
    let mut observer = engine
        .prepare_run_observer()
        .expect("manual-clock observer");
    let prompt = [0_u32, 1];
    let offers = [BatchRequestSpec::absolute(RequestSpec::new(
        &prompt,
        2,
        SamplingPolicy::Greedy,
        None,
    ))];
    let admission = engine
        .prepare_submit_batch(&offers)
        .expect("prepare manual-clock batch")
        .commit_prepared_batch_observed_at_for_test(10, &mut observer)
        .expect("publish manual-clock batch");
    let request = admission
        .accepted()
        .next()
        .expect("manual-clock request")
        .request_id();
    let now = Cell::new(20_u64);
    let clock = || now.get();

    let first = engine
        .step_observed_with_clock_for_test(&clock, &mut observer)
        .expect("first manual-clock step");
    assert_eq!(first.committed_positions, 1);
    now.set(30);
    let second = engine
        .step_observed_with_clock_for_test(&clock, &mut observer)
        .expect("second manual-clock step");
    assert_eq!(second.committed_positions, 1);
    now.set(40);
    let third = engine
        .step_observed_with_clock_for_test(&clock, &mut observer)
        .expect("third manual-clock step");
    assert_eq!(third.committed_positions, 1);
    assert_eq!(third.terminal_decisions, 1);

    let read = observer.read();
    let observed = &read.requests()[0];
    assert_eq!(read.release_ns(), Some(10));
    assert_eq!(observed.admitted_ns(), 10);
    assert_eq!(observed.first_work_start_ns(), Some(20));
    assert_eq!(observed.prefill_complete_ns(), Some(30));
    assert_eq!(observed.first_decode_start_ns(), Some(40));
    assert_eq!(
        read.output_commit_ns(0).expect("manual output timestamps"),
        &[Some(30), Some(40)]
    );
    assert_eq!(observed.terminal_decided_ns(), Some(40));
    assert_eq!(observed.preemption_count(), 2);
    assert_eq!(observed.resume_count(), 2);
    assert_eq!(read.totals().state_live_token_sample_sum, 6);
    assert_eq!(read.totals().state_allocated_page_slot_sample_sum, 12);
    assert_eq!(read.totals().state_sample_count, 3);
    drop(read);

    now.set(50);
    assert_eq!(
        engine
            .drain_events_observed_with_clock_for_test(request, usize::MAX, &clock, &mut observer,)
            .expect("manual-clock output drain")
            .len(),
        2
    );
    let terminal = engine
        .take_terminal_observed_with_clock_for_test(request, &clock, &mut observer)
        .expect("manual-clock terminal read")
        .expect("manual-clock terminal");
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(
        observer.read().requests()[0].request_owned_zero_ns(),
        Some(50)
    );
    let status = observer.finish();
    assert!(
        status.healthy(),
        "manual-clock observer: {:?}",
        status.failure()
    );
}

#[test]
fn run_observer_manual_clock_records_inclusive_deadline_before_more_work() {
    let apply_count = Arc::new(AtomicUsize::new(0));
    let mut engine = new_engine(
        Arc::new(CommitGate::passthrough()),
        Arc::clone(&apply_count),
        1,
    );
    let mut observer = engine.prepare_run_observer().expect("deadline observer");
    let prompt = [0_u32, 1];
    let offers = [BatchRequestSpec::absolute(RequestSpec::new(
        &prompt,
        2,
        SamplingPolicy::Greedy,
        Some(30),
    ))];
    let admission = engine
        .prepare_submit_batch(&offers)
        .expect("prepare deadline batch")
        .commit_prepared_batch_observed_at_for_test(10, &mut observer)
        .expect("publish deadline batch");
    let request = admission
        .accepted()
        .next()
        .expect("deadline request")
        .request_id();
    let now = Cell::new(20_u64);
    let clock = || now.get();

    let first = engine
        .step_observed_with_clock_for_test(&clock, &mut observer)
        .expect("pre-deadline step");
    assert_eq!(first.committed_positions, 1);
    assert_eq!(apply_count.load(Ordering::Acquire), 1);
    now.set(30);
    let expired = engine
        .step_observed_with_clock_for_test(&clock, &mut observer)
        .expect("inclusive deadline step");
    assert_eq!(expired.committed_positions, 0);
    assert_eq!(expired.terminal_decisions, 1);
    assert_eq!(apply_count.load(Ordering::Acquire), 1);
    let observed = observer.read().requests()[0];
    assert_eq!(
        observed.terminal_outcome(),
        Some(TerminalOutcome::DeadlineExceeded)
    );
    assert_eq!(observed.terminal_decided_ns(), Some(30));
    assert_eq!(observed.committed_positions(), 1);
    assert_eq!(observed.cancel_linearized_ns(), None);
    assert_eq!(observed.worker_quiescent_ns(), None);

    now.set(40);
    assert!(
        engine
            .drain_events_observed_with_clock_for_test(request, usize::MAX, &clock, &mut observer,)
            .expect("deadline drain")
            .is_empty()
    );
    assert_eq!(
        engine
            .take_terminal_observed_with_clock_for_test(request, &clock, &mut observer)
            .expect("deadline terminal read")
            .expect("deadline terminal")
            .outcome(),
        TerminalOutcome::DeadlineExceeded
    );
    assert_eq!(
        observer.read().requests()[0].request_owned_zero_ns(),
        Some(40)
    );
    assert!(observer.finish().healthy());
}

#[test]
fn run_observer_manual_clock_separates_cancel_terminal_quiescent_and_zero() {
    let apply_count = Arc::new(AtomicUsize::new(0));
    let mut engine = new_engine(
        Arc::new(CommitGate::passthrough()),
        Arc::clone(&apply_count),
        1,
    );
    let mut observer = engine
        .prepare_run_observer()
        .expect("cancellation observer");
    let prompt = [0_u32, 1];
    let offers = [BatchRequestSpec::absolute(RequestSpec::new(
        &prompt,
        2,
        SamplingPolicy::Greedy,
        None,
    ))];
    let admission = engine
        .prepare_submit_batch(&offers)
        .expect("prepare cancellation batch")
        .commit_prepared_batch_observed_at_for_test(10, &mut observer)
        .expect("publish cancellation batch");
    let request = admission
        .accepted()
        .next()
        .expect("cancellation request")
        .request_id();
    let now = Cell::new(20_u64);
    let clock = || now.get();
    engine
        .step_observed_with_clock_for_test(&clock, &mut observer)
        .expect("pre-cancellation step");
    assert_eq!(apply_count.load(Ordering::Acquire), 1);

    now.set(25);
    assert_eq!(
        engine
            .cancel_observed_with_clock_for_test(request, &clock, &mut observer)
            .expect("manual cancellation"),
        CancelDisposition::Requested
    );
    now.set(30);
    let cancelled = engine
        .step_observed_with_clock_for_test(&clock, &mut observer)
        .expect("cancellation terminal step");
    assert_eq!(cancelled.committed_positions, 0);
    assert_eq!(cancelled.terminal_decisions, 1);
    assert_eq!(apply_count.load(Ordering::Acquire), 1);
    let observed = observer.read().requests()[0];
    assert_eq!(observed.cancel_linearized_ns(), Some(25));
    assert_eq!(observed.terminal_decided_ns(), Some(30));
    assert_eq!(observed.worker_quiescent_ns(), Some(30));
    assert_eq!(
        observed.terminal_outcome(),
        Some(TerminalOutcome::Cancelled)
    );

    now.set(40);
    assert!(
        engine
            .drain_events_observed_with_clock_for_test(request, usize::MAX, &clock, &mut observer,)
            .expect("cancellation drain")
            .is_empty()
    );
    assert_eq!(
        engine
            .take_terminal_observed_with_clock_for_test(request, &clock, &mut observer)
            .expect("cancellation terminal read")
            .expect("cancellation terminal")
            .outcome(),
        TerminalOutcome::Cancelled
    );
    assert_eq!(
        observer.read().requests()[0].request_owned_zero_ns(),
        Some(40)
    );
    assert!(observer.finish().healthy());
}

#[test]
fn run_observer_clock_regression_is_sticky_neutral_and_freezes_valid_prefix() {
    let mut engine = new_engine(
        Arc::new(CommitGate::passthrough()),
        Arc::new(AtomicUsize::new(0)),
        1,
    );
    let mut observer = engine.prepare_run_observer().expect("regression observer");
    let fingerprint = observer.allocation_fingerprint();
    let prompt = [0_u32];
    let offers = [BatchRequestSpec::absolute(RequestSpec::new(
        &prompt,
        2,
        SamplingPolicy::Greedy,
        None,
    ))];
    let request = engine
        .prepare_submit_batch(&offers)
        .expect("prepare regression batch")
        .commit_prepared_batch_observed_at_for_test(100, &mut observer)
        .expect("publish regression batch")
        .accepted()
        .next()
        .expect("accepted regression request")
        .request_id();
    let now = Cell::new(50_u64);
    let clock = || now.get();

    engine
        .step_observed_with_clock_for_test(&clock, &mut observer)
        .expect("regressing observer cannot fail scheduler work");
    assert_eq!(
        observer.read().status().failure(),
        Some(crate::RunObserverFailure::ClockRegression)
    );
    let frozen_request = observer.read().requests()[0];
    let frozen_totals = observer.read().totals();

    now.set(101);
    for _ in 0..4 {
        if engine.request_phase(request).expect("regression phase") == RequestPhase::Terminal {
            break;
        }
        engine
            .step_observed_with_clock_for_test(&clock, &mut observer)
            .expect("poisoned observer remains behavior-neutral");
    }
    assert_eq!(
        engine.request_phase(request).expect("terminal phase"),
        RequestPhase::Terminal
    );
    assert_eq!(observer.read().requests()[0], frozen_request);
    assert_eq!(observer.read().totals(), frozen_totals);

    now.set(102);
    assert_eq!(
        engine
            .drain_events_observed_with_clock_for_test(request, usize::MAX, &clock, &mut observer)
            .expect("regression output drain")
            .len(),
        2
    );
    assert_eq!(
        engine
            .take_terminal_observed_with_clock_for_test(request, &clock, &mut observer)
            .expect("regression terminal read")
            .expect("regression terminal")
            .outcome(),
        TerminalOutcome::Completed
    );
    assert_eq!(
        observer.finish().failure(),
        Some(crate::RunObserverFailure::ClockRegression)
    );
    assert_eq!(observer.allocation_fingerprint(), fingerprint);
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
}

#[test]
fn run_observer_saturated_clock_is_sticky_and_behavior_neutral() {
    let mut engine = new_engine(
        Arc::new(CommitGate::passthrough()),
        Arc::new(AtomicUsize::new(0)),
        1,
    );
    let mut observer = engine.prepare_run_observer().expect("saturation observer");
    let prompt = [0_u32];
    let offers = [BatchRequestSpec::absolute(RequestSpec::new(
        &prompt,
        1,
        SamplingPolicy::Greedy,
        None,
    ))];
    let request = engine
        .prepare_submit_batch(&offers)
        .expect("prepare saturation batch")
        .commit_prepared_batch_observed_at_for_test(10, &mut observer)
        .expect("publish saturation batch")
        .accepted()
        .next()
        .expect("accepted saturation request")
        .request_id();
    let clock = || u64::MAX;

    let report = engine
        .step_observed_with_clock_for_test(&clock, &mut observer)
        .expect("saturated observer cannot fail scheduler work");
    assert_eq!(report.committed_positions, 1);
    assert_eq!(report.terminal_decisions, 1);
    assert_eq!(
        observer.read().status().failure(),
        Some(crate::RunObserverFailure::ClockSaturated)
    );
    let frozen = observer.read().requests()[0];
    engine
        .drain_events_observed_with_clock_for_test(request, usize::MAX, &clock, &mut observer)
        .expect("saturation output drain");
    assert_eq!(
        engine
            .take_terminal_observed_with_clock_for_test(request, &clock, &mut observer)
            .expect("saturation terminal read")
            .expect("saturation terminal")
            .outcome(),
        TerminalOutcome::Completed
    );
    assert_eq!(observer.read().requests()[0], frozen);
    assert_eq!(
        observer.finish().failure(),
        Some(crate::RunObserverFailure::ClockSaturated)
    );
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
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
    assert_eq!(report.preempted_requests, 1);
    assert_eq!(report.terminal_decisions, 1);
    assert_eq!(apply_count.load(Ordering::Acquire), 1);
    assert_eq!(
        engine.request_phase(cancelled).expect("cancelled phase"),
        RequestPhase::Terminal
    );
    assert_eq!(
        engine.request_phase(sibling).expect("sibling phase"),
        RequestPhase::Preempted
    );

    let recovery = engine.step().expect("ring service after suppression");
    assert_eq!(recovery.selected_positions, 1);
    assert_eq!(recovery.committed_positions, 1);
    assert_eq!(recovery.resumed_requests, 1);
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
fn preempted_resume_audit_is_atomic_when_a_later_member_is_missing() {
    let apply_count = Arc::new(AtomicUsize::new(0));
    let expert_count = Arc::new(AtomicUsize::new(0));
    let mut engine =
        new_checkpoint_engine(2, 1, Arc::clone(&apply_count), Arc::clone(&expert_count));
    let requests = [0_u32, 1_u32].map(|token| {
        engine
            .try_submit(RequestSpec::new(&[token], 2, SamplingPolicy::Greedy, None))
            .expect("accepted resume-audit request")
    });

    let first = engine.step().expect("preempt both audit requests");
    assert_eq!(first.committed_positions, 2);
    assert_eq!(first.preempted_requests, 2);
    assert_eq!(first.terminal_decisions, 0);
    assert_eq!(apply_count.load(Ordering::Acquire), 2);
    assert_eq!(expert_count.load(Ordering::Acquire), 2);
    assert!(requests.iter().copied().all(|request| {
        engine.request_phase(request).expect("preempted phase") == RequestPhase::Preempted
    }));
    let before_snapshot = engine.snapshot();
    let before_ledger = engine.ledger_snapshot();
    let before_trace = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("pre-corruption service trace")
        .events()
        .to_vec();

    engine
        .remove_preempted_membership_for_test(requests[1])
        .expect("remove later resident membership");
    let error = engine
        .step()
        .expect_err("resume audit must reject incomplete membership");
    assert_eq!(error.category(), ErrorCategory::Internal);
    assert_eq!(engine.snapshot(), before_snapshot);
    assert_eq!(engine.ledger_snapshot(), before_ledger);
    assert_eq!(
        engine
            .service_trace_since(ServiceTraceCursor::origin())
            .expect("post-corruption service trace")
            .events(),
        before_trace
    );
    assert!(requests.iter().copied().all(|request| {
        engine.request_phase(request).expect("atomic audit phase") == RequestPhase::Preempted
    }));
    assert_eq!(apply_count.load(Ordering::Acquire), 2);
    assert_eq!(expert_count.load(Ordering::Acquire), 2);

    let shutdown = engine.shutdown().expect("corrupted audit shutdown");
    assert_eq!(shutdown.terminated_requests, 2);
    assert_eq!(shutdown.discarded_output_events, 2);
    assert_eq!(shutdown.remaining_shared_bytes, 0);
    assert_eq!(engine.ledger_snapshot().total_used(), 0);
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
            preempted_requests: 0,
            resumed_requests: 0,
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
fn concrete_checkpoint_plan_is_ordered_redacted_and_allocation_stable() {
    let apply_count = Arc::new(AtomicUsize::new(0));
    let expert_count = Arc::new(AtomicUsize::new(0));
    let mut engine =
        new_checkpoint_engine(1, 1, Arc::clone(&apply_count), Arc::clone(&expert_count));
    let pristine = engine.ledger_snapshot();
    let request = engine
        .try_submit(RequestSpec::new(&[0], 1, seeded_policy(17), None))
        .expect("accepted checkpoint ordering request");
    let directives = [
        CheckpointDirective::new(
            request,
            0,
            CheckpointPoint::PostFinalSnapshot,
            CheckpointAction::ObserveOnly,
        ),
        CheckpointDirective::new(
            request,
            0,
            CheckpointPoint::CompositePermitPreFinalSnapshot,
            CheckpointAction::ObserveOnly,
        ),
        CheckpointDirective::new(
            request,
            0,
            CheckpointPoint::ReadyToCommitPrePlan,
            CheckpointAction::ObserveOnly,
        ),
        CheckpointDirective::new(
            request,
            0,
            CheckpointPoint::PostRouterPreExpert,
            CheckpointAction::ObserveOnly,
        ),
    ];
    let mut plan = engine
        .prepare_checkpoint_plan(&directives)
        .expect("ordered checkpoint plan");
    let allocation = plan.allocation_fingerprint_for_test();
    assert!(format!("{plan:?}").contains("<redacted>"));
    assert!(format!("{:?}", directives[0]).contains("<redacted>"));

    let report = engine
        .step_with_checkpoint_plan(&mut plan)
        .expect("observed checkpoint step");
    assert_eq!(
        report,
        StepReport {
            promoted_requests: 1,
            waves: 1,
            selected_positions: 1,
            expert_tasks: 1,
            expert_groups: 1,
            committed_positions: 1,
            preempted_requests: 0,
            resumed_requests: 0,
            terminal_decisions: 1,
        }
    );
    assert_eq!(plan.allocation_fingerprint_for_test(), allocation);
    assert_eq!(plan.fired_count(), 4);
    assert!(plan.is_complete());
    plan.ensure_complete().expect("complete checkpoint plan");
    let records = plan.records().collect::<Vec<_>>();
    assert_eq!(
        records
            .iter()
            .map(|record| record.directive().point())
            .collect::<Vec<_>>(),
        [
            CheckpointPoint::PostRouterPreExpert,
            CheckpointPoint::ReadyToCommitPrePlan,
            CheckpointPoint::CompositePermitPreFinalSnapshot,
            CheckpointPoint::PostFinalSnapshot,
        ]
    );
    assert_eq!(
        records
            .iter()
            .map(|record| record.effect().expect("fired effect").fire_ordinal())
            .collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert!(records.iter().all(|record| {
        let effect = record.effect().expect("fired observation");
        effect.cancellation().is_none() && effect.deadline().is_none()
    }));
    assert_eq!(expert_count.load(Ordering::Acquire), 1);
    assert_eq!(apply_count.load(Ordering::Acquire), 1);
    assert!(engine.rng_state_for_test(request).unwrap().is_some());

    assert_eq!(
        engine
            .drain_events(request, usize::MAX)
            .expect("observed output")
            .len(),
        1
    );
    assert_eq!(
        engine
            .take_terminal(request)
            .expect("observed terminal query")
            .expect("observed terminal")
            .outcome(),
        TerminalOutcome::Completed
    );
    assert_all_request_ownership_reaped(&engine, pristine.total_used(), pristine.shared_used());
}

#[test]
fn checkpoint_plan_validation_rejects_duplicates_limits_deadlines_and_foreign_engines() {
    let mut engine = new_checkpoint_engine(
        1,
        1,
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let request = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("accepted checkpoint validation request");
    let directive = CheckpointDirective::new(
        request,
        0,
        CheckpointPoint::PostRouterPreExpert,
        CheckpointAction::ObserveOnly,
    );
    let duplicate = engine
        .prepare_checkpoint_plan(&[directive, directive])
        .expect_err("duplicate checkpoint boundary must fail");
    assert_eq!(duplicate.category(), ErrorCategory::InvalidRequest);
    let excessive = vec![directive; MAX_CHECKPOINT_PLAN_ENTRIES + 1];
    let excessive_error = engine
        .prepare_checkpoint_plan(&excessive)
        .expect_err("checkpoint count ceiling must fail before binding");
    assert_eq!(excessive_error.category(), ErrorCategory::InvalidRequest);
    let expiry_error = engine
        .prepare_checkpoint_plan(&[CheckpointDirective::new(
            request,
            0,
            CheckpointPoint::ReadyToCommitPrePlan,
            CheckpointAction::ExpireDeadline,
        )])
        .expect_err("deadline-free expiry action must fail");
    assert_eq!(expiry_error.category(), ErrorCategory::InvalidRequest);

    let mut plan = engine
        .prepare_checkpoint_plan(&[directive])
        .expect("valid engine-bound checkpoint plan");
    let mut foreign = new_checkpoint_engine(
        1,
        1,
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    foreign
        .try_submit(RequestSpec::new(&[1], 1, SamplingPolicy::Greedy, None))
        .expect("accepted foreign request with matching numeric ID");
    let before = foreign.snapshot();
    let foreign_error = foreign
        .step_with_checkpoint_plan(&mut plan)
        .expect_err("plan must not cross an engine control domain");
    assert_eq!(foreign_error.category(), ErrorCategory::InvalidRequest);
    assert_eq!(foreign.snapshot(), before);
    assert_eq!(plan.fired_count(), 0);
}

#[test]
fn stale_checkpoint_plan_cannot_redirect_to_a_reused_request_slot() {
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
    limits.waves_per_step = 1;
    let config = SchedulerConfig::new(&adapter, limits).expect("stale-plan config");
    let mut engine = SchedulerEngine::new(adapter, config).expect("stale-plan engine");
    let first = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("accepted first checkpoint generation");
    let mut stale_plan = engine
        .prepare_checkpoint_plan(&[CheckpointDirective::new(
            first,
            0,
            CheckpointPoint::PostRouterPreExpert,
            CheckpointAction::Cancel,
        )])
        .expect("first-generation checkpoint plan");
    assert_eq!(
        engine
            .step()
            .expect("complete first generation")
            .committed_positions,
        1
    );
    assert_eq!(engine.drain_events(first, usize::MAX).unwrap().len(), 1);
    assert_eq!(
        engine.take_terminal(first).unwrap().unwrap().outcome(),
        TerminalOutcome::Completed
    );

    let current = engine
        .try_submit(RequestSpec::new(&[1], 1, SamplingPolicy::Greedy, None))
        .expect("accepted reused checkpoint slot");
    assert_ne!(first, current);
    let report = engine
        .step_with_checkpoint_plan(&mut stale_plan)
        .expect("stale target is ignored rather than redirected");
    assert_eq!(report.committed_positions, 1);
    assert_eq!(report.terminal_decisions, 1);
    assert_eq!(stale_plan.fired_count(), 0);
    assert_eq!(
        stale_plan
            .ensure_complete()
            .expect_err("stale target remains visibly incomplete")
            .category(),
        ErrorCategory::InvalidRequest
    );
    assert_eq!(engine.drain_events(current, usize::MAX).unwrap().len(), 1);
    assert_eq!(
        engine.take_terminal(current).unwrap().unwrap().outcome(),
        TerminalOutcome::Completed
    );
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
}

#[test]
fn cancellation_and_deadline_actions_suppress_each_preapply_checkpoint_exactly() {
    for point in [
        CheckpointPoint::PostRouterPreExpert,
        CheckpointPoint::ReadyToCommitPrePlan,
        CheckpointPoint::CompositePermitPreFinalSnapshot,
    ] {
        for action in [
            CheckpointAction::Cancel,
            CheckpointAction::ExpireDeadline,
            CheckpointAction::CancelAndExpireDeadline,
        ] {
            let apply_count = Arc::new(AtomicUsize::new(0));
            let expert_count = Arc::new(AtomicUsize::new(0));
            let mut engine =
                new_checkpoint_engine(2, 1, Arc::clone(&apply_count), Arc::clone(&expert_count));
            let pristine = engine.ledger_snapshot();
            let expires = matches!(
                action,
                CheckpointAction::ExpireDeadline | CheckpointAction::CancelAndExpireDeadline
            );
            if expires {
                engine.advance_clock(9).expect("pre-deadline clock");
            }
            let target = engine
                .try_submit(RequestSpec::new(
                    &[0],
                    1,
                    seeded_policy(31),
                    expires.then_some(10),
                ))
                .expect("accepted checkpoint target");
            let sibling = engine
                .try_submit(RequestSpec::new(
                    &[1],
                    1,
                    seeded_policy(47),
                    expires.then_some(100),
                ))
                .expect("accepted checkpoint sibling");
            let mut plan = engine
                .prepare_checkpoint_plan(&[CheckpointDirective::new(target, 0, point, action)])
                .expect("preapply checkpoint plan");
            let allocation = plan.allocation_fingerprint_for_test();

            let report = engine
                .step_with_checkpoint_plan(&mut plan)
                .expect("preapply checkpoint suppression");
            assert_eq!(report.promoted_requests, 2);
            assert_eq!(report.waves, 1);
            assert_eq!(report.selected_positions, 2);
            assert_eq!(
                report.expert_tasks,
                usize::from(point != CheckpointPoint::PostRouterPreExpert) + 1
            );
            assert_eq!(report.expert_groups, 1);
            assert_eq!(report.committed_positions, 1);
            assert_eq!(report.terminal_decisions, 2);
            assert_eq!(plan.allocation_fingerprint_for_test(), allocation);
            plan.ensure_complete().expect("fired preapply plan");
            let effect = plan
                .records()
                .next()
                .expect("checkpoint record")
                .effect()
                .expect("checkpoint effect");
            assert_eq!(effect.fire_ordinal(), 0);
            assert_eq!(
                effect.cancellation(),
                matches!(
                    action,
                    CheckpointAction::Cancel | CheckpointAction::CancelAndExpireDeadline
                )
                .then_some(CancelDisposition::Requested)
            );
            assert_eq!(
                effect.deadline(),
                expires.then_some(DeadlineExpirationDisposition::AdvancedToDeadline)
            );
            assert_eq!(expert_count.load(Ordering::Acquire), report.expert_tasks);
            assert_eq!(apply_count.load(Ordering::Acquire), 1);
            assert_eq!(engine.rng_state_for_test(target).unwrap(), None);
            assert!(engine.rng_state_for_test(sibling).unwrap().is_some());
            assert_no_active_request_ownership(&engine);
            let service = engine
                .service_trace_since(ServiceTraceCursor::origin())
                .expect("preapply service trace");
            assert!(service.status().healthy());
            assert_eq!(service.events().len(), 1);
            assert_eq!(service.events()[0].request_id(), sibling);
            drop(service);

            assert!(
                engine
                    .drain_events(target, usize::MAX)
                    .expect("suppressed target output")
                    .is_empty()
            );
            let target_terminal = engine
                .take_terminal(target)
                .expect("suppressed target terminal query")
                .expect("suppressed target terminal");
            let expected_outcome = if action == CheckpointAction::ExpireDeadline {
                TerminalOutcome::DeadlineExceeded
            } else {
                TerminalOutcome::Cancelled
            };
            assert_eq!(target_terminal.outcome(), expected_outcome);
            assert_eq!(target_terminal.committed_positions(), 0);
            assert_eq!(target_terminal.emitted_tokens(), 0);
            assert_eq!(
                engine
                    .drain_events(sibling, usize::MAX)
                    .expect("checkpoint sibling output")
                    .len(),
                1
            );
            assert_eq!(
                engine
                    .take_terminal(sibling)
                    .expect("checkpoint sibling terminal query")
                    .expect("checkpoint sibling terminal")
                    .outcome(),
                TerminalOutcome::Completed
            );
            assert_all_request_ownership_reaped(
                &engine,
                pristine.total_used(),
                pristine.shared_used(),
            );
        }
    }
}

#[test]
fn post_final_checkpoint_actions_lose_exactly_one_emitting_position() {
    for action in [
        CheckpointAction::Cancel,
        CheckpointAction::ExpireDeadline,
        CheckpointAction::CancelAndExpireDeadline,
    ] {
        let apply_count = Arc::new(AtomicUsize::new(0));
        let expert_count = Arc::new(AtomicUsize::new(0));
        let mut engine =
            new_checkpoint_engine(1, 2, Arc::clone(&apply_count), Arc::clone(&expert_count));
        let pristine = engine.ledger_snapshot();
        let expires = matches!(
            action,
            CheckpointAction::ExpireDeadline | CheckpointAction::CancelAndExpireDeadline
        );
        if expires {
            engine.advance_clock(9).expect("pre-deadline clock");
        }
        let request = engine
            .try_submit(RequestSpec::new(
                &[0],
                2,
                seeded_policy(71),
                expires.then_some(10),
            ))
            .expect("accepted post-final target");
        let mut plan = engine
            .prepare_checkpoint_plan(&[CheckpointDirective::new(
                request,
                0,
                CheckpointPoint::PostFinalSnapshot,
                action,
            )])
            .expect("post-final checkpoint plan");

        let report = engine
            .step_with_checkpoint_plan(&mut plan)
            .expect("post-final checkpoint step");
        assert_eq!(
            report,
            StepReport {
                promoted_requests: 1,
                waves: 1,
                selected_positions: 1,
                expert_tasks: 1,
                expert_groups: 1,
                committed_positions: 1,
                preempted_requests: 0,
                resumed_requests: 0,
                terminal_decisions: 1,
            }
        );
        plan.ensure_complete().expect("fired post-final plan");
        let effect = plan.records().next().unwrap().effect().unwrap();
        assert_eq!(
            effect.cancellation(),
            matches!(
                action,
                CheckpointAction::Cancel | CheckpointAction::CancelAndExpireDeadline
            )
            .then_some(CancelDisposition::Requested)
        );
        assert_eq!(
            effect.deadline(),
            expires.then_some(DeadlineExpirationDisposition::AdvancedToDeadline)
        );
        assert_eq!(expert_count.load(Ordering::Acquire), 1);
        assert_eq!(apply_count.load(Ordering::Acquire), 1);
        assert!(engine.rng_state_for_test(request).unwrap().is_some());
        let service = engine
            .service_trace_since(ServiceTraceCursor::origin())
            .expect("post-final service trace");
        assert_eq!(service.events().len(), 1);
        assert_eq!(service.events()[0].request_id(), request);
        assert_eq!(service.events()[0].position(), 0);
        drop(service);
        let events = engine
            .drain_events(request, usize::MAX)
            .expect("post-final committed output");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].request_id(), request);
        assert_eq!(events[0].output_index(), 0);
        let terminal = engine
            .take_terminal(request)
            .expect("post-final terminal query")
            .expect("post-final terminal");
        assert_eq!(
            terminal.outcome(),
            if action == CheckpointAction::ExpireDeadline {
                TerminalOutcome::DeadlineExceeded
            } else {
                TerminalOutcome::Cancelled
            }
        );
        assert_eq!(terminal.committed_positions(), 1);
        assert_eq!(terminal.emitted_tokens(), 1);
        assert_all_request_ownership_reaped(&engine, pristine.total_used(), pristine.shared_used());
    }
}

#[test]
fn post_final_checkpoint_cancellation_cannot_replace_a_completed_position() {
    let apply_count = Arc::new(AtomicUsize::new(0));
    let mut engine = new_checkpoint_engine(
        1,
        1,
        Arc::clone(&apply_count),
        Arc::new(AtomicUsize::new(0)),
    );
    let pristine = engine.ledger_snapshot();
    let request = engine
        .try_submit(RequestSpec::new(&[0], 1, seeded_policy(83), None))
        .expect("accepted completing post-final target");
    let mut plan = engine
        .prepare_checkpoint_plan(&[CheckpointDirective::new(
            request,
            0,
            CheckpointPoint::PostFinalSnapshot,
            CheckpointAction::Cancel,
        )])
        .expect("completing post-final plan");
    let report = engine
        .step_with_checkpoint_plan(&mut plan)
        .expect("completing post-final step");
    assert_eq!(report.committed_positions, 1);
    assert_eq!(report.terminal_decisions, 1);
    assert_eq!(apply_count.load(Ordering::Acquire), 1);
    assert_eq!(
        plan.records()
            .next()
            .unwrap()
            .effect()
            .unwrap()
            .cancellation(),
        Some(CancelDisposition::Requested)
    );
    assert_eq!(engine.drain_events(request, usize::MAX).unwrap().len(), 1);
    let terminal = engine.take_terminal(request).unwrap().unwrap();
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(terminal.committed_positions(), 1);
    assert_eq!(terminal.emitted_tokens(), 1);
    assert_all_request_ownership_reaped(&engine, pristine.total_used(), pristine.shared_used());
}

#[test]
fn output_blocked_cancellation_and_deadline_preserve_the_committed_prefix() {
    for action in [
        CheckpointAction::Cancel,
        CheckpointAction::ExpireDeadline,
        CheckpointAction::CancelAndExpireDeadline,
    ] {
        let apply_count = Arc::new(AtomicUsize::new(0));
        let expert_count = Arc::new(AtomicUsize::new(0));
        let adapter = GatedAdapter::with_expert_count(
            Arc::new(CommitGate::passthrough()),
            Arc::clone(&apply_count),
            Arc::clone(&expert_count),
        );
        let mut limits = SchedulerLimits::tiny();
        limits.batch_width = 1;
        limits.waves_per_step = 1;
        limits.output_capacity_per_request = 1;
        let config = SchedulerConfig::new(&adapter, limits).expect("output-blocked config");
        let mut engine = SchedulerEngine::new(adapter, config).expect("output-blocked engine");
        let pristine = engine.ledger_snapshot();
        let expires = matches!(
            action,
            CheckpointAction::ExpireDeadline | CheckpointAction::CancelAndExpireDeadline
        );
        if expires {
            engine.advance_clock(9).expect("pre-deadline clock");
        }
        let request = engine
            .try_submit(RequestSpec::new(
                &[0],
                2,
                seeded_policy(89),
                expires.then_some(10),
            ))
            .expect("accepted output-blocked request");
        let first = engine.step().expect("fill one output slot");
        assert_eq!(first.committed_positions, 1);
        assert_eq!(first.terminal_decisions, 0);
        assert_eq!(
            engine.request_phase(request).expect("blocked phase"),
            RequestPhase::OutputBlocked
        );
        let rng_before = engine
            .rng_state_for_test(request)
            .expect("blocked RNG state")
            .expect("first sampled output commits RNG");
        let service_before = engine
            .service_trace_since(ServiceTraceCursor::origin())
            .expect("blocked service trace")
            .events()
            .to_vec();
        assert_eq!(service_before.len(), 1);

        if matches!(
            action,
            CheckpointAction::Cancel | CheckpointAction::CancelAndExpireDeadline
        ) {
            assert_eq!(
                engine.cancel(request).expect("cancel blocked request"),
                CancelDisposition::Requested
            );
        }
        if expires {
            engine
                .advance_clock(10)
                .expect("inclusive blocked deadline");
        }
        let resolved = engine.step().expect("resolve blocked control");
        assert_eq!(
            resolved,
            StepReport {
                promoted_requests: 0,
                waves: 0,
                selected_positions: 0,
                expert_tasks: 0,
                expert_groups: 0,
                committed_positions: 0,
                preempted_requests: 0,
                resumed_requests: 0,
                terminal_decisions: 1,
            }
        );
        assert_eq!(expert_count.load(Ordering::Acquire), 1);
        assert_eq!(apply_count.load(Ordering::Acquire), 1);
        assert_eq!(
            engine.rng_state_for_test(request).unwrap(),
            Some(rng_before)
        );
        assert_eq!(
            engine
                .service_trace_since(ServiceTraceCursor::origin())
                .expect("terminal blocked service trace")
                .events(),
            service_before
        );
        assert_no_active_request_ownership(&engine);

        let events = engine
            .drain_events(request, usize::MAX)
            .expect("drain committed blocked prefix");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].request_id(), request);
        assert_eq!(events[0].output_index(), 0);
        let terminal = engine
            .take_terminal(request)
            .expect("blocked terminal query")
            .expect("blocked terminal");
        assert_eq!(
            terminal.outcome(),
            if action == CheckpointAction::ExpireDeadline {
                TerminalOutcome::DeadlineExceeded
            } else {
                TerminalOutcome::Cancelled
            }
        );
        assert_eq!(terminal.committed_positions(), 1);
        assert_eq!(terminal.emitted_tokens(), 1);
        assert_all_request_ownership_reaped(&engine, pristine.total_used(), pristine.shared_used());
    }
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
fn stale_task_transaction_is_rejected_before_expert_execution_and_sibling_progresses() {
    let apply_count = Arc::new(AtomicUsize::new(0));
    let expert_count = Arc::new(AtomicUsize::new(0));
    let adapter = GatedAdapter::with_expert_count(
        Arc::new(CommitGate::passthrough()),
        Arc::clone(&apply_count),
        Arc::clone(&expert_count),
    );
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 2;
    limits.waves_per_step = 1;
    let config = SchedulerConfig::new(&adapter, limits).expect("stale-task scheduler config");
    let mut engine = SchedulerEngine::new(adapter, config).expect("stale-task scheduler engine");
    let pristine = engine.ledger_snapshot();
    let stale = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("accepted stale-task target");
    let sibling = engine
        .try_submit(RequestSpec::new(&[1], 1, SamplingPolicy::Greedy, None))
        .expect("accepted stale-task sibling");

    crate::engine::corrupt_next_wave_task_transaction_for_test();
    let report = engine
        .step()
        .expect("stale task fails one selection closed");
    assert_eq!(
        report,
        StepReport {
            promoted_requests: 2,
            waves: 1,
            selected_positions: 2,
            expert_tasks: 1,
            expert_groups: 1,
            committed_positions: 1,
            preempted_requests: 0,
            resumed_requests: 0,
            terminal_decisions: 2,
        }
    );
    assert_eq!(expert_count.load(Ordering::Acquire), 1);
    assert_eq!(apply_count.load(Ordering::Acquire), 1);
    assert_eq!(
        engine.request_phase(stale).expect("stale-task phase"),
        RequestPhase::Terminal
    );
    assert_eq!(
        engine.request_phase(sibling).expect("sibling phase"),
        RequestPhase::Terminal
    );
    let service = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("service trace after stale task");
    assert!(service.status().healthy());
    assert_eq!(service.events().len(), 1);
    assert_eq!(service.events()[0].request_id(), sibling);
    assert_eq!(service.events()[0].position(), 0);
    drop(service);
    assert_no_active_request_ownership(&engine);

    assert!(
        engine
            .drain_events(stale, usize::MAX)
            .expect("stale-task output")
            .is_empty()
    );
    let stale_terminal = engine
        .take_terminal(stale)
        .expect("stale-task terminal query")
        .expect("stale-task terminal");
    assert_eq!(
        stale_terminal.outcome(),
        TerminalOutcome::Failed {
            category: ErrorCategory::Internal,
        }
    );
    assert_eq!(stale_terminal.committed_positions(), 0);
    assert_eq!(stale_terminal.emitted_tokens(), 0);

    let sibling_events = engine
        .drain_events(sibling, usize::MAX)
        .expect("sibling output");
    assert_eq!(sibling_events.len(), 1);
    assert_eq!(sibling_events[0].request_id(), sibling);
    assert_eq!(sibling_events[0].output_index(), 0);
    let sibling_terminal = engine
        .take_terminal(sibling)
        .expect("sibling terminal query")
        .expect("sibling terminal");
    assert_eq!(sibling_terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(sibling_terminal.committed_positions(), 1);
    assert_eq!(sibling_terminal.emitted_tokens(), 1);
    let ledger_trace = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("ledger trace after stale task cleanup");
    assert!(ledger_trace.status().healthy());
    drop(ledger_trace);
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
