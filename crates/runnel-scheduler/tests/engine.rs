use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};

use runnel_runtime::{
    AdapterExecutionLayout, AdapterWorkIdentity, DecoderAdapter, Result as RuntimeResult,
    RuntimeError, SampleConfig, SamplingPolicy, StateLayoutAccounting,
};
use runnel_scheduler::{
    BatchRequestSpec, CancelDisposition, ErrorCategory, LedgerCategory, RequestPhase, RequestSpec,
    SchedulerConfig, SchedulerEngine, SchedulerError, SchedulerLimits, SchedulingPolicy,
    ServicePhase, ServiceTraceCursor, StepReport, TerminalOutcome,
};

static NEXT_MODEL_ID: CheckedCounter = CheckedCounter::new(1);

struct CheckedCounter {
    next: AtomicU64,
}

impl CheckedCounter {
    const fn new(first: u64) -> Self {
        Self {
            next: AtomicU64::new(first),
        }
    }

    fn issue(&self, exhausted: RuntimeError) -> RuntimeResult<u64> {
        let mut candidate = self.next.load(Ordering::Relaxed);
        loop {
            if candidate == 0 {
                return Err(exhausted);
            }
            let successor = candidate.checked_add(1).unwrap_or(0);
            match self.next.compare_exchange_weak(
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
}

#[derive(Clone, Copy)]
struct MockStateLayout {
    max_tokens: usize,
    payload_bytes: usize,
    charge_bytes: usize,
}

impl StateLayoutAccounting for MockStateLayout {
    fn payload_bytes(self) -> usize {
        self.payload_bytes
    }

    fn charge_bytes(self) -> usize {
        self.charge_bytes
    }
}

struct MockState {
    identity: u64,
    revision: u64,
    position: usize,
    tokens: Vec<u32>,
}

struct MockWorkspace;

struct MockPrepared {
    identity: AdapterWorkIdentity,
    input_token: u32,
}

struct MockTask {
    identity: AdapterWorkIdentity,
    input_token: u32,
    expert_id: u16,
}

struct MockContribution {
    identity: AdapterWorkIdentity,
    input_token: u32,
    expert_id: u16,
}

struct MockPending {
    identity: AdapterWorkIdentity,
    input_token: u32,
    logits: [f32; 5],
}

struct MockCommitPermit<'a> {
    state: &'a mut MockState,
    input_token: u32,
    next_position: usize,
    next_revision: u64,
}

struct MockAdapter {
    model_id: u64,
    next_state: CheckedCounter,
    next_transaction: CheckedCounter,
    fail_after_commit: bool,
    reuse_transaction_id: bool,
}

impl MockAdapter {
    fn new() -> Self {
        Self {
            model_id: NEXT_MODEL_ID
                .issue(RuntimeError::ModelIdentityExhausted)
                .expect("test model identity"),
            next_state: CheckedCounter::new(1),
            next_transaction: CheckedCounter::new(1),
            fail_after_commit: false,
            reuse_transaction_id: false,
        }
    }

    fn failing_after_commit() -> Self {
        Self {
            fail_after_commit: true,
            ..Self::new()
        }
    }

    fn reusing_transaction_identity() -> Self {
        Self {
            reuse_transaction_id: true,
            ..Self::new()
        }
    }
}

impl DecoderAdapter for MockAdapter {
    type StateLayout = MockStateLayout;
    type State = MockState;
    type Workspace = MockWorkspace;
    type PreparedToken = MockPrepared;
    type ExpertTask = MockTask;
    type ExpertTasks = std::array::IntoIter<MockTask, 1>;
    type ExpertContribution = MockContribution;
    type PendingStateCommit = MockPending;
    type StateCommitPermit<'a> = MockCommitPermit<'a>;

    fn execution_layout(&self) -> RuntimeResult<AdapterExecutionLayout> {
        AdapterExecutionLayout::new(
            1,
            size_of::<MockPrepared>(),
            size_of::<MockTask>(),
            size_of::<MockContribution>(),
            size_of::<MockPending>(),
            0,
            0,
        )
    }

    fn vocabulary_size(&self) -> usize {
        5
    }

    fn is_stop_token(&self, token: u32) -> bool {
        token == 4
    }

    fn state_layout(
        &self,
        max_tokens: usize,
        page_tokens: usize,
    ) -> RuntimeResult<MockStateLayout> {
        if max_tokens == 0 || page_tokens == 0 {
            return Err(RuntimeError::InvalidStateLayout(
                "mock state geometry must be nonzero",
            ));
        }
        let payload_bytes =
            max_tokens
                .checked_mul(size_of::<u32>())
                .ok_or(RuntimeError::ResourceSizeOverflow {
                    resource: "mock state payload",
                })?;
        let charge_bytes = payload_bytes
            .checked_add(63)
            .map(|value| value / 64 * 64)
            .ok_or(RuntimeError::ResourceSizeOverflow {
                resource: "mock state charge",
            })?;
        Ok(MockStateLayout {
            max_tokens,
            payload_bytes,
            charge_bytes,
        })
    }

    fn new_state(&self, layout: MockStateLayout) -> RuntimeResult<MockState> {
        let mut tokens = Vec::new();
        tokens.try_reserve_exact(layout.max_tokens).map_err(|_| {
            RuntimeError::ResourceExhausted {
                resource: "mock state tokens",
                bytes: layout.payload_bytes,
            }
        })?;
        tokens.resize(layout.max_tokens, 0);
        Ok(MockState {
            identity: self
                .next_state
                .issue(RuntimeError::StateIdentityExhausted)?,
            revision: 0,
            position: 0,
            tokens,
        })
    }

    fn new_workspace(&self) -> RuntimeResult<MockWorkspace> {
        Ok(MockWorkspace)
    }

    fn prepare_token(
        &self,
        state: &MockState,
        token: u32,
        _workspace: &mut MockWorkspace,
    ) -> RuntimeResult<MockPrepared> {
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
        let transaction_id = if self.reuse_transaction_id {
            1
        } else {
            self.next_transaction
                .issue(RuntimeError::AdapterTransactionIdentityExhausted)?
        };
        Ok(MockPrepared {
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

    fn prepared_identity(&self, prepared: &MockPrepared) -> AdapterWorkIdentity {
        prepared.identity
    }

    fn expert_tasks(&self, prepared: &MockPrepared) -> Self::ExpertTasks {
        [MockTask {
            identity: prepared.identity,
            input_token: prepared.input_token,
            expert_id: u16::try_from(prepared.input_token % 2).expect("mock expert identity"),
        }]
        .into_iter()
    }

    fn task_identity(&self, task: &MockTask) -> AdapterWorkIdentity {
        task.identity
    }

    fn task_router_rank(&self, _task: &MockTask) -> u16 {
        0
    }

    fn task_expert_id(&self, task: &MockTask) -> u16 {
        task.expert_id
    }

    fn execute_expert(
        &self,
        task: MockTask,
        _workspace: &mut MockWorkspace,
    ) -> RuntimeResult<MockContribution> {
        Ok(MockContribution {
            identity: task.identity,
            input_token: task.input_token,
            expert_id: task.expert_id,
        })
    }

    fn contribution_identity(&self, contribution: &MockContribution) -> AdapterWorkIdentity {
        contribution.identity
    }

    fn contribution_router_rank(&self, _contribution: &MockContribution) -> u16 {
        0
    }

    fn contribution_expert_id(&self, contribution: &MockContribution) -> u16 {
        contribution.expert_id
    }

    fn finish_token(
        &self,
        prepared: MockPrepared,
        contributions: &[MockContribution],
        _workspace: &mut MockWorkspace,
    ) -> RuntimeResult<MockPending> {
        let [contribution] = contributions else {
            return Err(RuntimeError::InvalidExpertContribution(
                "mock expects exactly one contribution",
            ));
        };
        if contribution.identity != prepared.identity
            || contribution.input_token != prepared.input_token
            || contribution.expert_id != u16::try_from(prepared.input_token % 2).unwrap_or(0)
        {
            return Err(RuntimeError::InvalidExpertContribution(
                "mock contribution envelope mismatch",
            ));
        }
        let sampled_token = if prepared.input_token == 4 {
            4
        } else {
            (prepared.input_token + 1) % 4
        };
        let next_token = usize::try_from(sampled_token)
            .map_err(|_| RuntimeError::InvalidAdapterWork("mock token conversion failed"))?;
        let mut logits = [0.0_f32; 5];
        logits[next_token] = 10.0;
        Ok(MockPending {
            identity: prepared.identity,
            input_token: prepared.input_token,
            logits,
        })
    }

    fn pending_identity(&self, pending: &MockPending) -> AdapterWorkIdentity {
        pending.identity
    }

    fn pending_logits<'a>(&self, pending: &'a MockPending) -> &'a [f32] {
        &pending.logits
    }

    fn with_validated_state_commit<R, F>(
        &self,
        state: &mut MockState,
        pending: &MockPending,
        apply: F,
    ) -> RuntimeResult<R>
    where
        F: for<'permit> FnOnce(MockCommitPermit<'permit>) -> R,
    {
        if pending.identity.model_instance_id() != self.model_id
            || pending.identity.state_id().get() != state.identity
            || pending.identity.state_revision() != state.revision
            || pending.identity.position() != state.position
        {
            return Err(RuntimeError::InvalidAdapterWork(
                "mock pending identity mismatch",
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
        let committed = apply(MockCommitPermit {
            state,
            input_token: pending.input_token,
            next_position,
            next_revision,
        });
        if self.fail_after_commit {
            return Err(RuntimeError::InvalidAdapterWork(
                "injected post-commit adapter failure",
            ));
        }
        Ok(committed)
    }

    fn apply_state_commit(permit: MockCommitPermit<'_>) {
        permit.state.tokens[permit.state.position] = permit.input_token;
        permit.state.position = permit.next_position;
        permit.state.revision = permit.next_revision;
    }
}

fn new_engine_with_limits(limits: SchedulerLimits) -> SchedulerEngine<MockAdapter> {
    new_engine_with_policy(limits, SchedulingPolicy::default())
}

fn new_engine_with_policy(
    limits: SchedulerLimits,
    policy: SchedulingPolicy,
) -> SchedulerEngine<MockAdapter> {
    let adapter = MockAdapter::new();
    let config = SchedulerConfig::new(&adapter, limits)
        .expect("mock scheduler config")
        .with_scheduling_policy(policy);
    SchedulerEngine::new(adapter, config).expect("mock scheduler engine")
}

fn tiny_engine() -> SchedulerEngine<MockAdapter> {
    new_engine_with_limits(SchedulerLimits::tiny())
}

fn drain_tokens(
    engine: &mut SchedulerEngine<MockAdapter>,
    id: runnel_scheduler::RequestId,
) -> Vec<u32> {
    engine
        .drain_events(id, usize::MAX)
        .expect("drain committed events")
        .into_iter()
        .map(|event| {
            assert_eq!(event.request_id(), id);
            event.token()
        })
        .collect()
}

fn phase(engine: &SchedulerEngine<MockAdapter>, id: runnel_scheduler::RequestId) -> RequestPhase {
    engine.request_phase(id).expect("request remains retained")
}

#[test]
fn public_service_trace_is_incremental_phase_exact_and_debug_redacted() {
    assert_eq!(ServicePhase::Prefill.evidence_bit(), 0);
    assert_eq!(ServicePhase::Decode.evidence_bit(), 1);
    assert_eq!(ServicePhase::Prefill.as_str(), "prefill");
    assert_eq!(ServicePhase::Decode.as_str(), "decode");

    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 1;
    limits.waves_per_step = 1;
    limits.max_active_requests = 1;
    limits.trace_capacity = 4;
    let mut engine = new_engine_with_limits(limits);
    let empty = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("initial trace view");
    assert!(empty.events().is_empty());
    assert_eq!(empty.start_cursor(), ServiceTraceCursor::origin());
    assert_eq!(empty.next_cursor(), ServiceTraceCursor::origin());
    assert_eq!(empty.status().retained_events(), 0);
    assert_eq!(empty.status().event_limit(), 4);
    assert!(empty.status().healthy());
    drop(empty);

    let id = engine
        .try_submit(RequestSpec::new(&[0, 1], 3, SamplingPolicy::Greedy, None))
        .expect("trace request accepted");
    let expected_phases = [
        ServicePhase::Prefill,
        ServicePhase::Prefill,
        ServicePhase::Decode,
        ServicePhase::Decode,
    ];
    let mut cursor = ServiceTraceCursor::origin();
    for (position, expected_phase) in expected_phases.into_iter().enumerate() {
        let report = engine.step().expect("single-position trace step");
        assert_eq!(report.selected_positions, 1);
        assert_eq!(report.committed_positions, 1);
        let read = engine
            .service_trace_since(cursor)
            .expect("incremental trace read");
        assert_eq!(read.events().len(), 1);
        let event = read.events()[0];
        assert_eq!(event.request_id(), id);
        assert_eq!(event.position(), position);
        assert_eq!(event.phase(), expected_phase);
        assert_eq!(read.status().retained_events(), position + 1);
        assert!(read.status().healthy());
        assert_eq!(
            format!("{event:?}"),
            format!(
                "ServiceTraceEvent {{ request_id: \"<redacted>\", position: \"<redacted>\", phase: {expected_phase:?} }}"
            )
        );
        assert!(!format!("{read:?}").contains("RequestId"));
        cursor = read.next_cursor();
    }

    let frontier = engine
        .service_trace_since(cursor)
        .expect("frontier trace read");
    assert!(frontier.events().is_empty());
    assert_eq!(frontier.start_cursor(), cursor);
    assert_eq!(frontier.next_cursor(), cursor);
    assert!(frontier.status().healthy());
    drop(frontier);

    let empty_engine = tiny_engine();
    let invalid = empty_engine
        .service_trace_since(cursor)
        .expect_err("cursor beyond retained prefix must fail closed");
    assert_eq!(invalid.category(), ErrorCategory::InvalidRequest);
    assert!(
        empty_engine
            .service_trace_since(ServiceTraceCursor::origin())
            .expect("invalid read is nonmutating")
            .events()
            .is_empty()
    );

    assert_eq!(drain_tokens(&mut engine, id), [2, 3, 0]);
    assert_eq!(
        engine
            .take_terminal(id)
            .expect("trace terminal query")
            .expect("trace terminal retained")
            .outcome(),
        TerminalOutcome::Completed
    );
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
    assert_eq!(
        engine
            .shutdown()
            .expect("trace engine shutdown")
            .remaining_shared_bytes,
        0
    );
    assert!(matches!(
        engine
            .service_trace_since(ServiceTraceCursor::origin())
            .expect_err("successful shutdown destroys the trace"),
        SchedulerError::SchedulerClosed
    ));
}

#[test]
fn trace_overflow_is_sticky_accounted_and_behavior_neutral() {
    let mut small_limits = SchedulerLimits::tiny();
    small_limits.batch_width = 1;
    small_limits.waves_per_step = 1;
    small_limits.max_active_requests = 1;
    small_limits.trace_capacity = 1;
    let mut small = new_engine_with_limits(small_limits);
    let trace_charge = small.config().shared_static_charges().trace_bytes();
    assert_eq!(trace_charge, 128);
    assert_eq!(
        small
            .ledger_snapshot()
            .category(LedgerCategory::Trace)
            .used(),
        trace_charge
    );
    let small_id = small
        .try_submit(RequestSpec::new(&[0], 2, SamplingPolicy::Greedy, None))
        .expect("small-trace request");
    let small_first = small.step().expect("exact-full trace commit");
    let exact_full = small
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("exact-full trace read");
    assert_eq!(exact_full.events().len(), 1);
    assert_eq!(exact_full.events()[0].request_id(), small_id);
    assert_eq!(exact_full.events()[0].position(), 0);
    assert!(exact_full.status().healthy());
    assert!(!exact_full.status().overflowed());
    let full_cursor = exact_full.next_cursor();
    drop(exact_full);

    let small_second = small.step().expect("overflowing service commit");
    let overflowed = small
        .service_trace_since(full_cursor)
        .expect("overflowed frontier remains readable");
    assert!(overflowed.events().is_empty());
    assert_eq!(overflowed.status().retained_events(), 1);
    assert!(overflowed.status().overflowed());
    assert!(!overflowed.status().healthy());
    drop(overflowed);
    assert_eq!(
        small.step().expect("idle after overflow"),
        StepReport::default()
    );
    assert!(
        small
            .service_trace_since(ServiceTraceCursor::origin())
            .expect("overflow flag is sticky")
            .status()
            .overflowed()
    );
    assert_eq!(
        small
            .ledger_snapshot()
            .category(LedgerCategory::Trace)
            .used(),
        trace_charge
    );
    let small_tokens = drain_tokens(&mut small, small_id);
    let small_terminal = small
        .take_terminal(small_id)
        .expect("small terminal query")
        .expect("small terminal retained");

    let mut ample_limits = small_limits;
    ample_limits.trace_capacity = 8;
    let mut ample = new_engine_with_limits(ample_limits);
    let ample_id = ample
        .try_submit(RequestSpec::new(&[0], 2, SamplingPolicy::Greedy, None))
        .expect("ample-trace request");
    let ample_first = ample.step().expect("ample first commit");
    let ample_second = ample.step().expect("ample second commit");
    assert_eq!([small_first, small_second], [ample_first, ample_second]);
    let ample_trace = ample
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("complete ample trace");
    assert_eq!(ample_trace.events().len(), 2);
    assert!(ample_trace.status().healthy());
    assert_eq!(
        ample_trace
            .events()
            .iter()
            .map(|event| (event.position(), event.phase()))
            .collect::<Vec<_>>(),
        [(0, ServicePhase::Prefill), (1, ServicePhase::Decode)]
    );
    drop(ample_trace);
    let ample_tokens = drain_tokens(&mut ample, ample_id);
    let ample_terminal = ample
        .take_terminal(ample_id)
        .expect("ample terminal query")
        .expect("ample terminal retained");
    assert_eq!(small_tokens, ample_tokens);
    assert_eq!(small_terminal.outcome(), ample_terminal.outcome());
    assert_eq!(
        small_terminal.committed_positions(),
        ample_terminal.committed_positions()
    );
    assert_eq!(
        small_terminal.emitted_tokens(),
        ample_terminal.emitted_tokens()
    );

    assert_eq!(
        small
            .shutdown()
            .expect("small shutdown")
            .remaining_shared_bytes,
        0
    );
    assert_eq!(
        small
            .ledger_snapshot()
            .category(LedgerCategory::Trace)
            .used(),
        0
    );
    assert_eq!(
        ample
            .shutdown()
            .expect("ample shutdown")
            .remaining_shared_bytes,
        0
    );
}

#[test]
fn suppressed_control_decisions_do_not_append_service_events() {
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 1;
    limits.waves_per_step = 1;
    let mut engine = new_engine_with_limits(limits);
    let cancelled = engine
        .try_submit(RequestSpec::new(&[0], 2, SamplingPolicy::Greedy, None))
        .expect("cancellable request");
    assert_eq!(
        engine.cancel(cancelled).expect("publish cancellation"),
        CancelDisposition::Requested
    );
    assert_eq!(
        engine
            .step()
            .expect("resolve cancellation")
            .committed_positions,
        0
    );
    assert!(
        engine
            .service_trace_since(ServiceTraceCursor::origin())
            .expect("trace after cancellation")
            .events()
            .is_empty()
    );
    assert_eq!(
        engine
            .take_terminal(cancelled)
            .expect("cancel terminal query")
            .expect("cancel terminal retained")
            .outcome(),
        TerminalOutcome::Cancelled
    );

    let expired = engine
        .try_submit(RequestSpec::new(&[0], 2, SamplingPolicy::Greedy, Some(1)))
        .expect("deadline request");
    engine.advance_clock(1).expect("reach inclusive deadline");
    assert_eq!(
        engine.step().expect("resolve deadline").committed_positions,
        0
    );
    let trace = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("trace after deadline");
    assert!(trace.events().is_empty());
    assert!(trace.status().healthy());
    drop(trace);
    assert_eq!(
        engine
            .take_terminal(expired)
            .expect("deadline terminal query")
            .expect("deadline terminal retained")
            .outcome(),
        TerminalOutcome::DeadlineExceeded
    );
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
    assert_eq!(
        engine
            .shutdown()
            .expect("control trace shutdown")
            .remaining_shared_bytes,
        0
    );
}

#[test]
fn explicit_policies_have_frozen_names_and_distinct_service_order() {
    assert_eq!(
        SchedulingPolicy::default(),
        SchedulingPolicy::DeficitContinuousExpertCoalesce
    );
    assert_eq!(
        SchedulingPolicy::FifoRunToCompletion.evidence_id(),
        "fifo-single-request-run-to-completion-v1"
    );
    assert_eq!(
        SchedulingPolicy::DeficitContinuousExpertCoalesce.evidence_id(),
        "deficit-continuous-expert-coalesce-v1"
    );

    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 2;
    limits.waves_per_step = 1;
    limits.max_active_requests = 2;
    limits.output_capacity_per_request = 4;

    let geometry_adapter = MockAdapter::new();
    let candidate_config =
        SchedulerConfig::new(&geometry_adapter, limits).expect("candidate geometry");
    let baseline_config =
        candidate_config.with_scheduling_policy(SchedulingPolicy::FifoRunToCompletion);
    assert_eq!(
        candidate_config.scheduling_policy(),
        SchedulingPolicy::DeficitContinuousExpertCoalesce
    );
    assert_eq!(candidate_config.limits(), baseline_config.limits());
    assert_eq!(
        candidate_config.shared_static_charges(),
        baseline_config.shared_static_charges()
    );
    assert_eq!(
        candidate_config.minimum_total_charge_bytes(),
        baseline_config.minimum_total_charge_bytes()
    );

    let mut baseline = new_engine_with_policy(limits, SchedulingPolicy::FifoRunToCompletion);
    assert_eq!(
        baseline.config().scheduling_policy(),
        SchedulingPolicy::FifoRunToCompletion
    );
    let baseline_ids = [0_u32, 1_u32].map(|token| {
        baseline
            .try_submit(RequestSpec::new(&[token], 3, SamplingPolicy::Greedy, None))
            .expect("baseline request")
    });
    let first = baseline.step().expect("first baseline position");
    assert_eq!(first.promoted_requests, 2);
    assert_eq!(first.selected_positions, 1);
    assert_eq!(first.committed_positions, 1);
    assert_eq!(phase(&baseline, baseline_ids[0]), RequestPhase::Ready);
    assert_eq!(phase(&baseline, baseline_ids[1]), RequestPhase::Ready);
    assert_eq!(drain_tokens(&mut baseline, baseline_ids[0]), vec![1]);
    assert!(drain_tokens(&mut baseline, baseline_ids[1]).is_empty());

    for expected_position in 2..=3 {
        let report = baseline.step().expect("continued baseline position");
        assert_eq!(report.selected_positions, 1);
        assert_eq!(report.committed_positions, 1);
        assert_eq!(phase(&baseline, baseline_ids[1]), RequestPhase::Ready);
        assert!(
            drain_tokens(&mut baseline, baseline_ids[1]).is_empty(),
            "later FIFO request ran before position {expected_position} completed"
        );
    }
    assert_eq!(phase(&baseline, baseline_ids[0]), RequestPhase::Terminal);
    let next = baseline.step().expect("next baseline request starts");
    assert_eq!(next.promoted_requests, 0);
    assert_eq!(next.committed_positions, 1);
    assert_eq!(phase(&baseline, baseline_ids[1]), RequestPhase::Ready);

    let mut candidate =
        new_engine_with_policy(limits, SchedulingPolicy::DeficitContinuousExpertCoalesce);
    assert_eq!(
        candidate.config().scheduling_policy(),
        SchedulingPolicy::DeficitContinuousExpertCoalesce
    );
    let candidate_ids = [0_u32, 1_u32].map(|token| {
        candidate
            .try_submit(RequestSpec::new(&[token], 3, SamplingPolicy::Greedy, None))
            .expect("candidate request")
    });
    let first = candidate.step().expect("first candidate wave");
    assert_eq!(first.promoted_requests, 2);
    assert_eq!(first.selected_positions, 2);
    assert_eq!(first.committed_positions, 2);
    for id in candidate_ids {
        assert_eq!(phase(&candidate, id), RequestPhase::Ready);
        assert_eq!(drain_tokens(&mut candidate, id).len(), 1);
    }
}

fn completed_tokens_for_policy(
    policy: SchedulingPolicy,
    sampling: [SamplingPolicy; 2],
) -> Vec<Vec<u32>> {
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 2;
    limits.waves_per_step = 1;
    limits.max_active_requests = 2;
    limits.output_capacity_per_request = 4;
    let mut engine = new_engine_with_policy(limits, policy);
    let prompts = [[0_u32], [0_u32]];
    let offers = [
        BatchRequestSpec::absolute(RequestSpec::new(&prompts[0], 3, sampling[0], None)),
        BatchRequestSpec::absolute(RequestSpec::new(&prompts[1], 3, sampling[1], None)),
    ];
    let admission = engine
        .prepare_submit_batch(&offers)
        .expect("policy parity batch prepare")
        .commit_prepared_batch(17)
        .expect("policy parity batch commit");
    assert_eq!(admission.accepted_count(), 2);
    assert_eq!(admission.rejected_count(), 0);
    let ids = admission
        .accepted()
        .map(|accepted| {
            assert_eq!(accepted.admitted_ns(), 17);
            assert_eq!(engine.admitted_ns(accepted.request_id()).unwrap(), 17);
            accepted.request_id()
        })
        .collect::<Vec<_>>();
    for _ in 0..8 {
        if ids
            .iter()
            .copied()
            .all(|id| phase(&engine, id) == RequestPhase::Terminal)
        {
            break;
        }
        engine.step().expect("policy parity step");
    }
    let tokens = ids
        .into_iter()
        .map(|id| {
            assert_eq!(phase(&engine, id), RequestPhase::Terminal);
            let tokens = drain_tokens(&mut engine, id);
            let terminal = engine
                .take_terminal(id)
                .expect("policy parity terminal query")
                .expect("policy parity terminal result");
            assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
            assert_eq!(terminal.emitted_tokens(), tokens.len());
            tokens
        })
        .collect();
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
    let shutdown = engine.shutdown().expect("policy parity shutdown");
    assert_eq!(shutdown.terminated_requests, 0);
    assert_eq!(shutdown.remaining_shared_bytes, 0);
    tokens
}

#[test]
fn fifo_and_continuous_policies_preserve_per_request_token_parity() {
    let greedy_baseline = completed_tokens_for_policy(
        SchedulingPolicy::FifoRunToCompletion,
        [SamplingPolicy::Greedy; 2],
    );
    let greedy_candidate = completed_tokens_for_policy(
        SchedulingPolicy::DeficitContinuousExpertCoalesce,
        [SamplingPolicy::Greedy; 2],
    );
    assert_eq!(greedy_baseline, greedy_candidate);

    let sampled = [0x5eed, 0xdecafbad].map(|seed| {
        SamplingPolicy::Sample(SampleConfig {
            seed,
            // Flatten the mock's 10-versus-0 logits so the frozen seeds
            // exercise non-argmax draws instead of the greedy path.
            temperature: 1_000_000.0,
            top_k: 5,
            top_p: 1.0,
        })
    });
    let sampled_baseline =
        completed_tokens_for_policy(SchedulingPolicy::FifoRunToCompletion, sampled);
    let sampled_candidate =
        completed_tokens_for_policy(SchedulingPolicy::DeficitContinuousExpertCoalesce, sampled);
    assert_eq!(sampled_baseline, sampled_candidate);
    assert_ne!(
        sampled_baseline, greedy_baseline,
        "seeded parity must exercise a genuinely non-greedy stream"
    );
    assert_ne!(
        sampled_baseline[0], sampled_baseline[1],
        "distinct request seeds must exercise distinct request-owned streams"
    );
}

#[test]
fn fifo_output_blocking_holds_the_head_while_non_head_cancellation_cleans_up() {
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 3;
    limits.waves_per_step = 1;
    limits.max_active_requests = 3;
    limits.output_capacity_per_request = 1;
    let mut engine = new_engine_with_policy(limits, SchedulingPolicy::FifoRunToCompletion);
    let ids = [0_u32, 1_u32, 2_u32].map(|token| {
        engine
            .try_submit(RequestSpec::new(&[token], 2, SamplingPolicy::Greedy, None))
            .expect("FIFO pressure request")
    });

    let first = engine.step().expect("fill FIFO head output");
    assert_eq!(first.promoted_requests, 3);
    assert_eq!(first.committed_positions, 1);
    assert_eq!(phase(&engine, ids[0]), RequestPhase::OutputBlocked);
    assert_eq!(phase(&engine, ids[1]), RequestPhase::Ready);
    assert!(drain_tokens(&mut engine, ids[1]).is_empty());
    let before_cancellation_reap = engine.ledger_snapshot().request_used();

    assert_eq!(
        engine.cancel(ids[2]).expect("cancel non-head"),
        CancelDisposition::Requested
    );
    let blocked = engine.step().expect("resolve non-head cancellation");
    assert_eq!(blocked.committed_positions, 0);
    assert_eq!(blocked.terminal_decisions, 1);
    assert_eq!(phase(&engine, ids[2]), RequestPhase::Terminal);
    assert!(drain_tokens(&mut engine, ids[1]).is_empty());
    assert!(drain_tokens(&mut engine, ids[2]).is_empty());
    assert_eq!(
        engine
            .take_terminal(ids[2])
            .expect("cancelled non-head terminal query")
            .expect("cancelled non-head terminal")
            .outcome(),
        TerminalOutcome::Cancelled
    );
    assert!(engine.ledger_snapshot().request_used() < before_cancellation_reap);

    assert_eq!(drain_tokens(&mut engine, ids[0]), vec![1]);
    let completed_head = engine.step().expect("unblocked FIFO head completes");
    assert_eq!(completed_head.committed_positions, 1);
    assert_eq!(phase(&engine, ids[0]), RequestPhase::Terminal);
    assert_eq!(drain_tokens(&mut engine, ids[0]), vec![2]);
    assert_eq!(
        engine
            .take_terminal(ids[0])
            .expect("FIFO head terminal query")
            .expect("FIFO head terminal")
            .outcome(),
        TerminalOutcome::Completed
    );
    assert!(drain_tokens(&mut engine, ids[1]).is_empty());

    let successor = engine.step().expect("FIFO successor starts");
    assert_eq!(successor.committed_positions, 1);
    assert_eq!(phase(&engine, ids[1]), RequestPhase::OutputBlocked);
    assert_eq!(drain_tokens(&mut engine, ids[1]), vec![2]);
    let terminal_successor = engine.step().expect("FIFO successor completes");
    assert_eq!(terminal_successor.committed_positions, 1);
    assert_eq!(phase(&engine, ids[1]), RequestPhase::Terminal);
    assert_eq!(drain_tokens(&mut engine, ids[1]), vec![3]);
    assert_eq!(
        engine
            .take_terminal(ids[1])
            .expect("FIFO successor terminal query")
            .expect("FIFO successor terminal")
            .outcome(),
        TerminalOutcome::Completed
    );
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
    let shutdown = engine.shutdown().expect("FIFO pressure shutdown");
    assert_eq!(shutdown.terminated_requests, 0);
    assert_eq!(shutdown.remaining_shared_bytes, 0);
}

#[test]
fn fifo_non_head_deadline_terminalizes_without_receiving_service() {
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 3;
    limits.waves_per_step = 1;
    limits.max_active_requests = 3;
    limits.output_capacity_per_request = 4;
    let mut engine = new_engine_with_policy(limits, SchedulingPolicy::FifoRunToCompletion);
    let head = engine
        .try_submit(RequestSpec::new(&[0], 3, SamplingPolicy::Greedy, None))
        .expect("FIFO deadline head");
    let expiring = engine
        .try_submit(RequestSpec::new(&[1], 3, SamplingPolicy::Greedy, Some(1)))
        .expect("FIFO expiring non-head");
    let tail = engine
        .try_submit(RequestSpec::new(&[2], 3, SamplingPolicy::Greedy, None))
        .expect("FIFO deadline tail");

    assert_eq!(engine.step().unwrap().committed_positions, 1);
    assert_eq!(drain_tokens(&mut engine, head), vec![1]);
    assert!(drain_tokens(&mut engine, expiring).is_empty());
    assert!(drain_tokens(&mut engine, tail).is_empty());
    engine.advance_clock(1).expect("expire non-head");
    let report = engine.step().expect("resolve non-head deadline");
    assert_eq!(report.terminal_decisions, 1);
    assert_eq!(report.committed_positions, 1);
    assert_eq!(phase(&engine, expiring), RequestPhase::Terminal);
    assert_eq!(phase(&engine, head), RequestPhase::Ready);
    assert_eq!(phase(&engine, tail), RequestPhase::Ready);
    assert!(drain_tokens(&mut engine, expiring).is_empty());
    assert!(drain_tokens(&mut engine, tail).is_empty());
    assert_eq!(
        engine
            .take_terminal(expiring)
            .unwrap()
            .expect("deadline terminal")
            .outcome(),
        TerminalOutcome::DeadlineExceeded
    );
}

#[derive(Debug, PartialEq, Eq)]
struct DeterministicScenario {
    reports: Vec<StepReport>,
    emissions: Vec<Vec<(usize, Vec<u32>)>>,
    terminals: Vec<(usize, TerminalOutcome, usize, usize)>,
}

fn run_deterministic_scenario() -> DeterministicScenario {
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 2;
    limits.waves_per_step = 1;
    limits.output_capacity_per_request = 4;
    let mut engine = new_engine_with_limits(limits);
    let prompts = [[0], [1], [2]];
    let ids: Vec<_> = prompts
        .iter()
        .map(|prompt| {
            engine
                .try_submit(RequestSpec::new(prompt, 2, SamplingPolicy::Greedy, None))
                .expect("accepted deterministic request")
        })
        .collect();
    assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));

    let mut reports = Vec::new();
    let mut emissions = Vec::new();
    for _ in 0..4 {
        reports.push(engine.step().expect("deterministic scheduler step"));
        let mut step_emissions = Vec::new();
        for (request_index, id) in ids.iter().copied().enumerate() {
            let tokens = drain_tokens(&mut engine, id);
            if !tokens.is_empty() {
                step_emissions.push((request_index, tokens));
            }
        }
        emissions.push(step_emissions);
    }
    assert!(
        ids.iter()
            .copied()
            .all(|id| phase(&engine, id) == RequestPhase::Terminal)
    );

    let terminals = ids
        .iter()
        .copied()
        .enumerate()
        .map(|(request_index, id)| {
            let terminal = engine
                .take_terminal(id)
                .expect("take deterministic terminal")
                .expect("terminal result retained");
            assert_eq!(terminal.request_id(), id);
            (
                request_index,
                terminal.outcome(),
                terminal.committed_positions(),
                terminal.emitted_tokens(),
            )
        })
        .collect();
    DeterministicScenario {
        reports,
        emissions,
        terminals,
    }
}

#[test]
fn construction_and_submission_are_public_and_accounted() {
    let mut engine = tiny_engine();
    let initial = engine.snapshot();
    assert_eq!(initial.ledger_peak_bytes, initial.ledger_used_bytes);

    let id = engine
        .try_submit(RequestSpec::new(&[0], 2, SamplingPolicy::Greedy, None))
        .expect("accepted request");
    assert_eq!(id.get(), 1);
    let submitted = engine.snapshot();
    assert_eq!(submitted.queued_requests, 1);
    assert!(submitted.ledger_used_bytes > initial.ledger_used_bytes);
    assert!(submitted.ledger_peak_bytes >= submitted.ledger_used_bytes);
}

fn pressure_limits(max_outstanding: u64, max_queued: u64) -> SchedulerLimits {
    let mut limits = SchedulerLimits::evidence();
    limits.max_outstanding_requests = max_outstanding;
    limits.max_active_requests = 16.min(max_outstanding);
    limits.max_queued_requests = max_queued;
    limits.max_retained_terminal_results = max_outstanding;
    limits
}

#[test]
fn prepared_batch_drop_is_exact_and_consumes_no_request_identity() {
    let mut engine = tiny_engine();
    let before_engine = engine.snapshot();
    let before_ledger = engine.ledger_snapshot();
    let prompt = [0_u32; 2];
    let offers =
        [BatchRequestSpec::release_relative(&prompt, 2, SamplingPolicy::Greedy, 20_000_000); 3];

    let prepared = engine
        .prepare_submit_batch(&offers)
        .expect("prepare unpublished batch");
    drop(prepared);

    assert_eq!(engine.snapshot(), before_engine);
    assert_eq!(engine.ledger_snapshot(), before_ledger);
    let id = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("identity remains unconsumed");
    assert_eq!(id.get(), 1);

    let malformed = [
        BatchRequestSpec::absolute(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None)),
        BatchRequestSpec::absolute(RequestSpec::new(&[], 1, SamplingPolicy::Greedy, None)),
    ];
    let before = engine.snapshot();
    let error = engine
        .prepare_submit_batch(&malformed)
        .expect_err("a malformed later offer aborts the complete prepare");
    assert!(matches!(
        error,
        SchedulerError::InvalidRequest {
            field: "prompt",
            ..
        }
    ));
    assert_eq!(engine.snapshot(), before);
}

#[test]
fn deadline_pressure_16_commits_one_release_without_promotion() {
    let mut engine = new_engine_with_limits(pressure_limits(16, 16));
    let prompt = [0_u32; 16];
    let offers =
        [BatchRequestSpec::release_relative(&prompt, 8, SamplingPolicy::Greedy, 20_000_000); 16];
    let release_ns = 1_000_000_u64;

    let result = engine
        .prepare_submit_batch(&offers)
        .expect("prepare pressure-16")
        .commit_prepared_batch(release_ns)
        .expect("commit pressure-16");

    assert_eq!(result.release_ns(), release_ns);
    assert_eq!(result.offered_count(), 16);
    assert_eq!(result.rejected_count(), 0);
    assert_eq!(result.accepted_count(), 16);
    for (index, accepted) in result.accepted().enumerate() {
        assert_eq!(accepted.offered_index(), index);
        assert_eq!(accepted.request_id().get(), index as u64 + 1);
        assert_eq!(accepted.admitted_ns(), release_ns);
        assert_eq!(
            engine.admitted_ns(accepted.request_id()).unwrap(),
            release_ns
        );
        assert_eq!(
            engine.deadline_ns(accepted.request_id()).unwrap(),
            Some(release_ns + 20_000_000)
        );
        assert_eq!(phase(&engine, accepted.request_id()), RequestPhase::Queued);
    }
    let snapshot = engine.snapshot();
    assert_eq!(snapshot.queued_requests, 16);
    assert_eq!(snapshot.active_requests, 0);
}

#[test]
fn deadline_pressure_24_accepts_fifo_prefix_and_rejects_suffix_exactly() {
    let mut engine = new_engine_with_limits(pressure_limits(16, 16));
    let prompt = [0_u32; 16];
    let offers =
        [BatchRequestSpec::release_relative(&prompt, 8, SamplingPolicy::Greedy, 20_000_000); 24];
    let release_ns = 7_000_u64;

    let result = engine
        .prepare_submit_batch(&offers)
        .expect("prepare pressure-24")
        .commit_prepared_batch(release_ns)
        .expect("commit pressure-24");

    assert_eq!(result.accepted_count(), 16);
    assert_eq!(result.rejected_count(), 8);
    for (index, accepted) in result.accepted().enumerate() {
        assert_eq!(accepted.offered_index(), index);
        assert_eq!(accepted.request_id().get(), index as u64 + 1);
        assert_eq!(accepted.admitted_ns(), release_ns);
    }
    for (offset, rejected) in result.rejected().enumerate() {
        assert_eq!(rejected.offered_index(), 16 + offset);
        match rejected.error() {
            SchedulerError::ResourceExhausted {
                resource,
                required,
                limit,
            } => {
                assert_eq!(*resource, "queued request count");
                assert_eq!(*required, 17);
                assert_eq!(*limit, 16);
            }
            error => panic!("unexpected pressure rejection: {error:?}"),
        }
    }
    assert_eq!(engine.snapshot().queued_requests, 16);
    assert_eq!(engine.snapshot().active_requests, 0);
}

#[test]
fn pressure_rejections_and_failed_release_checks_leave_no_id_gaps() {
    let mut engine = new_engine_with_limits(pressure_limits(24, 16));
    let prompt = [0_u32; 2];
    let offers =
        [BatchRequestSpec::release_relative(&prompt, 2, SamplingPolicy::Greedy, 20_000_000); 24];
    let result = engine
        .prepare_submit_batch(&offers)
        .unwrap()
        .commit_prepared_batch(10)
        .unwrap();
    assert_eq!(result.accepted_count(), 16);
    assert_eq!(result.rejected_count(), 8);
    assert_eq!(engine.step().unwrap().promoted_requests, 16);
    let next = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("rejected suffix consumed no identities");
    assert_eq!(next.get(), 17);

    let mut rejected_limits = SchedulerLimits::tiny();
    rejected_limits.max_outstanding_requests = 1;
    rejected_limits.max_active_requests = 1;
    rejected_limits.max_queued_requests = 1;
    rejected_limits.max_retained_terminal_results = 1;
    let mut fresh = new_engine_with_limits(rejected_limits);
    let rejected_suffix = [
        BatchRequestSpec::absolute(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None)),
        BatchRequestSpec::release_relative(&[1], 1, SamplingPolicy::Greedy, u64::MAX),
    ];
    let before = fresh.ledger_snapshot();
    let error = fresh
        .prepare_submit_batch(&rejected_suffix)
        .unwrap()
        .commit_prepared_batch(1)
        .expect_err("relative deadline overflow is checked before publication");
    assert!(matches!(
        error,
        SchedulerError::InvalidRequest {
            field: "deadline",
            ..
        }
    ));
    assert_eq!(fresh.ledger_snapshot(), before);
    let first = fresh
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .unwrap();
    assert_eq!(first.get(), 1);
}

#[test]
fn every_release_validation_failure_is_an_exact_prepublication_rollback() {
    let request = RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None);

    let mut backwards = tiny_engine();
    backwards.advance_clock(10).expect("advance batch clock");
    let before_engine = backwards.snapshot();
    let before_ledger = backwards.ledger_snapshot();
    let error = backwards
        .prepare_submit_batch(&[BatchRequestSpec::absolute(request)])
        .expect("prepare backwards-release batch")
        .commit_prepared_batch(9)
        .expect_err("release boundary cannot move backwards");
    assert!(matches!(
        error,
        SchedulerError::InvalidRequest {
            field: "release_ns",
            ..
        }
    ));
    assert_eq!(backwards.snapshot(), before_engine);
    assert_eq!(backwards.ledger_snapshot(), before_ledger);
    assert_eq!(backwards.try_submit(request).unwrap().get(), 1);

    let mut absolute = tiny_engine();
    let expiring =
        BatchRequestSpec::absolute(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, Some(10)));
    let before_engine = absolute.snapshot();
    let before_ledger = absolute.ledger_snapshot();
    let error = absolute
        .prepare_submit_batch(&[expiring])
        .expect("prepare not-yet-expired absolute deadline")
        .commit_prepared_batch(10)
        .expect_err("absolute deadline expires inclusively at release");
    assert_eq!(error.category(), ErrorCategory::DeadlineExceeded);
    assert_eq!(absolute.snapshot(), before_engine);
    assert_eq!(absolute.ledger_snapshot(), before_ledger);
    assert_eq!(absolute.try_submit(request).unwrap().get(), 1);

    let mut zero_relative = tiny_engine();
    let before_engine = zero_relative.snapshot();
    let before_ledger = zero_relative.ledger_snapshot();
    let error = zero_relative
        .prepare_submit_batch(&[BatchRequestSpec::release_relative(
            &[0],
            1,
            SamplingPolicy::Greedy,
            0,
        )])
        .expect_err("zero release-relative duration is invalid");
    assert!(matches!(
        error,
        SchedulerError::InvalidRequest {
            field: "deadline",
            ..
        }
    ));
    assert_eq!(zero_relative.snapshot(), before_engine);
    assert_eq!(zero_relative.ledger_snapshot(), before_ledger);
    assert_eq!(zero_relative.try_submit(request).unwrap().get(), 1);

    let mut overflowing = tiny_engine();
    let before_engine = overflowing.snapshot();
    let before_ledger = overflowing.ledger_snapshot();
    let error = overflowing
        .prepare_submit_batch(&[BatchRequestSpec::release_relative(
            &[0],
            1,
            SamplingPolicy::Greedy,
            u64::MAX,
        )])
        .expect("prepare overflowing relative deadline")
        .commit_prepared_batch(1)
        .expect_err("relative release deadline must not wrap");
    assert!(matches!(
        error,
        SchedulerError::InvalidRequest {
            field: "deadline",
            ..
        }
    ));
    assert_eq!(overflowing.snapshot(), before_engine);
    assert_eq!(overflowing.ledger_snapshot(), before_ledger);
    assert_eq!(overflowing.try_submit(request).unwrap().get(), 1);

    let mut limits = SchedulerLimits::tiny();
    limits.max_outstanding_requests = 2;
    limits.max_active_requests = 1;
    limits.max_queued_requests = 1;
    limits.max_retained_terminal_results = 2;
    let mut all_rejected = new_engine_with_limits(limits);
    let first = all_rejected
        .try_submit(request)
        .expect("queued pressure owner");
    let before_engine = all_rejected.snapshot();
    let before_ledger = all_rejected.ledger_snapshot();
    let error = all_rejected
        .prepare_submit_batch(&[BatchRequestSpec::release_relative(
            &[1],
            1,
            SamplingPolicy::Greedy,
            u64::MAX,
        )])
        .expect("prepare all-rejected batch")
        .commit_prepared_batch(1)
        .expect_err("rejected offer deadline still validates at release");
    assert!(matches!(
        error,
        SchedulerError::InvalidRequest {
            field: "deadline",
            ..
        }
    ));
    assert_eq!(all_rejected.snapshot(), before_engine);
    assert_eq!(all_rejected.ledger_snapshot(), before_ledger);
    all_rejected.step().expect("promote pressure owner");
    assert_eq!(
        all_rejected.try_submit(request).unwrap().get(),
        first.get() + 1
    );
}

#[test]
fn fifo_pressure_does_not_bypass_an_unfit_head_for_a_smaller_offer() {
    let mut geometry = SchedulerLimits::tiny();
    geometry.max_prompt_tokens = 17;
    geometry.max_new_tokens = 1;
    geometry.max_context_tokens = 17;

    let mut probe = new_engine_with_limits(geometry);
    let shared = probe.ledger_snapshot().total_used();
    probe
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .unwrap();
    let small_delta = probe.ledger_snapshot().total_used() - shared;
    let minimum = probe.config().minimum_total_charge_bytes();
    let target_limit = minimum.max(shared + small_delta * 2);

    geometry.logical_memory_limit_bytes = target_limit;
    let mut engine = new_engine_with_limits(geometry);
    engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("filler request");
    let large_prompt = [0_u32; 17];
    let offers = [
        BatchRequestSpec::absolute(RequestSpec::new(
            &large_prompt,
            1,
            SamplingPolicy::Greedy,
            None,
        )),
        BatchRequestSpec::absolute(RequestSpec::new(&[1], 1, SamplingPolicy::Greedy, None)),
    ];
    let result = engine
        .prepare_submit_batch(&offers)
        .expect("both descriptions are intrinsically valid")
        .commit_prepared_batch(1)
        .unwrap();
    assert_eq!(result.accepted_count(), 0);
    assert_eq!(result.rejected_count(), 2);
    let mut rejected = result.rejected();
    let first_rejection = rejected.next().expect("first rejection");
    let second_rejection = rejected.next().expect("second rejection");
    assert_eq!(
        first_rejection.error().category(),
        ErrorCategory::ResourceExhausted
    );
    assert_eq!(
        format!("{:?}", first_rejection.error()),
        format!("{:?}", second_rejection.error())
    );
    assert_eq!(engine.snapshot().queued_requests, 1);
}

#[test]
fn atomic_batch_matches_the_incremental_fifo_admission_sequence() {
    let limits = pressure_limits(24, 16);
    let mut batch_engine = new_engine_with_limits(limits);
    let mut incremental_engine = new_engine_with_limits(limits);
    incremental_engine.advance_clock(42).unwrap();
    let prompt = [0_u32; 2];
    let requests = [RequestSpec::new(&prompt, 2, SamplingPolicy::Greedy, None); 24];
    let offers = requests.map(BatchRequestSpec::absolute);

    let batch = batch_engine
        .prepare_submit_batch(&offers)
        .unwrap()
        .commit_prepared_batch(42)
        .unwrap();
    let incremental = requests.map(|request| incremental_engine.try_submit(request));

    for (index, result) in incremental.iter().enumerate() {
        if index < 16 {
            let id = result.as_ref().expect("incremental accepted prefix");
            let accepted = batch.accepted().nth(index).expect("batch accepted prefix");
            assert_eq!(id.get(), accepted.request_id().get());
            assert_eq!(accepted.offered_index(), index);
        } else {
            let error = result.as_ref().expect_err("incremental rejected suffix");
            let rejected = batch
                .rejected()
                .nth(index - 16)
                .expect("batch rejected suffix");
            assert_eq!(format!("{error:?}"), format!("{:?}", rejected.error()));
        }
    }
    assert_eq!(batch_engine.snapshot(), incremental_engine.snapshot());
    assert_eq!(
        batch_engine.ledger_snapshot(),
        incremental_engine.ledger_snapshot()
    );
}

#[test]
fn batch_requests_complete_cancel_reap_and_reuse_every_registry_slot() {
    let mut limits = SchedulerLimits::tiny();
    limits.max_outstanding_requests = 2;
    limits.max_active_requests = 2;
    limits.max_queued_requests = 2;
    limits.max_retained_terminal_results = 2;
    limits.batch_width = 2;
    limits.waves_per_step = 1;
    let mut engine = new_engine_with_limits(limits);
    let pristine = engine.ledger_snapshot();
    let prompts = [[0_u32], [1_u32]];
    let offers = [
        BatchRequestSpec::absolute(RequestSpec::new(
            &prompts[0],
            2,
            SamplingPolicy::Greedy,
            None,
        )),
        BatchRequestSpec::absolute(RequestSpec::new(
            &prompts[1],
            3,
            SamplingPolicy::Greedy,
            None,
        )),
    ];
    let first_batch = engine
        .prepare_submit_batch(&offers)
        .expect("prepare first lifecycle batch")
        .commit_prepared_batch(11)
        .expect("commit first lifecycle batch");
    let first_ids = first_batch
        .accepted()
        .map(|accepted| accepted.request_id())
        .collect::<Vec<_>>();
    assert_eq!(first_ids.len(), 2);
    assert_eq!(
        engine.cancel(first_ids[1]).expect("cancel batch request"),
        CancelDisposition::Requested
    );

    for _ in 0..8 {
        if first_ids
            .iter()
            .copied()
            .all(|id| phase(&engine, id) == RequestPhase::Terminal)
        {
            break;
        }
        engine.step().expect("first lifecycle batch step");
    }
    assert!(
        first_ids
            .iter()
            .copied()
            .all(|id| phase(&engine, id) == RequestPhase::Terminal)
    );
    assert_eq!(drain_tokens(&mut engine, first_ids[0]).len(), 2);
    assert!(drain_tokens(&mut engine, first_ids[1]).is_empty());
    let completed = engine
        .take_terminal(first_ids[0])
        .expect("completed batch terminal query")
        .expect("completed batch terminal");
    let cancelled = engine
        .take_terminal(first_ids[1])
        .expect("cancelled batch terminal query")
        .expect("cancelled batch terminal");
    assert_eq!(completed.outcome(), TerminalOutcome::Completed);
    assert_eq!(cancelled.outcome(), TerminalOutcome::Cancelled);
    let reaped = engine.ledger_snapshot();
    assert_eq!(reaped.request_used(), 0);
    assert_eq!(reaped.shared_used(), pristine.shared_used());
    assert_eq!(reaped.total_used(), pristine.total_used());

    let reused_batch = engine
        .prepare_submit_batch(&offers)
        .expect("prepare reused lifecycle batch")
        .commit_prepared_batch(12)
        .expect("commit reused lifecycle batch");
    assert_eq!(
        first_batch
            .accepted()
            .map(|accepted| accepted.request_id())
            .collect::<Vec<_>>(),
        first_ids,
        "retained compact results remain independent of scratch reuse"
    );
    let reused_ids = reused_batch
        .accepted()
        .map(|accepted| accepted.request_id())
        .collect::<Vec<_>>();
    assert_eq!(reused_ids.len(), 2);
    assert!(reused_ids[0] > first_ids[1]);
    for _ in 0..8 {
        if reused_ids
            .iter()
            .copied()
            .all(|id| phase(&engine, id) == RequestPhase::Terminal)
        {
            break;
        }
        engine.step().expect("reused lifecycle batch step");
    }
    for id in reused_ids {
        assert_eq!(phase(&engine, id), RequestPhase::Terminal);
        assert!(!drain_tokens(&mut engine, id).is_empty());
        assert_eq!(
            engine
                .take_terminal(id)
                .expect("reused terminal query")
                .expect("reused terminal")
                .outcome(),
            TerminalOutcome::Completed
        );
    }
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
    let shutdown = engine.shutdown().expect("batch lifecycle shutdown");
    assert_eq!(shutdown.terminated_requests, 0);
    assert_eq!(shutdown.remaining_shared_bytes, 0);
}

#[test]
fn intrinsically_unserviceable_fifo_head_is_rejected_before_admission() {
    let adapter = MockAdapter::new();
    let mut limits = SchedulerLimits::tiny();
    limits.max_prompt_tokens = 17;
    limits.max_new_tokens = 1;
    limits.max_context_tokens = 17;
    let unconstrained =
        SchedulerConfig::new(&adapter, limits).expect("unconstrained scheduler config");
    limits.logical_memory_limit_bytes = unconstrained.minimum_total_charge_bytes();
    let config = SchedulerConfig::new(&adapter, limits).expect("exact-minimum scheduler config");
    let mut engine = SchedulerEngine::new(adapter, config).expect("exact-minimum scheduler engine");
    let baseline = engine.snapshot();

    let oversized_prompt = [0_u32; 17];
    let error = engine
        .try_submit(RequestSpec::new(
            &oversized_prompt,
            1,
            SamplingPolicy::Greedy,
            None,
        ))
        .expect_err("request whose base fits but lifecycle cannot must be rejected");
    assert_eq!(error.category(), ErrorCategory::ResourceExhausted);
    assert_eq!(engine.snapshot(), baseline);

    let serviceable = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("minimum request remains serviceable");
    assert_eq!(
        engine
            .step()
            .expect("minimum request completes")
            .committed_positions,
        1
    );
    assert_eq!(phase(&engine, serviceable), RequestPhase::Terminal);
}

#[test]
fn adapter_error_after_commit_terminalizes_without_stranding_ownership() {
    for max_new_tokens in [1, 2] {
        let adapter = MockAdapter::failing_after_commit();
        let config =
            SchedulerConfig::new(&adapter, SchedulerLimits::tiny()).expect("scheduler config");
        let mut engine = SchedulerEngine::new(adapter, config).expect("scheduler engine");
        let baseline = engine.snapshot();
        let id = engine
            .try_submit(RequestSpec::new(
                &[0],
                max_new_tokens,
                SamplingPolicy::Greedy,
                None,
            ))
            .expect("request accepted");

        let report = engine.step().expect("post-commit failure is contained");
        assert_eq!(report.committed_positions, 1);
        assert_eq!(report.terminal_decisions, 1);
        assert_eq!(phase(&engine, id), RequestPhase::Terminal);
        let trace = engine
            .service_trace_since(ServiceTraceCursor::origin())
            .expect("post-commit failure trace");
        assert_eq!(trace.events().len(), 1);
        assert_eq!(trace.events()[0].request_id(), id);
        assert_eq!(trace.events()[0].position(), 0);
        assert_eq!(trace.events()[0].phase(), ServicePhase::Prefill);
        assert!(trace.status().healthy());
        drop(trace);
        assert_eq!(drain_tokens(&mut engine, id), [1]);
        let terminal = engine
            .take_terminal(id)
            .expect("take failed terminal")
            .expect("terminal retained");
        assert_eq!(
            terminal.outcome(),
            TerminalOutcome::Failed {
                category: ErrorCategory::Internal,
            }
        );
        assert_eq!(terminal.committed_positions(), 1);
        assert_eq!(terminal.emitted_tokens(), 1);
        assert_eq!(
            engine.snapshot().ledger_used_bytes,
            baseline.ledger_used_bytes
        );
        assert_eq!(
            engine.step().expect("ring remains usable"),
            StepReport::default()
        );
    }
}

#[test]
fn adapter_transaction_identity_reuse_across_requests_fails_closed() {
    let adapter = MockAdapter::reusing_transaction_identity();
    let config = SchedulerConfig::new(&adapter, SchedulerLimits::tiny()).expect("scheduler config");
    let mut engine = SchedulerEngine::new(adapter, config).expect("scheduler engine");
    let first = engine
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("first request accepted");
    let reused = engine
        .try_submit(RequestSpec::new(&[1], 1, SamplingPolicy::Greedy, None))
        .expect("second request admitted before adapter identity validation");

    let report = engine.step().expect("identity reuse is contained");
    assert_eq!(report.committed_positions, 1);
    assert_eq!(report.terminal_decisions, 2);
    assert_eq!(drain_tokens(&mut engine, first), [1]);
    assert!(drain_tokens(&mut engine, reused).is_empty());
    let first_terminal = engine
        .take_terminal(first)
        .expect("first terminal query")
        .expect("first terminal retained");
    assert_eq!(first_terminal.outcome(), TerminalOutcome::Completed);
    let reused_terminal = engine
        .take_terminal(reused)
        .expect("reused terminal query")
        .expect("reused terminal retained");
    assert_eq!(
        reused_terminal.outcome(),
        TerminalOutcome::Failed {
            category: ErrorCategory::Internal,
        }
    );
    assert_eq!(reused_terminal.committed_positions(), 0);
}

#[test]
fn multi_request_service_is_deterministic_and_fair() {
    let first = run_deterministic_scenario();
    let second = run_deterministic_scenario();
    assert_eq!(first, second);

    assert_eq!(
        first
            .reports
            .iter()
            .map(|report| report.committed_positions)
            .collect::<Vec<_>>(),
        [2, 1, 2, 1]
    );
    assert!(first.reports.iter().all(|report| {
        report.waves == 1
            && report.selected_positions == report.committed_positions
            && report.expert_tasks == report.committed_positions
    }));
    assert_eq!(
        first.emissions,
        [
            vec![(0, vec![1]), (1, vec![2])],
            vec![(2, vec![3])],
            vec![(0, vec![2]), (1, vec![3])],
            vec![(2, vec![0])],
        ]
    );
    assert_eq!(
        first.terminals,
        [
            (0, TerminalOutcome::Completed, 2, 2),
            (1, TerminalOutcome::Completed, 2, 2),
            (2, TerminalOutcome::Completed, 2, 2),
        ]
    );
}

#[test]
fn prompt_and_decode_positions_publish_contiguous_output() {
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 1;
    limits.waves_per_step = 4;
    let mut engine = new_engine_with_limits(limits);
    let id = engine
        .try_submit(RequestSpec::new(&[0, 2], 3, SamplingPolicy::Greedy, None))
        .expect("accepted prompt/decode request");

    assert_eq!(engine.take_terminal(id).expect("preterminal query"), None);
    let report = engine.step().expect("complete prompt/decode request");
    assert_eq!(report.committed_positions, 4);
    assert_eq!(report.selected_positions, 4);
    assert_eq!(report.terminal_decisions, 1);

    let events = engine
        .drain_events(id, usize::MAX)
        .expect("drain generated tokens");
    assert_eq!(events.len(), 3);
    for (index, event) in events.iter().copied().enumerate() {
        assert_eq!(event.request_id(), id);
        assert_eq!(event.output_index(), index);
    }
    assert_eq!(
        events
            .iter()
            .copied()
            .map(runnel_scheduler::OutputEvent::token)
            .collect::<Vec<_>>(),
        [3, 0, 1]
    );
    let terminal = engine
        .take_terminal(id)
        .expect("take terminal")
        .expect("terminal result retained");
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(terminal.committed_positions(), 4);
    assert_eq!(terminal.emitted_tokens(), 3);
}

#[test]
fn bounded_output_backpressure_blocks_only_full_requests_and_resumes_after_drain() {
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 2;
    limits.waves_per_step = 1;
    limits.output_capacity_per_request = 1;
    let mut engine = new_engine_with_limits(limits);
    let ids: Vec<_> = [0, 1, 2]
        .into_iter()
        .map(|token| {
            engine
                .try_submit(RequestSpec::new(&[token], 2, SamplingPolicy::Greedy, None))
                .expect("accepted backpressure request")
        })
        .collect();

    assert_eq!(
        engine
            .step()
            .expect("first service round")
            .committed_positions,
        2
    );
    assert_eq!(phase(&engine, ids[0]), RequestPhase::OutputBlocked);
    assert_eq!(phase(&engine, ids[1]), RequestPhase::OutputBlocked);
    assert_eq!(phase(&engine, ids[2]), RequestPhase::Ready);

    assert_eq!(
        engine
            .step()
            .expect("finish first service round")
            .committed_positions,
        1
    );
    assert!(
        ids.iter()
            .copied()
            .all(|id| phase(&engine, id) == RequestPhase::OutputBlocked)
    );
    let blocked = engine.snapshot();
    assert_eq!(blocked.output_blocked_requests, 3);

    assert!(
        engine
            .drain_events(ids[1], 0)
            .expect("zero drain")
            .is_empty()
    );
    assert_eq!(phase(&engine, ids[1]), RequestPhase::OutputBlocked);
    let stalled = engine.step().expect("all-full step is a no-op");
    assert_eq!(stalled.committed_positions, 0);
    assert_eq!(stalled.selected_positions, 0);
    assert_eq!(engine.snapshot(), blocked);

    let ledger_before_drain = engine.snapshot();
    assert_eq!(drain_tokens(&mut engine, ids[1]), [2]);
    assert_eq!(phase(&engine, ids[1]), RequestPhase::Ready);
    assert_eq!(phase(&engine, ids[0]), RequestPhase::OutputBlocked);
    assert_eq!(phase(&engine, ids[2]), RequestPhase::OutputBlocked);
    let after_drain = engine.snapshot();
    assert_eq!(
        after_drain.ledger_used_bytes,
        ledger_before_drain.ledger_used_bytes
    );
    assert_eq!(
        after_drain.ledger_peak_bytes,
        ledger_before_drain.ledger_peak_bytes
    );

    let resumed = engine.step().expect("selectively resumed request");
    assert_eq!(resumed.committed_positions, 1);
    assert_eq!(phase(&engine, ids[1]), RequestPhase::Terminal);
    assert_eq!(drain_tokens(&mut engine, ids[0]), [1]);
    assert_eq!(drain_tokens(&mut engine, ids[2]), [3]);

    let siblings = engine.step().expect("resume remaining siblings");
    assert_eq!(siblings.committed_positions, 2);
    assert_eq!(drain_tokens(&mut engine, ids[0]), [2]);
    assert_eq!(drain_tokens(&mut engine, ids[1]), [3]);
    assert_eq!(drain_tokens(&mut engine, ids[2]), [0]);
    for id in ids {
        let terminal = engine
            .take_terminal(id)
            .expect("take completed terminal")
            .expect("completed terminal retained");
        assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
        assert_eq!(terminal.committed_positions(), 2);
        assert_eq!(terminal.emitted_tokens(), 2);
    }
}

#[test]
fn cancellation_and_deadlines_are_inclusive_with_cancellation_precedence() {
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 1;
    limits.waves_per_step = 1;
    let mut engine = new_engine_with_limits(limits);

    let cancelled = engine
        .try_submit(RequestSpec::new(&[0], 2, SamplingPolicy::Greedy, Some(5)))
        .expect("accepted cancellable request");
    engine.advance_clock(5).expect("advance to deadline");
    assert_eq!(
        engine.cancel(cancelled).expect("request cancellation"),
        CancelDisposition::Requested
    );
    assert_eq!(
        engine.cancel(cancelled).expect("repeated cancellation"),
        CancelDisposition::AlreadyRequested
    );
    let cancelled_report = engine.step().expect("resolve cancellation");
    assert_eq!(cancelled_report.committed_positions, 0);
    assert_eq!(cancelled_report.terminal_decisions, 1);
    assert_eq!(
        engine.cancel(cancelled).expect("terminal cancellation"),
        CancelDisposition::AlreadyTerminal
    );
    let cancelled_result = engine
        .take_terminal(cancelled)
        .expect("take cancellation terminal")
        .expect("cancellation terminal retained");
    assert_eq!(cancelled_result.outcome(), TerminalOutcome::Cancelled);
    assert_eq!(cancelled_result.committed_positions(), 0);
    assert_eq!(cancelled_result.emitted_tokens(), 0);

    let expired = engine
        .try_submit(RequestSpec::new(&[1], 2, SamplingPolicy::Greedy, Some(6)))
        .expect("accepted future-deadline request");
    engine
        .advance_clock(6)
        .expect("advance to inclusive deadline");
    let deadline_report = engine.step().expect("resolve deadline");
    assert_eq!(deadline_report.committed_positions, 0);
    assert_eq!(deadline_report.terminal_decisions, 1);
    let deadline_result = engine
        .take_terminal(expired)
        .expect("take deadline terminal")
        .expect("deadline terminal retained");
    assert_eq!(deadline_result.outcome(), TerminalOutcome::DeadlineExceeded);
    assert_eq!(deadline_result.committed_positions(), 0);
    assert_eq!(deadline_result.emitted_tokens(), 0);

    let partially_committed = engine
        .try_submit(RequestSpec::new(&[2], 2, SamplingPolicy::Greedy, Some(8)))
        .expect("accepted partial request");
    assert_eq!(
        engine
            .step()
            .expect("commit first position")
            .committed_positions,
        1
    );
    engine
        .advance_clock(8)
        .expect("advance partial request to deadline");
    assert_eq!(
        engine
            .cancel(partially_committed)
            .expect("partial request cancellation"),
        CancelDisposition::Requested
    );
    assert_eq!(
        engine
            .step()
            .expect("cancel partial request")
            .committed_positions,
        0
    );
    assert_eq!(drain_tokens(&mut engine, partially_committed), [3]);
    let partial_result = engine
        .take_terminal(partially_committed)
        .expect("take partial cancellation")
        .expect("partial cancellation retained");
    assert_eq!(partial_result.outcome(), TerminalOutcome::Cancelled);
    assert_eq!(partial_result.committed_positions(), 1);
    assert_eq!(partial_result.emitted_tokens(), 1);

    let before_backwards_clock = engine.snapshot();
    let backwards = engine
        .advance_clock(7)
        .expect_err("monotonic clock cannot move backwards");
    assert_eq!(backwards.category(), ErrorCategory::InvalidRequest);
    assert_eq!(engine.snapshot(), before_backwards_clock);
}

#[test]
fn zero_generation_and_stop_token_completion_are_distinct() {
    let mut limits = SchedulerLimits::tiny();
    limits.batch_width = 1;
    limits.waves_per_step = 4;
    let mut engine = new_engine_with_limits(limits);

    let zero_generation = engine
        .try_submit(RequestSpec::new(&[0, 1], 0, SamplingPolicy::Greedy, None))
        .expect("accepted zero-generation request");
    let zero_report = engine.step().expect("evaluate zero-generation prompt");
    assert_eq!(zero_report.committed_positions, 2);
    assert!(drain_tokens(&mut engine, zero_generation).is_empty());
    let zero_result = engine
        .take_terminal(zero_generation)
        .expect("take zero-generation terminal")
        .expect("zero-generation terminal retained");
    assert_eq!(zero_result.outcome(), TerminalOutcome::Completed);
    assert_eq!(zero_result.committed_positions(), 2);
    assert_eq!(zero_result.emitted_tokens(), 0);

    let stop = engine
        .try_submit(RequestSpec::new(&[4], 4, SamplingPolicy::Greedy, None))
        .expect("accepted stop-token request");
    let stop_report = engine.step().expect("stop on first generated token");
    assert_eq!(stop_report.committed_positions, 1);
    assert_eq!(stop_report.terminal_decisions, 1);
    assert_eq!(drain_tokens(&mut engine, stop), [4]);
    let stop_result = engine
        .take_terminal(stop)
        .expect("take stop terminal")
        .expect("stop terminal retained");
    assert_eq!(stop_result.outcome(), TerminalOutcome::Completed);
    assert_eq!(stop_result.committed_positions(), 1);
    assert_eq!(stop_result.emitted_tokens(), 1);

    let before_idle = engine.snapshot();
    assert_eq!(engine.step().expect("idle step"), StepReport::default());
    assert_eq!(engine.snapshot(), before_idle);
}

#[test]
fn terminal_reaping_restores_accounting_and_shutdown_releases_mixed_state() {
    let mut engine = tiny_engine();
    let baseline = engine.snapshot();
    let id = engine
        .try_submit(RequestSpec::new(&[3], 1, SamplingPolicy::Greedy, None))
        .expect("accepted reap request");
    engine.step().expect("complete reap request");
    let retained = engine.snapshot();
    assert_eq!(retained.retained_terminal_results, 1);
    assert!(retained.ledger_used_bytes > baseline.ledger_used_bytes);

    let terminal = engine
        .take_terminal(id)
        .expect("take terminal before output")
        .expect("terminal retained independently of output");
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(engine.snapshot().retained_terminal_results, 0);
    assert_eq!(drain_tokens(&mut engine, id), [0]);
    assert_eq!(
        engine.snapshot().ledger_used_bytes,
        baseline.ledger_used_bytes
    );
    assert_eq!(
        engine
            .request_phase(id)
            .expect_err("fully consumed request is reaped")
            .category(),
        ErrorCategory::InvalidRequest
    );

    let mut limits = SchedulerLimits::tiny();
    limits.max_active_requests = 1;
    limits.batch_width = 1;
    limits.waves_per_step = 1;
    limits.output_capacity_per_request = 1;
    let mut mixed = new_engine_with_limits(limits);
    let completed = mixed
        .try_submit(RequestSpec::new(&[3], 1, SamplingPolicy::Greedy, None))
        .expect("accepted soon-terminal request");
    let blocked = mixed
        .try_submit(RequestSpec::new(&[0], 2, SamplingPolicy::Greedy, None))
        .expect("accepted soon-blocked request");
    let queued = mixed
        .try_submit(RequestSpec::new(&[1], 2, SamplingPolicy::Greedy, None))
        .expect("accepted queued request");
    mixed.step().expect("produce retained terminal");
    mixed.step().expect("produce blocked output");
    assert_eq!(phase(&mixed, completed), RequestPhase::Terminal);
    assert_eq!(phase(&mixed, blocked), RequestPhase::OutputBlocked);
    assert_eq!(phase(&mixed, queued), RequestPhase::Queued);

    let shutdown = mixed.shutdown().expect("shutdown mixed engine");
    assert_eq!(shutdown.terminated_requests, 2);
    assert_eq!(shutdown.discarded_output_events, 2);
    assert!(shutdown.released_request_bytes > 0);
    assert_eq!(shutdown.remaining_shared_bytes, 0);
    let closed = mixed.snapshot();
    assert!(closed.closed);
    assert_eq!(closed.ledger_used_bytes, 0);
    assert_eq!(closed.active_requests, 0);
    assert_eq!(closed.queued_requests, 0);

    for request_id in [completed, blocked, queued] {
        assert_eq!(
            mixed
                .request_phase(request_id)
                .expect_err("shutdown reaps every request")
                .category(),
            ErrorCategory::InvalidRequest
        );
    }
    assert_eq!(
        mixed
            .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None,))
            .expect_err("closed engine rejects submission")
            .category(),
        ErrorCategory::Cancelled
    );
    assert_eq!(
        mixed
            .step()
            .expect_err("closed engine rejects steps")
            .category(),
        ErrorCategory::Cancelled
    );
    assert_eq!(
        mixed
            .shutdown()
            .expect("idempotent shutdown")
            .terminated_requests,
        0
    );
}
