use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};

use runnel_runtime::{
    AdapterExecutionLayout, AdapterWorkIdentity, DecoderAdapter, Result as RuntimeResult,
    RuntimeError, SamplingPolicy, StateLayoutAccounting,
};
use runnel_scheduler::{
    CancelDisposition, ErrorCategory, RequestPhase, RequestSpec, SchedulerConfig, SchedulerEngine,
    SchedulerLimits, StepReport, TerminalOutcome,
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
    let adapter = MockAdapter::new();
    let config = SchedulerConfig::new(&adapter, limits).expect("mock scheduler config");
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
