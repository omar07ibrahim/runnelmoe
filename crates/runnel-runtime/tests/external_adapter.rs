use std::sync::atomic::{AtomicU64, Ordering};

use runnel_runtime::{
    AdapterExecutionLayout, AdapterWorkIdentity, DecoderAdapter, Result, RuntimeError,
    StateLayoutAccounting,
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

    fn next(&self, exhausted: fn() -> RuntimeError) -> Result<u64> {
        let mut candidate = self.next.load(Ordering::Relaxed);
        loop {
            if candidate == 0 {
                return Err(exhausted());
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
struct MockLayout {
    max_tokens: usize,
}

impl StateLayoutAccounting for MockLayout {
    fn payload_bytes(self) -> usize {
        64
    }

    fn charge_bytes(self) -> usize {
        64
    }
}

struct MockState {
    state_id: u64,
    position: usize,
    revision: u64,
    max_tokens: usize,
}

struct MockPrepared(AdapterWorkIdentity);
struct MockTask(AdapterWorkIdentity);
struct MockContribution(AdapterWorkIdentity);
struct MockWorkspace;
struct MockPending {
    identity: AdapterWorkIdentity,
    logits: [f32; 3],
}

struct MockPermit<'a> {
    state: &'a mut MockState,
    next_position: usize,
    next_revision: u64,
}

struct ExternalMockAdapter {
    model_id: u64,
    next_transaction: CheckedCounter,
    next_state: CheckedCounter,
}

impl ExternalMockAdapter {
    fn new() -> Result<Self> {
        Ok(Self {
            model_id: NEXT_MODEL_ID.next(|| RuntimeError::ModelIdentityExhausted)?,
            next_transaction: CheckedCounter::new(1),
            next_state: CheckedCounter::new(1),
        })
    }

    fn next_transaction(&self) -> Result<u64> {
        self.next_transaction
            .next(|| RuntimeError::AdapterTransactionIdentityExhausted)
    }

    fn next_state(&self) -> Result<u64> {
        self.next_state
            .next(|| RuntimeError::StateIdentityExhausted)
    }
}

impl DecoderAdapter for ExternalMockAdapter {
    type StateLayout = MockLayout;
    type State = MockState;
    type Workspace = MockWorkspace;
    type PreparedToken = MockPrepared;
    type ExpertTask = MockTask;
    type ExpertTasks = std::array::IntoIter<MockTask, 1>;
    type ExpertContribution = MockContribution;
    type PendingStateCommit = MockPending;
    type StateCommitPermit<'a> = MockPermit<'a>;

    fn execution_layout(&self) -> Result<AdapterExecutionLayout> {
        AdapterExecutionLayout::new(1, 8, 8, 8, 24, 0, 0)
    }

    fn vocabulary_size(&self) -> usize {
        3
    }

    fn is_stop_token(&self, token: u32) -> bool {
        token == 2
    }

    fn state_layout(&self, max_tokens: usize, _page_tokens: usize) -> Result<MockLayout> {
        if max_tokens == 0 {
            return Err(RuntimeError::InvalidStateLayout(
                "mock maximum token count must be nonzero",
            ));
        }
        Ok(MockLayout { max_tokens })
    }

    fn new_state(&self, layout: MockLayout) -> Result<MockState> {
        Ok(MockState {
            state_id: self.next_state()?,
            position: 0,
            revision: 0,
            max_tokens: layout.max_tokens,
        })
    }

    fn new_workspace(&self) -> Result<MockWorkspace> {
        Ok(MockWorkspace)
    }

    fn prepare_token(
        &self,
        state: &MockState,
        token: u32,
        _workspace: &mut MockWorkspace,
    ) -> Result<MockPrepared> {
        if token >= self.vocabulary_size() as u32 {
            return Err(RuntimeError::InvalidToken {
                vocab_size: self.vocabulary_size(),
            });
        }
        if state.position >= state.max_tokens {
            return Err(RuntimeError::ContextLimit {
                limit: state.max_tokens,
            });
        }
        Ok(MockPrepared(AdapterWorkIdentity::try_new(
            self.next_transaction()?,
            self.model_id,
            state.state_id,
            state.revision,
            state.position,
        )?))
    }

    fn prepared_identity(&self, prepared: &MockPrepared) -> AdapterWorkIdentity {
        prepared.0
    }

    fn expert_tasks(&self, prepared: &MockPrepared) -> Self::ExpertTasks {
        [MockTask(prepared.0)].into_iter()
    }

    fn task_identity(&self, task: &MockTask) -> AdapterWorkIdentity {
        task.0
    }

    fn task_router_rank(&self, _task: &MockTask) -> u16 {
        0
    }

    fn task_expert_id(&self, _task: &MockTask) -> u16 {
        0
    }

    fn execute_expert(
        &self,
        task: MockTask,
        _workspace: &mut MockWorkspace,
    ) -> Result<MockContribution> {
        Ok(MockContribution(task.0))
    }

    fn contribution_identity(&self, contribution: &MockContribution) -> AdapterWorkIdentity {
        contribution.0
    }

    fn contribution_router_rank(&self, _contribution: &MockContribution) -> u16 {
        0
    }

    fn contribution_expert_id(&self, _contribution: &MockContribution) -> u16 {
        0
    }

    fn finish_token(
        &self,
        prepared: MockPrepared,
        contributions: &[MockContribution],
        _workspace: &mut MockWorkspace,
    ) -> Result<MockPending> {
        if contributions.len() != 1 || contributions[0].0 != prepared.0 {
            return Err(RuntimeError::InvalidExpertContribution(
                "mock contribution identity mismatch",
            ));
        }
        Ok(MockPending {
            identity: prepared.0,
            logits: [0.0, 1.0, 2.0],
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
    ) -> Result<R>
    where
        F: for<'permit> FnOnce(MockPermit<'permit>) -> R,
    {
        if pending.identity.model_instance_id() != self.model_id
            || pending.identity.state_id().get() != state.state_id
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
            .ok_or(RuntimeError::ContextLimit {
                limit: state.max_tokens,
            })?;
        if next_position > state.max_tokens {
            return Err(RuntimeError::ContextLimit {
                limit: state.max_tokens,
            });
        }
        let next_revision = state
            .revision
            .checked_add(1)
            .ok_or(RuntimeError::StateRevisionExhausted)?;
        Ok(apply(MockPermit {
            state,
            next_position,
            next_revision,
        }))
    }

    fn apply_state_commit(permit: MockPermit<'_>) {
        permit.state.position = permit.next_position;
        permit.state.revision = permit.next_revision;
    }
}

#[test]
fn external_adapter_can_implement_the_complete_transaction_contract() {
    let adapter = ExternalMockAdapter::new().unwrap();
    assert_eq!(adapter.vocabulary_size(), 3);
    assert!(adapter.is_stop_token(2));
    assert!(!adapter.is_stop_token(1));

    let layout = adapter.state_layout(2, 1).unwrap();
    let mut state = adapter.new_state(layout).unwrap();
    let other_state = adapter.new_state(layout).unwrap();
    assert_ne!(state.state_id, other_state.state_id);
    assert_ne!(state.state_id, 0);
    let mut workspace = adapter.new_workspace().unwrap();
    let prepared = adapter.prepare_token(&state, 1, &mut workspace).unwrap();
    let prepared_identity = adapter.prepared_identity(&prepared);
    let mut tasks = adapter.expert_tasks(&prepared);
    assert_eq!(tasks.len(), 1);
    let task = tasks.next().unwrap();
    assert_eq!(adapter.task_identity(&task), prepared_identity);
    assert_eq!(adapter.task_router_rank(&task), 0);
    let contribution = adapter.execute_expert(task, &mut workspace).unwrap();
    let pending = adapter
        .finish_token(prepared, &[contribution], &mut workspace)
        .unwrap();
    assert_eq!(adapter.pending_identity(&pending), prepared_identity);
    assert_eq!(adapter.pending_logits(&pending), [0.0, 1.0, 2.0]);

    adapter
        .with_validated_state_commit(
            &mut state,
            &pending,
            ExternalMockAdapter::apply_state_commit,
        )
        .unwrap();
    assert_eq!(state.position, 1);
    assert_eq!(state.revision, 1);
}
