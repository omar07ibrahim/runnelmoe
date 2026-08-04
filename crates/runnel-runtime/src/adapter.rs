//! Model-agnostic transactional decoder phases.
//!
//! The scheduler owns request policy and control state. A decoder adapter owns
//! only model work and a fallible-state-commit permit. Every artifact produced
//! before `apply_state_commit` is inert when dropped.

use std::num::NonZeroU64;

use crate::{Result, RuntimeError, StateId, StateLayout};

const LEDGER_ALIGNMENT: usize = 64;

/// Exact semantic accounting required from an adapter-owned state layout.
pub trait StateLayoutAccounting: Copy {
    fn payload_bytes(self) -> usize;
    fn charge_bytes(self) -> usize;
}

impl StateLayoutAccounting for StateLayout {
    fn payload_bytes(self) -> usize {
        StateLayout::payload_bytes(self)
    }

    fn charge_bytes(self) -> usize {
        StateLayout::charge_bytes(self)
    }
}

/// A checked, nonzero identity for one adapter token transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AdapterTransactionId(pub(crate) NonZeroU64);

impl AdapterTransactionId {
    /// Constructs an adapter-owned transaction identity.
    ///
    /// External adapters are responsible for issuing values from one strictly
    /// increasing lifetime counter and for permanently exhausting it instead
    /// of wrapping or reusing an earlier value.
    pub fn try_new(value: u64) -> Result<Self> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(RuntimeError::InvalidAdapterWork(
                "adapter transaction identity must be nonzero",
            ))
    }

    #[must_use]
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

/// Complete runtime identity carried by prepared work and every completion.
///
/// This is a validation tag, not an unforgeable capability. Private fields
/// prevent in-place mutation of the tag; conforming adapters must also keep
/// the identity binding inside task and contribution payloads opaque. A
/// scheduler validates the complete tag at every phase boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AdapterWorkIdentity {
    pub(crate) transaction_id: AdapterTransactionId,
    pub(crate) model_instance_id: u64,
    pub(crate) state_id: StateId,
    pub(crate) state_revision: u64,
    pub(crate) position: usize,
}

impl AdapterWorkIdentity {
    /// Constructs the immutable identity carried through one adapter token
    /// transaction.
    ///
    /// Adapter implementations must keep model-instance and state identities
    /// unique within their own lifetime and issue transaction identities in
    /// strictly increasing order. Constructing a tag does not authorize work
    /// or provide access to an adapter-owned task or contribution payload.
    pub fn try_new(
        transaction_id: u64,
        model_instance_id: u64,
        state_id: u64,
        state_revision: u64,
        position: usize,
    ) -> Result<Self> {
        let transaction_id = AdapterTransactionId::try_new(transaction_id)?;
        if model_instance_id == 0 {
            return Err(RuntimeError::InvalidAdapterWork(
                "adapter model identity must be nonzero",
            ));
        }
        let state_id = NonZeroU64::new(state_id).map(StateId::from_nonzero).ok_or(
            RuntimeError::InvalidAdapterWork("adapter state identity must be nonzero"),
        )?;
        Ok(Self {
            transaction_id,
            model_instance_id,
            state_id,
            state_revision,
            position,
        })
    }

    #[must_use]
    pub fn transaction_id(self) -> AdapterTransactionId {
        self.transaction_id
    }

    #[must_use]
    pub fn model_instance_id(self) -> u64 {
        self.model_instance_id
    }

    #[must_use]
    pub fn state_id(self) -> StateId {
        self.state_id
    }

    #[must_use]
    pub fn state_revision(self) -> u64 {
        self.state_revision
    }

    #[must_use]
    pub fn position(self) -> usize {
        self.position
    }
}

/// Adapter-owned payload capacities used by scheduler admission accounting.
///
/// These are semantic payload bytes, not allocator or RSS estimates. The
/// scheduler separately accounts for its envelopes and other fixed metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdapterExecutionLayout {
    max_tasks_per_token: usize,
    prepared_payload_bytes: usize,
    task_payload_bytes: usize,
    contribution_payload_bytes: usize,
    pending_payload_bytes: usize,
    workspace_payload_bytes: usize,
    workspace_charge_bytes: usize,
    model_resident_payload_bytes: usize,
    model_resident_charge_bytes: usize,
}

impl AdapterExecutionLayout {
    /// Validates payload sizes and rounds the complete workspace once to the
    /// scheduler ledger's 64-byte unit.
    pub fn new(
        max_tasks_per_token: usize,
        prepared_payload_bytes: usize,
        task_payload_bytes: usize,
        contribution_payload_bytes: usize,
        pending_payload_bytes: usize,
        workspace_payload_bytes: usize,
        model_resident_payload_bytes: usize,
    ) -> Result<Self> {
        if max_tasks_per_token == 0 || max_tasks_per_token > usize::from(u16::MAX) + 1 {
            return Err(RuntimeError::InvalidConfig(
                "adapter task count must fit the u16 router-rank identity".into(),
            ));
        }
        let workspace_charge_bytes = round_up_ledger(workspace_payload_bytes, "workspace")?;
        let model_resident_charge_bytes =
            round_up_ledger(model_resident_payload_bytes, "model-resident partition")?;
        Ok(Self {
            max_tasks_per_token,
            prepared_payload_bytes,
            task_payload_bytes,
            contribution_payload_bytes,
            pending_payload_bytes,
            workspace_payload_bytes,
            workspace_charge_bytes,
            model_resident_payload_bytes,
            model_resident_charge_bytes,
        })
    }

    #[must_use]
    pub fn max_tasks_per_token(self) -> usize {
        self.max_tasks_per_token
    }

    #[must_use]
    pub fn prepared_payload_bytes(self) -> usize {
        self.prepared_payload_bytes
    }

    #[must_use]
    pub fn task_payload_bytes(self) -> usize {
        self.task_payload_bytes
    }

    #[must_use]
    pub fn contribution_payload_bytes(self) -> usize {
        self.contribution_payload_bytes
    }

    #[must_use]
    pub fn pending_payload_bytes(self) -> usize {
        self.pending_payload_bytes
    }

    #[must_use]
    pub fn workspace_payload_bytes(self) -> usize {
        self.workspace_payload_bytes
    }

    #[must_use]
    pub fn workspace_charge_bytes(self) -> usize {
        self.workspace_charge_bytes
    }

    #[must_use]
    pub fn model_resident_payload_bytes(self) -> usize {
        self.model_resident_payload_bytes
    }

    #[must_use]
    pub fn model_resident_charge_bytes(self) -> usize {
        self.model_resident_charge_bytes
    }
}

fn round_up_ledger(bytes: usize, resource: &'static str) -> Result<usize> {
    bytes
        .checked_add(LEDGER_ALIGNMENT - 1)
        .map(|bytes| bytes / LEDGER_ALIGNMENT * LEDGER_ALIGNMENT)
        .ok_or_else(|| {
            RuntimeError::InvalidConfig(format!("adapter {resource} charge overflows usize"))
        })
}

/// Synchronous decoder phases consumed by a generic scheduler engine.
///
/// Implementations must lifetime-bind the commit permit to the exact mutable
/// state and immutable pending value validated by
/// `with_validated_state_commit`. For a conforming implementation, its
/// higher-ranked callback prevents safe scheduler code from returning the
/// permit in a future or otherwise retaining it across an outer asynchronous
/// yield.
///
/// This is a trusted model-extension boundary. Implementations must honor the
/// documented task ordering, identity, allocation, and commit contracts; a
/// scheduler still validates all observable envelopes before publication.
pub trait DecoderAdapter: Send + Sync {
    type StateLayout: StateLayoutAccounting + Send + Sync;
    type State: Send;
    type Workspace: Send;
    type PreparedToken: Send;
    type ExpertTask: Send;
    type ExpertTasks: ExactSizeIterator<Item = Self::ExpertTask> + Send;
    type ExpertContribution: Send;
    type PendingStateCommit: Send;
    type StateCommitPermit<'a>
    where
        Self: 'a;

    fn execution_layout(&self) -> Result<AdapterExecutionLayout>;

    /// Returns the fixed, nonzero vocabulary size accepted by this adapter.
    /// The largest valid token ID must fit in `u32`. The value must remain
    /// stable for the adapter's lifetime.
    fn vocabulary_size(&self) -> usize;

    /// Reports whether an already vocabulary-validated token ends generation.
    /// The answer must be deterministic and stable for the adapter's lifetime.
    fn is_stop_token(&self, token: u32) -> bool;

    fn state_layout(&self, max_tokens: usize, page_tokens: usize) -> Result<Self::StateLayout>;

    fn new_state(&self, layout: Self::StateLayout) -> Result<Self::State>;

    fn new_workspace(&self) -> Result<Self::Workspace>;

    fn prepare_token(
        &self,
        state: &Self::State,
        token: u32,
        workspace: &mut Self::Workspace,
    ) -> Result<Self::PreparedToken>;

    fn prepared_identity(&self, prepared: &Self::PreparedToken) -> AdapterWorkIdentity;

    /// Returns one or more tasks in contiguous router-rank order.
    ///
    /// The exact iterator length must be in
    /// `1..=execution_layout().max_tasks_per_token()`. Its ranks must be
    /// unique and exactly `0..len`, in iteration order, and every task identity
    /// must equal `prepared_identity(prepared)`. A scheduler validates this
    /// contract before indexing its preallocated scatter lanes.
    fn expert_tasks(&self, prepared: &Self::PreparedToken) -> Self::ExpertTasks;

    fn task_identity(&self, task: &Self::ExpertTask) -> AdapterWorkIdentity;

    fn task_router_rank(&self, task: &Self::ExpertTask) -> u16;

    fn task_expert_id(&self, task: &Self::ExpertTask) -> u16;

    fn execute_expert(
        &self,
        task: Self::ExpertTask,
        workspace: &mut Self::Workspace,
    ) -> Result<Self::ExpertContribution>;

    /// Returns the complete identity copied from the input expert task.
    fn contribution_identity(&self, contribution: &Self::ExpertContribution)
    -> AdapterWorkIdentity;

    /// Returns the router rank copied from the input expert task.
    fn contribution_router_rank(&self, contribution: &Self::ExpertContribution) -> u16;

    /// Returns the expert ID copied from the input expert task.
    fn contribution_expert_id(&self, contribution: &Self::ExpertContribution) -> u16;

    fn finish_token(
        &self,
        prepared: Self::PreparedToken,
        contributions: &[Self::ExpertContribution],
        workspace: &mut Self::Workspace,
    ) -> Result<Self::PendingStateCommit>;

    fn pending_identity(&self, pending: &Self::PendingStateCommit) -> AdapterWorkIdentity;

    /// Returns finite logits whose length is exactly `vocabulary_size()`.
    /// `pending_identity(pending)` must equal the prepared identity used to
    /// create this pending value.
    fn pending_logits<'a>(&self, pending: &'a Self::PendingStateCommit) -> &'a [f32];

    /// Validates state and invokes `apply` with a non-escaping, single-use
    /// permit for the exact pending value.
    ///
    /// The callback result is independent of the fresh permit lifetime. Safe
    /// callers can therefore apply or drop the permit synchronously, but
    /// cannot return it inside a future or store it beyond this call.
    /// Callers must complete every fallible scheduler check and reservation
    /// before this callback. Once it applies the permit, only infallible,
    /// non-panicking publication of already validated control state may follow.
    fn with_validated_state_commit<R, F>(
        &self,
        state: &mut Self::State,
        pending: &Self::PendingStateCommit,
        apply: F,
    ) -> Result<R>
    where
        F: for<'permit> FnOnce(Self::StateCommitPermit<'permit>) -> R;

    /// Applies a previously validated state transition without allocation or
    /// another replaceable model, state, or pending-data argument.
    fn apply_state_commit<'a>(permit: Self::StateCommitPermit<'a>);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_layout_rounds_the_complete_workspace_once() {
        let layout = AdapterExecutionLayout::new(2, 152, 32, 32, 216, 192, 5_600).unwrap();
        assert_eq!(layout.max_tasks_per_token(), 2);
        assert_eq!(layout.prepared_payload_bytes(), 152);
        assert_eq!(layout.task_payload_bytes(), 32);
        assert_eq!(layout.contribution_payload_bytes(), 32);
        assert_eq!(layout.pending_payload_bytes(), 216);
        assert_eq!(layout.workspace_payload_bytes(), 192);
        assert_eq!(layout.workspace_charge_bytes(), 192);
        assert_eq!(layout.model_resident_payload_bytes(), 5_600);
        assert_eq!(layout.model_resident_charge_bytes(), 5_632);

        let rounded = AdapterExecutionLayout::new(1, 0, 0, 0, 0, 193, 65).unwrap();
        assert_eq!(rounded.workspace_charge_bytes(), 256);
        assert_eq!(rounded.model_resident_charge_bytes(), 128);
    }

    #[test]
    fn execution_layout_rejects_zero_tasks_and_charge_overflow() {
        assert!(AdapterExecutionLayout::new(0, 0, 0, 0, 0, 0, 0).is_err());
        assert!(AdapterExecutionLayout::new(usize::from(u16::MAX) + 2, 0, 0, 0, 0, 0, 0).is_err());
        assert!(AdapterExecutionLayout::new(1, 0, 0, 0, 0, usize::MAX, 0).is_err());
        assert!(AdapterExecutionLayout::new(1, 0, 0, 0, 0, 0, usize::MAX).is_err());
    }

    #[test]
    fn public_adapter_identities_are_checked_and_preserve_fields() {
        let identity = AdapterWorkIdentity::try_new(11, 22, 33, 44, 55).unwrap();
        assert_eq!(identity.transaction_id().get(), 11);
        assert_eq!(identity.model_instance_id(), 22);
        assert_eq!(identity.state_id().get(), 33);
        assert_eq!(identity.state_revision(), 44);
        assert_eq!(identity.position(), 55);

        assert_eq!(
            AdapterTransactionId::try_new(0).unwrap_err(),
            RuntimeError::InvalidAdapterWork("adapter transaction identity must be nonzero")
        );
        assert_eq!(
            AdapterWorkIdentity::try_new(1, 0, 1, 0, 0).unwrap_err(),
            RuntimeError::InvalidAdapterWork("adapter model identity must be nonzero")
        );
        assert_eq!(
            AdapterWorkIdentity::try_new(1, 1, 0, 0, 0).unwrap_err(),
            RuntimeError::InvalidAdapterWork("adapter state identity must be nonzero")
        );
    }
}
