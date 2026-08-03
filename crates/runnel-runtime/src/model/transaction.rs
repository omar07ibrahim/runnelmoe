//! Fixed-capacity transactional execution for the frozen tiny adapter.

use std::{
    fmt,
    mem::size_of,
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

use runnel_kernels::{
    BackendKind, BackendRequest, FiniteInput, GemvWorkspace, KernelError, PreparedGemv,
};

use super::{Experts, TinyModel};
use crate::{
    AdapterExecutionLayout, AdapterTransactionId, AdapterWorkIdentity, DecoderAdapter, EOS_TOKEN,
    Result, RuntimeError, StateLayout, Tensor,
    attention::streaming_causal_attention,
    state::{SequenceState, StateAppendPermit},
};

const HIDDEN_SIZE: usize = 8;
const EXPERT_HIDDEN_SIZE: usize = 12;
const NUM_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const VOCAB_SIZE: usize = 32;
const RMS_EPSILON: f32 = 1.0 / 4096.0;

const PREPARED_F32_ELEMENTS: usize = HIDDEN_SIZE * 4 + NUM_EXPERTS + TOP_K;
const TASK_F32_ELEMENTS: usize = HIDDEN_SIZE;
const CONTRIBUTION_F32_ELEMENTS: usize = HIDDEN_SIZE;
const PENDING_F32_ELEMENTS: usize = HIDDEN_SIZE * 2 + VOCAB_SIZE + NUM_EXPERTS + TOP_K;
const WORKSPACE_F32_ELEMENTS: usize = EXPERT_HIDDEN_SIZE * 4;

static NEXT_ADAPTER_TRANSACTION_ID: CheckedAdapterTransactionCounter =
    CheckedAdapterTransactionCounter::new(1);

/// Fixed scratch for one tiny-adapter worker.
///
/// The three lanes hold gate, up, and activated expert intermediates. The
/// fourth 12-element lane lives inside `GemvWorkspace`, for a total semantic
/// payload of 192 bytes. Construction is the only fallible allocation point;
/// phase methods never grow any buffer.
pub struct TinyWorkspace {
    lanes: [[f32; EXPERT_HIDDEN_SIZE]; 3],
    gemv: GemvWorkspace,
}

impl fmt::Debug for TinyWorkspace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TinyWorkspace")
            .field("lane_count", &self.lanes.len())
            .field("lane_elements", &EXPERT_HIDDEN_SIZE)
            .field("gemv_max_rows", &self.gemv.max_rows())
            .field("payload", &Redacted)
            .finish()
    }
}

/// Immutable output of the pre-expert phase.
pub struct TinyPreparedToken {
    identity: AdapterWorkIdentity,
    input_token: u32,
    attention_residual: [f32; HIDDEN_SIZE],
    expert_input: [f32; HIDDEN_SIZE],
    key: [f32; HIDDEN_SIZE],
    value: [f32; HIDDEN_SIZE],
    router_scores: [f32; NUM_EXPERTS],
    expert_ids: [u16; TOP_K],
    route_weights: [f32; TOP_K],
}

impl fmt::Debug for TinyPreparedToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TinyPreparedToken")
            .field("identity", &self.identity)
            .field("input_token", &Redacted)
            .field("model_payload", &Redacted)
            .finish()
    }
}

/// One owned, scheduler-sortable expert invocation.
pub struct TinyExpertTask {
    identity: AdapterWorkIdentity,
    router_rank: u16,
    expert_id: u16,
    input_len: u16,
    input: [f32; HIDDEN_SIZE],
}

impl fmt::Debug for TinyExpertTask {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TinyExpertTask")
            .field("identity", &self.identity)
            .field("router_rank", &self.router_rank)
            .field("expert_id", &self.expert_id)
            .field("input_len", &self.input_len)
            .field("input", &Redacted)
            .finish()
    }
}

/// One owned expert result. Route weights are deliberately not worker-owned.
pub struct TinyExpertContribution {
    identity: AdapterWorkIdentity,
    router_rank: u16,
    expert_id: u16,
    output_len: u16,
    output: [f32; HIDDEN_SIZE],
}

impl fmt::Debug for TinyExpertContribution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TinyExpertContribution")
            .field("identity", &self.identity)
            .field("router_rank", &self.router_rank)
            .field("expert_id", &self.expert_id)
            .field("output_len", &self.output_len)
            .field("output", &Redacted)
            .finish()
    }
}

/// Immutable, fully finished model output awaiting state validation.
pub struct TinyPendingStateCommit {
    identity: AdapterWorkIdentity,
    input_token: u32,
    key: [f32; HIDDEN_SIZE],
    value: [f32; HIDDEN_SIZE],
    logits: [f32; VOCAB_SIZE],
    router_scores: [f32; NUM_EXPERTS],
    expert_ids: [u16; TOP_K],
    route_weights: [f32; TOP_K],
}

impl TinyPendingStateCommit {
    pub(super) fn input_token(&self) -> u32 {
        self.input_token
    }

    pub(super) fn logits_array(&self) -> &[f32; VOCAB_SIZE] {
        &self.logits
    }

    pub(super) fn router_scores(&self) -> &[f32; NUM_EXPERTS] {
        &self.router_scores
    }

    pub(super) fn expert_ids(&self) -> &[u16; TOP_K] {
        &self.expert_ids
    }

    pub(super) fn route_weights(&self) -> &[f32; TOP_K] {
        &self.route_weights
    }
}

impl fmt::Debug for TinyPendingStateCommit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TinyPendingStateCommit")
            .field("identity", &self.identity)
            .field("input_token", &Redacted)
            .field("model_payload", &Redacted)
            .finish()
    }
}

/// Single-use capability for the exact pending K/V pair validated against a
/// mutable state. Apply accepts no substitute data and has no error path.
pub struct TinyStateCommitPermit<'a> {
    append: StateAppendPermit<'a, 'a>,
}

impl fmt::Debug for TinyStateCommitPermit<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TinyStateCommitPermit")
            .field("validated_payload", &Redacted)
            .finish_non_exhaustive()
    }
}

struct Redacted;

impl fmt::Debug for Redacted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted>")
    }
}

impl DecoderAdapter for TinyModel {
    type StateLayout = StateLayout;
    type State = SequenceState;
    type Workspace = TinyWorkspace;
    type PreparedToken = TinyPreparedToken;
    type ExpertTask = TinyExpertTask;
    type ExpertTasks = std::array::IntoIter<TinyExpertTask, TOP_K>;
    type ExpertContribution = TinyExpertContribution;
    type PendingStateCommit = TinyPendingStateCommit;
    type StateCommitPermit<'a>
        = TinyStateCommitPermit<'a>
    where
        Self: 'a;

    fn execution_layout(&self) -> Result<AdapterExecutionLayout> {
        validate_frozen_geometry(self)?;
        AdapterExecutionLayout::new(
            TOP_K,
            PREPARED_F32_ELEMENTS * size_of::<f32>(),
            TASK_F32_ELEMENTS * size_of::<f32>(),
            CONTRIBUTION_F32_ELEMENTS * size_of::<f32>(),
            PENDING_F32_ELEMENTS * size_of::<f32>(),
            WORKSPACE_F32_ELEMENTS * size_of::<f32>(),
            model_resident_payload_bytes(self)?,
        )
    }

    fn vocabulary_size(&self) -> usize {
        VOCAB_SIZE
    }

    fn is_stop_token(&self, token: u32) -> bool {
        token == EOS_TOKEN
    }

    fn state_layout(&self, max_tokens: usize, page_tokens: usize) -> Result<StateLayout> {
        TinyModel::state_layout(self, max_tokens, page_tokens)
    }

    fn new_state(&self, layout: StateLayout) -> Result<SequenceState> {
        self.new_sequence_state(layout)
    }

    fn new_workspace(&self) -> Result<TinyWorkspace> {
        validate_frozen_geometry(self)?;
        let gemv = GemvWorkspace::try_new(EXPERT_HIDDEN_SIZE).map_err(|source| match source {
            KernelError::AllocationFailure {
                requested_bytes, ..
            } => RuntimeError::ResourceExhausted {
                resource: "tiny adapter expert workspace",
                bytes: requested_bytes,
            },
            source => RuntimeError::ExpertKernel {
                operation: "expert workspace allocation",
                source,
            },
        })?;
        Ok(TinyWorkspace {
            lanes: [[0.0; EXPERT_HIDDEN_SIZE]; 3],
            gemv,
        })
    }

    fn prepare_token(
        &self,
        state: &SequenceState,
        token: u32,
        _workspace: &mut TinyWorkspace,
    ) -> Result<TinyPreparedToken> {
        validate_frozen_geometry(self)?;
        let token_index = validate_token(token)?;
        let (state_id, state_revision, position) = validate_state_for_prepare(self, state)?;
        let identity = AdapterWorkIdentity::try_new(
            NEXT_ADAPTER_TRANSACTION_ID.next()?.get(),
            self.instance_id,
            state_id.get(),
            state_revision,
            position,
        )?;

        let mut attention_residual = [0.0; HIDDEN_SIZE];
        attention_residual.copy_from_slice(tensor_row::<HIDDEN_SIZE>(
            &self.weights.token_embedding,
            token_index,
            VOCAB_SIZE,
            "token embedding",
        )?);

        let mut normalized = [0.0; HIDDEN_SIZE];
        rms_norm_into(
            &attention_residual,
            &self.weights.attn_norm,
            &mut normalized,
            "attention RMS normalization",
        )?;
        let mut query = [0.0; HIDDEN_SIZE];
        let mut key = [0.0; HIDDEN_SIZE];
        let mut value = [0.0; HIDDEN_SIZE];
        linear_into(
            &self.weights.attn_q,
            &normalized,
            &mut query,
            "attention query",
        )?;
        linear_into(&self.weights.attn_k, &normalized, &mut key, "attention key")?;
        linear_into(
            &self.weights.attn_v,
            &normalized,
            &mut value,
            "attention value",
        )?;

        let mut attended = [0.0; HIDDEN_SIZE];
        streaming_causal_attention(
            &query,
            state.history(),
            &key,
            &value,
            self.config.num_heads,
            &mut attended,
        )?;
        let mut attention_output = [0.0; HIDDEN_SIZE];
        linear_into(
            &self.weights.attn_out,
            &attended,
            &mut attention_output,
            "attention output",
        )?;
        add_in_place_checked(
            &mut attention_residual,
            &attention_output,
            "attention residual",
        )?;

        let mut expert_input = [0.0; HIDDEN_SIZE];
        rms_norm_into(
            &attention_residual,
            &self.weights.ffn_norm,
            &mut expert_input,
            "expert RMS normalization",
        )?;
        let mut router_scores = [0.0; NUM_EXPERTS];
        linear_into(
            &self.weights.router,
            &expert_input,
            &mut router_scores,
            "router",
        )?;
        let expert_indices = stable_top_two(&router_scores);
        let expert_ids = [
            u16::try_from(expert_indices[0])
                .map_err(|_| RuntimeError::InvalidAdapterWork("expert ID exceeds u16"))?,
            u16::try_from(expert_indices[1])
                .map_err(|_| RuntimeError::InvalidAdapterWork("expert ID exceeds u16"))?,
        ];
        let selected_scores = [
            *router_scores
                .get(expert_indices[0])
                .ok_or(RuntimeError::InvalidAdapterWork(
                    "router selection is out of range",
                ))?,
            *router_scores
                .get(expert_indices[1])
                .ok_or(RuntimeError::InvalidAdapterWork(
                    "router selection is out of range",
                ))?,
        ];
        let route_weights = softmax_two(selected_scores)?;

        Ok(TinyPreparedToken {
            identity,
            input_token: token,
            attention_residual,
            expert_input,
            key,
            value,
            router_scores,
            expert_ids,
            route_weights,
        })
    }

    fn prepared_identity(&self, prepared: &TinyPreparedToken) -> AdapterWorkIdentity {
        prepared.identity
    }

    fn expert_tasks(&self, prepared: &TinyPreparedToken) -> Self::ExpertTasks {
        let [first_expert, second_expert] = prepared.expert_ids;
        [
            TinyExpertTask {
                identity: prepared.identity,
                router_rank: 0,
                expert_id: first_expert,
                input_len: HIDDEN_SIZE as u16,
                input: prepared.expert_input,
            },
            TinyExpertTask {
                identity: prepared.identity,
                router_rank: 1,
                expert_id: second_expert,
                input_len: HIDDEN_SIZE as u16,
                input: prepared.expert_input,
            },
        ]
        .into_iter()
    }

    fn task_identity(&self, task: &TinyExpertTask) -> AdapterWorkIdentity {
        task.identity
    }

    fn task_router_rank(&self, task: &TinyExpertTask) -> u16 {
        task.router_rank
    }

    fn task_expert_id(&self, task: &TinyExpertTask) -> u16 {
        task.expert_id
    }

    fn execute_expert(
        &self,
        task: TinyExpertTask,
        workspace: &mut TinyWorkspace,
    ) -> Result<TinyExpertContribution> {
        validate_frozen_geometry(self)?;
        validate_task(self, &task)?;
        let expert_index = usize::from(task.expert_id);
        let mut output = [0.0; HIDDEN_SIZE];
        let TinyWorkspace { lanes, gemv } = workspace;
        let [gate, up, activated] = lanes;

        match &self.weights.experts {
            Experts::F32(experts) => {
                let expert = experts
                    .get(expert_index)
                    .ok_or(RuntimeError::InvalidAdapterWork(
                        "expert task refers to an unavailable f32 expert",
                    ))?;
                linear_into(&expert.gate, &task.input, gate, "f32 expert gate")?;
                linear_into(&expert.up, &task.input, up, "f32 expert up")?;
                activate_into(gate, up, activated)?;
                linear_into(&expert.down, activated, &mut output, "f32 expert down")?;
            }
            Experts::Bf16 {
                weights: experts,
                dispatch,
            } => {
                let expert = experts
                    .get(expert_index)
                    .ok_or(RuntimeError::InvalidAdapterWork(
                        "expert task refers to an unavailable BF16 expert",
                    ))?;
                let input = FiniteInput::new(&task.input)
                    .map_err(|source| expert_kernel_error("expert input validation", source))?;
                run_bf16_into(
                    &expert.gate,
                    input,
                    *dispatch,
                    gemv,
                    gate,
                    "expert gate GEMV",
                )?;
                run_bf16_into(&expert.up, input, *dispatch, gemv, up, "expert up GEMV")?;
                activate_into(gate, up, activated)?;
                let activated_input = FiniteInput::new(activated)
                    .map_err(|source| expert_kernel_error("expert activation", source))?;
                run_bf16_into(
                    &expert.down,
                    activated_input,
                    *dispatch,
                    gemv,
                    &mut output,
                    "expert down GEMV",
                )?;
            }
        }
        ensure_finite(&output, "expert output")?;

        Ok(TinyExpertContribution {
            identity: task.identity,
            router_rank: task.router_rank,
            expert_id: task.expert_id,
            output_len: HIDDEN_SIZE as u16,
            output,
        })
    }

    fn contribution_identity(&self, contribution: &TinyExpertContribution) -> AdapterWorkIdentity {
        contribution.identity
    }

    fn contribution_router_rank(&self, contribution: &TinyExpertContribution) -> u16 {
        contribution.router_rank
    }

    fn contribution_expert_id(&self, contribution: &TinyExpertContribution) -> u16 {
        contribution.expert_id
    }

    fn finish_token(
        &self,
        prepared: TinyPreparedToken,
        contributions: &[TinyExpertContribution],
        _workspace: &mut TinyWorkspace,
    ) -> Result<TinyPendingStateCommit> {
        validate_frozen_geometry(self)?;
        validate_prepared(self, &prepared)?;
        if contributions.len() != TOP_K {
            return Err(RuntimeError::InvalidExpertContribution(
                "contribution count does not match top-k",
            ));
        }

        // First pass validates every completion independently. Nothing is
        // scattered or numerically reduced until the entire set is known safe.
        for contribution in contributions {
            validate_contribution(self, &prepared, contribution)?;
        }

        // Second pass scatters by router rank and rejects duplicates/missing
        // ranks without depending on completion order.
        let mut by_rank: [Option<&TinyExpertContribution>; TOP_K] = [None, None];
        for contribution in contributions {
            let rank = usize::from(contribution.router_rank);
            let slot = by_rank
                .get_mut(rank)
                .ok_or(RuntimeError::InvalidExpertContribution(
                    "router rank is out of range",
                ))?;
            if slot.replace(contribution).is_some() {
                return Err(RuntimeError::InvalidExpertContribution(
                    "router rank appears more than once",
                ));
            }
        }
        if by_rank.iter().any(Option::is_none) {
            return Err(RuntimeError::InvalidExpertContribution(
                "required router rank is missing",
            ));
        }

        // Mixture accumulation is always router-rank order, irrespective of
        // task sort or completion order.
        let mut mixture = [0.0_f32; HIDDEN_SIZE];
        for rank in 0..TOP_K {
            let contribution = by_rank.get(rank).and_then(|entry| *entry).ok_or(
                RuntimeError::InvalidExpertContribution("required router rank is missing"),
            )?;
            let weight =
                *prepared
                    .route_weights
                    .get(rank)
                    .ok_or(RuntimeError::InvalidAdapterWork(
                        "route weight rank is unavailable",
                    ))?;
            for (destination, value) in mixture.iter_mut().zip(&contribution.output) {
                *destination += weight * value;
            }
        }
        ensure_finite(&mixture, "expert mixture")?;

        let mut residual = prepared.attention_residual;
        add_in_place_checked(&mut residual, &mixture, "expert residual")?;
        let mut normalized = [0.0; HIDDEN_SIZE];
        rms_norm_into(
            &residual,
            &self.weights.final_norm,
            &mut normalized,
            "final RMS normalization",
        )?;
        let mut logits = [0.0; VOCAB_SIZE];
        linear_into(&self.weights.lm_head, &normalized, &mut logits, "lm head")?;

        Ok(TinyPendingStateCommit {
            identity: prepared.identity,
            input_token: prepared.input_token,
            key: prepared.key,
            value: prepared.value,
            logits,
            router_scores: prepared.router_scores,
            expert_ids: prepared.expert_ids,
            route_weights: prepared.route_weights,
        })
    }

    fn pending_identity(&self, pending: &TinyPendingStateCommit) -> AdapterWorkIdentity {
        pending.identity
    }

    fn pending_logits<'a>(&self, pending: &'a TinyPendingStateCommit) -> &'a [f32] {
        &pending.logits
    }

    fn with_validated_state_commit<R, F>(
        &self,
        state: &mut SequenceState,
        pending: &TinyPendingStateCommit,
        apply: F,
    ) -> Result<R>
    where
        F: for<'permit> FnOnce(TinyStateCommitPermit<'permit>) -> R,
    {
        validate_frozen_geometry(self)?;
        validate_pending(self, pending)?;
        match state.model_instance_id() {
            Some(instance_id)
                if instance_id == self.instance_id
                    && instance_id == pending.identity.model_instance_id => {}
            Some(_) => return Err(RuntimeError::StateMismatch),
            None => {
                return Err(RuntimeError::InvalidState(
                    "bound sequence state is missing its model identity",
                ));
            }
        }
        let layout = state.layout().ok_or(RuntimeError::InvalidState(
            "sequence state must have a bound layout",
        ))?;
        self.validate_state_layout(layout)?;
        let append = state.validate_append(
            pending.identity.state_id,
            pending.identity.state_revision,
            pending.identity.position,
            &pending.key,
            &pending.value,
        )?;
        Ok(apply(TinyStateCommitPermit { append }))
    }

    fn apply_state_commit<'a>(permit: TinyStateCommitPermit<'a>) {
        permit.append.apply();
    }
}

fn validate_frozen_geometry(model: &TinyModel) -> Result<()> {
    let config = &model.config;
    if config.hidden_size != HIDDEN_SIZE
        || config.expert_hidden_size != EXPERT_HIDDEN_SIZE
        || config.num_experts != NUM_EXPERTS
        || config.top_k != TOP_K
        || config.vocab_size != VOCAB_SIZE
    {
        return Err(RuntimeError::InvalidConfig(
            "transactional tiny execution requires frozen adapter geometry".into(),
        ));
    }
    Ok(())
}

fn model_resident_payload_bytes(model: &TinyModel) -> Result<usize> {
    let common = [
        &model.weights.token_embedding,
        &model.weights.attn_norm,
        &model.weights.attn_q,
        &model.weights.attn_k,
        &model.weights.attn_v,
        &model.weights.attn_out,
        &model.weights.ffn_norm,
        &model.weights.router,
        &model.weights.final_norm,
        &model.weights.lm_head,
    ];
    let mut total = 0_usize;
    for tensor in common {
        total = add_payload_bytes(total, tensor.data().len(), size_of::<f32>())?;
    }
    match &model.weights.experts {
        Experts::F32(experts) => {
            for expert in experts {
                for tensor in [&expert.gate, &expert.up, &expert.down] {
                    total = add_payload_bytes(total, tensor.data().len(), size_of::<f32>())?;
                }
            }
        }
        Experts::Bf16 { weights, .. } => {
            for expert in weights {
                for matrix in [&expert.gate, &expert.up, &expert.down] {
                    total = add_payload_bytes(total, matrix.words().len(), size_of::<u16>())?;
                }
            }
        }
    }
    Ok(total)
}

fn add_payload_bytes(total: usize, elements: usize, element_bytes: usize) -> Result<usize> {
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or(RuntimeError::ResourceSizeOverflow {
            resource: "tiny model resident payload",
        })?;
    total
        .checked_add(bytes)
        .ok_or(RuntimeError::ResourceSizeOverflow {
            resource: "tiny model resident payload",
        })
}

fn validate_token(token: u32) -> Result<usize> {
    let token_index = usize::try_from(token).map_err(|_| RuntimeError::InvalidToken {
        vocab_size: VOCAB_SIZE,
    })?;
    if token_index >= VOCAB_SIZE {
        return Err(RuntimeError::InvalidToken {
            vocab_size: VOCAB_SIZE,
        });
    }
    Ok(token_index)
}

fn validate_state_for_prepare(
    model: &TinyModel,
    state: &SequenceState,
) -> Result<(crate::StateId, u64, usize)> {
    let layout = state.layout().ok_or(RuntimeError::InvalidState(
        "sequence state must have a bound layout",
    ))?;
    match state.model_instance_id() {
        Some(instance_id) if instance_id == model.instance_id => {}
        Some(_) => return Err(RuntimeError::StateMismatch),
        None => {
            return Err(RuntimeError::InvalidState(
                "bound sequence state is missing its model identity",
            ));
        }
    }
    model.validate_state_layout(layout)?;
    let state_id = state.state_id().ok_or(RuntimeError::InvalidState(
        "bound sequence state is missing its state identity",
    ))?;
    let position = state.len();
    if position >= layout.max_tokens() {
        return Err(RuntimeError::ContextLimit {
            limit: layout.max_tokens(),
        });
    }
    let revision = state.revision();
    revision
        .checked_add(1)
        .ok_or(RuntimeError::StateRevisionExhausted)?;
    Ok((state_id, revision, position))
}

fn validate_identity(model: &TinyModel, identity: AdapterWorkIdentity) -> Result<()> {
    if identity.model_instance_id == 0 || identity.model_instance_id != model.instance_id {
        return Err(RuntimeError::InvalidAdapterWork(
            "work belongs to a different model instance",
        ));
    }
    identity
        .state_revision
        .checked_add(1)
        .ok_or(RuntimeError::StateRevisionExhausted)?;
    Ok(())
}

fn validate_prepared(model: &TinyModel, prepared: &TinyPreparedToken) -> Result<()> {
    validate_identity(model, prepared.identity)?;
    validate_token(prepared.input_token)?;
    ensure_finite(&prepared.attention_residual, "prepared attention residual")?;
    ensure_finite(&prepared.expert_input, "prepared expert input")?;
    ensure_finite(&prepared.key, "prepared key")?;
    ensure_finite(&prepared.value, "prepared value")?;
    ensure_finite(&prepared.router_scores, "prepared router scores")?;
    ensure_route(&prepared.expert_ids, &prepared.route_weights)
}

fn validate_task(model: &TinyModel, task: &TinyExpertTask) -> Result<()> {
    validate_identity(model, task.identity)?;
    if usize::from(task.router_rank) >= TOP_K {
        return Err(RuntimeError::InvalidAdapterWork(
            "expert task router rank is out of range",
        ));
    }
    if usize::from(task.expert_id) >= NUM_EXPERTS {
        return Err(RuntimeError::InvalidAdapterWork(
            "expert task expert ID is out of range",
        ));
    }
    if usize::from(task.input_len) != HIDDEN_SIZE {
        return Err(RuntimeError::InvalidAdapterWork(
            "expert task input width is invalid",
        ));
    }
    ensure_finite(&task.input, "expert task input")
}

fn validate_contribution(
    model: &TinyModel,
    prepared: &TinyPreparedToken,
    contribution: &TinyExpertContribution,
) -> Result<()> {
    if contribution.identity.model_instance_id != model.instance_id
        || contribution.identity != prepared.identity
    {
        return Err(RuntimeError::InvalidExpertContribution(
            "contribution identity does not match the prepared token",
        ));
    }
    let rank = usize::from(contribution.router_rank);
    if rank >= TOP_K {
        return Err(RuntimeError::InvalidExpertContribution(
            "router rank is out of range",
        ));
    }
    let expected_expert =
        prepared
            .expert_ids
            .get(rank)
            .ok_or(RuntimeError::InvalidExpertContribution(
                "router rank is unavailable",
            ))?;
    if contribution.expert_id != *expected_expert {
        return Err(RuntimeError::InvalidExpertContribution(
            "expert ID does not match the prepared router rank",
        ));
    }
    if usize::from(contribution.output_len) != HIDDEN_SIZE {
        return Err(RuntimeError::InvalidExpertContribution(
            "expert output width is invalid",
        ));
    }
    if contribution.output.iter().any(|value| !value.is_finite()) {
        return Err(RuntimeError::InvalidExpertContribution(
            "expert output is nonfinite",
        ));
    }
    Ok(())
}

fn validate_pending(model: &TinyModel, pending: &TinyPendingStateCommit) -> Result<()> {
    validate_identity(model, pending.identity)?;
    validate_token(pending.input_token)?;
    ensure_finite(&pending.key, "pending key")?;
    ensure_finite(&pending.value, "pending value")?;
    ensure_finite(&pending.logits, "pending logits")?;
    ensure_finite(&pending.router_scores, "pending router scores")?;
    ensure_route(&pending.expert_ids, &pending.route_weights)
}

fn ensure_route(expert_ids: &[u16; TOP_K], route_weights: &[f32; TOP_K]) -> Result<()> {
    if expert_ids.iter().any(|id| usize::from(*id) >= NUM_EXPERTS) {
        return Err(RuntimeError::InvalidAdapterWork(
            "route expert ID is out of range",
        ));
    }
    if expert_ids[0] == expert_ids[1] {
        return Err(RuntimeError::InvalidAdapterWork(
            "route expert IDs are not unique",
        ));
    }
    let total_weight = route_weights.iter().copied().sum::<f32>();
    if route_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
        || !total_weight.is_finite()
        || total_weight <= 0.0
    {
        return Err(RuntimeError::InvalidAdapterWork(
            "route weights must be nonnegative, finite, and have a positive sum",
        ));
    }
    Ok(())
}

fn tensor_row<'a, const WIDTH: usize>(
    tensor: &'a Tensor,
    row: usize,
    expected_rows: usize,
    operation: &'static str,
) -> Result<&'a [f32; WIDTH]> {
    if tensor.shape() != [expected_rows, WIDTH] {
        return Err(RuntimeError::InvalidAdapterWork(operation));
    }
    let start = row
        .checked_mul(WIDTH)
        .ok_or(RuntimeError::InvalidAdapterWork(
            "tensor row offset overflows",
        ))?;
    let end = start
        .checked_add(WIDTH)
        .ok_or(RuntimeError::InvalidAdapterWork(
            "tensor row range overflows",
        ))?;
    tensor
        .data()
        .get(start..end)
        .and_then(|values| values.try_into().ok())
        .ok_or(RuntimeError::InvalidAdapterWork(operation))
}

fn rms_norm_into(
    input: &[f32],
    weight: &Tensor,
    output: &mut [f32],
    operation: &'static str,
) -> Result<()> {
    if input.is_empty() || input.len() != output.len() || weight.shape() != [input.len()] {
        return Err(RuntimeError::InvalidAdapterWork(operation));
    }
    let mut sum_squares = 0.0_f32;
    for value in input {
        sum_squares += value * value;
    }
    if !sum_squares.is_finite() {
        return Err(RuntimeError::NonFinite(operation));
    }
    let inverse = (sum_squares / input.len() as f32 + RMS_EPSILON)
        .sqrt()
        .recip();
    for ((destination, value), coefficient) in output.iter_mut().zip(input).zip(weight.data()) {
        *destination = value * inverse * coefficient;
    }
    ensure_finite(output, operation)
}

fn linear_into(
    weight: &Tensor,
    input: &[f32],
    output: &mut [f32],
    operation: &'static str,
) -> Result<()> {
    if input.is_empty() || output.is_empty() || weight.shape() != [output.len(), input.len()] {
        return Err(RuntimeError::InvalidAdapterWork(operation));
    }
    for (row, destination) in output.iter_mut().enumerate() {
        let start = row
            .checked_mul(input.len())
            .ok_or(RuntimeError::InvalidAdapterWork(
                "linear row offset overflows",
            ))?;
        let end = start
            .checked_add(input.len())
            .ok_or(RuntimeError::InvalidAdapterWork(
                "linear row range overflows",
            ))?;
        let coefficients = weight
            .data()
            .get(start..end)
            .ok_or(RuntimeError::InvalidAdapterWork(operation))?;
        let mut sum = 0.0_f32;
        for (value, coefficient) in input.iter().zip(coefficients) {
            sum += coefficient * value;
        }
        *destination = sum;
    }
    ensure_finite(output, operation)
}

fn add_in_place_checked(
    destination: &mut [f32],
    source: &[f32],
    operation: &'static str,
) -> Result<()> {
    if destination.len() != source.len() {
        return Err(RuntimeError::InvalidAdapterWork(operation));
    }
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += source;
    }
    ensure_finite(destination, operation)
}

fn activate_into(
    gate: &[f32; EXPERT_HIDDEN_SIZE],
    up: &[f32; EXPERT_HIDDEN_SIZE],
    output: &mut [f32; EXPERT_HIDDEN_SIZE],
) -> Result<()> {
    for ((destination, gate), up) in output.iter_mut().zip(gate).zip(up) {
        *destination = silu(*gate) * up;
    }
    ensure_finite(output, "expert activation")
}

fn stable_top_two(scores: &[f32; NUM_EXPERTS]) -> [usize; TOP_K] {
    let mut indices = [0_usize, 1, 2, 3];
    indices.sort_unstable_by(|left, right| {
        scores[*right]
            .partial_cmp(&scores[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    [indices[0], indices[1]]
}

fn softmax_two(scores: [f32; TOP_K]) -> Result<[f32; TOP_K]> {
    let maximum = scores[0].max(scores[1]);
    let mut output = [(scores[0] - maximum).exp(), (scores[1] - maximum).exp()];
    let sum = output[0] + output[1];
    if !sum.is_finite() || sum <= 0.0 {
        return Err(RuntimeError::NonFinite("router softmax"));
    }
    output[0] /= sum;
    output[1] /= sum;
    ensure_finite(&output, "router softmax")?;
    Ok(output)
}

fn run_bf16_into(
    matrix: &runnel_kernels::Bf16Matrix,
    input: FiniteInput<'_>,
    dispatch: super::ExpertDispatch,
    workspace: &mut GemvWorkspace,
    output: &mut [f32],
    operation: &'static str,
) -> Result<()> {
    let request = match dispatch.backend {
        BackendKind::Scalar => BackendRequest::Scalar,
        BackendKind::Avx2 => BackendRequest::Avx2,
    };
    let prepared = PreparedGemv::with_capabilities(matrix, input, request, dispatch.capabilities)
        .map_err(|source| expert_kernel_error(operation, source))?;
    workspace
        .set_rows(matrix.rows())
        .map_err(|source| expert_kernel_error(operation, source))?;
    prepared
        .run(workspace, output)
        .map_err(|source| expert_kernel_error(operation, source))
}

fn expert_kernel_error(operation: &'static str, source: KernelError) -> RuntimeError {
    RuntimeError::ExpertKernel { operation, source }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn ensure_finite(values: &[f32], operation: &'static str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(RuntimeError::NonFinite(operation))
    } else {
        Ok(())
    }
}

struct CheckedAdapterTransactionCounter {
    next: AtomicU64,
}

impl CheckedAdapterTransactionCounter {
    const fn new(first: u64) -> Self {
        Self {
            next: AtomicU64::new(first),
        }
    }

    fn next(&self) -> Result<AdapterTransactionId> {
        let mut candidate = self.next.load(Ordering::Relaxed);
        loop {
            let identity = NonZeroU64::new(candidate)
                .ok_or(RuntimeError::AdapterTransactionIdentityExhausted)?;
            let successor = candidate.checked_add(1).unwrap_or(0);
            match self.next.compare_exchange_weak(
                candidate,
                successor,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(AdapterTransactionId(identity)),
                Err(observed) => candidate = observed,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        sync::{Arc, Barrier},
    };

    use super::*;
    use runnel_fixture::FixtureArtifact;
    use runnel_format::{Artifact, Limits};
    use runnel_kernels::Capabilities;

    fn fixture(version: u64) -> FixtureArtifact {
        match version {
            1 => FixtureArtifact::build(),
            2 => FixtureArtifact::build_v2(),
            3 => FixtureArtifact::build_v3(),
            _ => panic!("test requested unsupported fixture version"),
        }
    }

    fn fixture_model_with_backend(version: u64, backend: BackendRequest) -> TinyModel {
        let artifact =
            Artifact::from_bytes(fixture(version).to_parts(), Limits::default()).unwrap();
        TinyModel::from_artifact_with_backend(&artifact, backend).unwrap()
    }

    fn fixture_model(version: u64) -> TinyModel {
        let backend = if version == 1 {
            BackendRequest::Auto
        } else {
            BackendRequest::Scalar
        };
        fixture_model_with_backend(version, backend)
    }

    fn pattern_token(position: usize) -> u32 {
        if position == 0 {
            1
        } else {
            [14, 16, 6][(position - 1) % 3]
        }
    }

    fn prepared_with_contributions(
        model: &TinyModel,
        state: &SequenceState,
        token: u32,
        workspace: &mut TinyWorkspace,
    ) -> (TinyPreparedToken, Vec<TinyExpertContribution>) {
        let prepared = model.prepare_token(state, token, workspace).unwrap();
        let contributions = model
            .expert_tasks(&prepared)
            .map(|task| model.execute_expert(task, workspace).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(contributions.len(), TOP_K);
        (prepared, contributions)
    }

    fn finish_one(
        model: &TinyModel,
        state: &SequenceState,
        token: u32,
        workspace: &mut TinyWorkspace,
        reverse: bool,
    ) -> TinyPendingStateCommit {
        let prepared = model.prepare_token(state, token, workspace).unwrap();
        let mut tasks = model.expert_tasks(&prepared).collect::<Vec<_>>();
        if reverse {
            tasks.reverse();
        }
        let contributions = tasks
            .into_iter()
            .map(|task| model.execute_expert(task, workspace).unwrap())
            .collect::<Vec<_>>();
        model
            .finish_token(prepared, &contributions, workspace)
            .unwrap()
    }

    fn copy_contribution(source: &TinyExpertContribution) -> TinyExpertContribution {
        TinyExpertContribution {
            identity: source.identity,
            router_rank: source.router_rank,
            expert_id: source.expert_id,
            output_len: source.output_len,
            output: source.output,
        }
    }

    fn apply_pending(
        model: &TinyModel,
        state: &mut SequenceState,
        pending: &TinyPendingStateCommit,
    ) -> Result<()> {
        model.with_validated_state_commit(state, pending, TinyModel::apply_state_commit)
    }

    fn assert_commit_rejected(
        model: &TinyModel,
        state: &mut SequenceState,
        pending: &TinyPendingStateCommit,
        expected: RuntimeError,
    ) {
        let invoked = Cell::new(false);
        assert_eq!(
            model
                .with_validated_state_commit(state, pending, |_| invoked.set(true))
                .unwrap_err(),
            expected
        );
        assert!(
            !invoked.get(),
            "validation failure invoked the commit callback"
        );
    }

    fn assert_contribution_rejected<F>(
        model: &TinyModel,
        state: &SequenceState,
        workspace: &mut TinyWorkspace,
        mutate: F,
        expected: RuntimeError,
    ) where
        F: FnOnce(&TinyPreparedToken, &mut Vec<TinyExpertContribution>),
    {
        let before = state.test_fingerprint();
        let (prepared, mut contributions) =
            prepared_with_contributions(model, state, 16, workspace);
        mutate(&prepared, &mut contributions);
        assert_eq!(
            model
                .finish_token(prepared, &contributions, workspace)
                .unwrap_err(),
            expected
        );
        assert_eq!(state.test_fingerprint(), before);
    }

    fn assert_task_rejected<F>(
        model: &TinyModel,
        state: &SequenceState,
        workspace: &mut TinyWorkspace,
        mutate: F,
        expected: RuntimeError,
    ) where
        F: FnOnce(&mut TinyExpertTask),
    {
        let before = state.test_fingerprint();
        let prepared = model.prepare_token(state, 14, workspace).unwrap();
        let mut task = model.expert_tasks(&prepared).next().unwrap();
        mutate(&mut task);
        assert_eq!(model.execute_expert(task, workspace).unwrap_err(), expected);
        assert_eq!(state.test_fingerprint(), before);
    }

    fn assert_pending_rejected<F>(
        model: &TinyModel,
        state: &mut SequenceState,
        workspace: &mut TinyWorkspace,
        mutate: F,
        expected: RuntimeError,
    ) where
        F: FnOnce(&mut TinyPendingStateCommit),
    {
        let before = state.test_fingerprint();
        let mut pending = finish_one(model, state, 1, workspace, false);
        assert_eq!(state.test_fingerprint(), before);
        mutate(&mut pending);
        assert_commit_rejected(model, state, &pending, expected);
        assert_eq!(state.test_fingerprint(), before);
    }

    fn assert_pending_matches_step(pending: &TinyPendingStateCommit, expected: &crate::StepOutput) {
        assert_eq!(pending.input_token, expected.input_token);
        assert_eq!(pending.logits.as_slice(), expected.logits.as_slice());
        assert_eq!(
            pending.router_scores.as_slice(),
            expected.route.scores.as_slice()
        );
        assert_eq!(
            pending
                .expert_ids
                .iter()
                .map(|expert| usize::from(*expert))
                .collect::<Vec<_>>(),
            expected.route.expert_ids
        );
        assert_eq!(
            pending.route_weights.as_slice(),
            expected.route.weights.as_slice()
        );
    }

    fn assert_phased_parity(version: u64, backend: BackendRequest, tokens: &[u32]) {
        let model = fixture_model_with_backend(version, backend);
        let reference = fixture_model_with_backend(version, backend);
        let expected = reference.run_tokens(tokens).unwrap();
        let layout = model.state_layout(tokens.len(), 16).unwrap();
        let mut state = model.new_state(layout).unwrap();
        let mut workspace = model.new_workspace().unwrap();

        for (position, (token, expected)) in tokens.iter().zip(&expected).enumerate() {
            let before = state.test_fingerprint();
            let pending = finish_one(&model, &state, *token, &mut workspace, false);
            assert_eq!(state.test_fingerprint(), before);
            assert_pending_matches_step(&pending, expected);
            apply_pending(&model, &mut state, &pending).unwrap();
            assert_eq!(state.len(), position + 1);
            assert_eq!(state.revision(), (position + 1) as u64);
        }
    }

    #[test]
    fn transaction_id_exhausts_without_wrapping_or_reuse() {
        let ids = CheckedAdapterTransactionCounter::new(u64::MAX);
        assert_eq!(ids.next().unwrap().get(), u64::MAX);
        assert_eq!(
            ids.next().unwrap_err(),
            RuntimeError::AdapterTransactionIdentityExhausted
        );
        assert_eq!(
            ids.next().unwrap_err(),
            RuntimeError::AdapterTransactionIdentityExhausted
        );
    }

    #[test]
    fn transaction_ids_are_unique_under_concurrent_allocation() {
        const THREADS: usize = 4;
        const IDS_PER_THREAD: usize = 256;

        let ids = Arc::new(CheckedAdapterTransactionCounter::new(1));
        let start = Arc::new(Barrier::new(THREADS));
        let workers = (0..THREADS)
            .map(|_| {
                let ids = Arc::clone(&ids);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    (0..IDS_PER_THREAD)
                        .map(|_| ids.next().unwrap().get())
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();

        let mut observed = workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        observed.sort_unstable();
        let count = THREADS * IDS_PER_THREAD;
        assert_eq!(observed, (1..=count as u64).collect::<Vec<_>>());
        assert_eq!(ids.next().unwrap().get(), count as u64 + 1);
    }

    #[test]
    fn fixed_top_two_preserves_low_id_ties() {
        assert_eq!(stable_top_two(&[1.0, 2.0, 2.0, -1.0]), [1, 2]);
        assert_eq!(stable_top_two(&[-0.0, 0.0, 0.0, -0.0]), [0, 1]);
        assert_eq!(stable_top_two(&[3.0, 2.0, 2.0, 2.0]), [0, 1]);
    }

    #[test]
    fn two_way_softmax_matches_frozen_visit_order() {
        assert_eq!(softmax_two([0.0, 0.0]).unwrap(), [0.5, 0.5]);
        let actual = softmax_two([1.0, -1.0]).unwrap();
        let maximum = 1.0_f32;
        let expected_weights = [(1.0 - maximum).exp(), (-1.0 - maximum).exp()];
        let sum = expected_weights[0] + expected_weights[1];
        assert_eq!(
            actual,
            [expected_weights[0] / sum, expected_weights[1] / sum]
        );
    }

    #[test]
    fn workspace_layout_is_the_preregistered_192_byte_payload() {
        assert_eq!(WORKSPACE_F32_ELEMENTS * size_of::<f32>(), 192);
        assert_eq!(PREPARED_F32_ELEMENTS * size_of::<f32>(), 152);
        assert_eq!(TASK_F32_ELEMENTS * size_of::<f32>(), 32);
        assert_eq!(CONTRIBUTION_F32_ELEMENTS * size_of::<f32>(), 32);
        assert_eq!(PENDING_F32_ELEMENTS * size_of::<f32>(), 216);

        for (version, resident_payload, resident_charge) in
            [(1, 7_904, 7_936), (2, 5_600, 5_632), (3, 5_600, 5_632)]
        {
            let model = fixture_model(version);
            let layout = model.execution_layout().unwrap();
            assert_eq!(layout.max_tasks_per_token(), TOP_K);
            assert_eq!(layout.prepared_payload_bytes(), 152);
            assert_eq!(layout.task_payload_bytes(), 32);
            assert_eq!(layout.contribution_payload_bytes(), 32);
            assert_eq!(layout.pending_payload_bytes(), 216);
            assert_eq!(layout.workspace_payload_bytes(), 192);
            assert_eq!(layout.workspace_charge_bytes(), 192);
            assert_eq!(layout.model_resident_payload_bytes(), resident_payload);
            assert_eq!(layout.model_resident_charge_bytes(), resident_charge);

            let mut workspace = model.new_workspace().unwrap();
            assert_eq!(workspace.lanes.len(), 3);
            assert!(
                workspace
                    .lanes
                    .iter()
                    .all(|lane| lane.len() == EXPERT_HIDDEN_SIZE)
            );
            assert_eq!(workspace.gemv.max_rows(), EXPERT_HIDDEN_SIZE);
            assert_eq!(workspace.gemv.rows(), EXPERT_HIDDEN_SIZE);
            let capacity = workspace.gemv.capacity();
            assert!(capacity >= EXPERT_HIDDEN_SIZE);
            workspace.gemv.set_rows(HIDDEN_SIZE).unwrap();
            assert_eq!(workspace.gemv.rows(), HIDDEN_SIZE);
            assert_eq!(workspace.gemv.capacity(), capacity);
            workspace.gemv.set_rows(EXPERT_HIDDEN_SIZE).unwrap();
            assert_eq!(workspace.gemv.capacity(), capacity);
        }
    }

    #[test]
    fn direct_phases_match_compatibility_for_every_scalar_profile_and_v3_page_boundary() {
        let short = [1, 14, 16, 6, 9, 2];
        assert_phased_parity(1, BackendRequest::Auto, &short);
        assert_phased_parity(2, BackendRequest::Scalar, &short);
        let long = (0..17).map(pattern_token).collect::<Vec<_>>();
        assert_phased_parity(3, BackendRequest::Scalar, &long);
    }

    #[test]
    fn direct_phases_match_compatibility_for_eligible_avx2_profiles() {
        if !Capabilities::detected().avx2_available() {
            return;
        }
        let tokens = [1, 14, 16, 6, 9, 2];
        assert_phased_parity(2, BackendRequest::Avx2, &tokens);
        assert_phased_parity(3, BackendRequest::Avx2, &tokens);
    }

    #[test]
    fn completion_permutation_is_exact_and_stale_pending_is_rejected() {
        for version in [1, 2, 3] {
            let model = fixture_model(version);
            let layout = model.state_layout(16, 16).unwrap();
            let mut state = model.new_state(layout).unwrap();
            let mut workspace = model.new_workspace().unwrap();
            let before = state.test_fingerprint();
            let ordered = finish_one(&model, &state, 14, &mut workspace, false);
            let reversed = finish_one(&model, &state, 14, &mut workspace, true);
            assert_eq!(state.test_fingerprint(), before);
            assert_ne!(
                ordered.identity.transaction_id,
                reversed.identity.transaction_id
            );
            assert_eq!(ordered.key, reversed.key);
            assert_eq!(ordered.value, reversed.value);
            assert_eq!(ordered.logits, reversed.logits);
            assert_eq!(ordered.router_scores, reversed.router_scores);
            assert_eq!(ordered.expert_ids, reversed.expert_ids);
            assert_eq!(ordered.route_weights, reversed.route_weights);

            apply_pending(&model, &mut state, &ordered).unwrap();
            let committed = state.test_fingerprint();
            assert_commit_rejected(
                &model,
                &mut state,
                &reversed,
                RuntimeError::StateRevisionMismatch {
                    expected: 0,
                    actual: 1,
                },
            );
            assert_eq!(state.test_fingerprint(), committed);
        }
    }

    #[test]
    fn every_malformed_contribution_set_is_rejected_before_scatter() {
        let model = fixture_model(2);
        let layout = model.state_layout(16, 16).unwrap();
        let state = model.new_state(layout).unwrap();
        let mut workspace = model.new_workspace().unwrap();

        for retained in [0, 1] {
            assert_contribution_rejected(
                &model,
                &state,
                &mut workspace,
                |_, contributions| contributions.truncate(retained),
                RuntimeError::InvalidExpertContribution("contribution count does not match top-k"),
            );
        }
        assert_contribution_rejected(
            &model,
            &state,
            &mut workspace,
            |_, contributions| {
                let extra = copy_contribution(&contributions[0]);
                contributions.push(extra);
            },
            RuntimeError::InvalidExpertContribution("contribution count does not match top-k"),
        );
        assert_contribution_rejected(
            &model,
            &state,
            &mut workspace,
            |_, contributions| {
                let duplicate = copy_contribution(&contributions[0]);
                contributions[1] = duplicate;
            },
            RuntimeError::InvalidExpertContribution("router rank appears more than once"),
        );
        assert_contribution_rejected(
            &model,
            &state,
            &mut workspace,
            |_, contributions| contributions[0].router_rank = TOP_K as u16,
            RuntimeError::InvalidExpertContribution("router rank is out of range"),
        );
        assert_contribution_rejected(
            &model,
            &state,
            &mut workspace,
            |prepared, contributions| contributions[0].expert_id = prepared.expert_ids[1],
            RuntimeError::InvalidExpertContribution(
                "expert ID does not match the prepared router rank",
            ),
        );
        assert_contribution_rejected(
            &model,
            &state,
            &mut workspace,
            |_, contributions| contributions[0].expert_id = u16::MAX,
            RuntimeError::InvalidExpertContribution(
                "expert ID does not match the prepared router rank",
            ),
        );

        for width in [7, 9] {
            assert_contribution_rejected(
                &model,
                &state,
                &mut workspace,
                |_, contributions| contributions[0].output_len = width,
                RuntimeError::InvalidExpertContribution("expert output width is invalid"),
            );
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_contribution_rejected(
                &model,
                &state,
                &mut workspace,
                |_, contributions| contributions[0].output[0] = value,
                RuntimeError::InvalidExpertContribution("expert output is nonfinite"),
            );
        }
    }

    #[test]
    fn mixed_transaction_state_and_model_contributions_are_rejected_exactly() {
        let model = fixture_model(2);
        let layout = model.state_layout(16, 16).unwrap();
        let state = model.new_state(layout).unwrap();
        let mut workspace = model.new_workspace().unwrap();

        let before = state.test_fingerprint();
        let (prepared, mut contributions) =
            prepared_with_contributions(&model, &state, 16, &mut workspace);
        let (_other_prepared, mut other) =
            prepared_with_contributions(&model, &state, 16, &mut workspace);
        contributions[1] = other.remove(1);
        assert_eq!(
            model
                .finish_token(prepared, &contributions, &mut workspace)
                .unwrap_err(),
            RuntimeError::InvalidExpertContribution(
                "contribution identity does not match the prepared token",
            )
        );
        assert_eq!(state.test_fingerprint(), before);

        let other_state = model.new_state(layout).unwrap();
        let state_before = state.test_fingerprint();
        let other_before = other_state.test_fingerprint();
        let (prepared, mut contributions) =
            prepared_with_contributions(&model, &state, 16, &mut workspace);
        let (_other_prepared, mut other) =
            prepared_with_contributions(&model, &other_state, 16, &mut workspace);
        contributions[1] = other.remove(1);
        assert_eq!(
            model
                .finish_token(prepared, &contributions, &mut workspace)
                .unwrap_err(),
            RuntimeError::InvalidExpertContribution(
                "contribution identity does not match the prepared token",
            )
        );
        assert_eq!(state.test_fingerprint(), state_before);
        assert_eq!(other_state.test_fingerprint(), other_before);

        let foreign_model = fixture_model(2);
        let foreign_layout = foreign_model.state_layout(16, 16).unwrap();
        let foreign_state = foreign_model.new_state(foreign_layout).unwrap();
        let mut foreign_workspace = foreign_model.new_workspace().unwrap();
        let state_before = state.test_fingerprint();
        let foreign_before = foreign_state.test_fingerprint();
        let (prepared, mut contributions) =
            prepared_with_contributions(&model, &state, 16, &mut workspace);
        let (_foreign_prepared, mut foreign) =
            prepared_with_contributions(&foreign_model, &foreign_state, 16, &mut foreign_workspace);
        contributions[1] = foreign.remove(1);
        assert_eq!(
            model
                .finish_token(prepared, &contributions, &mut workspace)
                .unwrap_err(),
            RuntimeError::InvalidExpertContribution(
                "contribution identity does not match the prepared token",
            )
        );
        assert_eq!(state.test_fingerprint(), state_before);
        assert_eq!(foreign_state.test_fingerprint(), foreign_before);

        for mutate in [
            |identity: &mut AdapterWorkIdentity| identity.state_revision += 1,
            |identity: &mut AdapterWorkIdentity| identity.position += 1,
        ] {
            assert_contribution_rejected(
                &model,
                &state,
                &mut workspace,
                |_, contributions| mutate(&mut contributions[0].identity),
                RuntimeError::InvalidExpertContribution(
                    "contribution identity does not match the prepared token",
                ),
            );
        }
    }

    #[test]
    fn malformed_and_foreign_tasks_fail_without_state_mutation() {
        let model = fixture_model(2);
        let layout = model.state_layout(16, 16).unwrap();
        let state = model.new_state(layout).unwrap();
        let mut workspace = model.new_workspace().unwrap();

        assert_task_rejected(
            &model,
            &state,
            &mut workspace,
            |task| task.router_rank = TOP_K as u16,
            RuntimeError::InvalidAdapterWork("expert task router rank is out of range"),
        );
        assert_task_rejected(
            &model,
            &state,
            &mut workspace,
            |task| task.expert_id = NUM_EXPERTS as u16,
            RuntimeError::InvalidAdapterWork("expert task expert ID is out of range"),
        );
        for width in [7, 9] {
            assert_task_rejected(
                &model,
                &state,
                &mut workspace,
                |task| task.input_len = width,
                RuntimeError::InvalidAdapterWork("expert task input width is invalid"),
            );
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_task_rejected(
                &model,
                &state,
                &mut workspace,
                |task| task.input[0] = value,
                RuntimeError::NonFinite("expert task input"),
            );
        }

        let prepared = model.prepare_token(&state, 16, &mut workspace).unwrap();
        let task = model.expert_tasks(&prepared).next().unwrap();
        let foreign_model = fixture_model(2);
        let mut foreign_workspace = foreign_model.new_workspace().unwrap();
        let before = state.test_fingerprint();
        assert_eq!(
            foreign_model
                .execute_expert(task, &mut foreign_workspace)
                .unwrap_err(),
            RuntimeError::InvalidAdapterWork("work belongs to a different model instance")
        );
        assert_eq!(state.test_fingerprint(), before);
    }

    #[test]
    fn tiny_adapter_exposes_stable_vocabulary_stop_and_task_contracts() {
        for version in [1, 2, 3] {
            let model = fixture_model(version);
            assert_eq!(model.vocabulary_size(), VOCAB_SIZE);
            assert!(model.is_stop_token(EOS_TOKEN));
            for token in 1..VOCAB_SIZE as u32 {
                assert!(!model.is_stop_token(token));
            }

            let layout = model.execution_layout().unwrap();
            let state_layout = model.state_layout(1, 1).unwrap();
            let state = model.new_state(state_layout).unwrap();
            let mut workspace = model.new_workspace().unwrap();
            let prepared = model.prepare_token(&state, 1, &mut workspace).unwrap();
            let identity = model.prepared_identity(&prepared);
            let tasks = model.expert_tasks(&prepared);
            assert!(tasks.len() >= 1);
            assert!(tasks.len() <= layout.max_tasks_per_token());
            for (rank, task) in tasks.enumerate() {
                assert_eq!(model.task_identity(&task), identity);
                assert_eq!(usize::from(model.task_router_rank(&task)), rank);
            }
        }
    }

    #[test]
    fn malformed_pending_commits_fail_with_exact_rollback() {
        let model = fixture_model(2);
        let layout = model.state_layout(16, 16).unwrap();
        let mut state = model.new_state(layout).unwrap();
        let mut workspace = model.new_workspace().unwrap();

        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.input_token = VOCAB_SIZE as u32,
            RuntimeError::InvalidToken {
                vocab_size: VOCAB_SIZE,
            },
        );
        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.key[0] = f32::NAN,
            RuntimeError::NonFinite("pending key"),
        );
        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.value[0] = f32::INFINITY,
            RuntimeError::NonFinite("pending value"),
        );
        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.logits[0] = f32::NEG_INFINITY,
            RuntimeError::NonFinite("pending logits"),
        );
        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.router_scores[0] = f32::NAN,
            RuntimeError::NonFinite("pending router scores"),
        );
        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.expert_ids[0] = NUM_EXPERTS as u16,
            RuntimeError::InvalidAdapterWork("route expert ID is out of range"),
        );
        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.expert_ids[1] = pending.expert_ids[0],
            RuntimeError::InvalidAdapterWork("route expert IDs are not unique"),
        );
        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.route_weights = [0.0; TOP_K],
            RuntimeError::InvalidAdapterWork(
                "route weights must be nonnegative, finite, and have a positive sum",
            ),
        );
        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.route_weights[0] = -0.25,
            RuntimeError::InvalidAdapterWork(
                "route weights must be nonnegative, finite, and have a positive sum",
            ),
        );
        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.identity.state_revision += 1,
            RuntimeError::StateRevisionMismatch {
                expected: 1,
                actual: 0,
            },
        );
        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.identity.position += 1,
            RuntimeError::InvalidState("position must equal the logical state length"),
        );

        let other_state = model.new_state(layout).unwrap();
        let other_id = other_state.state_id().unwrap();
        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.identity.state_id = other_id,
            RuntimeError::StateMismatch,
        );
        let foreign_model = fixture_model(2);
        assert_pending_rejected(
            &model,
            &mut state,
            &mut workspace,
            |pending| pending.identity.model_instance_id = foreign_model.instance_id,
            RuntimeError::InvalidAdapterWork("work belongs to a different model instance"),
        );
    }

    #[test]
    fn dropped_and_rejected_commit_capabilities_preserve_complete_state() {
        let model = fixture_model(2);
        let layout = model.state_layout(16, 16).unwrap();
        let mut state = model.new_state(layout).unwrap();
        let mut workspace = model.new_workspace().unwrap();

        let before = state.test_fingerprint();
        {
            let _pending = finish_one(&model, &state, 1, &mut workspace, false);
        }
        assert_eq!(state.test_fingerprint(), before);

        {
            let pending = finish_one(&model, &state, 1, &mut workspace, false);
            model
                .with_validated_state_commit(&mut state, &pending, |_permit| {})
                .unwrap();
            assert_eq!(state.test_fingerprint(), before);
        }
        assert_eq!(state.test_fingerprint(), before);

        let source_pending = finish_one(&model, &state, 1, &mut workspace, false);
        let mut other_state = model.new_state(layout).unwrap();
        let source_before = state.test_fingerprint();
        let other_before = other_state.test_fingerprint();
        assert_commit_rejected(
            &model,
            &mut other_state,
            &source_pending,
            RuntimeError::StateMismatch,
        );
        assert_eq!(state.test_fingerprint(), source_before);
        assert_eq!(other_state.test_fingerprint(), other_before);

        let foreign_model = fixture_model(2);
        let foreign_layout = foreign_model.state_layout(16, 16).unwrap();
        let mut foreign_state = foreign_model.new_state(foreign_layout).unwrap();
        let foreign_before = foreign_state.test_fingerprint();
        assert_commit_rejected(
            &foreign_model,
            &mut foreign_state,
            &source_pending,
            RuntimeError::InvalidAdapterWork("work belongs to a different model instance"),
        );
        assert_eq!(state.test_fingerprint(), source_before);
        assert_eq!(foreign_state.test_fingerprint(), foreign_before);

        let first = finish_one(&model, &state, 1, &mut workspace, false);
        let stale = finish_one(&model, &state, 1, &mut workspace, false);
        apply_pending(&model, &mut state, &first).unwrap();
        let committed = state.test_fingerprint();
        assert_commit_rejected(
            &model,
            &mut state,
            &stale,
            RuntimeError::StateRevisionMismatch {
                expected: 0,
                actual: 1,
            },
        );
        assert_eq!(state.test_fingerprint(), committed);
    }

    #[test]
    fn one_workspace_serves_the_complete_v3_context_without_growth() {
        let model = fixture_model(3);
        let layout = model.state_layout(1_024, 16).unwrap();
        let mut state = model.new_state(layout).unwrap();
        let mut workspace = model.new_workspace().unwrap();
        let capacity = workspace.gemv.capacity();
        assert!(capacity >= EXPERT_HIDDEN_SIZE);
        assert_eq!(workspace.gemv.max_rows(), EXPERT_HIDDEN_SIZE);
        assert_eq!(workspace.gemv.rows(), EXPERT_HIDDEN_SIZE);

        for position in 0..1_024 {
            let pending = finish_one(
                &model,
                &state,
                pattern_token(position),
                &mut workspace,
                false,
            );
            assert_eq!(pending.identity.position, position);
            assert_eq!(workspace.gemv.max_rows(), EXPERT_HIDDEN_SIZE);
            assert_eq!(workspace.gemv.rows(), HIDDEN_SIZE);
            assert_eq!(workspace.gemv.capacity(), capacity);
            apply_pending(&model, &mut state, &pending).unwrap();
            if [0, 14, 15, 16, 254, 255, 256, 1_022, 1_023].contains(&position) {
                assert_eq!(state.len(), position + 1);
                assert_eq!(state.revision(), (position + 1) as u64);
                assert!(state.history().key_at(position).is_some());
                assert!(state.history().value_at(position).is_some());
            }
        }

        let full = state.test_fingerprint();
        assert_eq!(
            model.prepare_token(&state, 14, &mut workspace).unwrap_err(),
            RuntimeError::ContextLimit { limit: 1_024 }
        );
        assert_eq!(state.test_fingerprint(), full);
        assert_eq!(workspace.gemv.max_rows(), EXPERT_HIDDEN_SIZE);
        assert_eq!(workspace.gemv.rows(), HIDDEN_SIZE);
        assert_eq!(workspace.gemv.capacity(), capacity);
    }
}
