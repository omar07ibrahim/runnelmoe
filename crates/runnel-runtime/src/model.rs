use std::{
    collections::BTreeMap,
    fmt,
    mem::size_of,
    sync::atomic::{AtomicU64, Ordering},
};

use runnel_format::{Artifact, DType, Manifest, TensorRecord};
use runnel_kernels::{BackendKind, BackendRequest, Bf16Matrix, Capabilities, select_backend};

use crate::{
    DecoderAdapter, EOS_TOKEN, Result, RuntimeError, StateLayout, Tensor, TensorCatalog,
    state::SequenceState,
};

#[cfg(test)]
use crate::attention::streaming_causal_attention;
#[cfg(test)]
use runnel_kernels::KernelError;

mod transaction;

pub use transaction::{
    TinyExpertContribution, TinyExpertTask, TinyPendingStateCommit, TinyPreparedToken,
    TinyStateCommitPermit, TinyWorkspace,
};

#[cfg(test)]
const RMS_EPSILON: f32 = 1.0 / 4096.0;
const COMPATIBILITY_PAGE_TOKENS: usize = 16;
static NEXT_MODEL_INSTANCE_ID: CheckedModelIdCounter = CheckedModelIdCounter::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TinyConfig {
    pub context_length: usize,
    pub expert_hidden_size: usize,
    pub hidden_size: usize,
    pub num_experts: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub top_k: usize,
    pub vocab_size: usize,
}

impl TinyConfig {
    #[must_use]
    pub const fn reference() -> Self {
        Self {
            context_length: 16,
            expert_hidden_size: 12,
            hidden_size: 8,
            num_experts: 4,
            num_heads: 2,
            num_layers: 1,
            top_k: 2,
            vocab_size: 32,
        }
    }

    /// Frozen adapter-v3 geometry. Tensor equations and topology are otherwise
    /// identical to the compact BF16 adapter-v2 profile.
    #[must_use]
    pub const fn reference_v3() -> Self {
        Self {
            context_length: 1_024,
            ..Self::reference()
        }
    }

    fn validate_for(&self, profile: AdapterProfile) -> Result<()> {
        if self.context_length == 0
            || self.expert_hidden_size == 0
            || self.hidden_size == 0
            || self.num_experts == 0
            || self.num_heads == 0
            || self.num_layers == 0
            || self.top_k == 0
            || self.vocab_size == 0
        {
            return Err(RuntimeError::InvalidConfig(
                "all dimensions must be positive".into(),
            ));
        }
        if self.num_layers != 1 {
            return Err(RuntimeError::InvalidConfig(
                "the tiny adapter requires exactly one layer".into(),
            ));
        }
        if self.vocab_size != 32 {
            return Err(RuntimeError::InvalidConfig(
                "the tiny adapter requires the fixed 32-token vocabulary".into(),
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(RuntimeError::InvalidConfig(
                "hidden size must be divisible by head count".into(),
            ));
        }
        if self.top_k > self.num_experts {
            return Err(RuntimeError::InvalidConfig(
                "top-k cannot exceed expert count".into(),
            ));
        }
        if self != &profile.expected_config() {
            return Err(RuntimeError::InvalidConfig(format!(
                "tiny adapter version {} requires its frozen reference dimensions",
                profile.version()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdapterProfile {
    V1,
    V2,
    V3,
}

impl AdapterProfile {
    fn from_version(version: u64) -> Option<Self> {
        match version {
            1 => Some(Self::V1),
            2 => Some(Self::V2),
            3 => Some(Self::V3),
            _ => None,
        }
    }

    const fn version(self) -> u64 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }

    const fn expected_config(self) -> TinyConfig {
        match self {
            Self::V1 | Self::V2 => TinyConfig::reference(),
            Self::V3 => TinyConfig::reference_v3(),
        }
    }

    const fn uses_bf16_experts(self) -> bool {
        matches!(self, Self::V2 | Self::V3)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouteDecision {
    pub scores: Vec<f32>,
    pub expert_ids: Vec<usize>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepOutput {
    pub input_token: u32,
    pub logits: Vec<f32>,
    pub route: RouteDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    pub generated_tokens: Vec<u32>,
    pub steps: Vec<StepOutput>,
}

pub struct TinyModel {
    config: TinyConfig,
    instance_id: u64,
    weights: Weights,
}

impl fmt::Debug for TinyModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TinyModel")
            .field("config", &self.config)
            .field("instance_id", &self.instance_id)
            .field("weights", &"<redacted>")
            .finish()
    }
}

impl TinyModel {
    pub fn from_artifact(artifact: &Artifact) -> Result<Self> {
        Self::from_artifact_with_backend(artifact, BackendRequest::Auto)
    }

    /// Builds a model while explicitly choosing a compact-expert backend.
    ///
    /// Adapter v1 accepts only [`BackendRequest::Auto`] because it retains its
    /// original f32 implementation. Adapters v2 and v3 use compact BF16
    /// experts and accept automatic or explicit dispatch.
    pub fn from_artifact_with_backend(
        artifact: &Artifact,
        request: BackendRequest,
    ) -> Result<Self> {
        Self::from_verified_tensor_bytes_with_backend(artifact.manifest(), request, |descriptor| {
            artifact
                .tensor_bytes(descriptor.id)
                .map(|bytes| bytes.to_vec())
                .ok_or_else(|| {
                    RuntimeError::InvalidArtifact(format!(
                        "verified bytes are unavailable for tensor {}",
                        descriptor.role
                    ))
                })
        })
    }

    /// Builds a supported tiny adapter from an authenticated manifest and
    /// caller-supplied tensor bytes, using automatic compact-expert dispatch.
    ///
    /// This is the adapter boundary used by the out-of-core store: the caller
    /// retains responsibility for byte authentication, while this method
    /// independently validates the complete adapter catalog, shapes, dtypes,
    /// finite values, and required roles before constructing executable state.
    pub fn from_verified_tensor_bytes<F>(manifest: &Manifest, tensor_bytes: F) -> Result<Self>
    where
        F: FnMut(&TensorRecord) -> Result<Vec<u8>>,
    {
        Self::from_verified_tensor_bytes_with_backend(manifest, BackendRequest::Auto, tensor_bytes)
    }

    /// Builds a supported tiny adapter with an explicit compact-expert backend
    /// request from independently authenticated tensor bytes.
    pub fn from_verified_tensor_bytes_with_backend<F>(
        manifest: &Manifest,
        request: BackendRequest,
        mut tensor_bytes: F,
    ) -> Result<Self>
    where
        F: FnMut(&TensorRecord) -> Result<Vec<u8>>,
    {
        let adapter_version = manifest.adapter.version;
        if manifest.adapter.id != "runnel.tiny-causal-moe" {
            return Err(RuntimeError::InvalidArtifact(
                "unsupported adapter identity".into(),
            ));
        }
        let profile = AdapterProfile::from_version(adapter_version).ok_or_else(|| {
            RuntimeError::InvalidArtifact(format!("unsupported adapter version {adapter_version}"))
        })?;
        if profile == AdapterProfile::V1 && request != BackendRequest::Auto {
            return Err(RuntimeError::BackendIncompatibleWithAdapter {
                adapter_version,
                request,
            });
        }
        if manifest.tokenizer.id != "runnel.ascii32"
            || manifest.tokenizer.version != 1
            || manifest.tokenizer.vocab_size != 32
        {
            return Err(RuntimeError::InvalidArtifact(format!(
                "adapter version {adapter_version} requires runnel.ascii32 version 1 with vocabulary 32"
            )));
        }

        let model = &manifest.model;
        let config = TinyConfig {
            context_length: dimension(model.context_length, "context_length")?,
            expert_hidden_size: dimension(model.expert_hidden_size, "expert_hidden_size")?,
            hidden_size: dimension(model.hidden_size, "hidden_size")?,
            num_experts: dimension(model.num_experts, "num_experts")?,
            num_heads: dimension(model.num_heads, "num_heads")?,
            num_layers: dimension(model.num_layers, "num_layers")?,
            top_k: dimension(model.top_k, "top_k")?,
            vocab_size: dimension(model.vocab_size, "vocab_size")?,
        };
        config.validate_for(profile)?;

        let expected = expected_tensors(&config, profile);
        if manifest.tensors.len() != expected.len() {
            return Err(RuntimeError::InvalidArtifact(format!(
                "adapter version {adapter_version} requires exactly {} tensors, found {}",
                expected.len(),
                manifest.tensors.len()
            )));
        }
        for (descriptor, expected) in manifest.tensors.iter().zip(&expected) {
            if descriptor.role != expected.role {
                return Err(RuntimeError::InvalidArtifact(format!(
                    "tensor ID {} must have role {}, found {}",
                    descriptor.id, expected.role, descriptor.role
                )));
            }
            if descriptor.dtype != expected.dtype {
                return Err(RuntimeError::InvalidArtifact(format!(
                    "tensor {} has unsupported dtype {}; adapter version {adapter_version} requires {}",
                    descriptor.role,
                    descriptor.dtype.as_str(),
                    expected.dtype.as_str()
                )));
            }
            if descriptor.shape.len() != expected.shape.len()
                || descriptor
                    .shape
                    .iter()
                    .zip(&expected.shape)
                    .any(|(actual, expected)| u64::from(*actual) != *expected as u64)
            {
                return Err(RuntimeError::InvalidArtifact(format!(
                    "tensor {} must have shape {:?}, found {:?}",
                    descriptor.role, expected.shape, descriptor.shape
                )));
            }
        }

        let dispatch = if profile.uses_bf16_experts() {
            let capabilities = Capabilities::detected();
            let backend = select_backend(request, capabilities).map_err(|source| {
                RuntimeError::ExpertKernel {
                    operation: "backend selection",
                    source,
                }
            })?;
            Some(ExpertDispatch {
                backend,
                capabilities,
            })
        } else {
            None
        };

        let mut f32_tensors = TensorCatalog::new();
        let mut bf16_tensors = BTreeMap::new();
        for descriptor in &manifest.tensors {
            let bytes = tensor_bytes(descriptor)?;
            let expected_length = usize::try_from(descriptor.length).map_err(|_| {
                RuntimeError::InvalidArtifact(format!(
                    "tensor {} length cannot be represented on this host",
                    descriptor.role
                ))
            })?;
            if bytes.len() != expected_length {
                return Err(RuntimeError::InvalidArtifact(format!(
                    "tensor {} supplied {} bytes, expected {expected_length}",
                    descriptor.role,
                    bytes.len()
                )));
            }
            let shape = descriptor
                .shape
                .iter()
                .map(|dimension| usize::try_from(*dimension))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| {
                    RuntimeError::InvalidArtifact(format!(
                        "tensor {} shape cannot be represented on this host",
                        descriptor.role
                    ))
                })?;

            match descriptor.dtype {
                DType::F32Le => {
                    let tensor = decode_f32_tensor(&descriptor.role, shape, &bytes)?;
                    if f32_tensors
                        .insert(descriptor.role.clone(), tensor)
                        .is_some()
                    {
                        return Err(RuntimeError::InvalidArtifact(format!(
                            "duplicate tensor role {}",
                            descriptor.role
                        )));
                    }
                }
                DType::Bf16Le => {
                    let [rows, columns] = shape.as_slice() else {
                        return Err(RuntimeError::InvalidArtifact(format!(
                            "BF16 tensor {} must be a matrix",
                            descriptor.role
                        )));
                    };
                    let matrix =
                        Bf16Matrix::from_le_bytes(*rows, *columns, &bytes).map_err(|source| {
                            RuntimeError::InvalidBf16Tensor {
                                role: descriptor.role.clone(),
                                source,
                            }
                        })?;
                    if bf16_tensors
                        .insert(descriptor.role.clone(), matrix)
                        .is_some()
                    {
                        return Err(RuntimeError::InvalidArtifact(format!(
                            "duplicate tensor role {}",
                            descriptor.role
                        )));
                    }
                }
                DType::U8 | DType::I8 => unreachable!("dtype was validated before byte loading"),
            }
        }

        match (profile, dispatch) {
            (AdapterProfile::V1, None) => Self::from_catalog(config, f32_tensors),
            (AdapterProfile::V2 | AdapterProfile::V3, Some(dispatch)) => {
                Self::from_bf16_catalog(config, f32_tensors, bf16_tensors, dispatch, profile)
            }
            _ => unreachable!("supported adapter versions have a fixed representation"),
        }
    }

    /// Builds adapter v1 directly from finite f32 tensors.
    pub fn from_catalog(config: TinyConfig, mut tensors: TensorCatalog) -> Result<Self> {
        config.validate_for(AdapterProfile::V1)?;

        let mut experts = Vec::with_capacity(config.num_experts);
        for expert in 0..config.num_experts {
            experts.push(F32ExpertWeights {
                gate: take(
                    &mut tensors,
                    &format!("layers.0.experts.{expert}.gate"),
                    &[config.expert_hidden_size, config.hidden_size],
                )?,
                up: take(
                    &mut tensors,
                    &format!("layers.0.experts.{expert}.up"),
                    &[config.expert_hidden_size, config.hidden_size],
                )?,
                down: take(
                    &mut tensors,
                    &format!("layers.0.experts.{expert}.down"),
                    &[config.hidden_size, config.expert_hidden_size],
                )?,
            });
        }
        Self::from_components(config, tensors, Experts::F32(experts))
    }

    fn from_bf16_catalog(
        config: TinyConfig,
        tensors: TensorCatalog,
        mut bf16_tensors: BTreeMap<String, Bf16Matrix>,
        dispatch: ExpertDispatch,
        profile: AdapterProfile,
    ) -> Result<Self> {
        debug_assert!(profile.uses_bf16_experts());
        config.validate_for(profile)?;
        let mut experts = Vec::with_capacity(config.num_experts);
        for expert in 0..config.num_experts {
            experts.push(Bf16ExpertWeights {
                gate: take_bf16(
                    &mut bf16_tensors,
                    &format!("layers.0.experts.{expert}.gate"),
                    config.expert_hidden_size,
                    config.hidden_size,
                )?,
                up: take_bf16(
                    &mut bf16_tensors,
                    &format!("layers.0.experts.{expert}.up"),
                    config.expert_hidden_size,
                    config.hidden_size,
                )?,
                down: take_bf16(
                    &mut bf16_tensors,
                    &format!("layers.0.experts.{expert}.down"),
                    config.hidden_size,
                    config.expert_hidden_size,
                )?,
            });
        }
        if let Some(role) = bf16_tensors.into_keys().next() {
            return Err(RuntimeError::UnexpectedTensor(role));
        }
        Self::from_components(
            config,
            tensors,
            Experts::Bf16 {
                weights: experts,
                dispatch,
            },
        )
    }

    fn from_components(
        config: TinyConfig,
        mut tensors: TensorCatalog,
        experts: Experts,
    ) -> Result<Self> {
        let hidden = config.hidden_size;
        let token_embedding = take(
            &mut tensors,
            "token_embedding",
            &[config.vocab_size, hidden],
        )?;
        let attn_norm = take(&mut tensors, "layers.0.attn_norm", &[hidden])?;
        let attn_q = take(&mut tensors, "layers.0.attn_q", &[hidden, hidden])?;
        let attn_k = take(&mut tensors, "layers.0.attn_k", &[hidden, hidden])?;
        let attn_v = take(&mut tensors, "layers.0.attn_v", &[hidden, hidden])?;
        let attn_out = take(&mut tensors, "layers.0.attn_out", &[hidden, hidden])?;
        let ffn_norm = take(&mut tensors, "layers.0.ffn_norm", &[hidden])?;
        let router = take(
            &mut tensors,
            "layers.0.router",
            &[config.num_experts, hidden],
        )?;
        let final_norm = take(&mut tensors, "final_norm", &[hidden])?;
        let lm_head = take(&mut tensors, "lm_head", &[config.vocab_size, hidden])?;

        if let Some(role) = tensors.into_keys().next() {
            return Err(RuntimeError::UnexpectedTensor(role));
        }

        Ok(Self {
            config,
            instance_id: NEXT_MODEL_INSTANCE_ID.next()?,
            weights: Weights {
                token_embedding,
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_out,
                ffn_norm,
                router,
                experts,
                final_norm,
                lm_head,
            },
        })
    }

    #[must_use]
    pub fn config(&self) -> &TinyConfig {
        &self.config
    }

    /// Reports the selected compact expert backend for adapters v2 and v3.
    /// Adapter v1 returns `None` because it always executes its f32 path.
    #[must_use]
    pub fn expert_backend(&self) -> Option<BackendKind> {
        match &self.weights.experts {
            Experts::F32(_) => None,
            Experts::Bf16 { dispatch, .. } => Some(dispatch.backend),
        }
    }

    /// Checks request-bounded state geometry without exposing model identity.
    pub fn state_layout(&self, max_tokens: usize, page_tokens: usize) -> Result<StateLayout> {
        if max_tokens > self.config.context_length {
            return Err(RuntimeError::ContextLimit {
                limit: self.config.context_length,
            });
        }
        StateLayout::new(max_tokens, page_tokens, self.config.hidden_size)
    }

    /// Eagerly allocates a state whose complete layout was checked for this model.
    pub fn new_sequence_state(&self, layout: StateLayout) -> Result<SequenceState> {
        self.validate_state_layout(layout)?;
        SequenceState::try_new(layout, self.instance_id)
    }

    pub fn forward_token(&self, state: &mut SequenceState, token: u32) -> Result<StepOutput> {
        let token_index = usize::try_from(token).map_err(|_| RuntimeError::InvalidToken {
            vocab_size: self.config.vocab_size,
        })?;
        if token_index >= self.config.vocab_size {
            return Err(RuntimeError::InvalidToken {
                vocab_size: self.config.vocab_size,
            });
        }
        let mut workspace = <Self as DecoderAdapter>::new_workspace(self)?;

        if state.layout().is_none() {
            if state.model_instance_id().is_some()
                || state.state_id().is_some()
                || state.revision() != 0
                || !state.is_empty()
            {
                return Err(RuntimeError::InvalidState(
                    "unbound sequence shell is internally inconsistent",
                ));
            }
            let layout =
                self.state_layout(self.config.context_length, COMPATIBILITY_PAGE_TOKENS)?;
            let mut candidate = self.new_sequence_state(layout)?;
            let output = self.forward_token_bound(&mut candidate, token, &mut workspace)?;
            *state = candidate;
            return Ok(output);
        }

        self.forward_token_bound(state, token, &mut workspace)
    }

    fn forward_token_bound(
        &self,
        state: &mut SequenceState,
        token: u32,
        workspace: &mut TinyWorkspace,
    ) -> Result<StepOutput> {
        let prepared = <Self as DecoderAdapter>::prepare_token(self, state, token, workspace)?;
        let mut tasks = <Self as DecoderAdapter>::expert_tasks(self, &prepared);
        let first_task = tasks.next().ok_or(RuntimeError::InvalidAdapterWork(
            "first expert task is missing",
        ))?;
        let second_task = tasks.next().ok_or(RuntimeError::InvalidAdapterWork(
            "second expert task is missing",
        ))?;
        if tasks.next().is_some() {
            return Err(RuntimeError::InvalidAdapterWork(
                "expert task count exceeds frozen top-k",
            ));
        }
        let contributions = [
            <Self as DecoderAdapter>::execute_expert(self, first_task, workspace)?,
            <Self as DecoderAdapter>::execute_expert(self, second_task, workspace)?,
        ];
        let pending =
            <Self as DecoderAdapter>::finish_token(self, prepared, &contributions, workspace)?;

        // Compatibility output owns Vecs. Reserve and populate every one
        // before validating/applying state so allocation failure cannot follow
        // the state linearization point.
        let mut logits = try_vec_with_capacity(pending.logits_array().len(), "step logits")?;
        logits.extend_from_slice(pending.logits_array());
        let mut scores =
            try_vec_with_capacity(pending.router_scores().len(), "step router scores")?;
        scores.extend_from_slice(pending.router_scores());
        let mut expert_ids = try_vec_with_capacity(pending.expert_ids().len(), "step expert IDs")?;
        expert_ids.extend(
            pending
                .expert_ids()
                .iter()
                .map(|expert| usize::from(*expert)),
        );
        let mut weights =
            try_vec_with_capacity(pending.route_weights().len(), "step route weights")?;
        weights.extend_from_slice(pending.route_weights());
        let output = StepOutput {
            input_token: pending.input_token(),
            logits,
            route: RouteDecision {
                scores,
                expert_ids,
                weights,
            },
        };

        <Self as DecoderAdapter>::with_validated_state_commit(
            self,
            state,
            &pending,
            <Self as DecoderAdapter>::apply_state_commit,
        )?;
        Ok(output)
    }

    fn validate_state_layout(&self, layout: StateLayout) -> Result<()> {
        if layout.hidden_size() != self.config.hidden_size {
            return Err(RuntimeError::InvalidStateLayout(
                "state hidden size does not match the model",
            ));
        }
        if layout.max_tokens() > self.config.context_length {
            return Err(RuntimeError::ContextLimit {
                limit: self.config.context_length,
            });
        }
        Ok(())
    }

    pub fn run_tokens(&self, tokens: &[u32]) -> Result<Vec<StepOutput>> {
        if tokens.is_empty() {
            return Err(RuntimeError::EmptySequence);
        }
        if tokens.len() > self.config.context_length {
            return Err(RuntimeError::ContextLimit {
                limit: self.config.context_length,
            });
        }
        let mut outputs = try_vec_with_capacity(tokens.len(), "token-step output")?;
        let layout = self.state_layout(self.config.context_length, COMPATIBILITY_PAGE_TOKENS)?;
        let mut state = self.new_sequence_state(layout)?;
        let mut workspace = <Self as DecoderAdapter>::new_workspace(self)?;
        for token in tokens {
            outputs.push(self.forward_token_bound(&mut state, *token, &mut workspace)?);
        }
        Ok(outputs)
    }

    pub fn generate_greedy(&self, prompt: &[u32], max_new_tokens: usize) -> Result<Generation> {
        if prompt.is_empty() {
            return Err(RuntimeError::EmptySequence);
        }
        let required = prompt
            .len()
            .checked_add(max_new_tokens.saturating_sub(1))
            .ok_or(RuntimeError::ContextLimit {
                limit: self.config.context_length,
            })?;
        if required > self.config.context_length {
            return Err(RuntimeError::ContextLimit {
                limit: self.config.context_length,
            });
        }

        let mut steps = try_vec_with_capacity(required, "generation step output")?;
        let mut generated_tokens = try_vec_with_capacity(max_new_tokens, "generated token output")?;
        let layout = self.state_layout(self.config.context_length, COMPATIBILITY_PAGE_TOKENS)?;
        let mut state = self.new_sequence_state(layout)?;
        let mut workspace = <Self as DecoderAdapter>::new_workspace(self)?;
        for token in prompt {
            steps.push(self.forward_token_bound(&mut state, *token, &mut workspace)?);
        }

        for index in 0..max_new_tokens {
            let last = steps.last().ok_or(RuntimeError::EmptySequence)?;
            let next = greedy_token(&last.logits)?;
            generated_tokens.push(next);
            if next == EOS_TOKEN || index + 1 == max_new_tokens {
                break;
            }
            steps.push(self.forward_token_bound(&mut state, next, &mut workspace)?);
        }

        Ok(Generation {
            generated_tokens,
            steps,
        })
    }
}

fn dimension(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        RuntimeError::InvalidArtifact(format!(
            "model dimension {name} cannot be represented on this host"
        ))
    })
}

fn try_vec_with_capacity<T>(capacity: usize, resource: &'static str) -> Result<Vec<T>> {
    let bytes = capacity
        .checked_mul(size_of::<T>())
        .ok_or(RuntimeError::ResourceSizeOverflow { resource })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| RuntimeError::ResourceExhausted { resource, bytes })?;
    Ok(values)
}

struct CheckedModelIdCounter {
    next: AtomicU64,
}

impl CheckedModelIdCounter {
    const fn new(first: u64) -> Self {
        Self {
            next: AtomicU64::new(first),
        }
    }

    fn next(&self) -> Result<u64> {
        let mut candidate = self.next.load(Ordering::Relaxed);
        loop {
            if candidate == 0 {
                return Err(RuntimeError::ModelIdentityExhausted);
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

#[derive(Debug)]
struct ExpectedTensor {
    role: String,
    shape: Vec<usize>,
    dtype: DType,
}

fn expected_tensor(role: impl Into<String>, shape: Vec<usize>, dtype: DType) -> ExpectedTensor {
    ExpectedTensor {
        role: role.into(),
        shape,
        dtype,
    }
}

fn expected_tensors(config: &TinyConfig, profile: AdapterProfile) -> Vec<ExpectedTensor> {
    let hidden = config.hidden_size;
    let expert_hidden = config.expert_hidden_size;
    let mut tensors = vec![
        expected_tensor(
            "token_embedding",
            vec![config.vocab_size, hidden],
            DType::F32Le,
        ),
        expected_tensor("layers.0.attn_norm", vec![hidden], DType::F32Le),
        expected_tensor("layers.0.attn_q", vec![hidden, hidden], DType::F32Le),
        expected_tensor("layers.0.attn_k", vec![hidden, hidden], DType::F32Le),
        expected_tensor("layers.0.attn_v", vec![hidden, hidden], DType::F32Le),
        expected_tensor("layers.0.attn_out", vec![hidden, hidden], DType::F32Le),
        expected_tensor("layers.0.ffn_norm", vec![hidden], DType::F32Le),
        expected_tensor(
            "layers.0.router",
            vec![config.num_experts, hidden],
            DType::F32Le,
        ),
    ];
    let expert_dtype = if profile.uses_bf16_experts() {
        DType::Bf16Le
    } else {
        DType::F32Le
    };
    for expert in 0..config.num_experts {
        tensors.push(expected_tensor(
            format!("layers.0.experts.{expert}.gate"),
            vec![expert_hidden, hidden],
            expert_dtype,
        ));
        tensors.push(expected_tensor(
            format!("layers.0.experts.{expert}.up"),
            vec![expert_hidden, hidden],
            expert_dtype,
        ));
        tensors.push(expected_tensor(
            format!("layers.0.experts.{expert}.down"),
            vec![hidden, expert_hidden],
            expert_dtype,
        ));
    }
    tensors.push(expected_tensor("final_norm", vec![hidden], DType::F32Le));
    tensors.push(expected_tensor(
        "lm_head",
        vec![config.vocab_size, hidden],
        DType::F32Le,
    ));
    tensors
}

fn decode_f32_tensor(role: &str, shape: Vec<usize>, bytes: &[u8]) -> Result<Tensor> {
    let mut chunks = bytes.chunks_exact(size_of::<f32>());
    let values: Vec<f32> = chunks
        .by_ref()
        .map(|chunk| {
            f32::from_le_bytes(
                chunk
                    .try_into()
                    .expect("chunks_exact yields one f32 encoding"),
            )
        })
        .collect();
    if !chunks.remainder().is_empty() {
        return Err(RuntimeError::InvalidArtifact(format!(
            "tensor {role} has a partial f32 encoding"
        )));
    }
    Tensor::new(role, shape, values)
}

#[must_use]
pub fn stable_top_k(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|left, right| {
        scores[*right]
            .partial_cmp(&scores[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    indices.truncate(k.min(indices.len()));
    indices
}

fn greedy_token(logits: &[f32]) -> Result<u32> {
    if logits.is_empty() {
        return Err(RuntimeError::NonFinite("empty greedy logits"));
    }
    let mut best = 0_usize;
    for candidate in 1..logits.len() {
        if logits[candidate] > logits[best] {
            best = candidate;
        }
    }
    u32::try_from(best).map_err(|_| {
        RuntimeError::InvalidConfig("vocabulary cannot be represented by u32 token IDs".into())
    })
}

#[cfg(test)]
fn rms_norm(input: &[f32], weight: &Tensor) -> Result<Vec<f32>> {
    let mut sum_squares = 0.0_f32;
    for value in input {
        sum_squares += value * value;
    }
    if !sum_squares.is_finite() {
        return Err(RuntimeError::NonFinite("RMS normalization"));
    }
    let inverse = (sum_squares / input.len() as f32 + RMS_EPSILON)
        .sqrt()
        .recip();
    let output: Vec<f32> = input
        .iter()
        .zip(weight.data())
        .map(|(value, weight)| value * inverse * weight)
        .collect();
    ensure_finite(&output, "RMS normalization")?;
    Ok(output)
}

#[cfg(test)]
fn linear(weight: &Tensor, input: &[f32]) -> Vec<f32> {
    let rows = weight.shape()[0];
    let columns = weight.shape()[1];
    debug_assert_eq!(columns, input.len());
    let mut output = Vec::with_capacity(rows);
    for coefficients in weight.data().chunks_exact(columns).take(rows) {
        let mut sum = 0.0_f32;
        for (value, coefficient) in input.iter().zip(coefficients) {
            sum += coefficient * value;
        }
        output.push(sum);
    }
    output
}

#[cfg(test)]
fn softmax(input: &[f32]) -> Result<Vec<f32>> {
    let maximum = input
        .iter()
        .copied()
        .reduce(f32::max)
        .ok_or(RuntimeError::NonFinite("empty softmax"))?;
    let mut output: Vec<f32> = input.iter().map(|value| (*value - maximum).exp()).collect();
    let sum: f32 = output.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(RuntimeError::NonFinite("softmax"));
    }
    for value in &mut output {
        *value /= sum;
    }
    ensure_finite(&output, "softmax")?;
    Ok(output)
}

#[cfg(test)]
fn ensure_finite(values: &[f32], operation: &'static str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(RuntimeError::NonFinite(operation))
    } else {
        Ok(())
    }
}

fn take(tensors: &mut BTreeMap<String, Tensor>, role: &str, shape: &[usize]) -> Result<Tensor> {
    let tensor = tensors
        .remove(role)
        .ok_or_else(|| RuntimeError::MissingTensor(role.into()))?;
    if tensor.shape() != shape {
        return Err(RuntimeError::InvalidTensor {
            role: role.into(),
            reason: format!("expected shape {shape:?}, found {:?}", tensor.shape()),
        });
    }
    Ok(tensor)
}

fn take_bf16(
    tensors: &mut BTreeMap<String, Bf16Matrix>,
    role: &str,
    rows: usize,
    columns: usize,
) -> Result<Bf16Matrix> {
    let matrix = tensors
        .remove(role)
        .ok_or_else(|| RuntimeError::MissingTensor(role.into()))?;
    if matrix.rows() != rows || matrix.columns() != columns {
        return Err(RuntimeError::InvalidTensor {
            role: role.into(),
            reason: format!(
                "expected shape [{rows}, {columns}], found [{}, {}]",
                matrix.rows(),
                matrix.columns()
            ),
        });
    }
    Ok(matrix)
}

#[derive(Clone, Copy, Debug)]
struct ExpertDispatch {
    backend: BackendKind,
    capabilities: Capabilities,
}

#[derive(Debug)]
struct F32ExpertWeights {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

#[derive(Debug)]
struct Bf16ExpertWeights {
    gate: Bf16Matrix,
    up: Bf16Matrix,
    down: Bf16Matrix,
}

#[derive(Debug)]
enum Experts {
    F32(Vec<F32ExpertWeights>),
    Bf16 {
        weights: Vec<Bf16ExpertWeights>,
        dispatch: ExpertDispatch,
    },
}

#[derive(Debug)]
struct Weights {
    token_embedding: Tensor,
    attn_norm: Tensor,
    attn_q: Tensor,
    attn_k: Tensor,
    attn_v: Tensor,
    attn_out: Tensor,
    ffn_norm: Tensor,
    router: Tensor,
    experts: Experts,
    final_norm: Tensor,
    lm_head: Tensor,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use runnel_fixture::{FixtureArtifact, tensor_recipes};
    use runnel_format::{Artifact, Limits};

    fn fixture_model() -> TinyModel {
        let mut catalog = TensorCatalog::new();
        for recipe in tensor_recipes() {
            let values = recipe.values();
            let tensor = Tensor::new(&recipe.role, recipe.shape, values).unwrap();
            catalog.insert(recipe.role, tensor);
        }
        TinyModel::from_catalog(TinyConfig::reference(), catalog).unwrap()
    }

    fn fixture_artifact(version_two: bool) -> Artifact {
        let fixture = if version_two {
            FixtureArtifact::build_v2()
        } else {
            FixtureArtifact::build()
        };
        Artifact::from_bytes(fixture.to_parts(), Limits::default()).unwrap()
    }

    fn fixture_v3_artifact() -> Artifact {
        Artifact::from_bytes(FixtureArtifact::build_v3().to_parts(), Limits::default()).unwrap()
    }

    fn v2_model_with_mutation<F>(request: BackendRequest, mut mutate: F) -> Result<TinyModel>
    where
        F: FnMut(&TensorRecord, &mut Vec<u8>),
    {
        let artifact = fixture_artifact(true);
        TinyModel::from_verified_tensor_bytes_with_backend(
            artifact.manifest(),
            request,
            |descriptor| {
                let mut bytes = artifact.tensor_bytes(descriptor.id).unwrap().to_vec();
                mutate(descriptor, &mut bytes);
                Ok(bytes)
            },
        )
    }

    #[test]
    fn model_debug_redacts_all_weight_payloads() {
        let model = fixture_model();
        let debug = format!("{model:?}");
        assert!(debug.contains("TinyModel"));
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("token_embedding"));
        assert!(!debug.contains("F32("));
    }

    #[test]
    fn stable_top_k_uses_low_expert_id_for_ties() {
        assert_eq!(stable_top_k(&[1.0, 2.0, 2.0, -1.0], 2), [1, 2]);
        assert_eq!(stable_top_k(&[-0.0, 0.0, 0.0, -0.0], 4), [0, 1, 2, 3]);
        assert_eq!(stable_top_k(&[3.0, 2.0, 2.0, 2.0], 2), [0, 1]);
        assert_eq!(stable_top_k(&[1.0, 1.0, 1.0, 1.0], 2), [0, 1]);
    }

    #[test]
    fn greedy_uses_low_token_id_for_numeric_ties() {
        assert_eq!(greedy_token(&[-0.0, 0.0, 0.0, -0.0]).unwrap(), 0);
        assert_eq!(greedy_token(&[1.0, 3.0, 3.0, 2.0]).unwrap(), 1);
        assert!(greedy_token(&[]).is_err());
    }

    #[test]
    fn scalar_primitives_preserve_orientation_and_head_boundaries() {
        let matrix = Tensor::new("matrix", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(linear(&matrix, &[1.0, 10.0, 100.0]), [321.0, 654.0]);

        let norm_weight = Tensor::new("norm", vec![2], vec![2.0, 3.0]).unwrap();
        let normalized = rms_norm(&[1.0, 1.0], &norm_weight).unwrap();
        let inverse = (1.0_f32 + RMS_EPSILON).sqrt().recip();
        assert_eq!(normalized, [2.0 * inverse, 3.0 * inverse]);

        let state = SequenceState::try_new(StateLayout::new(1, 1, 4).unwrap(), 91).unwrap();
        let mut attended = [0.0; 4];
        streaming_causal_attention(
            &[7.0, 8.0, 9.0, 10.0],
            state.history(),
            &[0.0; 4],
            &[1.0, 2.0, 3.0, 4.0],
            2,
            &mut attended,
        )
        .unwrap();
        assert_eq!(attended, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(softmax(&[0.0, 0.0]).unwrap(), [0.5, 0.5]);
    }

    #[test]
    fn reference_model_is_deterministic() {
        let model = fixture_model();
        let left = model.generate_greedy(&[1, 14, 16, 6], 4).unwrap();
        let right = model.generate_greedy(&[1, 14, 16, 6], 4).unwrap();
        assert_eq!(left, right);
        assert!(!left.generated_tokens.is_empty());
    }

    #[test]
    fn context_limit_fails_before_partial_generation() {
        let model = fixture_model();
        let error = model.generate_greedy(&[1; 16], 2).unwrap_err();
        assert_eq!(error, RuntimeError::ContextLimit { limit: 16 });
        assert!(model.generate_greedy(&[1; 16], 1).is_ok());
        assert!(model.generate_greedy(&[1; 15], 2).is_ok());
        assert_eq!(
            model.generate_greedy(&[1; 15], 3).unwrap_err(),
            RuntimeError::ContextLimit { limit: 16 }
        );
        assert!(model.generate_greedy(&[1; 16], 0).is_ok());
        assert_eq!(model.run_tokens(&[1; 16]).unwrap().len(), 16);
        assert_eq!(
            model.run_tokens(&[1; 17]).unwrap_err(),
            RuntimeError::ContextLimit { limit: 16 }
        );
    }

    #[test]
    fn causal_outputs_do_not_depend_on_future_tokens() {
        let model = fixture_model();
        let left = model.run_tokens(&[1, 2, 3]).unwrap();
        let right = model.run_tokens(&[1, 2, 4]).unwrap();
        assert_eq!(left[..2], right[..2]);
    }

    #[test]
    fn tensor_shape_mismatch_is_rejected() {
        let mut catalog = TensorCatalog::new();
        for recipe in tensor_recipes() {
            let shape = if recipe.id == 7 {
                vec![1, recipe.element_count()]
            } else {
                recipe.shape.clone()
            };
            let tensor = Tensor::new(&recipe.role, shape, recipe.values()).unwrap();
            catalog.insert(recipe.role, tensor);
        }
        let error = TinyModel::from_catalog(TinyConfig::reference(), catalog).unwrap_err();
        assert!(matches!(error, RuntimeError::InvalidTensor { .. }));
    }

    #[test]
    fn artifact_adapter_boundary_rejects_incompatible_metadata() {
        let artifact = fixture_artifact(false);
        let mut wrong_tokenizer = artifact.manifest().clone();
        wrong_tokenizer.tokenizer.id = "runnel.ascii31".into();
        let mut wrong_hidden_size = artifact.manifest().clone();
        wrong_hidden_size.model.hidden_size = 4;
        let mut wrong_role = artifact.manifest().clone();
        wrong_role.tensors[20].role = "extra_norm".into();
        let mut wrong_shape = artifact.manifest().clone();
        wrong_shape.tensors[21].shape = vec![16, 16];

        for (manifest, expected) in [
            (wrong_tokenizer, "requires runnel.ascii32"),
            (wrong_hidden_size, "frozen reference"),
            (wrong_role, "must have role final_norm"),
            (wrong_shape, "must have shape [32, 8]"),
        ] {
            let error = TinyModel::from_verified_tensor_bytes(&manifest, |descriptor| {
                Ok(artifact.tensor_bytes(descriptor.id).unwrap().to_vec())
            })
            .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn adapter_profiles_are_closed_before_any_tensor_callback() {
        let version_two = fixture_artifact(true);
        let mut v2_with_v3_context = version_two.manifest().clone();
        v2_with_v3_context.model.context_length = 1_024;

        let version_three = fixture_v3_artifact();
        let mut v3_with_v2_context = version_three.manifest().clone();
        v3_with_v2_context.model.context_length = 16;
        let mut unsupported_version = version_three.manifest().clone();
        unsupported_version.adapter.version = 4;

        for manifest in [v2_with_v3_context, v3_with_v2_context] {
            let error = TinyModel::from_verified_tensor_bytes(&manifest, |_| -> Result<Vec<u8>> {
                panic!("profile rejection must precede tensor loading")
            })
            .unwrap_err();
            assert!(error.to_string().contains("frozen reference"), "{error}");
        }
        let error =
            TinyModel::from_verified_tensor_bytes(&unsupported_version, |_| -> Result<Vec<u8>> {
                panic!("version rejection must precede tensor loading")
            })
            .unwrap_err();
        assert!(error.to_string().contains("unsupported adapter version 4"));

        assert!(matches!(
            TinyModel::from_catalog(TinyConfig::reference_v3(), TensorCatalog::new()),
            Err(RuntimeError::InvalidConfig(_))
        ));
    }

    #[test]
    fn artifact_adapter_boundary_rejects_unsupported_dtype() {
        let fixture = FixtureArtifact::build();
        let mut parts = fixture.to_parts();
        let manifest = String::from_utf8(parts.manifest).unwrap();
        let replaced = manifest
            .replacen("\"dtype\":\"f32-le\"", "\"dtype\":\"u8\"", 1)
            .replacen("\"shape\":[32,8]", "\"shape\":[1024]", 1);
        parts.manifest = replaced.into_bytes();
        let artifact = Artifact::from_bytes(parts, Limits::default()).unwrap();
        let error = TinyModel::from_artifact(&artifact).unwrap_err();
        assert!(error.to_string().contains("unsupported dtype u8"));
    }

    #[test]
    fn tensor_constructor_rejects_nonfinite_artifact_values() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                Tensor::new("weight", vec![1], vec![value]),
                Err(RuntimeError::InvalidTensor { .. })
            ));
        }
    }

    #[test]
    fn state_cannot_cross_model_instances() {
        let first = fixture_model();
        let second = fixture_model();
        let mut state = SequenceState::new();
        first.forward_token(&mut state, 1).unwrap();
        let error = second.forward_token(&mut state, 1).unwrap_err();
        assert_eq!(error, RuntimeError::StateMismatch);
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn failed_forward_does_not_advance_state() {
        let mut catalog = TensorCatalog::new();
        for recipe in tensor_recipes() {
            let mut values = recipe.values();
            if recipe.id == 0 {
                let poisoned_row = 2 * TinyConfig::reference().hidden_size;
                values[poisoned_row..poisoned_row + TinyConfig::reference().hidden_size]
                    .fill(f32::MAX);
            }
            let tensor = Tensor::new(&recipe.role, recipe.shape, values).unwrap();
            catalog.insert(recipe.role, tensor);
        }
        let model = TinyModel::from_catalog(TinyConfig::reference(), catalog).unwrap();
        let mut state = SequenceState::new();
        model.forward_token(&mut state, 1).unwrap();
        let before_id = state.state_id();
        let before_revision = state.revision();
        let before_layout = state.layout();
        let before_key = state.history().key_at(0).unwrap().to_vec();
        let before_value = state.history().value_at(0).unwrap().to_vec();
        assert!(model.forward_token(&mut state, 2).is_err());
        assert_eq!(state.state_id(), before_id);
        assert_eq!(state.revision(), before_revision);
        assert_eq!(state.layout(), before_layout);
        assert_eq!(state.len(), 1);
        assert_eq!(state.history().key_at(0).unwrap(), before_key);
        assert_eq!(state.history().value_at(0).unwrap(), before_value);
    }

    #[test]
    fn nonfinite_logits_leave_an_eager_bound_state_uncommitted() {
        let mut catalog = TensorCatalog::new();
        for recipe in tensor_recipes() {
            let mut values = recipe.values();
            if recipe.id == 21 {
                values.fill(f32::MAX);
            }
            let tensor = Tensor::new(&recipe.role, recipe.shape, values).unwrap();
            catalog.insert(recipe.role, tensor);
        }
        let model = TinyModel::from_catalog(TinyConfig::reference(), catalog).unwrap();
        let layout = model.state_layout(16, 16).unwrap();
        let mut state = model.new_sequence_state(layout).unwrap();
        let identity = state.state_id();
        let payload = state.accounted_payload_bytes();
        let charge = state.accounted_charge_bytes();

        assert_eq!(
            model.forward_token(&mut state, 1).unwrap_err(),
            RuntimeError::NonFinite("lm head")
        );
        assert_eq!(state.state_id(), identity);
        assert_eq!(state.layout(), Some(layout));
        assert_eq!(state.revision(), 0);
        assert_eq!(state.len(), 0);
        assert_eq!(state.accounted_payload_bytes(), payload);
        assert_eq!(state.accounted_charge_bytes(), charge);
    }

    #[test]
    fn adapter_v1_accepts_auto_only_and_reports_no_expert_backend() {
        let artifact = fixture_artifact(false);
        let model = TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Auto).unwrap();
        assert_eq!(model.expert_backend(), None);

        for request in [BackendRequest::Scalar, BackendRequest::Avx2] {
            assert_eq!(
                TinyModel::from_artifact_with_backend(&artifact, request).unwrap_err(),
                RuntimeError::BackendIncompatibleWithAdapter {
                    adapter_version: 1,
                    request,
                }
            );
        }
    }

    #[test]
    fn adapter_v2_keeps_compact_experts_and_selects_requested_backend() {
        let artifact = fixture_artifact(true);
        let scalar =
            TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar).unwrap();
        assert_eq!(scalar.expert_backend(), Some(BackendKind::Scalar));
        match &scalar.weights.experts {
            Experts::Bf16 { weights, .. } => {
                assert_eq!(weights.len(), 4);
                for expert in weights {
                    assert_eq!((expert.gate.rows(), expert.gate.columns()), (12, 8));
                    assert_eq!((expert.up.rows(), expert.up.columns()), (12, 8));
                    assert_eq!((expert.down.rows(), expert.down.columns()), (8, 12));
                    assert_eq!(expert.gate.words().len(), 96);
                    assert_eq!(expert.up.words().len(), 96);
                    assert_eq!(expert.down.words().len(), 96);
                }
            }
            Experts::F32(_) => panic!("adapter v2 widened its expert matrices"),
        }

        let automatic = TinyModel::from_artifact(&artifact).unwrap();
        let capabilities = Capabilities::detected();
        let expected = if capabilities.avx2_available() {
            BackendKind::Avx2
        } else {
            BackendKind::Scalar
        };
        assert_eq!(automatic.expert_backend(), Some(expected));

        match TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Avx2) {
            Ok(model) => {
                assert!(capabilities.avx2_available());
                assert_eq!(model.expert_backend(), Some(BackendKind::Avx2));
            }
            Err(RuntimeError::ExpertKernel {
                operation: "backend selection",
                source: KernelError::BackendUnavailable,
            }) => assert!(!capabilities.avx2_available()),
            Err(error) => panic!("unexpected forced-AVX2 result: {error}"),
        }
    }

    #[test]
    fn adapter_v2_scalar_is_bit_exact_with_v1_for_exact_fixture_values() {
        let version_one = TinyModel::from_artifact(&fixture_artifact(false)).unwrap();
        let version_two =
            TinyModel::from_artifact_with_backend(&fixture_artifact(true), BackendRequest::Scalar)
                .unwrap();
        let tokens = [1, 14, 16, 6, 9, 2];
        assert_eq!(
            version_two.run_tokens(&tokens).unwrap(),
            version_one.run_tokens(&tokens).unwrap()
        );
        assert_eq!(
            version_two.generate_greedy(&tokens[..3], 4).unwrap(),
            version_one.generate_greedy(&tokens[..3], 4).unwrap()
        );
    }

    #[test]
    fn adapter_v2_avx2_preserves_routes_tokens_and_logit_tolerance() {
        if !Capabilities::detected().avx2_available() {
            return;
        }
        let artifact = fixture_artifact(true);
        let scalar =
            TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar).unwrap();
        let avx2 = TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Avx2).unwrap();
        let tokens = [1, 14, 16, 6, 9, 2];
        let scalar_outputs = scalar.run_tokens(&tokens).unwrap();
        let avx2_outputs = avx2.run_tokens(&tokens).unwrap();
        assert_eq!(scalar_outputs.len(), avx2_outputs.len());
        for (scalar_output, avx2_output) in scalar_outputs.iter().zip(&avx2_outputs) {
            assert_eq!(scalar_output.input_token, avx2_output.input_token);
            assert_eq!(scalar_output.route, avx2_output.route);
            for (&expected, &actual) in scalar_output.logits.iter().zip(&avx2_output.logits) {
                let allowed = 1.0e-5_f32 + 1.0e-4_f32 * expected.abs();
                assert!((actual - expected).abs() <= allowed);
            }
        }

        let scalar_generation = scalar.generate_greedy(&tokens[..3], 4).unwrap();
        let avx2_generation = avx2.generate_greedy(&tokens[..3], 4).unwrap();
        assert_eq!(
            scalar_generation.generated_tokens,
            avx2_generation.generated_tokens
        );
        assert_eq!(
            avx2_generation,
            avx2.generate_greedy(&tokens[..3], 4).unwrap()
        );
    }

    #[test]
    fn adapter_dtype_contract_is_validated_before_requesting_bytes() {
        let mut version_one = fixture_artifact(false).manifest().clone();
        version_one.tensors[8].dtype = DType::Bf16Le;
        let error = TinyModel::from_verified_tensor_bytes(&version_one, |_| -> Result<Vec<u8>> {
            panic!("metadata rejection must precede tensor loading")
        })
        .unwrap_err();
        assert!(error.to_string().contains("requires f32-le"), "{error}");

        let mut version_two_expert = fixture_artifact(true).manifest().clone();
        version_two_expert.tensors[8].dtype = DType::F32Le;
        let error =
            TinyModel::from_verified_tensor_bytes(&version_two_expert, |_| -> Result<Vec<u8>> {
                panic!("metadata rejection must precede tensor loading")
            })
            .unwrap_err();
        assert!(error.to_string().contains("requires bf16-le"), "{error}");

        let mut version_two_common = fixture_artifact(true).manifest().clone();
        version_two_common.tensors[0].dtype = DType::Bf16Le;
        let error =
            TinyModel::from_verified_tensor_bytes(&version_two_common, |_| -> Result<Vec<u8>> {
                panic!("metadata rejection must precede tensor loading")
            })
            .unwrap_err();
        assert!(error.to_string().contains("requires f32-le"), "{error}");

        let mut version_three_expert = fixture_v3_artifact().manifest().clone();
        version_three_expert.tensors[8].dtype = DType::F32Le;
        let error =
            TinyModel::from_verified_tensor_bytes(&version_three_expert, |_| -> Result<Vec<u8>> {
                panic!("metadata rejection must precede tensor loading")
            })
            .unwrap_err();
        assert!(error.to_string().contains("requires bf16-le"), "{error}");
    }

    #[test]
    fn adapter_v2_rejects_nonfinite_bf16_words_with_context() {
        for bits in [0x7f80_u16, 0xff80, 0x7f81, 0x7fc1, 0xffff] {
            let error = v2_model_with_mutation(BackendRequest::Scalar, |descriptor, bytes| {
                if descriptor.id == 8 {
                    bytes[..2].copy_from_slice(&bits.to_le_bytes());
                }
            })
            .unwrap_err();
            assert_eq!(
                error,
                RuntimeError::InvalidBf16Tensor {
                    role: "layers.0.experts.0.gate".into(),
                    source: KernelError::NonFiniteWeight { index: 0, bits },
                }
            );
        }
    }

    #[test]
    fn failed_bf16_kernel_never_commits_sequence_state() {
        for request in [BackendRequest::Scalar, BackendRequest::Avx2] {
            if request == BackendRequest::Avx2 && !Capabilities::detected().avx2_available() {
                continue;
            }
            let model = v2_model_with_mutation(request, |descriptor, bytes| {
                if descriptor.role.ends_with(".gate") {
                    for word in bytes.chunks_exact_mut(2) {
                        word.copy_from_slice(&0x7f7f_u16.to_le_bytes());
                    }
                }
            })
            .unwrap();
            let mut state = SequenceState::new();
            assert!(matches!(
                model.forward_token(&mut state, 1),
                Err(RuntimeError::ExpertKernel { .. })
            ));
            assert_eq!(state.state_id(), None);
            assert_eq!(state.model_instance_id(), None);
            assert_eq!(state.layout(), None);
            assert_eq!(state.revision(), 0);
            assert!(state.is_empty());
        }
    }

    #[test]
    fn model_identities_are_nonzero_and_exhaust_instead_of_wrapping() {
        let ids = CheckedModelIdCounter::new(u64::MAX);
        assert_eq!(ids.next().unwrap(), u64::MAX);
        assert_eq!(
            ids.next().unwrap_err(),
            RuntimeError::ModelIdentityExhausted
        );
        assert_eq!(
            ids.next().unwrap_err(),
            RuntimeError::ModelIdentityExhausted
        );
    }

    #[test]
    fn adapter_v2_model_is_safe_for_concurrent_independent_sequences() {
        let model = Arc::new(TinyModel::from_artifact(&fixture_artifact(true)).unwrap());
        let tokens = [1, 14, 16, 6, 9, 2];
        let expected = model.run_tokens(&tokens).unwrap();
        let workers = (0..4)
            .map(|_| {
                let model = Arc::clone(&model);
                std::thread::spawn(move || model.run_tokens(&tokens).unwrap())
            })
            .collect::<Vec<_>>();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), expected);
        }
    }
}
