use std::{
    collections::BTreeMap,
    mem::size_of,
    sync::atomic::{AtomicU64, Ordering},
};

use runnel_format::{Artifact, DType};

use crate::{EOS_TOKEN, Result, RuntimeError, Tensor, TensorCatalog};

const RMS_EPSILON: f32 = 1.0 / 4096.0;
static NEXT_MODEL_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

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

    fn validate(&self) -> Result<()> {
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
                "adapter version 1 requires exactly one layer".into(),
            ));
        }
        if self.vocab_size != 32 {
            return Err(RuntimeError::InvalidConfig(
                "adapter version 1 requires the fixed 32-token vocabulary".into(),
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
        if self != &Self::reference() {
            return Err(RuntimeError::InvalidConfig(
                "adapter version 1 requires the frozen tiny-v1 dimensions".into(),
            ));
        }
        Ok(())
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

#[derive(Debug, Clone, PartialEq)]
pub struct SequenceState {
    model_instance_id: Option<u64>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl SequenceState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_instance_id: None,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl Default for SequenceState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct TinyModel {
    config: TinyConfig,
    instance_id: u64,
    weights: Weights,
}

impl TinyModel {
    pub fn from_artifact(artifact: &Artifact) -> Result<Self> {
        let manifest = artifact.manifest();
        if manifest.adapter.id != "runnel.tiny-causal-moe" || manifest.adapter.version != 1 {
            return Err(RuntimeError::InvalidArtifact(
                "unsupported adapter identity".into(),
            ));
        }
        if manifest.tokenizer.id != "runnel.ascii32"
            || manifest.tokenizer.version != 1
            || manifest.tokenizer.vocab_size != 32
        {
            return Err(RuntimeError::InvalidArtifact(
                "adapter version 1 requires runnel.ascii32 version 1 with vocabulary 32".into(),
            ));
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
        config.validate()?;

        let expected = expected_tensors(&config);
        if manifest.tensors.len() != expected.len() {
            return Err(RuntimeError::InvalidArtifact(format!(
                "adapter version 1 requires exactly {} tensors, found {}",
                expected.len(),
                manifest.tensors.len()
            )));
        }
        for (descriptor, (role, shape)) in manifest.tensors.iter().zip(&expected) {
            if descriptor.role != *role {
                return Err(RuntimeError::InvalidArtifact(format!(
                    "tensor ID {} must have role {role}, found {}",
                    descriptor.id, descriptor.role
                )));
            }
            if descriptor.dtype != DType::F32Le {
                return Err(RuntimeError::InvalidArtifact(format!(
                    "tensor {} has unsupported dtype {}",
                    descriptor.role,
                    descriptor.dtype.as_str()
                )));
            }
            if descriptor.shape.len() != shape.len()
                || descriptor
                    .shape
                    .iter()
                    .zip(shape)
                    .any(|(actual, expected)| u64::from(*actual) != *expected as u64)
            {
                return Err(RuntimeError::InvalidArtifact(format!(
                    "tensor {} must have shape {shape:?}, found {:?}",
                    descriptor.role, descriptor.shape
                )));
            }
        }

        let mut catalog = TensorCatalog::new();
        for descriptor in &manifest.tensors {
            let bytes = artifact.tensor_bytes(descriptor.id).ok_or_else(|| {
                RuntimeError::InvalidArtifact(format!(
                    "verified bytes are unavailable for tensor {}",
                    descriptor.role
                ))
            })?;
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
                    "tensor {} has a partial f32 encoding",
                    descriptor.role
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
            let tensor = Tensor::new(&descriptor.role, shape, values)?;
            if catalog.insert(descriptor.role.clone(), tensor).is_some() {
                return Err(RuntimeError::InvalidArtifact(format!(
                    "duplicate tensor role {}",
                    descriptor.role
                )));
            }
        }
        Self::from_catalog(config, catalog)
    }

    pub fn from_catalog(config: TinyConfig, mut tensors: TensorCatalog) -> Result<Self> {
        config.validate()?;

        let hidden = config.hidden_size;
        let expert_hidden = config.expert_hidden_size;
        let mut experts = Vec::with_capacity(config.num_experts);

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

        for expert in 0..config.num_experts {
            experts.push(ExpertWeights {
                gate: take(
                    &mut tensors,
                    &format!("layers.0.experts.{expert}.gate"),
                    &[expert_hidden, hidden],
                )?,
                up: take(
                    &mut tensors,
                    &format!("layers.0.experts.{expert}.up"),
                    &[expert_hidden, hidden],
                )?,
                down: take(
                    &mut tensors,
                    &format!("layers.0.experts.{expert}.down"),
                    &[hidden, expert_hidden],
                )?,
            });
        }

        let final_norm = take(&mut tensors, "final_norm", &[hidden])?;
        let lm_head = take(&mut tensors, "lm_head", &[config.vocab_size, hidden])?;

        if let Some(role) = tensors.into_keys().next() {
            return Err(RuntimeError::UnexpectedTensor(role));
        }

        Ok(Self {
            config,
            instance_id: NEXT_MODEL_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
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

    pub fn forward_token(&self, state: &mut SequenceState, token: u32) -> Result<StepOutput> {
        let token_index = usize::try_from(token).map_err(|_| RuntimeError::InvalidToken {
            token,
            vocab_size: self.config.vocab_size,
        })?;
        if token_index >= self.config.vocab_size {
            return Err(RuntimeError::InvalidToken {
                token,
                vocab_size: self.config.vocab_size,
            });
        }
        if state.len() >= self.config.context_length {
            return Err(RuntimeError::ContextLimit {
                limit: self.config.context_length,
            });
        }
        if state
            .model_instance_id
            .is_some_and(|instance_id| instance_id != self.instance_id)
        {
            return Err(RuntimeError::StateMismatch);
        }

        let mut residual = self.weights.token_embedding.row(token_index).to_vec();
        let normalized = rms_norm(&residual, &self.weights.attn_norm)?;
        let query = linear(&self.weights.attn_q, &normalized);
        let key = linear(&self.weights.attn_k, &normalized);
        let value = linear(&self.weights.attn_v, &normalized);
        let mut pending_state = state.clone();
        pending_state.model_instance_id = Some(self.instance_id);
        pending_state.keys.push(key);
        pending_state.values.push(value);

        let attended = causal_attention(&query, &pending_state, self.config.num_heads)?;
        let attention_output = linear(&self.weights.attn_out, &attended);
        add_in_place(&mut residual, &attention_output);

        let normalized = rms_norm(&residual, &self.weights.ffn_norm)?;
        let scores = linear(&self.weights.router, &normalized);
        ensure_finite(&scores, "router")?;
        let expert_ids = stable_top_k(&scores, self.config.top_k);
        let selected_scores: Vec<f32> = expert_ids.iter().map(|id| scores[*id]).collect();
        let weights = softmax(&selected_scores)?;

        let mut mixture = vec![0.0_f32; self.config.hidden_size];
        for (expert_id, route_weight) in expert_ids.iter().zip(&weights) {
            let expert = &self.weights.experts[*expert_id];
            let gate = linear(&expert.gate, &normalized);
            let up = linear(&expert.up, &normalized);
            let activated: Vec<f32> = gate
                .iter()
                .zip(up)
                .map(|(gate, up)| silu(*gate) * up)
                .collect();
            let output = linear(&expert.down, &activated);
            for (destination, value) in mixture.iter_mut().zip(output) {
                *destination += *route_weight * value;
            }
        }
        add_in_place(&mut residual, &mixture);

        let normalized = rms_norm(&residual, &self.weights.final_norm)?;
        let logits = linear(&self.weights.lm_head, &normalized);
        ensure_finite(&logits, "lm head")?;
        *state = pending_state;

        Ok(StepOutput {
            input_token: token,
            logits,
            route: RouteDecision {
                scores,
                expert_ids,
                weights,
            },
        })
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
        let mut state = SequenceState::new();
        let mut outputs = Vec::with_capacity(tokens.len());
        for token in tokens {
            outputs.push(self.forward_token(&mut state, *token)?);
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

        let mut state = SequenceState::new();
        let mut steps = Vec::with_capacity(required);
        for token in prompt {
            steps.push(self.forward_token(&mut state, *token)?);
        }

        let mut generated_tokens = Vec::with_capacity(max_new_tokens);
        for index in 0..max_new_tokens {
            let next = greedy_token(&steps.last().expect("prompt is nonempty").logits);
            generated_tokens.push(next);
            if next == EOS_TOKEN || index + 1 == max_new_tokens {
                break;
            }
            steps.push(self.forward_token(&mut state, next)?);
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

fn expected_tensors(config: &TinyConfig) -> Vec<(String, Vec<usize>)> {
    let hidden = config.hidden_size;
    let expert_hidden = config.expert_hidden_size;
    let mut tensors = vec![
        ("token_embedding".into(), vec![config.vocab_size, hidden]),
        ("layers.0.attn_norm".into(), vec![hidden]),
        ("layers.0.attn_q".into(), vec![hidden, hidden]),
        ("layers.0.attn_k".into(), vec![hidden, hidden]),
        ("layers.0.attn_v".into(), vec![hidden, hidden]),
        ("layers.0.attn_out".into(), vec![hidden, hidden]),
        ("layers.0.ffn_norm".into(), vec![hidden]),
        ("layers.0.router".into(), vec![config.num_experts, hidden]),
    ];
    for expert in 0..config.num_experts {
        tensors.push((
            format!("layers.0.experts.{expert}.gate"),
            vec![expert_hidden, hidden],
        ));
        tensors.push((
            format!("layers.0.experts.{expert}.up"),
            vec![expert_hidden, hidden],
        ));
        tensors.push((
            format!("layers.0.experts.{expert}.down"),
            vec![hidden, expert_hidden],
        ));
    }
    tensors.push(("final_norm".into(), vec![hidden]));
    tensors.push(("lm_head".into(), vec![config.vocab_size, hidden]));
    tensors
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

fn greedy_token(logits: &[f32]) -> u32 {
    let mut best = 0_usize;
    for candidate in 1..logits.len() {
        if logits[candidate] > logits[best] {
            best = candidate;
        }
    }
    u32::try_from(best).expect("vocabulary fits u32")
}

fn causal_attention(query: &[f32], state: &SequenceState, num_heads: usize) -> Result<Vec<f32>> {
    let hidden = query.len();
    let head_size = hidden / num_heads;
    let scale = 1.0_f32 / (head_size as f32).sqrt();
    let mut output = vec![0.0_f32; hidden];

    for head in 0..num_heads {
        let start = head * head_size;
        let end = start + head_size;
        let mut scores = Vec::with_capacity(state.len());
        for key in &state.keys {
            let mut score = 0.0_f32;
            for index in start..end {
                score += query[index] * key[index];
            }
            scores.push(score * scale);
        }
        let probabilities = softmax(&scores)?;
        for (probability, value) in probabilities.iter().zip(&state.values) {
            for index in start..end {
                output[index] += *probability * value[index];
            }
        }
    }
    ensure_finite(&output, "attention")?;
    Ok(output)
}

fn rms_norm(input: &[f32], weight: &Tensor) -> Result<Vec<f32>> {
    let mut sum_squares = 0.0_f32;
    for value in input {
        sum_squares += value * value;
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

fn linear(weight: &Tensor, input: &[f32]) -> Vec<f32> {
    let rows = weight.shape()[0];
    let columns = weight.shape()[1];
    debug_assert_eq!(columns, input.len());
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut sum = 0.0_f32;
        for (value, coefficient) in input.iter().zip(weight.row(row)) {
            sum += coefficient * value;
        }
        output.push(sum);
    }
    output
}

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

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn add_in_place(destination: &mut [f32], source: &[f32]) {
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += source;
    }
}

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

#[derive(Debug)]
struct ExpertWeights {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
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
    experts: Vec<ExpertWeights>,
    final_norm: Tensor,
    lm_head: Tensor,
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn stable_top_k_uses_low_expert_id_for_ties() {
        assert_eq!(stable_top_k(&[1.0, 2.0, 2.0, -1.0], 2), [1, 2]);
        assert_eq!(stable_top_k(&[-0.0, 0.0, 0.0, -0.0], 4), [0, 1, 2, 3]);
        assert_eq!(stable_top_k(&[3.0, 2.0, 2.0, 2.0], 2), [0, 1]);
        assert_eq!(stable_top_k(&[1.0, 1.0, 1.0, 1.0], 2), [0, 1]);
    }

    #[test]
    fn greedy_uses_low_token_id_for_numeric_ties() {
        assert_eq!(greedy_token(&[-0.0, 0.0, 0.0, -0.0]), 0);
        assert_eq!(greedy_token(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn scalar_primitives_preserve_orientation_and_head_boundaries() {
        let matrix = Tensor::new("matrix", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(linear(&matrix, &[1.0, 10.0, 100.0]), [321.0, 654.0]);

        let norm_weight = Tensor::new("norm", vec![2], vec![2.0, 3.0]).unwrap();
        let normalized = rms_norm(&[1.0, 1.0], &norm_weight).unwrap();
        let inverse = (1.0_f32 + RMS_EPSILON).sqrt().recip();
        assert_eq!(normalized, [2.0 * inverse, 3.0 * inverse]);

        let state = SequenceState {
            model_instance_id: None,
            keys: vec![vec![0.0; 4]],
            values: vec![vec![1.0, 2.0, 3.0, 4.0]],
        };
        assert_eq!(
            causal_attention(&[7.0, 8.0, 9.0, 10.0], &state, 2).unwrap(),
            [1.0, 2.0, 3.0, 4.0]
        );
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
        for (from, to, expected) in [
            (
                "runnel.ascii32",
                "runnel.ascii31",
                "requires runnel.ascii32",
            ),
            ("\"hidden_size\":8", "\"hidden_size\":4", "frozen tiny-v1"),
            ("final_norm", "extra_norm", "must have role final_norm"),
            ("[32,8]", "[16,16]", "must have shape [32, 8]"),
        ] {
            let fixture = FixtureArtifact::build();
            let mut parts = fixture.to_parts();
            let manifest = String::from_utf8(parts.manifest).unwrap();
            let replaced = manifest.replacen(from, to, 1);
            assert_ne!(manifest, replaced);
            parts.manifest = replaced.into_bytes();
            let artifact = Artifact::from_bytes(parts, Limits::default()).unwrap();
            let error = TinyModel::from_artifact(&artifact).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
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
            if recipe.id == 2 {
                values.fill(f32::MAX);
            }
            let tensor = Tensor::new(&recipe.role, recipe.shape, values).unwrap();
            catalog.insert(recipe.role, tensor);
        }
        let model = TinyModel::from_catalog(TinyConfig::reference(), catalog).unwrap();
        let mut state = SequenceState::new();
        assert!(model.forward_token(&mut state, 1).is_err());
        assert!(state.is_empty());
    }
}
