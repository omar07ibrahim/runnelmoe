//! Deterministic greedy and seeded sampling primitives.
//!
//! Sampling is deliberately a preview operation. It returns the RNG state that
//! a surrounding token transaction may commit, but never mutates committed RNG
//! state itself.

use std::{cmp::Ordering, fmt, mem::size_of};

use thiserror::Error;

const SPLITMIX64_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const SPLITMIX64_MIX_1: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX64_MIX_2: u64 = 0x94d0_49bb_1331_11eb;
const TWO_POW_NEGATIVE_53: f64 = 1.0 / 9_007_199_254_740_992.0;
const SAMPLING_LEDGER_ALIGNMENT: usize = 64;
const CANDIDATE_SLOT_BYTES: usize = 32;

pub type SamplingResult<T> = std::result::Result<T, SamplingError>;

/// Errors produced before a sampling preview can be published.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SamplingError {
    #[error("sampling vocabulary must be nonempty")]
    EmptyVocabulary,
    #[error("sampling logits must be nonempty")]
    EmptyLogits,
    #[error("vocabulary size {vocab_size} cannot be represented by u32 token IDs")]
    VocabularyTooLarge { vocab_size: usize },
    #[error("sampling workspace has vocabulary size {workspace}, but logits have length {logits}")]
    WorkspaceSizeMismatch { workspace: usize, logits: usize },
    #[error("sampling logit is not finite")]
    NonFiniteLogit,
    #[error("sampling temperature must be finite and greater than zero")]
    InvalidTemperature,
    #[error("top-k {top_k} is outside 1..={vocab_size}")]
    InvalidTopK { top_k: usize, vocab_size: usize },
    #[error("top-p must be finite and in (0, 1]")]
    InvalidTopP,
    #[error("greedy sampling cannot observe an RNG state")]
    UnexpectedGreedyRngState,
    #[error("sampling workspace allocation failed for {vocab_size} candidates")]
    WorkspaceAllocation { vocab_size: usize },
    #[error("sampling workspace size overflows for vocabulary {vocab_size}")]
    WorkspaceSizeOverflow { vocab_size: usize },
    #[error("non-finite or invalid sampling arithmetic during {stage}")]
    InvalidArithmetic { stage: &'static str },
}

/// Parameters for version-1 SplitMix64 sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleConfig {
    pub seed: u64,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl SampleConfig {
    /// Validates parameters independently so admission can reject bad requests
    /// before allocating per-request scheduler state.
    pub fn validate(self, vocab_size: usize) -> SamplingResult<()> {
        validate_vocab_size(vocab_size)?;
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(SamplingError::InvalidTemperature);
        }
        if self.top_k == 0 || self.top_k > vocab_size {
            return Err(SamplingError::InvalidTopK {
                top_k: self.top_k,
                vocab_size,
            });
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(SamplingError::InvalidTopP);
        }
        Ok(())
    }
}

/// Token selection policy for one request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingPolicy {
    Greedy,
    Sample(SampleConfig),
}

impl SamplingPolicy {
    /// Validates policy parameters against the adapter vocabulary.
    pub fn validate(self, vocab_size: usize) -> SamplingResult<()> {
        match self {
            Self::Greedy => validate_vocab_size(vocab_size),
            Self::Sample(config) => config.validate(vocab_size),
        }
    }
}

/// A compact token-selection result suitable for a transaction permit.
///
/// `observed_rng_state == None` means that no sampled token has committed yet;
/// the configured seed was used. A successful sampled commit publishes
/// `next_rng_state`. Greedy previews always carry `None` in both fields.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SamplingPreview {
    token: u32,
    observed_rng_state_value: u64,
    next_rng_state_value: u64,
    rng_state_flags: u8,
}

impl fmt::Debug for SamplingPreview {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SamplingPreview")
            .field("token", &"<redacted>")
            .field("rng_state", &"<redacted>")
            .finish()
    }
}

impl SamplingPreview {
    const HAS_OBSERVED_STATE: u8 = 1 << 0;
    const HAS_NEXT_STATE: u8 = 1 << 1;

    fn greedy(token: u32) -> Self {
        Self {
            token,
            observed_rng_state_value: 0,
            next_rng_state_value: 0,
            rng_state_flags: 0,
        }
    }

    fn sampled(token: u32, observed_rng_state: Option<u64>, next_rng_state: u64) -> Self {
        Self {
            token,
            observed_rng_state_value: observed_rng_state.unwrap_or(0),
            next_rng_state_value: next_rng_state,
            rng_state_flags: Self::HAS_NEXT_STATE
                | (u8::from(observed_rng_state.is_some()) * Self::HAS_OBSERVED_STATE),
        }
    }

    /// Returns the token bound to this preview's RNG transition.
    #[must_use]
    pub fn token(self) -> u32 {
        self.token
    }

    #[must_use]
    pub fn observed_rng_state(self) -> Option<u64> {
        (self.rng_state_flags & Self::HAS_OBSERVED_STATE != 0)
            .then_some(self.observed_rng_state_value)
    }

    #[must_use]
    pub fn next_rng_state(self) -> Option<u64> {
        (self.rng_state_flags & Self::HAS_NEXT_STATE != 0).then_some(self.next_rng_state_value)
    }
}

/// One retained candidate exposed without allocating a diagnostics vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetainedCandidate {
    pub token: u32,
    pub probability: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Candidate {
    token: u32,
    reserved: u32,
    scaled_logit: f64,
    weight: f64,
    probability: f64,
}

const _: [(); CANDIDATE_SLOT_BYTES] = [(); size_of::<Candidate>()];

impl Default for Candidate {
    fn default() -> Self {
        Self {
            token: 0,
            reserved: 0,
            scaled_logit: 0.0,
            weight: 0.0,
            probability: 0.0,
        }
    }
}

/// Checked candidate-storage geometry for one sampling workspace.
///
/// `payload_bytes` is the semantic byte capacity of the fixed candidate
/// array. `charge_bytes` rounds that complete payload once to the scheduler's
/// 64-byte logical-ledger unit; it is not an allocator or RSS estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplingWorkspaceLayout {
    vocab_size: usize,
    payload_bytes: usize,
    charge_bytes: usize,
}

impl SamplingWorkspaceLayout {
    #[must_use]
    pub fn vocab_size(self) -> usize {
        self.vocab_size
    }

    #[must_use]
    pub fn payload_bytes(self) -> usize {
        self.payload_bytes
    }

    #[must_use]
    pub fn charge_bytes(self) -> usize {
        self.charge_bytes
    }
}

/// Reusable, fallibly allocated scratch storage for one fixed vocabulary.
///
/// Construction creates every candidate slot. `preview` sorts and overwrites
/// those slots in place and never reserves, grows, or otherwise heap-allocates.
pub struct SamplingWorkspace {
    candidates: Vec<Candidate>,
    retained_len: usize,
}

impl fmt::Debug for SamplingWorkspace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SamplingWorkspace")
            .field("vocab_size", &self.candidates.len())
            .field("retained_len", &self.retained_len)
            .field("candidate_payload", &"<redacted>")
            .finish()
    }
}

impl SamplingWorkspace {
    /// Computes fixed candidate storage and logical charge without allocating.
    pub fn layout(vocab_size: usize) -> SamplingResult<SamplingWorkspaceLayout> {
        validate_vocab_size(vocab_size)?;
        let payload_bytes = vocab_size
            .checked_mul(CANDIDATE_SLOT_BYTES)
            .ok_or(SamplingError::WorkspaceSizeOverflow { vocab_size })?;
        if !workspace_payload_fits_host(payload_bytes) {
            return Err(SamplingError::WorkspaceSizeOverflow { vocab_size });
        }
        let charge_bytes = round_workspace_charge(payload_bytes)
            .ok_or(SamplingError::WorkspaceSizeOverflow { vocab_size })?;
        Ok(SamplingWorkspaceLayout {
            vocab_size,
            payload_bytes,
            charge_bytes,
        })
    }

    /// Allocates all sampling scratch space for `vocab_size` candidates.
    pub fn new(vocab_size: usize) -> SamplingResult<Self> {
        let layout = Self::layout(vocab_size)?;
        let mut candidates = Vec::new();
        candidates
            .try_reserve_exact(layout.vocab_size())
            .map_err(|_| SamplingError::WorkspaceAllocation { vocab_size })?;
        candidates.resize(layout.vocab_size(), Candidate::default());
        Ok(Self {
            candidates,
            retained_len: 0,
        })
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.candidates.len()
    }

    #[must_use]
    pub fn retained_len(&self) -> usize {
        self.retained_len
    }

    /// Iterates retained candidates in exact categorical visit order.
    pub fn retained_candidates(&self) -> impl ExactSizeIterator<Item = RetainedCandidate> + '_ {
        self.candidates[..self.retained_len]
            .iter()
            .map(|candidate| RetainedCandidate {
                token: candidate.token,
                probability: candidate.probability,
            })
    }

    /// Computes a token and optional RNG-state transition without mutating any
    /// committed request state.
    ///
    /// For sampled policy, `prior_rng_state == None` denotes the state before
    /// the first token commit and uses `SampleConfig::seed`. Passing `Some(s)`
    /// previews the transition after the already-committed state `s`.
    pub fn preview(
        &mut self,
        logits: &[f32],
        policy: SamplingPolicy,
        prior_rng_state: Option<u64>,
    ) -> SamplingResult<SamplingPreview> {
        self.retained_len = 0;
        validate_logits(logits, self.candidates.len())?;
        policy.validate(logits.len())?;

        match policy {
            SamplingPolicy::Greedy => self.preview_greedy(logits, prior_rng_state),
            SamplingPolicy::Sample(config) => self.preview_sample(logits, config, prior_rng_state),
        }
    }

    fn preview_greedy(
        &mut self,
        logits: &[f32],
        prior_rng_state: Option<u64>,
    ) -> SamplingResult<SamplingPreview> {
        if prior_rng_state.is_some() {
            return Err(SamplingError::UnexpectedGreedyRngState);
        }

        let mut best = 0_usize;
        for candidate in 1..logits.len() {
            if logits[candidate] > logits[best] {
                best = candidate;
            }
        }
        let token = token_id(best)?;
        self.candidates[0] = Candidate {
            token,
            reserved: 0,
            scaled_logit: f64::from(logits[best]),
            weight: 1.0,
            probability: 1.0,
        };
        self.retained_len = 1;
        Ok(SamplingPreview::greedy(token))
    }

    fn preview_sample(
        &mut self,
        logits: &[f32],
        config: SampleConfig,
        prior_rng_state: Option<u64>,
    ) -> SamplingResult<SamplingPreview> {
        let temperature = f64::from(config.temperature);
        for (index, (&logit, candidate)) in
            logits.iter().zip(self.candidates.iter_mut()).enumerate()
        {
            let scaled_logit = f64::from(logit) / temperature;
            if !scaled_logit.is_finite() {
                return Err(SamplingError::InvalidArithmetic {
                    stage: "scaled logit",
                });
            }
            *candidate = Candidate {
                token: token_id(index)?,
                reserved: 0,
                scaled_logit,
                weight: 0.0,
                probability: 0.0,
            };
        }

        self.candidates.sort_unstable_by(|left, right| {
            right
                .scaled_logit
                .partial_cmp(&left.scaled_logit)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.token.cmp(&right.token))
        });

        let candidates = &mut self.candidates[..config.top_k];
        let maximum = candidates[0].scaled_logit;
        let mut total_weight = 0.0_f64;
        for candidate in candidates.iter_mut() {
            candidate.weight = (candidate.scaled_logit - maximum).exp();
            if !candidate.weight.is_finite() {
                return Err(SamplingError::InvalidArithmetic {
                    stage: "exponential weight",
                });
            }
            total_weight += candidate.weight;
            if !total_weight.is_finite() {
                return Err(SamplingError::InvalidArithmetic {
                    stage: "total weight",
                });
            }
        }
        if total_weight <= 0.0 {
            return Err(SamplingError::InvalidArithmetic {
                stage: "total weight",
            });
        }

        let threshold = f64::from(config.top_p) * total_weight;
        if !threshold.is_finite() {
            return Err(SamplingError::InvalidArithmetic {
                stage: "top-p threshold",
            });
        }
        let mut cumulative_weight = 0.0_f64;
        let mut retained_len = candidates.len();
        for (index, candidate) in candidates.iter().enumerate() {
            cumulative_weight += candidate.weight;
            if !cumulative_weight.is_finite() {
                return Err(SamplingError::InvalidArithmetic {
                    stage: "top-p accumulation",
                });
            }
            if cumulative_weight >= threshold {
                retained_len = index + 1;
                break;
            }
        }

        let retained = &mut candidates[..retained_len];
        let mut retained_sum = 0.0_f64;
        for candidate in retained.iter() {
            retained_sum += candidate.weight;
            if !retained_sum.is_finite() {
                return Err(SamplingError::InvalidArithmetic {
                    stage: "retained weight",
                });
            }
        }
        if retained_sum <= 0.0 {
            return Err(SamplingError::InvalidArithmetic {
                stage: "retained weight",
            });
        }
        for candidate in retained.iter_mut() {
            candidate.probability = candidate.weight / retained_sum;
            if !candidate.probability.is_finite() {
                return Err(SamplingError::InvalidArithmetic {
                    stage: "retained probability",
                });
            }
        }

        let rng_input = prior_rng_state.unwrap_or(config.seed);
        let (next_rng_state, _, unit) = splitmix64_step(rng_input);
        let (token, _) = select_categorical(retained, unit)?;

        self.retained_len = retained_len;
        Ok(SamplingPreview::sampled(
            token,
            prior_rng_state,
            next_rng_state,
        ))
    }
}

fn round_workspace_charge(payload_bytes: usize) -> Option<usize> {
    payload_bytes
        .checked_add(SAMPLING_LEDGER_ALIGNMENT - 1)
        .map(|bytes| bytes / SAMPLING_LEDGER_ALIGNMENT * SAMPLING_LEDGER_ALIGNMENT)
}

fn workspace_payload_fits_host(payload_bytes: usize) -> bool {
    payload_bytes <= isize::MAX as usize
}

/// Selects from already-normalized candidates in visit order.
///
/// The boolean reports whether floating-point rounding required the specified
/// last-candidate fallback. The scan is deliberately strict (`cumulative > u`)
/// and allocation-free.
fn select_categorical(candidates: &[Candidate], unit: f64) -> SamplingResult<(u32, bool)> {
    let fallback = candidates
        .last()
        .ok_or(SamplingError::InvalidArithmetic {
            stage: "categorical candidates",
        })?
        .token;
    let mut cumulative_probability = 0.0_f64;
    for candidate in candidates {
        cumulative_probability += candidate.probability;
        if !cumulative_probability.is_finite() {
            return Err(SamplingError::InvalidArithmetic {
                stage: "categorical accumulation",
            });
        }
        if cumulative_probability > unit {
            return Ok((candidate.token, false));
        }
    }
    Ok((fallback, true))
}

fn validate_vocab_size(vocab_size: usize) -> SamplingResult<()> {
    if vocab_size == 0 {
        return Err(SamplingError::EmptyVocabulary);
    }
    if u32::try_from(vocab_size - 1).is_err() {
        return Err(SamplingError::VocabularyTooLarge { vocab_size });
    }
    Ok(())
}

fn validate_logits(logits: &[f32], workspace_size: usize) -> SamplingResult<()> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if logits.len() != workspace_size {
        return Err(SamplingError::WorkspaceSizeMismatch {
            workspace: workspace_size,
            logits: logits.len(),
        });
    }
    for logit in logits {
        if !logit.is_finite() {
            return Err(SamplingError::NonFiniteLogit);
        }
    }
    Ok(())
}

fn token_id(index: usize) -> SamplingResult<u32> {
    u32::try_from(index).map_err(|_| SamplingError::VocabularyTooLarge {
        vocab_size: index.saturating_add(1),
    })
}

fn splitmix64_step(state: u64) -> (u64, u64, f64) {
    let next_state = state.wrapping_add(SPLITMIX64_GAMMA);
    let mut word = next_state;
    word = (word ^ (word >> 30)).wrapping_mul(SPLITMIX64_MIX_1);
    word = (word ^ (word >> 27)).wrapping_mul(SPLITMIX64_MIX_2);
    word ^= word >> 31;
    let unit = ((word >> 11) as f64) * TWO_POW_NEGATIVE_53;
    (next_state, word, unit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    const ZERO_WORD_STATE: u64 = 0x61c8_8646_80b5_83eb;
    const HALF_WORD_STATE: u64 = 0x2fed_f1ef_ce1d_5545;
    const MAX_WORD_STATE: u64 = 0x3162_8af6_7b21_31ab;
    const PYTHON_SAMPLING_VECTORS: &str =
        include_str!("../../../fixtures/scheduler/sampling-v1.json");
    const DIAGNOSTIC_MAX_ULPS: u64 = 4;

    #[derive(Debug, Deserialize)]
    struct SamplingVectors {
        rng_cases: Vec<RngVector>,
        sample_cases: Vec<SampleVector>,
        invalid_cases: Vec<InvalidVector>,
        categorical_cases: Vec<CategoricalVector>,
    }

    #[derive(Debug, Deserialize)]
    struct RngVector {
        id: String,
        state_before_hex: String,
        state_after_hex: String,
        word_hex: String,
        unit_f64_hex: String,
    }

    #[derive(Debug, Deserialize)]
    struct SampleVector {
        id: String,
        input: SampleInput,
        output: SampleOutput,
    }

    #[derive(Debug, Deserialize)]
    struct SampleInput {
        logit_f32_bits: Vec<String>,
        state_before_hex: String,
        temperature_f32_bits: String,
        top_k: usize,
        top_p_f32_bits: String,
    }

    #[derive(Debug, Deserialize)]
    struct SampleOutput {
        categorical_cumulative_f64_hex: Vec<String>,
        chosen_token_id: u32,
        ordered_token_ids: Vec<u32>,
        random_unit_f64_hex: String,
        random_word_hex: String,
        retained_sum_f64_hex: String,
        retained_token_ids: Vec<u32>,
        scaled_logits_f64_hex: Vec<String>,
        softmax_total_f64_hex: String,
        softmax_weights_f64_hex: Vec<String>,
        state_after_hex: String,
        top_k_token_ids: Vec<u32>,
        top_p_threshold_f64_hex: String,
        used_rounding_fallback: bool,
    }

    #[derive(Debug, Deserialize)]
    struct InvalidVector {
        id: String,
        expected_error: String,
        input: SampleInput,
    }

    #[derive(Debug, Deserialize)]
    struct CategoricalVector {
        id: String,
        input: CategoricalInput,
        output: CategoricalOutput,
    }

    #[derive(Debug, Deserialize)]
    struct CategoricalInput {
        candidates: Vec<CategoricalCandidate>,
        unit_f64_hex: String,
    }

    #[derive(Debug, Deserialize)]
    struct CategoricalCandidate {
        token_id: u32,
        weight_f64_hex: String,
    }

    #[derive(Debug, Deserialize)]
    struct CategoricalOutput {
        chosen_token_id: u32,
        cumulative_f64_hex: Vec<String>,
        used_rounding_fallback: bool,
    }

    fn committed_vectors() -> SamplingVectors {
        serde_json::from_str(PYTHON_SAMPLING_VECTORS).expect("committed sampling vectors are JSON")
    }

    fn parse_hex_u64(text: &str) -> u64 {
        u64::from_str_radix(
            text.strip_prefix("0x").expect("hex integer has 0x prefix"),
            16,
        )
        .expect("hex integer fits u64")
    }

    fn parse_hex_u32(text: &str) -> u32 {
        u32::from_str_radix(
            text.strip_prefix("0x").expect("hex integer has 0x prefix"),
            16,
        )
        .expect("hex integer fits u32")
    }

    /// Parses the finite canonical strings emitted by Python `float.hex()`.
    /// Constructing IEEE-754 bits directly avoids decimal or libm round trips.
    fn parse_python_f64_hex(text: &str) -> Result<f64, String> {
        let (negative, magnitude) = text
            .strip_prefix('-')
            .map_or((false, text), |rest| (true, rest));
        let magnitude = magnitude
            .strip_prefix("0x")
            .ok_or_else(|| format!("missing 0x prefix in {text}"))?;
        let (significand, exponent_text) = magnitude
            .split_once('p')
            .ok_or_else(|| format!("missing binary exponent in {text}"))?;
        let exponent = exponent_text
            .parse::<i32>()
            .map_err(|error| format!("invalid exponent in {text}: {error}"))?;
        let (leading_text, fraction_text) = significand
            .split_once('.')
            .ok_or_else(|| format!("missing radix point in {text}"))?;
        if fraction_text.is_empty() || fraction_text.len() > 13 {
            return Err(format!("non-canonical fraction width in {text}"));
        }
        let leading = u8::from_str_radix(leading_text, 16)
            .map_err(|error| format!("invalid leading digit in {text}: {error}"))?;
        let mut fraction = u64::from_str_radix(fraction_text, 16)
            .map_err(|error| format!("invalid fraction in {text}: {error}"))?;
        fraction <<= 4 * (13 - fraction_text.len());
        let sign = u64::from(negative) << 63;

        let bits = match leading {
            0 if fraction == 0 => sign,
            0 if exponent == -1022 => sign | fraction,
            0 => return Err(format!("non-canonical subnormal exponent in {text}")),
            1 if (-1022..=1023).contains(&exponent) => {
                sign | ((exponent + 1023) as u64) << 52 | fraction
            }
            1 => return Err(format!("normal exponent outside f64 range in {text}")),
            _ => return Err(format!("non-canonical leading digit in {text}")),
        };
        Ok(f64::from_bits(bits))
    }

    fn ordered_f64_bits(value: f64) -> u64 {
        let bits = value.to_bits();
        if bits >> 63 == 0 {
            bits | (1_u64 << 63)
        } else {
            !bits
        }
    }

    fn assert_diagnostic_close(actual: f64, expected_hex: &str, label: &str) {
        let expected = parse_python_f64_hex(expected_hex).expect("valid Python f64 hex vector");
        assert!(actual.is_finite(), "{label}: Rust value is not finite");
        assert!(expected.is_finite(), "{label}: Python vector is not finite");
        let distance = ordered_f64_bits(actual).abs_diff(ordered_f64_bits(expected));
        assert!(
            distance <= DIAGNOSTIC_MAX_ULPS,
            "{label}: Rust {actual:?} differs from Python {expected:?} by {distance} ULPs (limit {DIAGNOSTIC_MAX_ULPS})"
        );
    }

    fn assert_diagnostic_slice(actual: &[f64], expected_hex: &[String], label: &str) {
        assert_eq!(actual.len(), expected_hex.len(), "{label}: length mismatch");
        for (index, (actual, expected)) in actual.iter().zip(expected_hex).enumerate() {
            assert_diagnostic_close(*actual, expected, &format!("{label}[{index}]"));
        }
    }

    fn sample_config(seed: u64, top_k: usize, top_p: f32) -> SampleConfig {
        SampleConfig {
            seed,
            temperature: 1.0,
            top_k,
            top_p,
        }
    }

    fn retained_tokens(workspace: &SamplingWorkspace) -> Vec<u32> {
        workspace
            .retained_candidates()
            .map(|candidate| candidate.token)
            .collect()
    }

    #[test]
    fn debug_redacts_preview_rng_token_and_workspace_candidates() {
        let preview = SamplingPreview::sampled(27, Some(123_456_789), 987_654_321);
        let preview_debug = format!("{preview:?}");
        assert!(preview_debug.contains("<redacted>"));
        assert!(!preview_debug.contains("27"));
        assert!(!preview_debug.contains("123456789"));
        assert!(!preview_debug.contains("987654321"));

        let mut workspace = SamplingWorkspace::new(4).unwrap();
        workspace
            .preview(
                &[12_345.0, 23_456.0, 34_567.0, 45_678.0],
                SamplingPolicy::Greedy,
                None,
            )
            .unwrap();
        let workspace_debug = format!("{workspace:?}");
        assert!(workspace_debug.contains("vocab_size: 4"));
        assert!(workspace_debug.contains("<redacted>"));
        for sentinel in ["12345", "23456", "34567", "45678"] {
            assert!(!workspace_debug.contains(sentinel));
        }
    }

    #[test]
    fn workspace_layout_reports_exact_payload_and_64_byte_charge() {
        assert_eq!(size_of::<Candidate>(), CANDIDATE_SLOT_BYTES);
        let cases = [
            (1, 32, 64),
            (2, 64, 64),
            (3, 96, 128),
            (4, 128, 128),
            (5, 160, 192),
        ];

        for (vocab_size, payload_bytes, charge_bytes) in cases {
            let layout = SamplingWorkspace::layout(vocab_size).unwrap();
            assert_eq!(layout.vocab_size(), vocab_size);
            assert_eq!(layout.payload_bytes(), payload_bytes);
            assert_eq!(layout.charge_bytes(), charge_bytes);

            let workspace = SamplingWorkspace::new(vocab_size).unwrap();
            assert_eq!(workspace.vocab_size(), layout.vocab_size());
            assert_eq!(
                workspace.candidates.len() * size_of::<Candidate>(),
                layout.payload_bytes()
            );
        }
    }

    #[test]
    fn workspace_layout_rejects_invalid_vocabularies_without_payload_details() {
        assert_eq!(
            SamplingWorkspace::layout(0).unwrap_err(),
            SamplingError::EmptyVocabulary
        );

        if let Ok(too_large) = usize::try_from(u64::from(u32::MAX) + 2) {
            let error = SamplingWorkspace::layout(too_large).unwrap_err();
            assert_eq!(
                error,
                SamplingError::VocabularyTooLarge {
                    vocab_size: too_large
                }
            );
            let debug = format!("{error:?}");
            assert!(!debug.contains("candidate"));
            assert!(!debug.contains("payload"));
        }

        assert_eq!(round_workspace_charge(usize::MAX), None);
        assert!(!workspace_payload_fits_host(isize::MAX as usize + 1));
    }

    #[test]
    fn splitmix64_matches_exact_version_one_vectors() {
        let vectors = [
            (
                0,
                0x9e37_79b9_7f4a_7c15,
                0xe220_a839_7b1d_cdaf,
                0x3fec_4415_072f_63b9,
            ),
            (
                0x9e37_79b9_7f4a_7c15,
                0x3c6e_f372_fe94_f82a,
                0x6e78_9e6a_a1b9_65f4,
                0x3fdb_9e27_9aa8_6e58,
            ),
            (ZERO_WORD_STATE, 0, 0, 0),
            (
                HALF_WORD_STATE,
                0xce25_6ba9_4d67_d15a,
                0x8000_0000_0000_0000,
                0x3fe0_0000_0000_0000,
            ),
            (
                MAX_WORD_STATE,
                0xcf9a_04af_fa6b_adc0,
                u64::MAX,
                0x3fef_ffff_ffff_ffff,
            ),
        ];

        for (state, expected_state, expected_word, expected_unit_bits) in vectors {
            let (next_state, word, unit) = splitmix64_step(state);
            assert_eq!(next_state, expected_state);
            assert_eq!(word, expected_word);
            assert_eq!(unit.to_bits(), expected_unit_bits);
            assert!((0.0..1.0).contains(&unit));
        }
    }

    #[test]
    fn splitmix64_matches_committed_python_vectors() {
        let vectors = committed_vectors();
        assert_eq!(
            vectors.rng_cases.len(),
            4,
            "all RNG vectors must be exercised"
        );

        for case in vectors.rng_cases {
            let state_before = parse_hex_u64(&case.state_before_hex);
            let (state_after, word, unit) = splitmix64_step(state_before);
            assert_eq!(
                state_after,
                parse_hex_u64(&case.state_after_hex),
                "{}: next state",
                case.id
            );
            assert_eq!(word, parse_hex_u64(&case.word_hex), "{}: word", case.id);
            assert_diagnostic_close(unit, &case.unit_f64_hex, &format!("{}: unit", case.id));
        }
    }

    #[test]
    fn sampling_matches_all_committed_python_vectors() {
        let vectors = committed_vectors();
        assert_eq!(
            vectors.sample_cases.len(),
            7,
            "all sampling vectors must be exercised"
        );

        for case in vectors.sample_cases {
            let logits: Vec<f32> = case
                .input
                .logit_f32_bits
                .iter()
                .map(|bits| f32::from_bits(parse_hex_u32(bits)))
                .collect();
            let state_before = parse_hex_u64(&case.input.state_before_hex);
            let config = SampleConfig {
                seed: state_before,
                temperature: f32::from_bits(parse_hex_u32(&case.input.temperature_f32_bits)),
                top_k: case.input.top_k,
                top_p: f32::from_bits(parse_hex_u32(&case.input.top_p_f32_bits)),
            };
            let mut workspace = SamplingWorkspace::new(logits.len()).unwrap();
            let preview = workspace
                .preview(&logits, SamplingPolicy::Sample(config), None)
                .unwrap_or_else(|error| panic!("{}: Rust rejected valid vector: {error}", case.id));

            assert_eq!(preview.token(), case.output.chosen_token_id, "{}", case.id);
            assert_eq!(preview.observed_rng_state(), None, "{}", case.id);
            assert_eq!(
                preview.next_rng_state(),
                Some(parse_hex_u64(&case.output.state_after_hex)),
                "{}: next state",
                case.id
            );

            let ordered_ids: Vec<u32> = workspace
                .candidates
                .iter()
                .map(|candidate| candidate.token)
                .collect();
            assert_eq!(
                ordered_ids, case.output.ordered_token_ids,
                "{}: complete order",
                case.id
            );
            let top_k = &workspace.candidates[..config.top_k];
            let top_k_ids: Vec<u32> = top_k.iter().map(|candidate| candidate.token).collect();
            assert_eq!(
                top_k_ids, case.output.top_k_token_ids,
                "{}: top-k order",
                case.id
            );
            assert_eq!(
                retained_tokens(&workspace),
                case.output.retained_token_ids,
                "{}: retained order",
                case.id
            );

            let scaled_logits: Vec<f64> = top_k
                .iter()
                .map(|candidate| candidate.scaled_logit)
                .collect();
            let weights: Vec<f64> = top_k.iter().map(|candidate| candidate.weight).collect();
            assert_diagnostic_slice(
                &scaled_logits,
                &case.output.scaled_logits_f64_hex,
                &format!("{}: scaled logits", case.id),
            );
            assert_diagnostic_slice(
                &weights,
                &case.output.softmax_weights_f64_hex,
                &format!("{}: softmax weights", case.id),
            );

            let softmax_total = top_k
                .iter()
                .fold(0.0_f64, |sum, candidate| sum + candidate.weight);
            assert_diagnostic_close(
                softmax_total,
                &case.output.softmax_total_f64_hex,
                &format!("{}: softmax total", case.id),
            );
            assert_diagnostic_close(
                f64::from(config.top_p) * softmax_total,
                &case.output.top_p_threshold_f64_hex,
                &format!("{}: top-p threshold", case.id),
            );

            let retained = &workspace.candidates[..workspace.retained_len];
            let retained_sum = retained
                .iter()
                .fold(0.0_f64, |sum, candidate| sum + candidate.weight);
            assert_diagnostic_close(
                retained_sum,
                &case.output.retained_sum_f64_hex,
                &format!("{}: retained sum", case.id),
            );
            let mut cumulative = 0.0_f64;
            let cumulative: Vec<f64> = retained
                .iter()
                .map(|candidate| {
                    cumulative += candidate.probability;
                    cumulative
                })
                .collect();
            assert_diagnostic_slice(
                &cumulative,
                &case.output.categorical_cumulative_f64_hex,
                &format!("{}: categorical cumulative", case.id),
            );

            let (_, expected_word, unit) = splitmix64_step(state_before);
            assert_eq!(
                expected_word,
                parse_hex_u64(&case.output.random_word_hex),
                "{}: random word",
                case.id
            );
            assert_diagnostic_close(
                unit,
                &case.output.random_unit_f64_hex,
                &format!("{}: random unit", case.id),
            );
            let (selected_again, used_fallback) = select_categorical(retained, unit).unwrap();
            assert_eq!(selected_again, case.output.chosen_token_id, "{}", case.id);
            assert_eq!(
                used_fallback, case.output.used_rounding_fallback,
                "{}: rounding fallback",
                case.id
            );
        }
    }

    #[test]
    fn rust_rejects_all_committed_invalid_python_vectors() {
        let vectors = committed_vectors();
        assert_eq!(
            vectors.invalid_cases.len(),
            14,
            "all invalid vectors must be exercised"
        );

        for case in vectors.invalid_cases {
            assert_eq!(case.expected_error, "invalid_request", "{}", case.id);
            let logits: Vec<f32> = case
                .input
                .logit_f32_bits
                .iter()
                .map(|bits| f32::from_bits(parse_hex_u32(bits)))
                .collect();
            let config = SampleConfig {
                seed: parse_hex_u64(&case.input.state_before_hex),
                temperature: f32::from_bits(parse_hex_u32(&case.input.temperature_f32_bits)),
                top_k: case.input.top_k,
                top_p: f32::from_bits(parse_hex_u32(&case.input.top_p_f32_bits)),
            };
            let mut workspace = SamplingWorkspace::new(logits.len().max(1)).unwrap();
            assert!(
                workspace
                    .preview(&logits, SamplingPolicy::Sample(config), None)
                    .is_err(),
                "{}: Rust accepted an invalid Python vector",
                case.id
            );
        }
    }

    #[test]
    fn categorical_scan_matches_all_committed_boundary_vectors() {
        let vectors = committed_vectors();
        assert_eq!(
            vectors.categorical_cases.len(),
            5,
            "all categorical boundary vectors must be exercised"
        );

        for case in vectors.categorical_cases {
            let weights: Vec<(u32, f64)> = case
                .input
                .candidates
                .iter()
                .map(|candidate| {
                    (
                        candidate.token_id,
                        parse_python_f64_hex(&candidate.weight_f64_hex)
                            .expect("valid Python candidate weight"),
                    )
                })
                .collect();
            let total = weights
                .iter()
                .fold(0.0_f64, |sum, (_, weight)| sum + weight);
            let candidates: Vec<Candidate> = weights
                .iter()
                .map(|(token, weight)| Candidate {
                    token: *token,
                    reserved: 0,
                    scaled_logit: 0.0,
                    weight: *weight,
                    probability: *weight / total,
                })
                .collect();
            let unit = parse_python_f64_hex(&case.input.unit_f64_hex)
                .expect("valid Python categorical unit");
            let (token, used_fallback) = select_categorical(&candidates, unit).unwrap();
            assert_eq!(token, case.output.chosen_token_id, "{}", case.id);
            assert_eq!(
                used_fallback, case.output.used_rounding_fallback,
                "{}: rounding fallback",
                case.id
            );

            let mut cumulative = 0.0_f64;
            let cumulative: Vec<f64> = candidates
                .iter()
                .map(|candidate| {
                    cumulative += candidate.probability;
                    cumulative
                })
                .collect();
            assert_diagnostic_slice(
                &cumulative,
                &case.output.cumulative_f64_hex,
                &format!("{}: categorical cumulative", case.id),
            );
        }
    }

    #[test]
    fn greedy_chooses_lowest_id_on_ties_without_rng_transition() {
        let mut workspace = SamplingWorkspace::new(4).unwrap();
        let preview = workspace
            .preview(&[-0.0, 3.0, 3.0, 1.0], SamplingPolicy::Greedy, None)
            .unwrap();

        assert_eq!(preview.token(), 1);
        assert_eq!(preview.observed_rng_state(), None);
        assert_eq!(preview.next_rng_state(), None);
        assert_eq!(retained_tokens(&workspace), [1]);
        assert_eq!(
            workspace
                .preview(&[1.0; 4], SamplingPolicy::Greedy, Some(9))
                .unwrap_err(),
            SamplingError::UnexpectedGreedyRngState
        );
    }

    #[test]
    fn stable_ties_are_ordered_by_ascending_token_id() {
        let mut workspace = SamplingWorkspace::new(4).unwrap();
        let policy = SamplingPolicy::Sample(sample_config(ZERO_WORD_STATE, 4, 1.0));
        let preview = workspace.preview(&[2.0; 4], policy, None).unwrap();

        assert_eq!(retained_tokens(&workspace), [0, 1, 2, 3]);
        assert_eq!(preview.token(), 0);
        assert_eq!(
            workspace
                .retained_candidates()
                .map(|candidate| candidate.probability)
                .collect::<Vec<_>>(),
            [0.25; 4]
        );
    }

    #[test]
    fn top_k_one_always_selects_the_greatest_logit() {
        let mut workspace = SamplingWorkspace::new(4).unwrap();
        let policy = SamplingPolicy::Sample(sample_config(MAX_WORD_STATE, 1, f32::MIN_POSITIVE));
        let preview = workspace
            .preview(&[1.0, 9.0, 3.0, 9.0], policy, None)
            .unwrap();

        assert_eq!(preview.token(), 1);
        assert_eq!(retained_tokens(&workspace), [1]);
    }

    #[test]
    fn top_p_uses_the_shortest_prefix_at_f32_boundaries() {
        let below_half = f32::from_bits(0x3eff_ffff);
        let above_half = f32::from_bits(0x3f00_0001);
        let cases = [(below_half, 1), (0.5, 1), (above_half, 2)];
        let mut workspace = SamplingWorkspace::new(2).unwrap();

        for (top_p, expected_len) in cases {
            let policy = SamplingPolicy::Sample(sample_config(ZERO_WORD_STATE, 2, top_p));
            workspace.preview(&[0.0, 0.0], policy, None).unwrap();
            assert_eq!(workspace.retained_len(), expected_len, "top_p={top_p}");
        }
    }

    #[test]
    fn categorical_strictly_greater_handles_zero_and_one_adjacent_u() {
        let mut workspace = SamplingWorkspace::new(2).unwrap();
        let zero_policy = SamplingPolicy::Sample(sample_config(ZERO_WORD_STATE, 2, 1.0));
        let half_policy = SamplingPolicy::Sample(sample_config(HALF_WORD_STATE, 2, 1.0));
        let almost_one_policy = SamplingPolicy::Sample(sample_config(MAX_WORD_STATE, 2, 1.0));

        assert_eq!(
            workspace
                .preview(&[0.0, 0.0], zero_policy, None)
                .unwrap()
                .token(),
            0
        );
        assert_eq!(
            workspace
                .preview(&[0.0, 0.0], half_policy, None)
                .unwrap()
                .token(),
            1,
            "a cumulative probability equal to u must not win"
        );
        assert_eq!(
            workspace
                .preview(&[0.0, 0.0], almost_one_policy, None)
                .unwrap()
                .token(),
            1
        );
    }

    #[test]
    fn smallest_positive_temperature_and_underflow_are_well_defined() {
        let minimum = f32::from_bits(1);
        let mut workspace = SamplingWorkspace::new(2).unwrap();
        let minimum_policy = SamplingPolicy::Sample(SampleConfig {
            seed: ZERO_WORD_STATE,
            temperature: minimum,
            top_k: 2,
            top_p: 1.0,
        });
        assert_eq!(
            workspace
                .preview(&[0.0, minimum], minimum_policy, None)
                .unwrap()
                .token(),
            1
        );

        let underflow_policy = SamplingPolicy::Sample(sample_config(MAX_WORD_STATE, 2, 1.0));
        assert_eq!(
            workspace
                .preview(&[0.0, -1_000.0], underflow_policy, None)
                .unwrap()
                .token(),
            0
        );
        assert_eq!(retained_tokens(&workspace), [0]);
        assert_eq!(
            workspace.retained_candidates().next().unwrap().probability,
            1.0
        );
    }

    #[test]
    fn invalid_config_and_logits_are_rejected_before_sampling() {
        assert_eq!(
            SamplingWorkspace::new(0).unwrap_err(),
            SamplingError::EmptyVocabulary
        );
        let valid = sample_config(1, 2, 1.0);
        let invalid_temperatures = [0.0, -1.0, f32::NAN, f32::INFINITY];
        for temperature in invalid_temperatures {
            assert_eq!(
                SampleConfig {
                    temperature,
                    ..valid
                }
                .validate(2)
                .unwrap_err(),
                SamplingError::InvalidTemperature
            );
        }
        for top_k in [0, 3] {
            assert_eq!(
                SampleConfig { top_k, ..valid }.validate(2).unwrap_err(),
                SamplingError::InvalidTopK {
                    top_k,
                    vocab_size: 2,
                }
            );
        }
        for top_p in [
            0.0,
            -0.5,
            f32::from_bits(0x3f80_0001),
            f32::NAN,
            f32::INFINITY,
        ] {
            assert_eq!(
                SampleConfig { top_p, ..valid }.validate(2).unwrap_err(),
                SamplingError::InvalidTopP
            );
        }

        let policy = SamplingPolicy::Sample(valid);
        let mut workspace = SamplingWorkspace::new(2).unwrap();
        assert_eq!(
            workspace.preview(&[], policy, None).unwrap_err(),
            SamplingError::EmptyLogits
        );
        assert_eq!(
            workspace.preview(&[0.0], policy, None).unwrap_err(),
            SamplingError::WorkspaceSizeMismatch {
                workspace: 2,
                logits: 1,
            }
        );
        for bad in [
            [f32::NAN, 0.0],
            [0.0, f32::INFINITY],
            [0.0, f32::NEG_INFINITY],
        ] {
            assert_eq!(
                workspace.preview(&bad, policy, None).unwrap_err(),
                SamplingError::NonFiniteLogit
            );
            assert_eq!(workspace.retained_len(), 0);
        }
    }

    #[test]
    fn preview_records_stale_check_state_without_mutating_it() {
        let mut workspace = SamplingWorkspace::new(3).unwrap();
        let config = sample_config(0, 3, 1.0);
        let policy = SamplingPolicy::Sample(config);

        let initial = workspace.preview(&[1.0, 2.0, 3.0], policy, None).unwrap();
        let retry = workspace.preview(&[1.0, 2.0, 3.0], policy, None).unwrap();
        assert_eq!(initial, retry);
        assert_eq!(initial.observed_rng_state(), None);
        assert_eq!(initial.next_rng_state(), Some(SPLITMIX64_GAMMA));

        let explicit_zero = workspace
            .preview(&[1.0, 2.0, 3.0], policy, Some(0))
            .unwrap();
        assert_eq!(explicit_zero.observed_rng_state(), Some(0));
        assert_eq!(explicit_zero.next_rng_state(), Some(SPLITMIX64_GAMMA));

        let committed = initial.next_rng_state();
        let following = workspace
            .preview(&[1.0, 2.0, 3.0], policy, committed)
            .unwrap();
        assert_eq!(following.observed_rng_state(), committed);
        assert_eq!(following.next_rng_state(), Some(0x3c6e_f372_fe94_f82a));
        assert!(std::mem::size_of::<SamplingPreview>() <= 24);
    }

    #[test]
    fn repeated_previews_do_not_grow_or_move_workspace_storage() {
        let mut workspace = SamplingWorkspace::new(16).unwrap();
        let pointer = workspace.candidates.as_ptr();
        let capacity = workspace.candidates.capacity();
        let logits: Vec<f32> = (0_u8..16).map(f32::from).collect();
        let policy = SamplingPolicy::Sample(sample_config(17, 12, 0.9));

        let mut state = None;
        for _ in 0..128 {
            let preview = workspace.preview(&logits, policy, state).unwrap();
            state = preview.next_rng_state();
            assert_eq!(workspace.candidates.as_ptr(), pointer);
            assert_eq!(workspace.candidates.capacity(), capacity);
            assert_eq!(workspace.candidates.len(), 16);
        }
    }
}
