use std::{
    env,
    io::{self, Write},
    process::ExitCode,
};

use runnel_fixture::{FixtureArtifact, FixtureIdentity};
use runnel_format::{Artifact, DType, Digest, Limits};
use runnel_kernels::KernelError;
use runnel_runtime::{
    BackendKind, BackendRequest, Generation, RuntimeError, StepOutput, TinyModel, TinyTokenizer,
};
use serde::{Deserialize, Serialize};

const SCHEMA: &str = "runnel.m4-correctness/1";
const PROMPT: &str = "moe";
const GENERATED_TEXT: &str = "njsh";
const INPUT_IDS: [u32; 4] = [1, 14, 16, 6];
const GENERATED_IDS: [u32; 4] = [15, 11, 20, 9];
const FULL_IDS: [u32; 8] = [1, 14, 16, 6, 15, 11, 20, 9];
const POSITIONS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
const REPETITIONS: u32 = 2;
const VOCAB_SIZE: usize = 32;
const ROUTER_WIDTH: usize = 4;
const TOP_K: usize = 2;
const MAX_GOLDEN_BYTES: usize = 128 * 1024;
const STDERR_LIMIT: usize = 512;

const V1_FILES: GoldenFiles = GoldenFiles {
    tokens: include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/tiny/golden_tokens.json"
    )),
    routes: include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/tiny/golden_routes.json"
    )),
    logits: include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/tiny/golden_logits.json"
    )),
    metadata: include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/tiny/golden_metadata.json"
    )),
};

const V2_FILES: GoldenFiles = GoldenFiles {
    tokens: include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/tiny-v2/golden_tokens.json"
    )),
    routes: include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/tiny-v2/golden_routes.json"
    )),
    logits: include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/tiny-v2/golden_logits.json"
    )),
    metadata: include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/tiny-v2/golden_metadata.json"
    )),
};

type CheckResult<T> = Result<T, String>;

#[derive(Clone, Copy)]
struct GoldenFiles {
    tokens: &'static [u8],
    routes: &'static [u8],
    logits: &'static [u8],
    metadata: &'static [u8],
}

#[derive(Debug, Deserialize)]
struct TokenGolden {
    fixture: String,
    full_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    generated_text: String,
    input_ids: Vec<u32>,
    max_new_tokens: usize,
    prompt: String,
    stop_reason: String,
}

#[derive(Debug, Deserialize)]
struct RoutesGolden {
    fixture: String,
    generated_ids: Vec<u32>,
    input_ids: Vec<u32>,
    positions: Vec<RoutePosition>,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct RoutePosition {
    position: usize,
    router_scores: Vec<f32>,
    selected_experts: Vec<usize>,
    selected_weights: Vec<f32>,
    token_id: u32,
}

#[derive(Debug, Deserialize)]
struct LogitsGolden {
    comparison: Comparison,
    fixture: String,
    positions: Vec<LogitPosition>,
    prompt: String,
    steps: Vec<LogitStep>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
struct Comparison {
    atol: f64,
    rtol: f64,
}

#[derive(Debug, Deserialize)]
struct LogitPosition {
    logits: Vec<f32>,
    position: usize,
    token_id: u32,
}

#[derive(Debug, Deserialize)]
struct LogitStep {
    input_ids: Vec<u32>,
    logits: Vec<f32>,
    next_token_id: u32,
    step: usize,
}

#[derive(Debug, Deserialize)]
struct MetadataGolden {
    adapter: Option<AdapterGolden>,
    artifact: ArtifactGolden,
    comparison: Comparison,
    fixture: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct AdapterGolden {
    id: String,
    version: u64,
}

#[derive(Debug, Deserialize)]
struct ArtifactGolden {
    artifact_id: String,
    object_digest: String,
    object_length: u64,
    page_table_digest: String,
    page_table_length: u64,
}

struct GoldenBundle {
    tokens: TokenGolden,
    routes: RoutesGolden,
    logits: LogitsGolden,
    metadata: MetadataGolden,
    digests: GoldenDigests,
}

#[derive(Clone, Debug, Serialize)]
struct GoldenDigests {
    logits_sha256: String,
    routes_sha256: String,
    tokens_sha256: String,
    metadata_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct FixtureEvidence {
    artifact_id: String,
    object_digest: String,
    object_length: u64,
    page_table_digest: String,
    page_table_length: u64,
}

#[derive(Debug, Serialize)]
struct ErrorMetrics {
    max_abs: f64,
    max_rel: f64,
    max_tolerance_ratio: f64,
    worst_position: usize,
    worst_index: usize,
}

#[derive(Debug, Serialize)]
struct Metrics {
    adapter_version: u64,
    representation: &'static str,
    requested_backend: &'static str,
    selected_backend: Option<&'static str>,
    fixture: FixtureEvidence,
    goldens: GoldenDigests,
    prompt: String,
    input_ids: Vec<u32>,
    full_ids: Vec<u32>,
    positions: Vec<usize>,
    repetitions: u32,
    tokens_exact: bool,
    expert_ids_exact: bool,
    deterministic: bool,
    tolerance: Comparison,
    logits_error: Option<ErrorMetrics>,
    router_score_error: Option<ErrorMetrics>,
    route_weight_error: Option<ErrorMetrics>,
}

#[derive(Debug, Serialize)]
struct EvidenceRow {
    schema: &'static str,
    check_id: &'static str,
    kind: &'static str,
    case_id: Option<&'static str>,
    backend: &'static str,
    status: &'static str,
    failure: Option<&'static str>,
    metrics: Metrics,
}

#[derive(Clone, Copy)]
struct CheckSpec {
    check_id: &'static str,
    backend: &'static str,
    adapter_version: u64,
    representation: &'static str,
    requested_backend: &'static str,
    expected_backend: Option<BackendKind>,
    backend_unavailable_is_unsupported: bool,
}

const V1_CHECK: CheckSpec = CheckSpec {
    check_id: "tiny-v1-preservation",
    backend: "v1-f32",
    adapter_version: 1,
    representation: "f32",
    requested_backend: "f32-preservation",
    expected_backend: None,
    backend_unavailable_is_unsupported: false,
};

const V2_SCALAR_CHECK: CheckSpec = CheckSpec {
    check_id: "tiny-v2-scalar",
    backend: "scalar",
    adapter_version: 2,
    representation: "bf16-experts",
    requested_backend: "forced-scalar",
    expected_backend: Some(BackendKind::Scalar),
    backend_unavailable_is_unsupported: false,
};

const V2_AVX2_CHECK: CheckSpec = CheckSpec {
    check_id: "tiny-v2-avx2",
    backend: "avx2",
    adapter_version: 2,
    representation: "bf16-experts",
    requested_backend: "forced-avx2",
    expected_backend: Some(BackendKind::Avx2),
    backend_unavailable_is_unsupported: true,
};

struct ErrorAccumulator {
    metrics: ErrorMetrics,
    initialized: bool,
}

impl ErrorAccumulator {
    const fn new() -> Self {
        Self {
            metrics: ErrorMetrics {
                max_abs: 0.0,
                max_rel: 0.0,
                max_tolerance_ratio: 0.0,
                worst_position: 0,
                worst_index: 0,
            },
            initialized: false,
        }
    }

    fn observe(
        &mut self,
        actual: f32,
        expected: f32,
        position: usize,
        index: usize,
        tolerance: Comparison,
    ) -> CheckResult<()> {
        require_finite(actual, "runtime output")?;
        require_finite(expected, "oracle golden")?;

        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let absolute = (actual - expected).abs();
        let relative = if absolute == 0.0 {
            0.0
        } else {
            absolute / expected.abs().max(f64::MIN_POSITIVE)
        };
        let allowed = tolerance.atol + tolerance.rtol * expected.abs();
        let tolerance_ratio = absolute / allowed;
        if !absolute.is_finite() || !relative.is_finite() || !tolerance_ratio.is_finite() {
            return Err("non-finite comparison metric".to_owned());
        }

        self.metrics.max_abs = self.metrics.max_abs.max(absolute);
        self.metrics.max_rel = self.metrics.max_rel.max(relative);
        if !self.initialized || tolerance_ratio > self.metrics.max_tolerance_ratio {
            self.metrics.max_tolerance_ratio = tolerance_ratio;
            self.metrics.worst_position = position;
            self.metrics.worst_index = index;
            self.initialized = true;
        }
        Ok(())
    }

    fn finish(self, label: &str) -> CheckResult<ErrorMetrics> {
        if !self.initialized {
            return Err(format!("{label} comparison observed no components"));
        }
        Ok(self.metrics)
    }
}

impl GoldenBundle {
    fn parse(
        files: GoldenFiles,
        expected_fixture: &str,
        adapter_version: u64,
    ) -> CheckResult<Self> {
        let bundle = Self {
            tokens: parse_json(files.tokens, "token golden")?,
            routes: parse_json(files.routes, "route golden")?,
            logits: parse_json(files.logits, "logit golden")?,
            metadata: parse_json(files.metadata, "metadata golden")?,
            digests: GoldenDigests {
                logits_sha256: raw_sha256(files.logits),
                routes_sha256: raw_sha256(files.routes),
                tokens_sha256: raw_sha256(files.tokens),
                metadata_sha256: raw_sha256(files.metadata),
            },
        };
        bundle.validate(expected_fixture, adapter_version)?;
        Ok(bundle)
    }

    fn validate(&self, expected_fixture: &str, adapter_version: u64) -> CheckResult<()> {
        require(
            self.tokens.fixture == expected_fixture
                && self.routes.fixture == expected_fixture
                && self.logits.fixture == expected_fixture
                && self.metadata.fixture == expected_fixture,
            "golden fixture labels do not match the requested adapter",
        )?;
        require(
            self.tokens.prompt == PROMPT
                && self.routes.prompt == PROMPT
                && self.logits.prompt == PROMPT,
            "golden prompt is not the frozen prompt",
        )?;
        require(
            self.tokens.input_ids == INPUT_IDS
                && self.routes.input_ids == INPUT_IDS
                && self.tokens.generated_ids == GENERATED_IDS
                && self.routes.generated_ids == GENERATED_IDS
                && self.tokens.full_ids == FULL_IDS,
            "golden token vectors are not frozen",
        )?;
        require(
            self.tokens.generated_text == GENERATED_TEXT
                && self.tokens.max_new_tokens == GENERATED_IDS.len()
                && self.tokens.stop_reason == "max_new_tokens",
            "golden generation contract is not frozen",
        )?;
        require(
            self.logits.comparison == self.metadata.comparison,
            "golden tolerance metadata disagrees",
        )?;
        require(
            self.logits.comparison.atol.is_finite()
                && self.logits.comparison.rtol.is_finite()
                && self.logits.comparison.atol > 0.0
                && self.logits.comparison.rtol >= 0.0,
            "golden tolerance is invalid",
        )?;

        match (adapter_version, self.metadata.adapter.as_ref()) {
            (1, None) => {}
            (2, Some(adapter))
                if adapter.id == "runnel.tiny-causal-moe" && adapter.version == 2 => {}
            _ => return Err("golden adapter metadata is not frozen".to_owned()),
        }

        require(
            self.routes.positions.len() == FULL_IDS.len(),
            "route golden does not cover every frozen position",
        )?;
        for (position, expected) in self.routes.positions.iter().enumerate() {
            require(
                expected.position == position && expected.token_id == FULL_IDS[position],
                "route golden position or token is not frozen",
            )?;
            require(
                expected.router_scores.len() == ROUTER_WIDTH
                    && expected.selected_experts.len() == TOP_K
                    && expected.selected_weights.len() == TOP_K,
                "route golden has an invalid vector width",
            )?;
            require(
                expected
                    .selected_experts
                    .iter()
                    .all(|expert| *expert < ROUTER_WIDTH)
                    && expected.selected_experts[0] != expected.selected_experts[1],
                "route golden has an invalid expert selection",
            )?;
            require_all_finite(&expected.router_scores, "golden router scores")?;
            require_all_finite(&expected.selected_weights, "golden route weights")?;
        }

        require(
            self.logits.positions.len() == FULL_IDS.len(),
            "logit golden does not cover every frozen position",
        )?;
        for (position, expected) in self.logits.positions.iter().enumerate() {
            require(
                expected.position == position && expected.token_id == FULL_IDS[position],
                "logit golden position or token is not frozen",
            )?;
            require(
                expected.logits.len() == VOCAB_SIZE,
                "logit golden has an invalid vocabulary width",
            )?;
            require_all_finite(&expected.logits, "golden logits")?;
        }

        require(
            self.logits.steps.len() == GENERATED_IDS.len(),
            "logit golden does not cover every generation step",
        )?;
        for (step, expected) in self.logits.steps.iter().enumerate() {
            let prefix_length = INPUT_IDS.len() + step;
            require(
                expected.step == step
                    && expected.input_ids == FULL_IDS[..prefix_length]
                    && expected.next_token_id == GENERATED_IDS[step]
                    && expected.logits.len() == VOCAB_SIZE,
                "generation-step golden is not frozen",
            )?;
            require_all_finite(&expected.logits, "golden generation logits")?;
        }
        Ok(())
    }

    fn require_identity(&self, identity: FixtureIdentity) -> CheckResult<()> {
        require(
            self.metadata.artifact.artifact_id == identity.artifact_id.to_string()
                && self.metadata.artifact.object_digest == identity.object_digest.to_string()
                && self.metadata.artifact.object_length == identity.object_length
                && self.metadata.artifact.page_table_digest
                    == identity.page_table_digest.to_string()
                && self.metadata.artifact.page_table_length == identity.page_table_length,
            "authenticated fixture identity disagrees with golden metadata",
        )
    }
}

fn main() -> ExitCode {
    if env::args_os().count() != 1 {
        bounded_stderr("usage: runnel-m4-model-check");
        return ExitCode::FAILURE;
    }

    match build_evidence() {
        Ok(bytes) => {
            if let Err(error) = io::stdout().lock().write_all(&bytes) {
                bounded_stderr(&format!("could not write JSONL evidence: {error}"));
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            bounded_stderr(&error);
            ExitCode::FAILURE
        }
    }
}

fn build_evidence() -> CheckResult<Vec<u8>> {
    let v1_goldens = GoldenBundle::parse(V1_FILES, "runnel-tiny-causal-moe-v1", 1)?;
    let v2_goldens = GoldenBundle::parse(V2_FILES, "runnel-tiny-causal-moe-v2", 2)?;
    let (v1_artifact, v1_fixture) = authenticated_fixture(FixtureArtifact::build(), 1)?;
    let (v2_artifact, v2_fixture) = authenticated_fixture(FixtureArtifact::build_v2(), 2)?;
    v1_goldens.require_identity(FixtureArtifact::build().identity())?;
    v2_goldens.require_identity(FixtureArtifact::build_v2().identity())?;

    let rows = build_model_rows_with(
        v1_fixture,
        v2_fixture,
        &v1_goldens,
        &v2_goldens,
        || TinyModel::from_artifact_with_backend(&v1_artifact, BackendRequest::Auto),
        || TinyModel::from_artifact_with_backend(&v2_artifact, BackendRequest::Scalar),
        || TinyModel::from_artifact_with_backend(&v2_artifact, BackendRequest::Avx2),
    );
    serialize_rows(&rows)
}

#[allow(clippy::too_many_arguments)]
fn build_model_rows_with<V1, Scalar, Avx2>(
    v1_fixture: FixtureEvidence,
    v2_fixture: FixtureEvidence,
    v1_goldens: &GoldenBundle,
    v2_goldens: &GoldenBundle,
    construct_v1: V1,
    construct_scalar: Scalar,
    construct_avx2: Avx2,
) -> [EvidenceRow; 3]
where
    V1: FnOnce() -> Result<TinyModel, RuntimeError>,
    Scalar: FnOnce() -> Result<TinyModel, RuntimeError>,
    Avx2: FnOnce() -> Result<TinyModel, RuntimeError>,
{
    // Each check is attempted independently. A construction or execution failure
    // is evidence about that check, not a reason to truncate the closed protocol.
    let v1_row = execute_model_check(V1_CHECK, v1_fixture, v1_goldens, construct_v1());
    let scalar_row = execute_model_check(
        V2_SCALAR_CHECK,
        v2_fixture.clone(),
        v2_goldens,
        construct_scalar(),
    );
    let avx2_row = execute_model_check(V2_AVX2_CHECK, v2_fixture, v2_goldens, construct_avx2());
    [v1_row, scalar_row, avx2_row]
}

fn execute_model_check(
    spec: CheckSpec,
    fixture: FixtureEvidence,
    goldens: &GoldenBundle,
    model: Result<TinyModel, RuntimeError>,
) -> EvidenceRow {
    let model = match model {
        Ok(model) => model,
        Err(RuntimeError::ExpertKernel {
            operation: "backend selection",
            source: KernelError::BackendUnavailable,
        }) if spec.backend_unavailable_is_unsupported => {
            return unexecuted_model_row(
                spec,
                fixture,
                goldens,
                "unsupported",
                "backend_unavailable",
            );
        }
        Err(_) => {
            return unexecuted_model_row(spec, fixture, goldens, "failed", "model_execution_error");
        }
    };

    let selected_backend = match spec.expected_backend {
        None => match require(
            model.expert_backend().is_none(),
            "v1 unexpectedly selected a compact-expert backend",
        ) {
            Ok(()) => None,
            Err(_) => {
                return unexecuted_model_row(
                    spec,
                    fixture,
                    goldens,
                    "failed",
                    "model_execution_error",
                );
            }
        },
        Some(expected) => match require_selected_backend(&model, expected) {
            Ok(selected) => Some(selected),
            Err(_) => {
                return unexecuted_model_row(
                    spec,
                    fixture,
                    goldens,
                    "failed",
                    "model_execution_error",
                );
            }
        },
    };

    match evaluate_model(spec, selected_backend, fixture.clone(), goldens, &model) {
        Ok(row) => row,
        Err(_) => unexecuted_model_row(spec, fixture, goldens, "failed", "model_execution_error"),
    }
}

fn serialize_rows(rows: &[EvidenceRow]) -> CheckResult<Vec<u8>> {
    let mut output = Vec::with_capacity(6 * 1024);
    for row in rows {
        let value = serde_json::to_value(row)
            .map_err(|error| format!("could not construct correctness JSON: {error}"))?;
        serde_json::to_writer(&mut output, &value)
            .map_err(|error| format!("could not serialize correctness row: {error}"))?;
        output.push(b'\n');
    }
    Ok(output)
}

fn authenticated_fixture(
    fixture: FixtureArtifact,
    adapter_version: u64,
) -> CheckResult<(Artifact, FixtureEvidence)> {
    let identity = fixture.identity();
    let artifact = Artifact::from_bytes_with_expected_id(
        fixture.to_parts(),
        Limits::default(),
        identity.artifact_id,
    )
    .map_err(|error| format!("fixture authentication failed: {error}"))?;
    require(
        artifact.artifact_id() == identity.artifact_id
            && artifact.manifest().adapter.id == "runnel.tiny-causal-moe"
            && artifact.manifest().adapter.version == adapter_version,
        "authenticated fixture has an unexpected manifest identity",
    )?;
    require(
        artifact.manifest().objects.len() == 1,
        "authenticated fixture must contain one object",
    )?;
    let object = &artifact.manifest().objects[0];
    require(
        object.digest == identity.object_digest
            && object.length == identity.object_length
            && object.page_table == identity.page_table_digest
            && object.page_table_length == identity.page_table_length,
        "authenticated object identity disagrees with fixture identity",
    )?;
    require(
        artifact.manifest().tensors.len() == 22
            && artifact.manifest().tensors.iter().all(|tensor| {
                if adapter_version == 2 && matches!(tensor.id, 8..=19) {
                    tensor.dtype == DType::Bf16Le
                } else {
                    tensor.dtype == DType::F32Le
                }
            }),
        "authenticated fixture has an unexpected tensor representation",
    )?;

    Ok((
        artifact,
        FixtureEvidence {
            artifact_id: identity.artifact_id.to_string(),
            object_digest: identity.object_digest.to_string(),
            object_length: identity.object_length,
            page_table_digest: identity.page_table_digest.to_string(),
            page_table_length: identity.page_table_length,
        },
    ))
}

fn evaluate_model(
    spec: CheckSpec,
    selected_backend: Option<&'static str>,
    fixture: FixtureEvidence,
    goldens: &GoldenBundle,
    model: &TinyModel,
) -> CheckResult<EvidenceRow> {
    let check_id = spec.check_id;
    let first_generation = model
        .generate_greedy(&INPUT_IDS, GENERATED_IDS.len())
        .map_err(|error| format!("{check_id} generation failed: {error}"))?;
    let second_generation = model
        .generate_greedy(&INPUT_IDS, GENERATED_IDS.len())
        .map_err(|error| format!("{check_id} repeated generation failed: {error}"))?;
    let first_outputs = model
        .run_tokens(&FULL_IDS)
        .map_err(|error| format!("{check_id} full-position run failed: {error}"))?;
    let second_outputs = model
        .run_tokens(&FULL_IDS)
        .map_err(|error| format!("{check_id} repeated full-position run failed: {error}"))?;

    require_outputs_finite(&first_outputs)?;
    require_outputs_finite(&second_outputs)?;
    require_generation_finite(&first_generation)?;
    require_generation_finite(&second_generation)?;

    let encoded = TinyTokenizer
        .encode(PROMPT)
        .map_err(|error| format!("frozen prompt could not be encoded: {error}"))?;
    let decoded = TinyTokenizer
        .decode(&first_generation.generated_tokens)
        .map_err(|error| format!("generated tokens could not be decoded: {error}"))?;
    let tokens_exact = encoded == INPUT_IDS
        && first_generation.generated_tokens == GENERATED_IDS
        && second_generation.generated_tokens == GENERATED_IDS
        && decoded == GENERATED_TEXT
        && first_outputs.len() == FULL_IDS.len()
        && first_outputs
            .iter()
            .zip(FULL_IDS)
            .all(|(output, token)| output.input_token == token);
    let deterministic = first_generation == second_generation && first_outputs == second_outputs;

    require(
        first_outputs.len() == goldens.routes.positions.len()
            && first_outputs.len() == goldens.logits.positions.len(),
        "runtime output count disagrees with golden positions",
    )?;
    let expected_generation_outputs = INPUT_IDS.len() + GENERATED_IDS.len() - 1;
    require(
        first_generation.steps.len() == expected_generation_outputs
            && second_generation.steps.len() == expected_generation_outputs,
        "runtime generation output count disagrees with the frozen generation path",
    )?;
    let tokens_exact = tokens_exact
        && first_generation
            .steps
            .iter()
            .zip(&FULL_IDS[..expected_generation_outputs])
            .all(|(output, token)| output.input_token == *token)
        && second_generation
            .steps
            .iter()
            .zip(&FULL_IDS[..expected_generation_outputs])
            .all(|(output, token)| output.input_token == *token);
    let expert_ids_exact =
        first_outputs
            .iter()
            .zip(&goldens.routes.positions)
            .all(|(actual, expected)| actual.route.expert_ids == expected.selected_experts)
            && [&first_generation, &second_generation]
                .into_iter()
                .all(|generation| {
                    generation.steps.iter().zip(&goldens.routes.positions).all(
                        |(actual, expected)| actual.route.expert_ids == expected.selected_experts,
                    )
                });
    let mut logits_error = ErrorAccumulator::new();
    let mut router_score_error = ErrorAccumulator::new();
    let mut route_weight_error = ErrorAccumulator::new();
    for (position, ((actual, expected_logits), expected_route)) in first_outputs
        .iter()
        .zip(&goldens.logits.positions)
        .zip(&goldens.routes.positions)
        .enumerate()
    {
        observe_slice(
            &mut logits_error,
            &actual.logits,
            &expected_logits.logits,
            position,
            goldens.logits.comparison,
            "logits",
        )?;
        observe_slice(
            &mut router_score_error,
            &actual.route.scores,
            &expected_route.router_scores,
            position,
            goldens.logits.comparison,
            "router scores",
        )?;
        observe_slice(
            &mut route_weight_error,
            &actual.route.weights,
            &expected_route.selected_weights,
            position,
            goldens.logits.comparison,
            "route weights",
        )?;
    }

    // `golden_logits.steps` is the independent oracle's cache/generation
    // contract: each row is the decision output at prompt_len - 1 + step.
    // Route goldens cover the corresponding complete generation prefix. Fold
    // both repetitions into the retained diagnostics so a cache-only mismatch
    // cannot be hidden by a passing full-sequence `run_tokens` check.
    for generation in [&first_generation, &second_generation] {
        for (position, (actual, expected_route)) in generation
            .steps
            .iter()
            .zip(&goldens.routes.positions)
            .enumerate()
        {
            observe_slice(
                &mut router_score_error,
                &actual.route.scores,
                &expected_route.router_scores,
                position,
                goldens.logits.comparison,
                "generation router scores",
            )?;
            observe_slice(
                &mut route_weight_error,
                &actual.route.weights,
                &expected_route.selected_weights,
                position,
                goldens.logits.comparison,
                "generation route weights",
            )?;
        }
        for (step, expected) in goldens.logits.steps.iter().enumerate() {
            let position = INPUT_IDS.len() - 1 + step;
            observe_slice(
                &mut logits_error,
                &generation.steps[position].logits,
                &expected.logits,
                position,
                goldens.logits.comparison,
                "generation logits",
            )?;
        }
    }
    let logits_error = logits_error.finish("logits")?;
    let router_score_error = router_score_error.finish("router scores")?;
    let route_weight_error = route_weight_error.finish("route weights")?;
    let within_tolerance = logits_error.max_tolerance_ratio <= 1.0
        && router_score_error.max_tolerance_ratio <= 1.0
        && route_weight_error.max_tolerance_ratio <= 1.0;
    let is_ok = tokens_exact && expert_ids_exact && deterministic && within_tolerance;
    let status = if is_ok { "ok" } else { "failed" };
    let failure = if is_ok {
        None
    } else if !within_tolerance {
        Some("tolerance_exceeded")
    } else {
        Some("exactness_mismatch")
    };

    Ok(EvidenceRow {
        schema: SCHEMA,
        check_id: spec.check_id,
        kind: "model",
        case_id: None,
        backend: spec.backend,
        status,
        failure,
        metrics: Metrics {
            adapter_version: spec.adapter_version,
            representation: spec.representation,
            requested_backend: spec.requested_backend,
            selected_backend,
            fixture,
            goldens: goldens.digests.clone(),
            prompt: PROMPT.to_owned(),
            input_ids: INPUT_IDS.to_vec(),
            full_ids: FULL_IDS.to_vec(),
            positions: POSITIONS.to_vec(),
            repetitions: REPETITIONS,
            tokens_exact,
            expert_ids_exact,
            deterministic,
            tolerance: goldens.logits.comparison,
            logits_error: Some(logits_error),
            router_score_error: Some(router_score_error),
            route_weight_error: Some(route_weight_error),
        },
    })
}

fn unexecuted_model_row(
    spec: CheckSpec,
    fixture: FixtureEvidence,
    goldens: &GoldenBundle,
    status: &'static str,
    failure: &'static str,
) -> EvidenceRow {
    EvidenceRow {
        schema: SCHEMA,
        check_id: spec.check_id,
        kind: "model",
        case_id: None,
        backend: spec.backend,
        status,
        failure: Some(failure),
        metrics: Metrics {
            adapter_version: spec.adapter_version,
            representation: spec.representation,
            requested_backend: spec.requested_backend,
            selected_backend: None,
            fixture,
            goldens: goldens.digests.clone(),
            prompt: PROMPT.to_owned(),
            input_ids: INPUT_IDS.to_vec(),
            full_ids: FULL_IDS.to_vec(),
            positions: POSITIONS.to_vec(),
            repetitions: REPETITIONS,
            tokens_exact: false,
            expert_ids_exact: false,
            deterministic: false,
            tolerance: goldens.logits.comparison,
            logits_error: None,
            router_score_error: None,
            route_weight_error: None,
        },
    }
}

fn require_selected_backend(model: &TinyModel, expected: BackendKind) -> CheckResult<&'static str> {
    require(
        model.expert_backend() == Some(expected),
        "runtime reported an unexpected selected backend",
    )?;
    Ok(match expected {
        BackendKind::Scalar => "scalar",
        BackendKind::Avx2 => "avx2",
    })
}

fn observe_slice(
    accumulator: &mut ErrorAccumulator,
    actual: &[f32],
    expected: &[f32],
    position: usize,
    tolerance: Comparison,
    label: &str,
) -> CheckResult<()> {
    require(
        actual.len() == expected.len(),
        &format!("{label} vector width disagrees with golden"),
    )?;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        accumulator.observe(actual, expected, position, index, tolerance)?;
    }
    Ok(())
}

fn require_outputs_finite(outputs: &[StepOutput]) -> CheckResult<()> {
    for output in outputs {
        require_all_finite(&output.logits, "runtime logits")?;
        require_all_finite(&output.route.scores, "runtime router scores")?;
        require_all_finite(&output.route.weights, "runtime route weights")?;
    }
    Ok(())
}

fn require_generation_finite(generation: &Generation) -> CheckResult<()> {
    require_outputs_finite(&generation.steps)
}

fn require_all_finite(values: &[f32], label: &str) -> CheckResult<()> {
    for &value in values {
        require_finite(value, label)?;
    }
    Ok(())
}

fn require_finite(value: f32, label: &str) -> CheckResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{label} contains a non-finite value"))
    }
}

fn parse_json<T: for<'de> Deserialize<'de>>(bytes: &[u8], label: &str) -> CheckResult<T> {
    require(
        bytes.len() <= MAX_GOLDEN_BYTES,
        &format!("{label} exceeds the bounded input size"),
    )?;
    serde_json::from_slice(bytes).map_err(|error| format!("could not parse {label}: {error}"))
}

fn raw_sha256(bytes: &[u8]) -> String {
    Digest::of(bytes).path_component()
}

fn require(condition: bool, message: &str) -> CheckResult<()> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

fn bounded_stderr(message: &str) {
    let sanitized: String = message
        .chars()
        .map(|character| match character {
            '\n' | '\r' => ' ',
            other => other,
        })
        .take(STDERR_LIMIT)
        .collect();
    eprintln!("runnel-m4-model-check: {sanitized}");
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, collections::BTreeSet};

    use serde_json::Value;

    use super::*;

    #[test]
    fn evidence_is_exactly_three_closed_jsonl_rows() {
        let bytes = build_evidence().unwrap();
        assert_eq!(bytes.last(), Some(&b'\n'));
        let text = std::str::from_utf8(&bytes).unwrap();
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), 3);

        let expected_ids = ["tiny-v1-preservation", "tiny-v2-scalar", "tiny-v2-avx2"];
        let top_level_keys = BTreeSet::from([
            "backend", "case_id", "check_id", "failure", "kind", "metrics", "schema", "status",
        ]);
        let metric_keys = BTreeSet::from([
            "adapter_version",
            "deterministic",
            "expert_ids_exact",
            "fixture",
            "full_ids",
            "goldens",
            "input_ids",
            "logits_error",
            "positions",
            "prompt",
            "repetitions",
            "representation",
            "requested_backend",
            "route_weight_error",
            "router_score_error",
            "selected_backend",
            "tokens_exact",
            "tolerance",
        ]);

        for (index, line) in lines.iter().enumerate() {
            let value: Value = serde_json::from_str(line).unwrap();
            let object = value.as_object().unwrap();
            assert_eq!(
                object.keys().map(String::as_str).collect::<BTreeSet<_>>(),
                top_level_keys
            );
            assert_eq!(value["schema"], SCHEMA);
            assert_eq!(value["kind"], "model");
            assert_eq!(value["check_id"], expected_ids[index]);
            let metrics = value["metrics"].as_object().unwrap();
            assert_eq!(
                metrics.keys().map(String::as_str).collect::<BTreeSet<_>>(),
                metric_keys
            );
            assert_eq!(
                metrics["positions"],
                serde_json::to_value(POSITIONS).unwrap()
            );
            assert_eq!(metrics["repetitions"], REPETITIONS);
        }
        assert!(!text.contains("timing"));
    }

    #[test]
    fn semantic_failure_is_a_successfully_serialized_protocol_row() {
        let goldens = GoldenBundle::parse(V2_FILES, "runnel-tiny-causal-moe-v2", 2).unwrap();
        let fixture = FixtureArtifact::build_v2();
        let mut row = unexecuted_model_row(
            V2_AVX2_CHECK,
            FixtureEvidence {
                artifact_id: fixture.identity().artifact_id.to_string(),
                object_digest: fixture.identity().object_digest.to_string(),
                object_length: fixture.identity().object_length,
                page_table_digest: fixture.identity().page_table_digest.to_string(),
                page_table_length: fixture.identity().page_table_length,
            },
            &goldens,
            "unsupported",
            "backend_unavailable",
        );
        row.status = "failed";
        row.failure = Some("exactness_mismatch");
        row.metrics.selected_backend = Some("avx2");
        row.metrics.tokens_exact = true;
        row.metrics.expert_ids_exact = true;
        row.metrics.deterministic = false;
        row.metrics.logits_error = Some(ErrorMetrics {
            max_abs: 0.0,
            max_rel: 0.0,
            max_tolerance_ratio: 0.0,
            worst_position: 0,
            worst_index: 0,
        });
        row.metrics.router_score_error = Some(ErrorMetrics {
            max_abs: 0.0,
            max_rel: 0.0,
            max_tolerance_ratio: 0.0,
            worst_position: 0,
            worst_index: 0,
        });
        row.metrics.route_weight_error = Some(ErrorMetrics {
            max_abs: 0.0,
            max_rel: 0.0,
            max_tolerance_ratio: 0.0,
            worst_position: 0,
            worst_index: 0,
        });

        let bytes = serialize_rows(&[row]).unwrap();
        let value: Value = serde_json::from_slice(bytes.strip_suffix(b"\n").unwrap()).unwrap();
        assert_eq!(value["status"], "failed");
        assert_eq!(value["failure"], "exactness_mismatch");
        assert!(value["metrics"]["logits_error"].is_object());
    }

    #[test]
    fn construction_failure_is_retained_and_later_checks_are_attempted() {
        let v1_goldens = GoldenBundle::parse(V1_FILES, "runnel-tiny-causal-moe-v1", 1).unwrap();
        let v2_goldens = GoldenBundle::parse(V2_FILES, "runnel-tiny-causal-moe-v2", 2).unwrap();
        let (v1_artifact, v1_fixture) = authenticated_fixture(FixtureArtifact::build(), 1).unwrap();
        let (v2_artifact, v2_fixture) =
            authenticated_fixture(FixtureArtifact::build_v2(), 2).unwrap();
        let attempts = Cell::new(0_u8);

        let rows = build_model_rows_with(
            v1_fixture,
            v2_fixture,
            &v1_goldens,
            &v2_goldens,
            || {
                attempts.set(attempts.get() | 0b001);
                let _ = &v1_artifact;
                Err(RuntimeError::InvalidArtifact(
                    "injected construction failure".to_owned(),
                ))
            },
            || {
                attempts.set(attempts.get() | 0b010);
                TinyModel::from_artifact_with_backend(&v2_artifact, BackendRequest::Scalar)
            },
            || {
                attempts.set(attempts.get() | 0b100);
                TinyModel::from_artifact_with_backend(&v2_artifact, BackendRequest::Avx2)
            },
        );

        assert_eq!(attempts.get(), 0b111);
        assert_eq!(rows[0].check_id, "tiny-v1-preservation");
        assert_eq!(rows[0].status, "failed");
        assert_eq!(rows[0].failure, Some("model_execution_error"));
        assert!(!rows[0].metrics.tokens_exact);
        assert!(!rows[0].metrics.expert_ids_exact);
        assert!(!rows[0].metrics.deterministic);
        assert!(rows[0].metrics.logits_error.is_none());
        assert!(rows[0].metrics.router_score_error.is_none());
        assert!(rows[0].metrics.route_weight_error.is_none());
        assert_eq!(rows[1].check_id, "tiny-v2-scalar");
        assert_eq!(rows[1].status, "ok");
        assert_eq!(rows[2].check_id, "tiny-v2-avx2");
        assert!(matches!(rows[2].status, "ok" | "unsupported"));
    }

    #[test]
    fn only_forced_avx2_backend_unavailable_is_unsupported() {
        let goldens = GoldenBundle::parse(V2_FILES, "runnel-tiny-causal-moe-v2", 2).unwrap();
        let (_, fixture) = authenticated_fixture(FixtureArtifact::build_v2(), 2).unwrap();
        let backend_unavailable = || RuntimeError::ExpertKernel {
            operation: "backend selection",
            source: KernelError::BackendUnavailable,
        };

        let scalar = execute_model_check(
            V2_SCALAR_CHECK,
            fixture.clone(),
            &goldens,
            Err(backend_unavailable()),
        );
        let avx2 =
            execute_model_check(V2_AVX2_CHECK, fixture, &goldens, Err(backend_unavailable()));

        assert_eq!(scalar.status, "failed");
        assert_eq!(scalar.failure, Some("model_execution_error"));
        assert_eq!(avx2.status, "unsupported");
        assert_eq!(avx2.failure, Some("backend_unavailable"));
    }

    #[test]
    fn generation_step_logits_are_part_of_the_retained_oracle_proof() {
        let mut goldens = GoldenBundle::parse(V2_FILES, "runnel-tiny-causal-moe-v2", 2).unwrap();
        let (artifact, fixture) = authenticated_fixture(FixtureArtifact::build_v2(), 2).unwrap();
        let model = TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar);

        // This source is used only by the generation/cache path comparison;
        // the full-position golden remains unchanged.
        goldens.logits.steps[0].logits[0] += 1.0;
        let row = execute_model_check(V2_SCALAR_CHECK, fixture, &goldens, model);

        assert_eq!(row.status, "failed");
        assert_eq!(row.failure, Some("tolerance_exceeded"));
        assert!(
            row.metrics
                .logits_error
                .as_ref()
                .unwrap()
                .max_tolerance_ratio
                > 1.0
        );
    }
}
