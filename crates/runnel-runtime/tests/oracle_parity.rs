use std::{collections::BTreeMap, fs, path::PathBuf};

use runnel_fixture::FixtureArtifact;
use runnel_format::{Artifact, Limits};
use runnel_kernels::Capabilities;
use runnel_runtime::{BackendRequest, RuntimeError, TinyModel, TinyTokenizer};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TokenGolden {
    full_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    generated_text: String,
    input_ids: Vec<u32>,
    max_new_tokens: usize,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct RoutesGolden {
    positions: Vec<RoutePosition>,
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
    positions: Vec<LogitPosition>,
    steps: Vec<LogitStep>,
}

#[derive(Debug, Deserialize)]
struct Comparison {
    atol: f32,
    rtol: f32,
}

#[derive(Debug, Deserialize)]
struct LogitStep {
    input_ids: Vec<u32>,
    logits: Vec<f32>,
    next_token_id: u32,
    step: usize,
}

#[derive(Debug, Deserialize)]
struct LogitPosition {
    logits: Vec<f32>,
    position: usize,
    token_id: u32,
}

#[derive(Debug, Deserialize)]
struct MetadataGolden {
    artifact: ArtifactGolden,
}

#[derive(Debug, Deserialize)]
struct ArtifactGolden {
    artifact_id: String,
    object_digest: String,
    object_length: u64,
    page_table_digest: String,
    page_table_length: u64,
}

fn fixture_path(directory: &str, filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(directory)
        .join(filename)
}

fn read_json<T: for<'de> Deserialize<'de>>(directory: &str, filename: &str) -> T {
    let bytes = fs::read(fixture_path(directory, filename)).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn fixture_model(fixture: &FixtureArtifact, request: BackendRequest) -> TinyModel {
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("rmoa");
    fixture.write_new(&root).unwrap();
    let artifact = Artifact::open(root, Limits::default()).unwrap();
    TinyModel::from_artifact_with_backend(&artifact, request).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: &Comparison, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let allowed = tolerance.atol + tolerance.rtol * expected.abs();
        let difference = (actual - expected).abs();
        assert!(
            difference <= allowed,
            "{label}[{index}]: actual={actual:?} expected={expected:?} difference={difference:?} allowed={allowed:?}"
        );
    }
}

fn assert_runtime_matches_committed_oracle(
    directory: &str,
    fixture: &FixtureArtifact,
    request: BackendRequest,
) {
    let tokens: TokenGolden = read_json(directory, "golden_tokens.json");
    let routes: RoutesGolden = read_json(directory, "golden_routes.json");
    let logits: LogitsGolden = read_json(directory, "golden_logits.json");
    let metadata: MetadataGolden = read_json(directory, "golden_metadata.json");
    let identity = fixture.identity();
    assert_eq!(
        identity.artifact_id.to_string(),
        metadata.artifact.artifact_id
    );
    assert_eq!(
        identity.object_digest.to_string(),
        metadata.artifact.object_digest
    );
    assert_eq!(identity.object_length, metadata.artifact.object_length);
    assert_eq!(
        identity.page_table_digest.to_string(),
        metadata.artifact.page_table_digest
    );
    assert_eq!(
        identity.page_table_length,
        metadata.artifact.page_table_length
    );
    let model = fixture_model(fixture, request);

    assert_eq!(
        TinyTokenizer.encode(&tokens.prompt).unwrap(),
        tokens.input_ids
    );
    let generation = model
        .generate_greedy(&tokens.input_ids, tokens.max_new_tokens)
        .unwrap();
    assert_eq!(generation.generated_tokens, tokens.generated_ids);
    assert_eq!(
        TinyTokenizer.decode(&generation.generated_tokens).unwrap(),
        tokens.generated_text
    );

    let outputs = model.run_tokens(&tokens.full_ids).unwrap();
    assert_eq!(outputs.len(), routes.positions.len());
    for (position, (output, expected)) in outputs.iter().zip(&routes.positions).enumerate() {
        assert_eq!(expected.position, position);
        assert_eq!(output.input_token, expected.token_id);
        assert_eq!(output.route.expert_ids, expected.selected_experts);
        assert_close(
            &output.route.scores,
            &expected.router_scores,
            &logits.comparison,
            "router_scores",
        );
        assert_close(
            &output.route.weights,
            &expected.selected_weights,
            &logits.comparison,
            "selected_weights",
        );
    }

    assert_eq!(outputs.len(), logits.positions.len());
    for (position, (output, expected)) in outputs.iter().zip(&logits.positions).enumerate() {
        assert_eq!(expected.position, position);
        assert_eq!(expected.token_id, output.input_token);
        assert_close(
            &output.logits,
            &expected.logits,
            &logits.comparison,
            "position_logits",
        );
    }

    let mut expected_prefixes = BTreeMap::new();
    for expected in &logits.steps {
        expected_prefixes.insert(expected.step, expected);
    }
    for step in 0..tokens.generated_ids.len() {
        let expected = expected_prefixes[&step];
        let position = expected.input_ids.len() - 1;
        let output = &outputs[position];
        assert_close(
            &output.logits,
            &expected.logits,
            &logits.comparison,
            "logits",
        );
        let mut selected = 0_usize;
        for candidate in 1..output.logits.len() {
            if output.logits[candidate]
                .total_cmp(&output.logits[selected])
                .is_gt()
            {
                selected = candidate;
            }
        }
        assert_eq!(selected as u32, expected.next_token_id);
    }
}

#[test]
fn scalar_v1_runtime_matches_committed_pytorch_oracle() {
    assert_runtime_matches_committed_oracle(
        "tiny",
        &FixtureArtifact::build(),
        BackendRequest::Auto,
    );
}

#[test]
fn scalar_v2_runtime_matches_committed_bf16_pytorch_oracle() {
    assert_runtime_matches_committed_oracle(
        "tiny-v2",
        &FixtureArtifact::build_v2(),
        BackendRequest::Scalar,
    );
}

#[test]
fn avx2_v2_runtime_matches_committed_bf16_pytorch_oracle_when_available() {
    if !Capabilities::detected().avx2_available() {
        return;
    }
    assert_runtime_matches_committed_oracle(
        "tiny-v2",
        &FixtureArtifact::build_v2(),
        BackendRequest::Avx2,
    );
}

#[test]
fn scalar_v3_runtime_matches_committed_pytorch_oracle() {
    assert_runtime_matches_committed_oracle(
        "tiny-v3",
        &FixtureArtifact::build_v3(),
        BackendRequest::Scalar,
    );
}

#[test]
fn avx2_v3_runtime_matches_committed_pytorch_oracle_when_available() {
    if !Capabilities::detected().avx2_available() {
        return;
    }
    assert_runtime_matches_committed_oracle(
        "tiny-v3",
        &FixtureArtifact::build_v3(),
        BackendRequest::Avx2,
    );
}

#[test]
fn v3_p1024_crosses_every_page_and_rejects_position_1025_atomically() {
    let model = fixture_model(&FixtureArtifact::build_v3(), BackendRequest::Scalar);
    let layout = model.state_layout(1_024, 16).unwrap();
    assert_eq!(layout.page_count(), 64);
    let mut state = model.new_sequence_state(layout).unwrap();
    let identity = state.state_id().unwrap();

    for position in 0..1_024 {
        let token = if position == 0 {
            1
        } else {
            [14, 16, 6][(position - 1) % 3]
        };
        let output = model.forward_token(&mut state, token).unwrap();
        assert_eq!(output.input_token, token);
        let one_based = position + 1;
        if [1, 15, 16, 17, 255, 256, 257, 1_023, 1_024].contains(&one_based) {
            assert_eq!(state.len(), one_based);
            assert_eq!(state.revision(), one_based as u64);
            assert!(state.history().key_at(position).is_some());
            assert!(state.history().value_at(position).is_some());
        }
    }

    let first_key = state.history().key_at(0).unwrap().to_vec();
    let last_key = state.history().key_at(1_023).unwrap().to_vec();
    let error = model.forward_token(&mut state, 14).unwrap_err();
    assert_eq!(error, RuntimeError::ContextLimit { limit: 1_024 });
    assert_eq!(state.state_id(), Some(identity));
    assert_eq!(state.revision(), 1_024);
    assert_eq!(state.len(), 1_024);
    assert_eq!(state.history().key_at(0).unwrap(), first_key);
    assert_eq!(state.history().key_at(1_023).unwrap(), last_key);
    assert!(state.history().key_at(1_024).is_none());
}
