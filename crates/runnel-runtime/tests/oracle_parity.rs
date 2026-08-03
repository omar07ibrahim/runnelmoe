use std::{collections::BTreeMap, fs, path::PathBuf};

use runnel_fixture::FixtureArtifact;
use runnel_format::{Artifact, Limits};
use runnel_runtime::{TinyModel, TinyTokenizer};
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

fn fixture_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/tiny")
        .join(filename)
}

fn read_json<T: for<'de> Deserialize<'de>>(filename: &str) -> T {
    let bytes = fs::read(fixture_path(filename)).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn fixture_model() -> TinyModel {
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("rmoa");
    FixtureArtifact::build().write_new(&root).unwrap();
    let artifact = Artifact::open(root, Limits::default()).unwrap();
    TinyModel::from_artifact(&artifact).unwrap()
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

#[test]
fn scalar_runtime_matches_committed_pytorch_oracle() {
    let tokens: TokenGolden = read_json("golden_tokens.json");
    let routes: RoutesGolden = read_json("golden_routes.json");
    let logits: LogitsGolden = read_json("golden_logits.json");
    let metadata: MetadataGolden = read_json("golden_metadata.json");
    let identity = FixtureArtifact::build().identity();
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
    let model = fixture_model();

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
