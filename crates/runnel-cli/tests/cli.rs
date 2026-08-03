use std::{fs, process::Command};

use serde_json::Value;

fn run(arguments: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_runnel"))
        .args(arguments)
        .output()
        .unwrap()
}

fn generate_fixture() -> (tempfile::TempDir, String, Value) {
    let parent = tempfile::tempdir().unwrap();
    let artifact = parent.path().join("artifact");
    let output = run(&["fixture", "--output", artifact.to_str().unwrap(), "--json"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let identity = serde_json::from_slice(&output.stdout).unwrap();
    (parent, artifact.to_str().unwrap().to_owned(), identity)
}

#[test]
fn fixture_then_generate_matches_the_public_json_contract() {
    let (_parent, artifact, identity) = generate_fixture();
    assert_eq!(
        identity["artifact_id"],
        "sha256:e49321cefc980ab59cd449341edfc624ccfc4b0b703cd175184e056d296d9ed3"
    );
    assert_eq!(identity["object_length"], 7904);

    let output = run(&[
        "generate",
        "--artifact",
        &artifact,
        "--prompt",
        "moe",
        "--max-new-tokens",
        "4",
        "--strategy",
        "greedy",
        "--json",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["schema_version"], 1);
    assert_eq!(result["artifact_id"], identity["artifact_id"]);
    assert_eq!(result["adapter"]["id"], "runnel.tiny-causal-moe");
    assert_eq!(result["adapter"]["version"], 1);
    assert_eq!(result["input_token_count"], 4);
    assert_eq!(result["generated_ids"], serde_json::json!([15, 11, 20, 9]));
    assert_eq!(result["text"], "njsh");
    assert_eq!(result["stop_reason"], "max_new_tokens");
}

#[test]
fn corrupt_artifact_fails_without_partial_stdout() {
    let (_parent, artifact, identity) = generate_fixture();
    let digest = identity["object_digest"]
        .as_str()
        .unwrap()
        .strip_prefix("sha256:")
        .unwrap();
    let object_path = std::path::Path::new(&artifact)
        .join("objects")
        .join("sha256")
        .join(digest);
    let mut object = fs::read(&object_path).unwrap();
    object[0] ^= 1;
    fs::write(object_path, object).unwrap();

    let output = run(&[
        "generate",
        "--artifact",
        &artifact,
        "--prompt",
        "moe",
        "--json",
    ]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("digest mismatch"));
}

#[test]
fn request_validation_fails_without_partial_stdout() {
    let (_parent, artifact, _) = generate_fixture();
    let unsupported = run(&[
        "generate",
        "--artifact",
        &artifact,
        "--prompt",
        "🔑secret",
        "--json",
    ]);
    assert!(!unsupported.status.success());
    assert!(unsupported.stdout.is_empty());
    let error = String::from_utf8_lossy(&unsupported.stderr);
    assert!(error.contains("unsupported character"));
    assert!(!error.contains('🔑'));
    assert!(!error.contains("secret"));

    let context = run(&[
        "generate",
        "--artifact",
        &artifact,
        "--prompt",
        "abcdefghijklmno",
        "--max-new-tokens",
        "2",
        "--json",
    ]);
    assert!(!context.status.success());
    assert!(context.stdout.is_empty());
    assert!(String::from_utf8_lossy(&context.stderr).contains("context limit"));
}

#[test]
fn fixture_command_refuses_to_overwrite_an_existing_path() {
    let (_parent, artifact, _) = generate_fixture();
    let output = run(&["fixture", "--output", &artifact, "--json"]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}
