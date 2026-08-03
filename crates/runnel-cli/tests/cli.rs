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

#[test]
fn data_plane_demo_reports_deterministic_parity_and_accounting() {
    let output = run(&["data-plane-demo", "--json"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();

    assert_eq!(result["schema_version"], 2);
    assert_eq!(result["prompt"], "moe");
    assert_eq!(result["max_new_tokens"], 4);
    assert_eq!(result["generated_ids"], serde_json::json!([15, 11, 20, 9]));
    assert_eq!(result["generated_text"], "njsh");
    assert_eq!(result["parity"], true);
    assert_eq!(result["generation_cache"]["completed_generated_tokens"], 4);
    assert_eq!(result["generation_cache"]["physical_read_bytes"], 7_904);
    assert_eq!(
        result["generation_cache"]["bytes_per_generated_token"]["numerator_bytes"],
        7_904
    );
    assert_eq!(
        result["generation_cache"]["bytes_per_generated_token"]["denominator_tokens"],
        4
    );
    let forced = &result["forced_eviction_generation"];
    assert_eq!(forced["full_generation_parity"], true);
    assert_eq!(forced["cache_capacity_bytes"], 65_536);
    assert_eq!(forced["tensor_count"], 22);
    assert_eq!(forced["tensor_page_accesses"], 22);
    assert_eq!(forced["interference_page_accesses"], 1);
    assert_eq!(
        forced["tensor_page"]["object_digest"],
        result["fixtures"]["tiny"]["object_digest"]
    );
    assert_eq!(forced["tensor_page"]["page_size"], 65_536);
    assert_eq!(forced["tensor_page"]["page_index"], 0);
    assert_eq!(forced["tensor_page"]["logical_bytes"], 7_904);
    assert_eq!(
        forced["interference_page"]["object_digest"],
        result["fixtures"]["multi_page"]["object_digest"]
    );
    assert_eq!(forced["interference_page"]["page_size"], 65_536);
    assert_eq!(forced["interference_page"]["page_index"], 0);
    assert_eq!(forced["interference_page"]["logical_bytes"], 65_536);
    assert_eq!(forced["metrics"]["demand_bytes"], 239_424);
    assert_eq!(forced["metrics"]["physical_read_bytes"], 81_344);
    assert_eq!(forced["metrics"]["hits"], 20);
    assert_eq!(forced["metrics"]["misses"], 3);
    assert_eq!(forced["metrics"]["admissions"], 3);
    assert_eq!(forced["metrics"]["evictions"], 2);
    assert_eq!(forced["metrics"]["coalesced_demands"], 0);
    for metric in [
        "bytes",
        "coalesced",
        "late",
        "useful",
        "wasted",
        "redundant",
        "dropped",
    ] {
        assert_eq!(forced["metrics"]["prefetch"][metric], 0);
    }
    assert_eq!(forced["metrics"]["accounted"]["active_loads"], 0);
    assert_eq!(forced["metrics"]["accounted"]["page_pool_bytes"], 7_936);
    assert_eq!(forced["metrics"]["accounted"]["inflight_bytes"], 0);
    assert_eq!(forced["metrics"]["accounted"]["resident_bytes"], 7_936);
    assert_eq!(forced["metrics"]["accounted"]["retiring_bytes"], 0);
    assert_eq!(forced["metrics"]["accounted"]["leases"], 0);
    assert_eq!(forced["metrics"]["trace_events_dropped"], 0);
    assert_eq!(forced["trace"]["access"], "demand");
    assert_eq!(forced["trace"]["event_count"], 31);
    assert_eq!(forced["trace"]["outcomes"]["hit"], 20);
    assert_eq!(forced["trace"]["outcomes"]["miss"], 3);
    assert_eq!(forced["trace"]["outcomes"]["load_started"], 3);
    assert_eq!(forced["trace"]["outcomes"]["admitted"], 3);
    assert_eq!(forced["trace"]["outcomes"]["evicted"], 2);
    let forced_events = forced["trace"]["events"].as_array().unwrap();
    assert_eq!(forced_events.len(), 31);
    let prefix = [
        ("miss", false, 7_904),
        ("load_started", false, 7_904),
        ("admitted", false, 7_904),
        ("miss", true, 65_536),
        ("evicted", false, 7_904),
        ("load_started", true, 65_536),
        ("admitted", true, 65_536),
        ("miss", false, 7_904),
        ("evicted", true, 65_536),
        ("load_started", false, 7_904),
        ("admitted", false, 7_904),
    ];
    for (sequence, event) in forced_events.iter().enumerate() {
        let (outcome, interference, logical_bytes) = if sequence < prefix.len() {
            prefix[sequence]
        } else {
            ("hit", false, 7_904)
        };
        assert_eq!(event["sequence"], sequence as u64);
        assert_eq!(event["outcome"], outcome);
        assert_eq!(event["reason"], "demand");
        let expected_digest = if interference {
            result["fixtures"]["multi_page"]["object_digest"]
                .as_str()
                .unwrap()
        } else {
            result["fixtures"]["tiny"]["object_digest"]
                .as_str()
                .unwrap()
        };
        assert_eq!(event["object_digest"], expected_digest);
        assert_eq!(event["page_size"], 65_536);
        assert_eq!(event["page_index"], 0);
        assert_eq!(event["logical_bytes"], logical_bytes);
    }

    assert_eq!(
        result["fixtures"]["tiny"]["artifact_id"],
        "sha256:e49321cefc980ab59cd449341edfc624ccfc4b0b703cd175184e056d296d9ed3"
    );
    assert_eq!(result["fixtures"]["tiny"]["object_length"], 7_904);
    assert_eq!(
        result["fixtures"]["multi_page"]["artifact_id"],
        "sha256:15feda585327bf8e25c124692de1b2e101315185e427b811a07b8eaca54e1630"
    );
    assert_eq!(
        result["fixtures"]["multi_page"]["object_digest"],
        "sha256:9bb8fcc8e9d6f6ca3512bcb6daf6f89b6f0134874c9803f8782b100b39c854ae"
    );
    assert_eq!(result["fixtures"]["multi_page"]["object_length"], 131_089);

    assert_eq!(result["trace"]["access"], "demand");
    assert_eq!(
        result["trace"]["page_indices"],
        serde_json::json!([0, 1, 0, 2, 2])
    );
    assert_eq!(
        result["trace"]["page_lengths"],
        serde_json::json!([65_536, 65_536, 17])
    );
    assert_eq!(result["trace"]["event_count"], 16);
    let events = result["trace"]["events"].as_array().unwrap();
    assert_eq!(events.len(), 16);
    let expected = [
        ("miss", 0, 65_536),
        ("load_started", 0, 65_536),
        ("admitted", 0, 65_536),
        ("miss", 1, 65_536),
        ("evicted", 0, 65_536),
        ("load_started", 1, 65_536),
        ("admitted", 1, 65_536),
        ("miss", 0, 65_536),
        ("evicted", 1, 65_536),
        ("load_started", 0, 65_536),
        ("admitted", 0, 65_536),
        ("miss", 2, 17),
        ("evicted", 0, 65_536),
        ("load_started", 2, 17),
        ("admitted", 2, 17),
        ("hit", 2, 17),
    ];
    for (sequence, (event, (outcome, page_index, logical_bytes))) in
        events.iter().zip(expected).enumerate()
    {
        assert_eq!(event["sequence"], sequence as u64);
        assert_eq!(event["outcome"], outcome);
        assert_eq!(event["reason"], "demand");
        assert_eq!(
            event["object_digest"],
            result["fixtures"]["multi_page"]["object_digest"]
        );
        assert_eq!(event["page_size"], 65_536);
        assert_eq!(event["page_index"], page_index);
        assert_eq!(event["logical_bytes"], logical_bytes);
    }
    assert_eq!(result["trace"]["outcomes"]["hit"], 1);
    assert_eq!(result["trace"]["outcomes"]["miss"], 4);
    assert_eq!(result["trace"]["outcomes"]["load_started"], 4);
    assert_eq!(result["trace"]["outcomes"]["admitted"], 4);
    assert_eq!(result["trace"]["outcomes"]["evicted"], 3);
    for outcome in [
        "load_coalesced",
        "late_prefetch",
        "prefetch_coalesced",
        "retired",
        "load_failed",
        "cancelled",
        "prefetch_useful",
        "prefetch_wasted",
        "prefetch_redundant",
        "prefetch_dropped",
    ] {
        assert_eq!(result["trace"]["outcomes"][outcome], 0);
    }
    assert_eq!(result["cache_capacity_bytes"], 65_536);

    let metrics = &result["metrics"];
    assert_eq!(metrics["demand_bytes"], 196_642);
    assert_eq!(metrics["physical_read_bytes"], 196_625);
    assert_eq!(metrics["hits"], 1);
    assert_eq!(metrics["misses"], 4);
    assert_eq!(metrics["admissions"], 4);
    assert_eq!(metrics["evictions"], 3);
    assert_eq!(metrics["coalesced_demands"], 0);
    assert_eq!(metrics["prefetch"]["bytes"], 0);
    assert_eq!(metrics["prefetch"]["coalesced"], 0);
    assert_eq!(metrics["prefetch"]["late"], 0);
    assert_eq!(metrics["prefetch"]["useful"], 0);
    assert_eq!(metrics["prefetch"]["wasted"], 0);
    assert_eq!(metrics["prefetch"]["redundant"], 0);
    assert_eq!(metrics["prefetch"]["dropped"], 0);
    assert_eq!(metrics["accounted"]["active_loads"], 0);
    assert_eq!(metrics["accounted"]["page_pool_bytes"], 64);
    assert_eq!(metrics["accounted"]["inflight_bytes"], 0);
    assert_eq!(metrics["accounted"]["resident_bytes"], 64);
    assert_eq!(metrics["accounted"]["retiring_bytes"], 0);
    assert_eq!(metrics["accounted"]["leases"], 0);
    assert_eq!(metrics["trace_events_dropped"], 0);
    assert!(!String::from_utf8_lossy(&output.stdout).contains("/tmp/"));
}

#[test]
fn data_plane_demo_has_concise_text_output() {
    let output = run(&["data-plane-demo"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 6);
    assert!(stdout.contains("data-plane parity: true"));
    assert!(stdout.contains("generated IDs: [15, 11, 20, 9]"));
    assert!(stdout.contains("generation cache: 7904 physical bytes / 4 completed tokens"));
    assert!(stdout.contains("forced-eviction parity: true (2 evictions)"));
    assert!(!stdout.contains("/tmp/"));
}
