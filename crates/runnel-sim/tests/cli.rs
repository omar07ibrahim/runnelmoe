use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use runnel_sim::{
    PageClass, PageDescriptor, PageId, SimLimits, TinyLfuConfig, TraceEvent, TraceHeader,
    serialize_trace,
};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_runnel-cache-sim")
}

fn run(arguments: &[&str]) -> Output {
    Command::new(binary())
        .args(arguments)
        .stdin(Stdio::null())
        .output()
        .expect("run cache simulator")
}

fn run_with_deadline(arguments: &[&str], timeout: Duration) -> Output {
    let mut child = Command::new(binary())
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cache simulator");
    let deadline = Instant::now() + timeout;
    loop {
        if child.try_wait().expect("poll cache simulator").is_some() {
            return child
                .wait_with_output()
                .expect("collect cache simulator output");
        }
        if Instant::now() >= deadline {
            child.kill().expect("kill blocked cache simulator");
            let output = child
                .wait_with_output()
                .expect("reap blocked cache simulator");
            panic!(
                "cache simulator exceeded {:?}; stderr={}",
                timeout,
                String::from_utf8_lossy(&output.stderr)
            );
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn generated_trace_validates_and_simulates_offline() {
    let generated = run(&[
        "generate",
        "--family",
        "markov_clusters",
        "--replicate",
        "2",
        "--measured-steps",
        "16",
    ]);
    assert!(generated.status.success());
    assert!(generated.stderr.is_empty());
    assert!(generated.stdout.ends_with(b"\n"));

    let temporary = tempfile::tempdir().unwrap();
    let trace_path = temporary.path().join("trace.jsonl");
    std::fs::write(&trace_path, generated.stdout).unwrap();
    let trace = trace_path.to_str().unwrap();

    let validated = run(&["validate", trace]);
    assert!(validated.status.success());
    assert!(validated.stderr.is_empty());
    let validation: serde_json::Value = serde_json::from_slice(&validated.stdout).unwrap();
    assert_eq!(validation["schema"], "runnel.cache-validation/1");
    assert_eq!(validation["page_count"], 384);

    let simulated = run(&[
        "simulate",
        trace,
        "--policy",
        "tiny-lfu",
        "--capacity-bytes",
        "2097152",
    ]);
    assert!(simulated.status.success());
    assert!(simulated.stderr.is_empty());
    let result: serde_json::Value = serde_json::from_slice(&simulated.stdout).unwrap();
    assert_eq!(result["schema"], "runnel.cache-result/1");
    assert_eq!(result["policy"], "tiny-lfu");
    assert_eq!(result["metrics"]["demand_accesses"], 96);
}

#[test]
fn matrix_has_the_closed_eighteen_cell_projection() {
    let output = run(&[
        "matrix",
        "--family",
        "iid_uniform",
        "--replicate",
        "0",
        "--measured-steps",
        "8",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let matrix: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(matrix["schema"], "runnel.cache-matrix/1");
    let results = matrix["results"].as_array().unwrap();
    assert_eq!(results.len(), 18);

    let projection = results
        .iter()
        .map(|result| {
            (
                result["capacity_bytes"].as_u64().unwrap(),
                result["policy"].as_str().unwrap().to_owned(),
            )
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(projection.len(), 18);
}

#[test]
fn empty_canonical_trace_simulates_under_every_cli_policy() {
    let temporary = tempfile::tempdir().unwrap();
    let trace_path = temporary.path().join("empty.jsonl");
    let trace = serialize_trace(
        &TraceHeader {
            kind: "header".to_owned(),
            schema: "runnel.cache-trace/1".to_owned(),
            trace_id: "empty".to_owned(),
            page_count: 0,
            event_count: 0,
            charge_quantum: 1,
            prefetch_model: "instant-between-events-v1".to_owned(),
        },
        &[],
        &[],
        SimLimits::default(),
    )
    .unwrap();
    std::fs::write(&trace_path, trace).unwrap();

    for policy in [
        "no-cache",
        "lru",
        "slru",
        "tiny-lfu",
        "router-admit",
        "router-prefetch",
        "belady",
    ] {
        let output = run(&[
            "simulate",
            trace_path.to_str().unwrap(),
            "--policy",
            policy,
            "--capacity-bytes",
            "1",
        ]);
        assert!(
            output.status.success(),
            "policy {policy} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["policy"], policy);
        assert_eq!(result["metrics"]["demand_accesses"], 0);
        assert_eq!(result["metrics"]["total_physical_load_bytes"], 0);
    }
}

#[test]
fn router_admit_accepts_an_extreme_page_without_prefetch_limit_overflow() {
    let temporary = tempfile::tempdir().unwrap();
    let trace_path = temporary.path().join("extreme.jsonl");
    let trace = serialize_trace(
        &TraceHeader {
            kind: "header".to_owned(),
            schema: "runnel.cache-trace/1".to_owned(),
            trace_id: "extreme-router-admit".to_owned(),
            page_count: 1,
            event_count: 1,
            charge_quantum: 1,
            prefetch_model: "instant-between-events-v1".to_owned(),
        },
        &[PageDescriptor {
            id: PageId(0),
            logical_bytes: u64::MAX,
            charge_bytes: u64::MAX,
            class: PageClass::Shared,
        }],
        &[TraceEvent::Demand {
            sequence: 0,
            request: 0,
            step: 0,
            page: PageId(0),
        }],
        SimLimits::default(),
    )
    .unwrap();
    std::fs::write(&trace_path, trace).unwrap();

    let output = run(&[
        "simulate",
        trace_path.to_str().unwrap(),
        "--policy",
        "router-admit",
        "--capacity-bytes",
        "18446744073709551615",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["policy"], "router-admit");
    assert_eq!(
        result["policy_spec"]["config"]["max_prefetch_bytes_per_signal"],
        u64::MAX
    );
    assert_eq!(result["metrics"]["demand_accesses"], 1);
    assert_eq!(result["metrics"]["demand_load_bytes"], u64::MAX);
}

#[cfg(unix)]
#[test]
fn trace_paths_reject_symbolic_links() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let target = temporary.path().join("target.jsonl");
    let link = temporary.path().join("link.jsonl");
    std::fs::write(&target, b"not a trace\n").unwrap();
    symlink(&target, &link).unwrap();

    let output = run(&["validate", link.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("symbolic link"));
}

#[cfg(unix)]
#[test]
fn trace_paths_reject_fifos_without_blocking() {
    use rustix::fs::{Mode, mkfifoat};

    let temporary = tempfile::tempdir().unwrap();
    let fifo = temporary.path().join("trace.fifo");
    mkfifoat(rustix::fs::CWD, &fifo, Mode::RUSR | Mode::WUSR).unwrap();

    let output = run_with_deadline(
        &["validate", fifo.to_str().unwrap()],
        Duration::from_secs(2),
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("regular file"));
}

#[test]
fn extreme_tinylfu_capacity_returns_a_typed_error_without_panicking() {
    let accepted = TinyLfuConfig::m3(u64::MAX / 10).unwrap();
    assert_eq!(accepted.sample_accesses, (u64::MAX / 10) * 10);
    assert!(TinyLfuConfig::m3(u64::MAX / 10 + 1).is_err());

    let trace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/cache/variable-byte.jsonl");
    let output = run(&[
        "simulate",
        trace.to_str().unwrap(),
        "--policy",
        "tiny-lfu",
        "--capacity-bytes",
        "18446744073709551615",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("sample_accesses overflows u64"));
    assert!(!stderr.contains("panicked"));
}
