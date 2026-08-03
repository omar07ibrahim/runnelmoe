use std::process::Command;

use serde_json::Value;

fn model_check() -> Command {
    Command::new(env!("CARGO_BIN_EXE_runnel-m4-model-check"))
}

#[test]
fn canonical_invocation_emits_only_three_jsonl_rows() {
    let output = model_check().output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());

    let text = std::str::from_utf8(&output.stdout).unwrap();
    let rows: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter()
            .map(|row| row["check_id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["tiny-v1-preservation", "tiny-v2-scalar", "tiny-v2-avx2"]
    );
    assert_eq!(rows[0]["status"], "ok");
    assert_eq!(rows[1]["status"], "ok");
    assert!(matches!(
        rows[2]["status"].as_str(),
        Some("ok" | "unsupported")
    ));
}

#[test]
fn arguments_are_rejected_without_partial_stdout() {
    let output = model_check().arg("--help").output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(
        output.stderr,
        b"runnel-m4-model-check: usage: runnel-m4-model-check\n"
    );
    assert!(output.stderr.len() <= 560);
}
