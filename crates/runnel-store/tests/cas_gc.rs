use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::os::unix::fs::OpenOptionsExt as _;
use std::time::{Duration, Instant};

use runnel_fixture::FixtureArtifact;
use runnel_format::{Digest, Limits};
use runnel_store::{
    ArtifactSource, CancellationToken, Cas, CasConfig, Control, DiskBudget, GcPolicy, StoreError,
};
use rustix::fs::FlockOperation;
use tempfile::tempdir;

fn config() -> CasConfig {
    CasConfig {
        disk_budget: DiskBudget::new(1024 * 1024, 0),
        copy_buffer_bytes: 4_096,
    }
}

fn write_private(path: &std::path::Path, bytes: &[u8]) {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .unwrap();
    file.write_all(bytes).unwrap();
    file.sync_all().unwrap();
}

fn imported_fixture() -> (tempfile::TempDir, FixtureArtifact, Cas, std::path::PathBuf) {
    let temporary = tempdir().unwrap();
    let fixture = FixtureArtifact::build();
    let source_root = temporary.path().join("source");
    fixture.write_new(&source_root).unwrap();
    let source = ArtifactSource::open(&source_root).unwrap();
    let cas_root = temporary.path().join("cas");
    let cas = Cas::create(&cas_root, config()).unwrap();
    cas.import(
        &source,
        fixture.identity().artifact_id,
        Limits::default(),
        &Control::unbounded(),
    )
    .unwrap();
    (temporary, fixture, cas, cas_root)
}

#[test]
fn gc_dry_run_and_sweep_delete_only_verified_orphans() {
    let (_temporary, _fixture, cas, cas_root) = imported_fixture();
    let orphan_bytes = b"verified orphan";
    let orphan = Digest::of(orphan_bytes);
    let orphan_path = cas_root
        .join("objects/sha256")
        .join(orphan.path_component());
    write_private(&orphan_path, orphan_bytes);

    let dry = cas
        .gc(
            GcPolicy {
                dry_run: true,
                remove_staging: false,
            },
            Limits::default(),
            &Control::unbounded(),
        )
        .unwrap();
    assert_eq!(dry.candidate_blobs, 1);
    assert_eq!(dry.deleted_blobs, 0);
    assert!(orphan_path.exists());

    let swept = cas
        .gc(
            GcPolicy::default(),
            Limits::default(),
            &Control::unbounded(),
        )
        .unwrap();
    assert_eq!(swept.candidate_blobs, 1);
    assert_eq!(swept.deleted_blobs, 1);
    assert!(!orphan_path.exists());
}

#[test]
fn shared_blobs_remain_live_across_multiple_manifest_roots() {
    let (temporary, fixture, cas, cas_root) = imported_fixture();
    let second_root = temporary.path().join("source-two");
    fixture.write_new(&second_root).unwrap();
    let second_manifest = String::from_utf8(fixture.manifest_bytes().to_vec())
        .unwrap()
        .replace("runnel.ascii32", "runnel.ascii33")
        .into_bytes();
    fs::write(second_root.join("manifest.json"), &second_manifest).unwrap();
    let second_id = Digest::of(&second_manifest);
    let second_source = ArtifactSource::open(&second_root).unwrap();
    cas.import(
        &second_source,
        second_id,
        Limits::default(),
        &Control::unbounded(),
    )
    .unwrap();

    let report = cas
        .gc(
            GcPolicy::default(),
            Limits::default(),
            &Control::unbounded(),
        )
        .unwrap();
    assert_eq!(report.candidate_blobs, 0);
    assert!(
        cas_root
            .join("objects/sha256")
            .join(fixture.identity().object_digest.path_component())
            .exists()
    );
    assert_eq!(
        fs::read_dir(cas_root.join("manifests/sha256"))
            .unwrap()
            .count(),
        2
    );
}

#[test]
fn malformed_or_unexpected_entry_causes_zero_deletion() {
    let (_temporary, _fixture, cas, cas_root) = imported_fixture();
    let orphan_bytes = b"keep until a complete valid plan exists";
    let orphan = Digest::of(orphan_bytes);
    let orphan_path = cas_root
        .join("objects/sha256")
        .join(orphan.path_component());
    write_private(&orphan_path, orphan_bytes);
    let surprise = cas_root.join("objects/sha256/not-a-digest");
    write_private(&surprise, b"unexpected");

    assert!(matches!(
        cas.gc(
            GcPolicy::default(),
            Limits::default(),
            &Control::unbounded(),
        ),
        Err(StoreError::UnsafeLayout { .. })
    ));
    assert!(orphan_path.exists());

    fs::remove_file(surprise).unwrap();
    let malformed = b"{}\n";
    let malformed_id = Digest::of(malformed);
    write_private(
        &cas_root
            .join("manifests/sha256")
            .join(malformed_id.path_component()),
        malformed,
    );
    assert!(matches!(
        cas.gc(
            GcPolicy::default(),
            Limits::default(),
            &Control::unbounded(),
        ),
        Err(StoreError::Format(_))
    ));
    assert!(orphan_path.exists());
}

#[test]
fn cancelled_gc_is_mutation_free() {
    let (_temporary, _fixture, cas, cas_root) = imported_fixture();
    let orphan_bytes = b"cancelled orphan";
    let orphan = Digest::of(orphan_bytes);
    let orphan_path = cas_root
        .join("objects/sha256")
        .join(orphan.path_component());
    write_private(&orphan_path, orphan_bytes);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert_eq!(
        cas.gc(
            GcPolicy::default(),
            Limits::default(),
            &Control::with_cancellation(cancellation),
        )
        .unwrap_err(),
        StoreError::Cancelled
    );
    assert!(orphan_path.exists());
}

#[test]
fn transaction_lock_wait_observes_deadline() {
    let (_temporary, _fixture, cas, cas_root) = imported_fixture();
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .open(cas_root.join("transaction.lock"))
        .unwrap();
    rustix::fs::flock(&lock, FlockOperation::LockExclusive).unwrap();
    let result = cas.gc(
        GcPolicy::default(),
        Limits::default(),
        &Control::with_deadline(Instant::now() + Duration::from_millis(10)),
    );
    rustix::fs::flock(&lock, FlockOperation::Unlock).unwrap();
    assert_eq!(result.unwrap_err(), StoreError::DeadlineExceeded);
}
