use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::os::unix::fs::OpenOptionsExt as _;
use std::str::FromStr as _;
use std::sync::Arc;
use std::thread;

use runnel_fixture::FixtureArtifact;
use runnel_format::{Digest, Limits};
use runnel_store::{ArtifactSource, Cas, CasConfig, Control, DiskBudget, ResumeToken, StoreError};
use tempfile::tempdir;

fn config(max_bytes: u64, reserve_bytes: u64) -> CasConfig {
    CasConfig {
        disk_budget: DiskBudget::new(max_bytes, reserve_bytes),
        copy_buffer_bytes: 4_096,
    }
}

fn fixture_source() -> (tempfile::TempDir, FixtureArtifact, ArtifactSource) {
    let temporary = tempdir().unwrap();
    let fixture = FixtureArtifact::build();
    fixture.write_new(temporary.path().join("source")).unwrap();
    let source = ArtifactSource::open(temporary.path().join("source")).unwrap();
    (temporary, fixture, source)
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

fn resume_token(kind: &str, digest: Digest, length: u64) -> ResumeToken {
    let random = "0123456789abcdef0123456789abcdef";
    let stage = format!(
        "rmoa-stage-v1-{kind}-{}-{length}-{random}",
        digest.path_component()
    );
    ResumeToken::from_str(&format!(
        "rmoa-resume-v1|{kind}|{}|{length}|{stage}",
        digest.path_component()
    ))
    .unwrap()
}

fn multi_page_source() -> (tempfile::TempDir, ArtifactSource, Digest, Digest, Vec<u8>) {
    const PAGE_SIZE: usize = 65_536;
    let temporary = tempdir().unwrap();
    let root = temporary.path().join("source");
    fs::create_dir(&root).unwrap();
    fs::create_dir(root.join("objects")).unwrap();
    fs::create_dir(root.join("objects/sha256")).unwrap();
    fs::create_dir(root.join("page-tables")).unwrap();
    fs::create_dir(root.join("page-tables/sha256")).unwrap();

    let object: Vec<u8> = (0..70_000).map(|index| (index % 251) as u8).collect();
    let object_digest = Digest::of(&object);
    let mut table = Vec::new();
    table.extend_from_slice(b"RMOAPG1\n");
    table.extend_from_slice(&1_u32.to_le_bytes());
    table.extend_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
    table.extend_from_slice(&(object.len() as u64).to_le_bytes());
    table.extend_from_slice(&2_u64.to_le_bytes());
    table.extend_from_slice(object_digest.as_bytes());
    for page in object.chunks(PAGE_SIZE) {
        table.extend_from_slice(Digest::of(page).as_bytes());
    }
    let table_digest = Digest::of(&table);
    let mut manifest = String::new();
    write!(
        manifest,
        concat!(
            "{{",
            "\"adapter\":{{\"id\":\"runnel.tiny-causal-moe\",\"version\":1}},",
            "\"format\":\"rmoa\",",
            "\"model\":{{\"context_length\":16,\"expert_hidden_size\":12,",
            "\"hidden_size\":8,\"num_experts\":4,\"num_heads\":2,",
            "\"num_layers\":1,\"top_k\":2,\"vocab_size\":32}},",
            "\"objects\":[{{\"digest\":\"{}\",\"length\":{},",
            "\"page_size\":65536,\"page_table\":\"{}\",",
            "\"page_table_length\":{}}}],",
            "\"tensors\":[{{\"dtype\":\"u8\",\"id\":0,\"length\":{},",
            "\"object\":\"{}\",\"offset\":0,\"role\":\"payload\",",
            "\"shape\":[{}]}}],",
            "\"tokenizer\":{{\"id\":\"runnel.ascii32\",\"version\":1,",
            "\"vocab_size\":32}},\"version\":1}}\n"
        ),
        object_digest,
        object.len(),
        table_digest,
        table.len(),
        object.len(),
        object_digest,
        object.len(),
    )
    .unwrap();
    let manifest = manifest.into_bytes();
    let artifact_id = Digest::of(&manifest);
    write_private(
        &root
            .join("objects/sha256")
            .join(object_digest.path_component()),
        &object,
    );
    write_private(
        &root
            .join("page-tables/sha256")
            .join(table_digest.path_component()),
        &table,
    );
    write_private(&root.join("manifest.json"), &manifest);
    let source = ArtifactSource::open(&root).unwrap();
    (temporary, source, artifact_id, object_digest, object)
}

#[test]
fn import_is_manifest_last_idempotent_and_openable() {
    let (_source_root, fixture, source) = fixture_source();
    let temporary = tempdir().unwrap();
    let cas_root = temporary.path().join("cas");
    let cas = Cas::create(&cas_root, config(1024 * 1024, 0)).unwrap();
    let identity = fixture.identity();

    let first = cas
        .import(
            &source,
            identity.artifact_id,
            Limits::default(),
            &Control::unbounded(),
        )
        .unwrap();
    assert_eq!(first.artifact_id, identity.artifact_id);
    assert_eq!(first.published_blobs, 3);
    assert_eq!(first.reused_blobs, 0);
    assert!(
        cas_root
            .join("manifests/sha256")
            .join(identity.artifact_id.path_component())
            .is_file()
    );

    let second = cas
        .import(
            &source,
            identity.artifact_id,
            Limits::default(),
            &Control::unbounded(),
        )
        .unwrap();
    assert_eq!(second.published_blobs, 0);
    assert_eq!(second.reused_blobs, 3);

    let artifact = cas
        .open_artifact(
            identity.artifact_id,
            Limits::default(),
            &Control::unbounded(),
        )
        .unwrap();
    assert_eq!(artifact.artifact_id(), identity.artifact_id);
    let page = artifact.page_specs().next().unwrap().unwrap();
    let (verified, _) = artifact
        .reader()
        .read(&page, &Control::unbounded())
        .unwrap();
    assert!(!verified.is_empty());
}

#[test]
fn corrupt_preexisting_object_fails_before_manifest_commit() {
    let (_source_root, fixture, source) = fixture_source();
    let temporary = tempdir().unwrap();
    let cas_root = temporary.path().join("cas");
    let cas = Cas::create(&cas_root, config(1024 * 1024, 0)).unwrap();
    let identity = fixture.identity();
    let corrupt = vec![0x5a; usize::try_from(identity.object_length).unwrap()];
    write_private(
        &cas_root
            .join("objects/sha256")
            .join(identity.object_digest.path_component()),
        &corrupt,
    );

    assert!(matches!(
        cas.import(
            &source,
            identity.artifact_id,
            Limits::default(),
            &Control::unbounded(),
        ),
        Err(StoreError::Integrity { kind: "object" })
    ));
    assert!(
        !cas_root
            .join("manifests/sha256")
            .join(identity.artifact_id.path_component())
            .exists()
    );
}

#[test]
fn exact_resume_offset_succeeds_and_corrupt_prefix_fails_closed() {
    let (_source_root, fixture, source) = fixture_source();
    let identity = fixture.identity();
    let parts = fixture.to_parts();
    let object = parts.objects.get(&identity.object_digest).unwrap();

    let temporary = tempdir().unwrap();
    let cas_root = temporary.path().join("cas");
    let cas = Cas::create(&cas_root, config(1024 * 1024, 0)).unwrap();
    let token = resume_token("object", identity.object_digest, identity.object_length);
    write_private(
        &cas_root.join("staging").join(token.stage_name()),
        &object[..100],
    );
    assert_eq!(cas.resume_tokens().unwrap(), std::slice::from_ref(&token));
    cas.import_with_resumes(
        &source,
        identity.artifact_id,
        Limits::default(),
        &[token],
        &Control::unbounded(),
    )
    .unwrap();

    let temporary = tempdir().unwrap();
    let cas_root = temporary.path().join("cas");
    let cas = Cas::create(&cas_root, config(1024 * 1024, 0)).unwrap();
    let token = resume_token(
        "page-table",
        identity.page_table_digest,
        identity.page_table_length,
    );
    write_private(
        &cas_root.join("staging").join(token.stage_name()),
        &[0xff; 10],
    );
    assert!(matches!(
        cas.import_with_resumes(
            &source,
            identity.artifact_id,
            Limits::default(),
            &[token],
            &Control::unbounded(),
        ),
        Err(StoreError::Integrity {
            kind: "staging prefix"
        })
    ));
    assert!(
        !cas_root
            .join("manifests/sha256")
            .join(identity.artifact_id.path_component())
            .exists()
    );
}

#[test]
fn concurrent_imports_serialize_and_converge() {
    let (_source_root, fixture, source) = fixture_source();
    let temporary = tempdir().unwrap();
    let cas = Arc::new(Cas::create(temporary.path().join("cas"), config(1024 * 1024, 0)).unwrap());
    let expected = fixture.identity().artifact_id;
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let cas = Arc::clone(&cas);
            let source = source.clone();
            thread::spawn(move || {
                cas.import(&source, expected, Limits::default(), &Control::unbounded())
                    .unwrap()
            })
        })
        .collect();
    let mut receipts: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    receipts.sort_by_key(|receipt| receipt.published_blobs);
    assert_eq!(receipts[0].reused_blobs, 3);
    assert_eq!(receipts[1].published_blobs, 3);
}

#[test]
fn cancellation_before_import_is_mutation_free() {
    let (_source_root, fixture, source) = fixture_source();
    let temporary = tempdir().unwrap();
    let cas_root = temporary.path().join("cas");
    let cas = Cas::create(&cas_root, config(1024 * 1024, 0)).unwrap();
    let token = runnel_store::CancellationToken::new();
    token.cancel();
    let control = Control::with_cancellation(token);
    assert_eq!(
        cas.import(
            &source,
            fixture.identity().artifact_id,
            Limits::default(),
            &control,
        )
        .unwrap_err(),
        StoreError::Cancelled
    );
    assert_eq!(
        fs::read_dir(cas_root.join("manifests/sha256"))
            .unwrap()
            .count(),
        0
    );
    assert_eq!(fs::read_dir(cas_root.join("staging")).unwrap().count(), 0);
}

#[test]
fn resumed_stage_requires_exact_private_mode() {
    use std::os::unix::fs::PermissionsExt as _;

    let (_source_root, fixture, _source) = fixture_source();
    let temporary = tempdir().unwrap();
    let cas_root = temporary.path().join("cas");
    let cas = Cas::create(&cas_root, config(1024 * 1024, 0)).unwrap();
    let identity = fixture.identity();
    let token = resume_token("object", identity.object_digest, identity.object_length);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o644)
        .open(cas_root.join("staging").join(token.stage_name()))
        .unwrap();
    file.write_all(b"partial").unwrap();
    drop(file);
    fs::set_permissions(
        cas_root.join("staging").join(token.stage_name()),
        fs::Permissions::from_mode(0o644),
    )
    .unwrap();
    assert_eq!(
        cas.resume_tokens().unwrap_err(),
        StoreError::UnsafeLayout {
            kind: "staging file"
        }
    );
}

#[test]
fn multi_page_resume_discards_incomplete_page_to_nonzero_boundary() {
    const PAGE_SIZE: usize = 65_536;
    let (_source_root, source, artifact_id, object_digest, object) = multi_page_source();
    let temporary = tempdir().unwrap();
    let cas_root = temporary.path().join("cas");
    let cas = Cas::create(&cas_root, config(1024 * 1024, 0)).unwrap();
    let token = resume_token("object", object_digest, object.len() as u64);
    let mut staged = object[..PAGE_SIZE].to_vec();
    staged.extend_from_slice(&[0xff; 100]);
    write_private(&cas_root.join("staging").join(token.stage_name()), &staged);

    cas.import_with_resumes(
        &source,
        artifact_id,
        Limits::default(),
        &[token],
        &Control::unbounded(),
    )
    .unwrap();
    let stored = cas
        .open_artifact(artifact_id, Limits::default(), &Control::unbounded())
        .unwrap();
    let pages: Vec<_> = stored.page_specs().collect::<Result<_, _>>().unwrap();
    assert_eq!(pages.len(), 2);
    for page in pages {
        stored.reader().read(&page, &Control::unbounded()).unwrap();
    }
}
