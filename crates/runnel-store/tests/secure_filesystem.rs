use std::fs::{self, File};
use std::os::unix::fs::symlink;

use runnel_fixture::MultiPageFixture;
use runnel_store::{ArtifactSource, Control, ErrorCategory, StoreError};
use rustix::fs::Mode;

fn write_fixture() -> (tempfile::TempDir, std::path::PathBuf, MultiPageFixture) {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("artifact");
    let fixture = MultiPageFixture::build();
    fixture.write_new(&root).unwrap();
    (temporary, root, fixture)
}

#[test]
fn rejects_manifest_symlink_without_path_disclosure() {
    let (temporary, root, _) = write_fixture();
    let manifest = root.join("manifest.json");
    let outside = temporary.path().join("outside");
    fs::rename(&manifest, &outside).unwrap();
    symlink(&outside, &manifest).unwrap();

    let error = ArtifactSource::open(&root).unwrap_err();
    assert_eq!(error, StoreError::UnsafeLayout { kind: "manifest" });
    assert_eq!(error.category(), ErrorCategory::UnsafeFilesystem);
    assert!(
        !error
            .to_string()
            .contains(temporary.path().to_str().unwrap())
    );
}

#[test]
fn rejects_object_symlink() {
    let (temporary, root, fixture) = write_fixture();
    let identity = fixture.identity();
    let object = root
        .join("objects/sha256")
        .join(identity.object_digest.path_component());
    let outside = temporary.path().join("outside-object");
    fs::rename(&object, &outside).unwrap();
    symlink(&outside, &object).unwrap();

    let source = ArtifactSource::open(&root).unwrap();
    assert_eq!(
        source
            .open_with_expected_id(
                runnel_format::Limits::default(),
                identity.artifact_id,
                &Control::default(),
            )
            .unwrap_err(),
        StoreError::UnsafeLayout { kind: "object" }
    );
}

#[test]
fn rejects_fifo_manifest_without_blocking() {
    let (_temporary, root, _) = write_fixture();
    fs::remove_file(root.join("manifest.json")).unwrap();
    let directory = File::open(&root).unwrap();
    rustix::fs::mkfifoat(&directory, "manifest.json", Mode::RUSR | Mode::WUSR).unwrap();

    assert_eq!(
        ArtifactSource::open(&root).unwrap_err(),
        StoreError::UnsafeLayout { kind: "manifest" }
    );
}

#[test]
fn rejects_special_page_table_leaf() {
    let (_temporary, root, fixture) = write_fixture();
    let identity = fixture.identity();
    let page_table = root
        .join("page-tables/sha256")
        .join(identity.page_table_digest.path_component());
    fs::remove_file(&page_table).unwrap();
    fs::create_dir(&page_table).unwrap();

    let source = ArtifactSource::open(&root).unwrap();
    assert_eq!(
        source
            .open_with_expected_id(
                runnel_format::Limits::default(),
                identity.artifact_id,
                &Control::default(),
            )
            .unwrap_err(),
        StoreError::UnsafeLayout { kind: "page table" }
    );
}

#[test]
fn rejects_symlink_in_ancestor_component() {
    let (temporary, root, _) = write_fixture();
    let real_parent = root.parent().unwrap();
    let link = temporary.path().join("ancestor-link");
    symlink(real_parent, &link).unwrap();
    let redirected = link.join(root.file_name().unwrap());

    assert_eq!(
        ArtifactSource::open(redirected).unwrap_err(),
        StoreError::UnsafeLayout {
            kind: "artifact root"
        }
    );
}

#[test]
fn retained_directories_survive_root_rename() {
    let (temporary, root, fixture) = write_fixture();
    let source = ArtifactSource::open(&root).unwrap();
    let moved = temporary.path().join("moved-artifact");
    fs::rename(&root, moved).unwrap();

    let artifact = source
        .open_with_expected_id(
            runnel_format::Limits::default(),
            fixture.identity().artifact_id,
            &Control::default(),
        )
        .unwrap();
    assert_eq!(artifact.artifact_id(), fixture.identity().artifact_id);
}

#[test]
fn page_table_corruption_is_rejected_during_open() {
    let (_temporary, root, fixture) = write_fixture();
    let identity = fixture.identity();
    let page_table = root
        .join("page-tables/sha256")
        .join(identity.page_table_digest.path_component());
    let mut bytes = fs::read(&page_table).unwrap();
    bytes[80] ^= 0x40;
    fs::write(page_table, bytes).unwrap();

    let source = ArtifactSource::open(&root).unwrap();
    let error = source
        .open_with_expected_id(
            runnel_format::Limits::default(),
            identity.artifact_id,
            &Control::default(),
        )
        .unwrap_err();
    assert!(matches!(error, StoreError::Format(_)));
    assert_eq!(error.category(), ErrorCategory::Integrity);
    assert_eq!(error.to_string(), "artifact format validation failed");
}

#[test]
fn page_table_truncation_is_rejected_during_open() {
    let (_temporary, root, fixture) = write_fixture();
    let identity = fixture.identity();
    let page_table = root
        .join("page-tables/sha256")
        .join(identity.page_table_digest.path_component());
    OpenOptionsExt::truncate_to(&page_table, identity.page_table_length - 1);

    let source = ArtifactSource::open(&root).unwrap();
    assert!(matches!(
        source
            .open_with_expected_id(
                runnel_format::Limits::default(),
                identity.artifact_id,
                &Control::default(),
            )
            .unwrap_err(),
        StoreError::LengthMismatch {
            kind: "page table",
            ..
        }
    ));
}

struct OpenOptionsExt;

impl OpenOptionsExt {
    fn truncate_to(path: &std::path::Path, length: u64) {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(length)
            .unwrap();
    }
}
