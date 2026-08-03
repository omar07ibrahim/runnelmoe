use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::time::{Duration, Instant};

use runnel_fixture::{MULTI_PAGE_OBJECT_LENGTH, MultiPageFixture, PAGE_SIZE};
use runnel_store::{
    ArtifactSource, CancellationToken, Control, ErrorCategory, PageKey, StoreError,
};
use tempfile::TempDir;

struct WrittenFixture {
    _temporary: TempDir,
    root: std::path::PathBuf,
    fixture: MultiPageFixture,
}

impl WrittenFixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("artifact");
        let fixture = MultiPageFixture::build();
        fixture.write_new(&root).unwrap();
        Self {
            _temporary: temporary,
            root,
            fixture,
        }
    }

    fn object_path(&self) -> std::path::PathBuf {
        self.root
            .join("objects/sha256")
            .join(self.fixture.identity().object_digest.path_component())
    }

    fn open(&self) -> (ArtifactSource, runnel_store::StoredArtifact) {
        let source = ArtifactSource::open(&self.root).unwrap();
        let artifact = source
            .open_with_expected_id(
                runnel_format::Limits::default(),
                self.fixture.identity().artifact_id,
                &Control::default(),
            )
            .unwrap();
        (source, artifact)
    }
}

#[test]
fn reads_full_pages_and_short_final_page() {
    let written = WrittenFixture::new();
    let (_, artifact) = written.open();
    let identity = written.fixture.identity();
    let reader = artifact.reader();

    for index in 0..3 {
        let specification = artifact
            .page_spec(PageKey::new(
                identity.object_digest,
                u64::from(PAGE_SIZE),
                index,
            ))
            .unwrap();
        let (page, stats) = reader.read(&specification, &Control::default()).unwrap();
        let start = index as usize * PAGE_SIZE as usize;
        let end = (start + PAGE_SIZE as usize).min(MULTI_PAGE_OBJECT_LENGTH);
        assert_eq!(page.key(), specification.key());
        assert_eq!(page.bytes(), &written.fixture.object_bytes()[start..end]);
        assert_eq!(stats.physical_bytes(), (end - start) as u64);
    }

    let final_specification = artifact
        .page_spec(PageKey::new(
            identity.object_digest,
            u64::from(PAGE_SIZE),
            2,
        ))
        .unwrap();
    assert_eq!(final_specification.length(), 17);
}

#[test]
fn out_of_range_and_wrong_geometry_are_rejected() {
    let written = WrittenFixture::new();
    let (_, artifact) = written.open();
    let digest = written.fixture.identity().object_digest;

    assert_eq!(
        artifact
            .page_spec(PageKey::new(digest, u64::from(PAGE_SIZE), 3))
            .unwrap_err(),
        StoreError::PageOutOfRange { index: 3, count: 3 }
    );
    assert_eq!(
        artifact
            .page_spec(PageKey::new(digest, u64::from(PAGE_SIZE) * 2, 0))
            .unwrap_err(),
        StoreError::Integrity {
            kind: "page geometry"
        }
    );
}

#[test]
fn corruption_after_open_fails_without_exposing_a_page() {
    let written = WrittenFixture::new();
    let (_, artifact) = written.open();
    let digest = written.fixture.identity().object_digest;
    let specification = artifact
        .page_spec(PageKey::new(digest, u64::from(PAGE_SIZE), 1))
        .unwrap();

    let mut object = OpenOptions::new()
        .write(true)
        .open(written.object_path())
        .unwrap();
    object
        .seek(SeekFrom::Start(u64::from(PAGE_SIZE) + 37))
        .unwrap();
    object.write_all(&[0xff]).unwrap();
    object.sync_data().unwrap();

    let error = artifact
        .reader()
        .read(&specification, &Control::default())
        .unwrap_err();
    assert_eq!(error, StoreError::Integrity { kind: "page" });
    assert_eq!(error.category(), ErrorCategory::Integrity);
}

#[test]
fn truncation_after_open_is_detected_before_exposure() {
    let written = WrittenFixture::new();
    let (_, artifact) = written.open();
    let digest = written.fixture.identity().object_digest;
    let specification = artifact
        .page_spec(PageKey::new(digest, u64::from(PAGE_SIZE), 0))
        .unwrap();
    OpenOptions::new()
        .write(true)
        .open(written.object_path())
        .unwrap()
        .set_len(PAGE_SIZE as u64)
        .unwrap();

    assert!(matches!(
        artifact
            .reader()
            .read(&specification, &Control::default())
            .unwrap_err(),
        StoreError::LengthMismatch { kind: "object", .. }
    ));
}

#[test]
fn extension_after_open_is_detected_before_exposure() {
    let written = WrittenFixture::new();
    let (_, artifact) = written.open();
    let digest = written.fixture.identity().object_digest;
    let specification = artifact
        .page_spec(PageKey::new(digest, u64::from(PAGE_SIZE), 0))
        .unwrap();
    let file = OpenOptions::new()
        .write(true)
        .open(written.object_path())
        .unwrap();
    file.set_len(MULTI_PAGE_OBJECT_LENGTH as u64 + 1).unwrap();

    assert!(matches!(
        artifact
            .reader()
            .read(&specification, &Control::default())
            .unwrap_err(),
        StoreError::LengthMismatch { kind: "object", .. }
    ));
}

#[test]
fn reordered_valid_pages_fail_their_bound_hashes() {
    let written = WrittenFixture::new();
    let (_, artifact) = written.open();
    let digest = written.fixture.identity().object_digest;
    let mut reordered = written.fixture.object_bytes().to_vec();
    let (first, rest) = reordered.split_at_mut(PAGE_SIZE as usize);
    let second = &mut rest[..PAGE_SIZE as usize];
    first.swap_with_slice(second);
    fs::write(written.object_path(), reordered).unwrap();

    for index in [0, 1] {
        let specification = artifact
            .page_spec(PageKey::new(digest, u64::from(PAGE_SIZE), index))
            .unwrap();
        assert_eq!(
            artifact
                .reader()
                .read(&specification, &Control::default())
                .unwrap_err(),
            StoreError::Integrity { kind: "page" }
        );
    }
}

#[test]
fn cancellation_and_deadline_stop_reads() {
    let written = WrittenFixture::new();
    let (_, artifact) = written.open();
    let digest = written.fixture.identity().object_digest;
    let specification = artifact
        .page_spec(PageKey::new(digest, u64::from(PAGE_SIZE), 0))
        .unwrap();

    let token = CancellationToken::new();
    token.cancel();
    let cancelled = Control::with_cancellation(token);
    assert_eq!(
        artifact
            .reader()
            .read(&specification, &cancelled)
            .unwrap_err(),
        StoreError::Cancelled
    );

    let expired = Control::with_deadline(Instant::now() - Duration::from_nanos(1));
    assert_eq!(
        artifact
            .reader()
            .read(&specification, &expired)
            .unwrap_err(),
        StoreError::DeadlineExceeded
    );
}

#[test]
fn retained_object_and_table_descriptors_survive_unlink() {
    let written = WrittenFixture::new();
    let (_, artifact) = written.open();
    let identity = written.fixture.identity();
    let specification = artifact
        .page_spec(PageKey::new(
            identity.object_digest,
            u64::from(PAGE_SIZE),
            2,
        ))
        .unwrap();

    fs::remove_file(written.object_path()).unwrap();
    fs::remove_file(
        written
            .root
            .join("page-tables/sha256")
            .join(identity.page_table_digest.path_component()),
    )
    .unwrap();

    let (page, _) = artifact
        .reader()
        .read(&specification, &Control::default())
        .unwrap();
    assert_eq!(
        page.bytes(),
        &written.fixture.object_bytes()[2 * PAGE_SIZE as usize..]
    );
}

#[test]
fn tensor_page_iteration_covers_exact_intersecting_pages() {
    let written = WrittenFixture::new();
    let (_, artifact) = written.open();
    let specifications = artifact
        .tensor_page_specs(0)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(specifications.len(), 3);
    assert_eq!(specifications[0].offset(), 0);
    assert_eq!(specifications[2].length(), 17);
}
