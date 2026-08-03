use runnel_fixture::FixtureArtifact;
use runnel_format::Limits;
use runnel_store::{ArtifactSource, Cas, CasConfig, Control, DiskBudget, StoreError};
use tempfile::tempdir;

fn rounded(length: u64, unit: u64) -> u64 {
    length.div_ceil(unit) * unit
}

fn fixture_source() -> (tempfile::TempDir, FixtureArtifact, ArtifactSource) {
    let temporary = tempdir().unwrap();
    let fixture = FixtureArtifact::build();
    fixture.write_new(temporary.path().join("source")).unwrap();
    let source = ArtifactSource::open(temporary.path().join("source")).unwrap();
    (temporary, fixture, source)
}

fn required_bytes(cas: &Cas, fixture: &FixtureArtifact) -> u64 {
    let usage = cas.usage().unwrap();
    let identity = fixture.identity();
    rounded(fixture.manifest_bytes().len() as u64, usage.allocation_unit)
        + rounded(identity.page_table_length, usage.allocation_unit)
        + rounded(identity.object_length, usage.allocation_unit)
}

#[test]
fn exact_cas_budget_succeeds_and_one_byte_less_fails() {
    let (_source_root, fixture, source) = fixture_source();
    let temporary = tempdir().unwrap();
    let root = temporary.path().join("cas");
    let bootstrap = Cas::create(
        &root,
        CasConfig {
            disk_budget: DiskBudget::new(u64::MAX, 0),
            copy_buffer_bytes: 4_096,
        },
    )
    .unwrap();
    let required = required_bytes(&bootstrap, &fixture);
    drop(bootstrap);
    let cas = Cas::open(
        &root,
        CasConfig {
            disk_budget: DiskBudget::new(required, 0),
            copy_buffer_bytes: 4_096,
        },
    )
    .unwrap();
    cas.import(
        &source,
        fixture.identity().artifact_id,
        Limits::default(),
        &Control::unbounded(),
    )
    .unwrap();

    let temporary = tempdir().unwrap();
    let root = temporary.path().join("cas");
    let bootstrap = Cas::create(
        &root,
        CasConfig {
            disk_budget: DiskBudget::new(u64::MAX, 0),
            copy_buffer_bytes: 4_096,
        },
    )
    .unwrap();
    let required = required_bytes(&bootstrap, &fixture);
    drop(bootstrap);
    let cas = Cas::open(
        &root,
        CasConfig {
            disk_budget: DiskBudget::new(required - 1, 0),
            copy_buffer_bytes: 4_096,
        },
    )
    .unwrap();
    assert_eq!(
        cas.import(
            &source,
            fixture.identity().artifact_id,
            Limits::default(),
            &Control::unbounded(),
        )
        .unwrap_err(),
        StoreError::BudgetExceeded {
            required,
            limit: required - 1,
        }
    );
}

#[test]
fn filesystem_reserve_accepts_zero_and_rejects_an_impossible_reserve() {
    let (_source_root, fixture, source) = fixture_source();
    for should_succeed in [true, false] {
        let temporary = tempdir().unwrap();
        let root = temporary.path().join("cas");
        let bootstrap = Cas::create(
            &root,
            CasConfig {
                disk_budget: DiskBudget::new(u64::MAX, 0),
                copy_buffer_bytes: 4_096,
            },
        )
        .unwrap();
        drop(bootstrap);
        let cas = Cas::open(
            &root,
            CasConfig {
                // A boundary derived from live f_bavail is inherently racy
                // with unrelated filesystem writers. Exact arithmetic is
                // covered by the injected DiskUsage unit test; this
                // integration test uses stable end points.
                disk_budget: DiskBudget::new(u64::MAX, if should_succeed { 0 } else { u64::MAX }),
                copy_buffer_bytes: 4_096,
            },
        )
        .unwrap();
        let result = cas.import(
            &source,
            fixture.identity().artifact_id,
            Limits::default(),
            &Control::unbounded(),
        );
        if should_succeed {
            result.unwrap();
        } else {
            assert!(
                matches!(result, Err(StoreError::FilesystemReserve { .. })),
                "unexpected reserve result: {result:?}"
            );
        }
    }
}
