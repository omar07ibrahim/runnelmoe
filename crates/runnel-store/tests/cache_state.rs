use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::sync::Arc;
use std::time::Duration;

use runnel_fixture::{MultiPageFixture, PAGE_SIZE};
use runnel_format::Limits;
use runnel_store::{
    ArtifactSource, AsyncReader, AsyncReaderConfig, CacheConfig, Control, ErrorCategory, PageCache,
    PageSpec, PrefetchOutcome,
};
use tempfile::TempDir;
use tokio::sync::Barrier;

struct OpenFixture {
    _temporary: TempDir,
    object_path: std::path::PathBuf,
    specs: Vec<PageSpec>,
}

fn open_fixture() -> OpenFixture {
    let temporary = tempfile::tempdir().expect("temporary root");
    let root = temporary.path().join("artifact");
    let fixture = MultiPageFixture::build();
    let identity = fixture.write_new(&root).expect("write fixture");
    let object_path = root
        .join("objects/sha256")
        .join(identity.object_digest.path_component());
    let source = ArtifactSource::open(&root).expect("open descriptor-safe source");
    let artifact = source
        .open_with_expected_id(
            Limits::default(),
            identity.artifact_id,
            &Control::unbounded(),
        )
        .expect("open stored artifact");
    let specs = artifact
        .page_specs()
        .collect::<Result<Vec<_>, _>>()
        .expect("page specifications");
    OpenFixture {
        _temporary: temporary,
        object_path,
        specs,
    }
}

fn cache_with(config: CacheConfig) -> PageCache {
    let reader = AsyncReader::new(
        runnel_store::SyncReader::new(),
        AsyncReaderConfig::new(2, 8).expect("async configuration"),
    )
    .expect("async reader");
    PageCache::new(reader, config).expect("page cache")
}

async fn wait_until(mut predicate: impl FnMut() -> bool) {
    for _ in 0..1_000 {
        if predicate() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("condition did not become true");
}

fn one_page_config() -> CacheConfig {
    CacheConfig::new(u64::from(PAGE_SIZE), u64::from(PAGE_SIZE))
        .expect("cache configuration")
        .with_max_inflight_bytes(u64::from(PAGE_SIZE))
        .with_max_loads(1)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_budget_never_reclaims_a_leased_page() {
    let fixture = open_fixture();
    let cache = cache_with(one_page_config());
    let first = cache
        .get(fixture.specs[0].clone(), Control::unbounded())
        .await
        .expect("first lease");
    let snapshot = cache.metrics();
    assert_eq!(snapshot.page_pool_bytes, u64::from(PAGE_SIZE));
    assert_eq!(snapshot.resident_bytes, u64::from(PAGE_SIZE));
    assert_eq!(snapshot.leases, 1);

    let blocked = cache
        .get(fixture.specs[1].clone(), Control::unbounded())
        .await
        .expect_err("leased page must not be reclaimed");
    assert_eq!(blocked.category(), ErrorCategory::ResourceExhausted);
    assert_eq!(cache.metrics().page_pool_bytes, u64::from(PAGE_SIZE));

    drop(first);
    let second = cache
        .get(fixture.specs[1].clone(), Control::unbounded())
        .await
        .expect("load after lease release");
    assert_eq!(second.len(), PAGE_SIZE as usize);
    assert_eq!(cache.metrics().evictions, 1);
    assert!(cache.metrics().page_pool_bytes <= u64::from(PAGE_SIZE));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_admission_keeps_planned_victims_and_ledger_unchanged() {
    const SHORT_PAGE_ALLOCATION: u64 = 64;

    let fixture = open_fixture();
    assert_eq!(fixture.specs[2].length(), 17);
    let full_page_bytes = u64::from(PAGE_SIZE);
    let capacity = full_page_bytes + SHORT_PAGE_ALLOCATION;
    let cache = cache_with(
        CacheConfig::new(capacity, full_page_bytes)
            .expect("full page plus aligned short-tail allocation")
            .with_max_inflight_bytes(full_page_bytes)
            .with_max_loads(1),
    );

    let pinned = cache
        .get(fixture.specs[0].clone(), Control::unbounded())
        .await
        .expect("pinned full page");
    let short = cache
        .get(fixture.specs[2].clone(), Control::unbounded())
        .await
        .expect("short resident page");
    drop(short);
    let before = cache.metrics();
    assert_eq!(before.page_pool_bytes, capacity);
    assert_eq!(before.resident_bytes, capacity);
    assert_eq!(before.inflight_bytes, 0);
    assert_eq!(before.retiring_bytes, 0);
    assert_eq!(before.leases, 1);
    assert_eq!(
        before.page_pool_bytes,
        before.inflight_bytes + before.resident_bytes + before.retiring_bytes
    );

    let rejected = cache
        .get(fixture.specs[1].clone(), Control::unbounded())
        .await
        .expect_err("evicting only the short tail cannot make a full page fit");
    assert_eq!(rejected.category(), ErrorCategory::ResourceExhausted);
    let after = cache.metrics();
    assert_eq!(after.page_pool_bytes, before.page_pool_bytes);
    assert_eq!(after.resident_bytes, before.resident_bytes);
    assert_eq!(after.inflight_bytes, 0);
    assert_eq!(after.retiring_bytes, 0);
    assert_eq!(after.leases, 1);
    assert_eq!(after.evictions, before.evictions);
    assert_eq!(
        after.page_pool_bytes,
        after.inflight_bytes + after.resident_bytes + after.retiring_bytes
    );

    let physical_before_hit = after.physical_read_bytes;
    let retained_short = cache
        .get(fixture.specs[2].clone(), Control::unbounded())
        .await
        .expect("failed admission must not evict a planned victim");
    assert_eq!(retained_short.len(), 17);
    assert_eq!(cache.metrics().physical_read_bytes, physical_before_hit);
    drop(retained_short);
    drop(pinned);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalidated_lease_remains_charged_until_drop() {
    let fixture = open_fixture();
    let cache = cache_with(one_page_config());
    let lease = cache
        .get(fixture.specs[0].clone(), Control::unbounded())
        .await
        .expect("lease");
    assert!(cache.invalidate(lease.key()));
    let retiring = cache.metrics();
    assert_eq!(retiring.resident_bytes, 0);
    assert_eq!(retiring.retiring_bytes, u64::from(PAGE_SIZE));
    assert_eq!(retiring.page_pool_bytes, u64::from(PAGE_SIZE));
    assert_eq!(retiring.leases, 1);

    drop(lease);
    let released = cache.metrics();
    assert_eq!(released.page_pool_bytes, 0);
    assert_eq!(released.retiring_bytes, 0);
    assert_eq!(released.leases, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn duplicate_demanders_share_one_physical_page() {
    const WAITERS: usize = 64;

    let fixture = open_fixture();
    let cache = cache_with(
        one_page_config()
            .with_waiter_limits(WAITERS + 1, WAITERS + 1)
            .with_max_leases(WAITERS + 1),
    );
    let barrier = Arc::new(Barrier::new(WAITERS + 1));
    let mut tasks = Vec::with_capacity(WAITERS);
    for _ in 0..WAITERS {
        let task_cache = cache.clone();
        let specification = fixture.specs[0].clone();
        let task_barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            task_barrier.wait().await;
            task_cache.get(specification, Control::unbounded()).await
        }));
    }
    barrier.wait().await;
    for task in tasks {
        let lease = task.await.expect("waiter task").expect("coalesced lease");
        assert_eq!(lease.len(), PAGE_SIZE as usize);
    }

    let snapshot = cache.metrics();
    assert_eq!(snapshot.physical_read_bytes, u64::from(PAGE_SIZE));
    assert_eq!(snapshot.admissions, 1);
    assert_eq!(snapshot.hits + snapshot.misses, WAITERS as u64);
    assert!(snapshot.coalesced_demands >= 1 || snapshot.hits >= 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefetch_counters_follow_resident_truth_table() {
    let fixture = open_fixture();
    let cache = cache_with(one_page_config());

    assert_eq!(
        cache
            .prefetch(fixture.specs[0].clone(), Control::unbounded())
            .await
            .expect("first prefetch"),
        PrefetchOutcome::Started
    );
    wait_until(|| cache.metrics().resident_bytes == u64::from(PAGE_SIZE)).await;
    assert_eq!(
        cache
            .prefetch(fixture.specs[0].clone(), Control::unbounded())
            .await
            .expect("redundant prefetch"),
        PrefetchOutcome::Redundant
    );
    let lease = cache
        .get(fixture.specs[0].clone(), Control::unbounded())
        .await
        .expect("use prefetched page");
    assert_eq!(cache.metrics().useful_prefetches, 1);
    drop(lease);

    assert_eq!(
        cache
            .prefetch(fixture.specs[1].clone(), Control::unbounded())
            .await
            .expect("second prefetch"),
        PrefetchOutcome::Started
    );
    wait_until(|| cache.metrics().admissions == 2).await;
    assert_eq!(
        cache
            .prefetch(fixture.specs[2].clone(), Control::unbounded())
            .await
            .expect("third prefetch"),
        PrefetchOutcome::Started
    );
    wait_until(|| cache.metrics().admissions == 3).await;
    let snapshot = cache.metrics();
    assert_eq!(snapshot.redundant_prefetches, 1);
    assert_eq!(snapshot.useful_prefetches, 1);
    assert_eq!(snapshot.wasted_prefetches, 1);
    assert_eq!(snapshot.dropped_prefetches, 0);
    assert!(snapshot.page_pool_bytes <= u64::from(PAGE_SIZE));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn corruption_failure_fans_out_and_releases_reserved_bytes() {
    const WAITERS: usize = 32;

    let fixture = open_fixture();
    let mut object = OpenOptions::new()
        .write(true)
        .open(&fixture.object_path)
        .expect("open object for fault injection");
    object.seek(SeekFrom::Start(0)).expect("seek object");
    object.write_all(&[0xff]).expect("corrupt first page");
    object.sync_all().expect("sync corruption");

    let cache = cache_with(
        one_page_config()
            .with_waiter_limits(WAITERS + 1, WAITERS + 1)
            .with_max_leases(WAITERS + 1),
    );
    let barrier = Arc::new(Barrier::new(WAITERS + 1));
    let mut tasks = Vec::with_capacity(WAITERS);
    for _ in 0..WAITERS {
        let task_cache = cache.clone();
        let specification = fixture.specs[0].clone();
        let task_barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            task_barrier.wait().await;
            task_cache.get(specification, Control::unbounded()).await
        }));
    }
    barrier.wait().await;
    for task in tasks {
        let error = task
            .await
            .expect("waiter task")
            .expect_err("corrupt page must fail");
        assert_eq!(error.category(), ErrorCategory::Integrity);
    }
    wait_until(|| cache.metrics().active_loads == 0).await;
    let snapshot = cache.metrics();
    assert_eq!(snapshot.admissions, 0);
    assert_eq!(snapshot.page_pool_bytes, 0);
    assert_eq!(snapshot.inflight_bytes, 0);
    assert_eq!(snapshot.leases, 0);
    assert!(snapshot.physical_read_bytes >= u64::from(PAGE_SIZE));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_rejects_new_work_and_eventually_releases_inflight_bytes() {
    let fixture = open_fixture();
    let cache = cache_with(one_page_config());
    let _outcome = cache
        .prefetch(fixture.specs[0].clone(), Control::unbounded())
        .await
        .expect("prefetch admission");
    cache.shutdown();

    let error = cache
        .get(fixture.specs[1].clone(), Control::unbounded())
        .await
        .expect_err("shutdown cache");
    assert_eq!(error.category(), ErrorCategory::Shutdown);
    wait_until(|| cache.metrics().active_loads == 0).await;
    let snapshot = cache.metrics();
    assert_eq!(snapshot.inflight_bytes, 0);
    assert_eq!(snapshot.page_pool_bytes, 0);
}
